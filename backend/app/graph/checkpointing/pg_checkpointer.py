"""
Graph checkpointer — wraps AsyncPostgresSaver with a MemorySaver fallback.

LangGraph persists the full AgentState after every node via the checkpointer.
This enables two things: resuming a paused graph after a HIL interrupt, and
time-travel replay from any historical checkpoint.

The factory returns AsyncPostgresSaver when Postgres is reachable and falls
back to an in-process MemorySaver when it is not.  The fallback lets the
server start and serve requests without a running database — checkpoints are
lost on restart but the conversation still works.

build_thread_id() and build_branch_thread_id() enforce the same
"{user_id}::{session_id}" namespace used by the memory layer so checkpoint
keys and memory keys are always co-located per user.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ── Process-wide singleton ────────────────────────────────────────────────────
_saver = None
_saver_lock = threading.Lock()
_setup_done = False
_async_pool = None  # dedicated AsyncConnectionPool for the checkpointer


# ─────────────────────────────────────────────────────────────────────────────
# Thread-ID helpers (tested by D-008 and D-009)
# ─────────────────────────────────────────────────────────────────────────────

def build_thread_id(user_id: str, session_id: str) -> str:
    """Return the namespaced LangGraph thread key: '{user_id}::{session_id}'.

    Mirrors the session namespace used by the STM layer so that graph
    checkpoints and conversation history are always co-located under the
    same key.
    """
    # Idempotent: if session_id already contains the namespace, return as-is
    if "::" in session_id and session_id.startswith(f"{user_id}::"):
        return session_id
    return f"{user_id}::{session_id}"


def build_branch_thread_id(parent_thread_id: str, checkpoint_id: str) -> str:
    """Return a branch thread ID derived from a historical checkpoint.

    Format: '{parent_thread_id}::branch::{checkpoint_id[:8]}'

    The short checkpoint prefix is enough to identify the branch origin
    while keeping the key at a manageable length for Postgres indexes.
    """
    short_cp = checkpoint_id[:8] if len(checkpoint_id) >= 8 else checkpoint_id
    return f"{parent_thread_id}::branch::{short_cp}"


def build_config(
    user_id: str,
    session_id: str,
    org_id: Optional[str] = None,
    checkpoint_id: Optional[str] = None,
) -> dict:
    """Build a LangGraph RunnableConfig dict with all required configurable keys.

    Args:
        user_id:        The authenticated user's ID.
        session_id:     Raw or already-namespaced session identifier.
        org_id:         Optional organisation scope (used as checkpoint_ns).
        checkpoint_id:  If set, resume execution from this checkpoint.

    Returns:
        Dict suitable for passing as the `config` argument to graph.ainvoke /
        graph.astream_events / graph.aupdate_state.
    """
    thread_id = build_thread_id(user_id, session_id)
    configurable: dict = {
        "thread_id": thread_id,
        "checkpoint_ns": org_id or "default",
    }
    if checkpoint_id:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable, "recursion_limit": 25}


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointer factory
# ─────────────────────────────────────────────────────────────────────────────

async def get_checkpointer():
    """Return the process-wide LangGraph checkpointer singleton.

    Preference order:
      1. AsyncPostgresSaver (backed by shared pg_pool from database.py).
      2. MemorySaver (in-process fallback when Postgres is unreachable).

    The saver is initialised exactly once; subsequent calls return the cached
    instance.  `setup()` (which creates the checkpoints table) is called on
    first creation — it is idempotent and safe to re-run.

    Returns:
        A LangGraph checkpoint saver object.  Always non-None.
    """
    global _saver, _setup_done, _async_pool

    if _saver is not None:
        return _saver

    with _saver_lock:
        if _saver is not None:
            return _saver

        # ── Try AsyncPostgresSaver (production path) ──────────────────────────
        # LangGraph's AsyncPostgresSaver requires an *async* psycopg connection
        # or AsyncConnectionPool — the shared psycopg_pool.ConnectionPool is
        # SYNC and passing it raises "Invalid connection type: ConnectionPool".
        # We therefore open a small dedicated AsyncConnectionPool here. Because
        # the graph is later driven by astream_events (async), an async saver is
        # mandatory; the previous sync-pool path silently fell back to an
        # ephemeral MemorySaver, so Postgres checkpointing never actually ran.
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # type: ignore
            from psycopg_pool import AsyncConnectionPool  # type: ignore
            from app.core.database import build_psycopg_dsn

            dsn = build_psycopg_dsn()
            # autocommit + no prepared statements is what LangGraph documents
            # for its Postgres savers.
            _async_pool = AsyncConnectionPool(
                conninfo=dsn,
                max_size=int(os.getenv("CHECKPOINTER_POOL_MAX", "4")),
                open=False,
                kwargs={"autocommit": True, "prepare_threshold": 0},
            )
            await _async_pool.open(wait=True, timeout=10.0)

            saver = AsyncPostgresSaver(_async_pool)  # type: ignore[arg-type]

            if not _setup_done:
                await saver.setup()  # idempotent — creates checkpoints tables
                _setup_done = True
                logger.info("[checkpointer] AsyncPostgresSaver setup complete")

            _saver = saver
            logger.info("[checkpointer] Using AsyncPostgresSaver ✓")
            return _saver

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[checkpointer] AsyncPostgresSaver unavailable (%s) — "
                "falling back to MemorySaver", exc
            )
            # Tear down a half-open pool so we don't leak connections.
            if _async_pool is not None:
                try:
                    await _async_pool.close()
                except Exception:  # noqa: BLE001
                    pass
                _async_pool = None

        # ── MemorySaver fallback ──────────────────────────────────────────────
        from langgraph.checkpoint.memory import MemorySaver  # type: ignore
        _saver = MemorySaver()
        logger.info("[checkpointer] Using MemorySaver (in-process fallback) ✓")
        return _saver


def get_checkpointer_sync():
    """Synchronous wrapper around get_checkpointer() for non-async contexts.

    Runs get_checkpointer() in the current or a new event loop.
    Prefer the async version wherever possible.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Cannot run a coroutine in an already-running loop from sync context.
            # Return MemorySaver as safe default.
            from langgraph.checkpoint.memory import MemorySaver  # type: ignore
            return MemorySaver()
        return loop.run_until_complete(get_checkpointer())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(get_checkpointer())
        finally:
            loop.close()


async def close_checkpointer() -> None:
    """Close the dedicated async checkpointer pool on application shutdown.

    Prevents connection leaks (the AsyncConnectionPool holds live Postgres
    sockets). Safe to call when no pool was ever opened.
    """
    global _async_pool, _saver, _setup_done
    if _async_pool is not None:
        try:
            await _async_pool.close()
            logger.info("[checkpointer] async pool closed")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[checkpointer] async pool close failed: %s", exc)
        finally:
            _async_pool = None
            _saver = None
            _setup_done = False

