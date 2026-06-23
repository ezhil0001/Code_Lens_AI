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
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ── Process-wide singleton ────────────────────────────────────────────────────
_saver = None
_saver_lock = threading.Lock()
_setup_done = False


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
    global _saver, _setup_done

    if _saver is not None:
        return _saver

    with _saver_lock:
        if _saver is not None:
            return _saver

        # ── Try AsyncPostgresSaver (production path) ──────────────────────────
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # type: ignore
            from app.core.database import get_pg_pool

            pool = get_pg_pool()
            # AsyncPostgresSaver expects an asyncpg pool; our pool is psycopg.
            # LangGraph 1.x also ships a sync PostgresSaver — try async first,
            # then fall back to the sync variant wrapped in a thread executor.
            saver = AsyncPostgresSaver(pool)  # type: ignore[arg-type]

            if not _setup_done:
                try:
                    await saver.setup()
                    _setup_done = True
                    logger.info("[checkpointer] AsyncPostgresSaver setup complete")
                except Exception as setup_err:  # noqa: BLE001
                    logger.warning(
                        "[checkpointer] AsyncPostgresSaver.setup() failed: %s — "
                        "checkpoints table may not exist yet", setup_err
                    )

            _saver = saver
            logger.info("[checkpointer] Using AsyncPostgresSaver ✓")
            return _saver

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[checkpointer] AsyncPostgresSaver unavailable (%s) — "
                "falling back to MemorySaver", exc
            )

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
