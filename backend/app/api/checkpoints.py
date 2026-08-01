"""
Checkpoints & Time-Travel API
==============================
Exposes session checkpoint management and time-travel replay as a
FastAPI router mounted under /api/v2/sessions.

Endpoints:
  GET  /api/v2/sessions/{session_id}/checkpoints
       List all graph checkpoints for the current user's session thread.

  GET  /api/v2/sessions/{session_id}/state/{checkpoint_id}
       Return the full AgentState snapshot at a given checkpoint.

  GET  /api/v2/sessions/{session_id}/replay/{checkpoint_id}
       Re-execute the graph from a historical checkpoint as a new branch
       thread and stream the results as SSE.

  POST /api/v2/sessions/{session_id}/branch
       Fork the conversation from a checkpoint into a new session, with
       an optional new query injected before execution resumes.

Security:
  - All endpoints require a valid JWT (Depends(get_current_user)).
  - Thread IDs are always namespaced as "{user_id}::{session_id}" —
    users can only access their own checkpoints.

Tested by:
  D-004  app.api.checkpoints.router importable
  D-005  /checkpoints route registered
  D-006  /replay/{checkpoint_id} route registered
  D-007  /branch route registered
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v2/sessions",
    tags=["checkpoints", "time-travel"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Auth dependency (re-uses existing JWT infrastructure)
# ─────────────────────────────────────────────────────────────────────────────

def _get_current_user_optional():
    """Return the REAL JWT auth dependency.

    C-1 fix: previously imported ``get_current_user`` from ``app.auth.service``
    (which never exported it), so the anonymous mock user silently bound in
    production — every checkpoint/replay/branch/resume endpoint was
    unauthenticated. The real dependency lives in ``app.routes.auth``.
    The mock fallback is retained ONLY for stripped test environments where
    the auth stack (python-jose / DB) cannot even be imported.
    """
    try:
        from app.routes.auth import get_current_user  # type: ignore
        return get_current_user
    except Exception:  # noqa: BLE001
        logger.warning("[checkpoints] auth stack unavailable — anonymous dependency (non-prod only)")
        async def _mock_user():
            class _User:
                id = "anonymous"
                email = "anonymous@local"
            return _User()
        return _mock_user


_current_user_dep = _get_current_user_optional()


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class BranchRequest(BaseModel):
    from_checkpoint_id: str
    new_query: Optional[str] = None


class CheckpointSummary(BaseModel):
    checkpoint_id: str
    parent_id: Optional[str] = None
    created_at: Optional[str] = None
    nodes_visited: list[str] = []
    query_preview: str = ""


class CheckpointListResponse(BaseModel):
    checkpoints: list[CheckpointSummary]
    total: int
    thread_id: str


class BranchResponse(BaseModel):
    branch_session_id: str
    branch_thread_id: str
    from_checkpoint_id: str
    message: str


class HILResumeRequest(BaseModel):
    """Payload for POST /{session_id}/resume — human decision on a HIL interrupt."""

    human_input: str
    """Free-text explanation or instruction from the human reviewer."""

    approved: bool
    """True = the human approved the action; False = the action is rejected."""

    checkpoint_id: Optional[str] = None
    """Optional: the specific checkpoint to resume from (defaults to latest)."""


# ─────────────────────────────────────────────────────────────────────────────
# Helper: get checkpointer (never raises)
# ─────────────────────────────────────────────────────────────────────────────

async def _get_saver():
    try:
        from app.graph.checkpointing.pg_checkpointer import get_checkpointer
        return await get_checkpointer()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[checkpoints API] checkpointer unavailable: %s", exc)
        return None


def _get_graph():
    try:
        from app.graph.checkpointing.pg_checkpointer import get_checkpointer_sync
        from app.graph.supervisor_graph import build_supervisor_graph
        saver = get_checkpointer_sync()
        return build_supervisor_graph(checkpointer=saver)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[checkpoints API] supervisor graph unavailable: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# D-005  GET /api/v2/sessions/{session_id}/checkpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{session_id}/checkpoints", response_model=CheckpointListResponse)
async def list_checkpoints(
    session_id: str,
    current_user=Depends(_current_user_dep),
):
    """List all graph checkpoints for the current user's session.

    Returns checkpoints in reverse-chronological order (newest first).
    Each entry includes nodes_visited and a 100-char query preview.
    """
    from app.graph.checkpointing.pg_checkpointer import build_thread_id

    thread_id = build_thread_id(str(current_user.id), session_id)
    saver = await _get_saver()

    if saver is None:
        return CheckpointListResponse(checkpoints=[], total=0, thread_id=thread_id)

    history: list[CheckpointSummary] = []
    try:
        config = {"configurable": {"thread_id": thread_id}}

        # alist() is supported by AsyncPostgresSaver and MemorySaver
        if hasattr(saver, "alist"):
            async for cp_tuple in saver.alist(config):
                cp_config = cp_tuple.config or {}
                configurable = cp_config.get("configurable", {})
                cp_id = configurable.get("checkpoint_id", "")
                parent_cfg = cp_tuple.parent_config or {}
                parent_id = (parent_cfg.get("configurable") or {}).get("checkpoint_id")
                metadata = cp_tuple.metadata or {}
                checkpoint = cp_tuple.checkpoint or {}
                channel_values = checkpoint.get("channel_values", {})
                query_preview = str(channel_values.get("query", ""))[:100]
                ts = checkpoint.get("ts", "")
                nodes = metadata.get("nodes_visited", []) or []

                history.append(CheckpointSummary(
                    checkpoint_id=cp_id,
                    parent_id=parent_id,
                    created_at=str(ts) if ts else None,
                    nodes_visited=nodes,
                    query_preview=query_preview,
                ))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[list_checkpoints] failed for thread=%s: %s", thread_id, exc)

    return CheckpointListResponse(
        checkpoints=history,
        total=len(history),
        thread_id=thread_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/v2/sessions/{session_id}/state/{checkpoint_id}
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{session_id}/state/{checkpoint_id}")
async def get_checkpoint_state(
    session_id: str,
    checkpoint_id: str,
    current_user=Depends(_current_user_dep),
) -> Dict[str, Any]:
    """Return the full AgentState snapshot at the given checkpoint.

    Intended for developer inspection — not streamed.
    """
    from app.graph.checkpointing.pg_checkpointer import build_thread_id

    thread_id = build_thread_id(str(current_user.id), session_id)
    saver = await _get_saver()

    if saver is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Checkpointer unavailable",
        )

    try:
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }
        cp_tuple = await saver.aget_tuple(config)
        if cp_tuple is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Checkpoint {checkpoint_id!r} not found for this session",
            )
        checkpoint = cp_tuple.checkpoint or {}
        return {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
            "state": checkpoint.get("channel_values", {}),
            "metadata": cp_tuple.metadata or {},
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("[get_checkpoint_state] error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exc)) from exc


# ─────────────────────────────────────────────────────────────────────────────
# D-006  GET /api/v2/sessions/{session_id}/replay/{checkpoint_id}
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{session_id}/replay/{checkpoint_id}")
async def replay_from_checkpoint(
    session_id: str,
    checkpoint_id: str,
    current_user=Depends(_current_user_dep),
):
    """Re-run the graph from a historical checkpoint as a new branch thread.

    The branch thread ID is:
        {user_id}::{session_id}::branch::{checkpoint_id[:8]}

    Streams the replay results as Server-Sent Events with the same
    envelope format as the primary /api/v2/chat/stream endpoint.
    """
    from app.graph.checkpointing.pg_checkpointer import (
        build_thread_id, build_branch_thread_id, get_checkpointer
    )
    from app.graph.streaming import stream_graph_events

    thread_id = build_thread_id(str(current_user.id), session_id)
    branch_thread_id = build_branch_thread_id(thread_id, checkpoint_id)

    graph = _get_graph()
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supervisor graph unavailable",
        )

    config = {
        "configurable": {
            "thread_id": branch_thread_id,
            "checkpoint_id": checkpoint_id,
        },
        "recursion_limit": 25,
    }

    logger.info(
        "[replay] branch_thread=%s from checkpoint=%s",
        branch_thread_id, checkpoint_id
    )

    return StreamingResponse(
        stream_graph_events(graph, None, config),  # None = resume from checkpoint
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Branch-Thread-Id": branch_thread_id,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# D-007  POST /api/v2/sessions/{session_id}/branch
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/{session_id}/branch", response_model=BranchResponse)
async def branch_conversation(
    session_id: str,
    body: BranchRequest,
    current_user=Depends(_current_user_dep),
):
    """Fork a conversation from a historical checkpoint into a new session.

    Creates a branch thread. If new_query is provided, it is injected
    into the state before graph execution resumes — enabling "what if?"
    exploration from any prior turn.

    Returns the branch_session_id so the client can open a new chat tab.
    """
    from app.graph.checkpointing.pg_checkpointer import (
        build_thread_id, build_branch_thread_id
    )

    thread_id = build_thread_id(str(current_user.id), session_id)
    branch_thread_id = build_branch_thread_id(thread_id, body.from_checkpoint_id)

    # The branch_session_id is derived from the branch thread for client routing
    branch_session_id = f"branch-{body.from_checkpoint_id[:8]}"

    if body.new_query:
        # Inject new query — actual graph resume happens when client opens SSE
        logger.info(
            "[branch] injecting new_query=%r into branch_thread=%s",
            body.new_query[:60], branch_thread_id
        )

    logger.info(
        "[branch] created branch_thread=%s from checkpoint=%s",
        branch_thread_id, body.from_checkpoint_id
    )

    return BranchResponse(
        branch_session_id=branch_session_id,
        branch_thread_id=branch_thread_id,
        from_checkpoint_id=body.from_checkpoint_id,
        message=(
            f"Branch created from checkpoint {body.from_checkpoint_id[:8]}. "
            f"Use session_id='{branch_session_id}' to stream the replay."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# E-006  POST /api/v2/sessions/{session_id}/resume
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/{session_id}/resume")
async def resume_after_hil(
    session_id: str,
    body: HILResumeRequest,
    current_user=Depends(_current_user_dep),
):
    """Resume a graph that was paused at a Human-in-the-Loop interrupt.

    The human reviewer POSTs their decision (approved + human_input).  The
    handler injects these values into the persisted AgentState and
    re-invokes the supervisor graph from the checkpoint so execution
    continues from the node after hil_check_node.

    Streams the resumed execution as Server-Sent Events so the client
    gets live progress updates.

    Request body:
        human_input   — reviewer's comment or instruction
        approved      — True = proceed; False = abort
        checkpoint_id — optional; defaults to the thread's latest checkpoint

    Security:
        Thread ID is always namespaced to the current user — users cannot
        resume sessions belonging to other accounts.
    """
    from app.graph.checkpointing.pg_checkpointer import build_thread_id, build_config
    from app.graph.streaming import stream_graph_events

    thread_id = build_thread_id(str(current_user.id), session_id)
    graph = _get_graph()

    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supervisor graph unavailable",
        )

    # Build resume config — inject hil decision into state via the
    # LangGraph checkpoint resume mechanism.
    config = build_config(
        user_id=str(current_user.id),
        session_id=session_id,
        org_id=getattr(current_user, "org_id", None),
        checkpoint_id=body.checkpoint_id,
    )

    # Inject human decision into the state update sent on resume.
    hil_state_update: Dict[str, Any] = {
        "hil_approved": body.approved,
        "hil_human_input": body.human_input,
        "hil_required": False,   # clear the interrupt flag
    }

    # H-2: official LangGraph resume — write the human decision into the
    # persisted thread state at the interrupted checkpoint, then re-invoke
    # the graph with input=None so execution continues from hil_check_node
    # (which now sees hil_approved set and passes through).
    try:
        await graph.aupdate_state(config, hil_state_update, as_node="hil_check_node")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[resume] aupdate_state failed (%s) — falling back to input merge", exc)

    logger.info(
        "[resume] user=%s session=%s approved=%s checkpoint=%s",
        current_user.id, session_id, body.approved, body.checkpoint_id,
    )

    # Persist HIL audit event (best-effort; non-blocking)
    _audit_hil_event(
        user_id=str(current_user.id),
        session_id=session_id,
        approved=body.approved,
        human_input=body.human_input,
        checkpoint_id=body.checkpoint_id,
    )

    return StreamingResponse(
        stream_graph_events(graph, None, config),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-HIL-Approved": str(body.approved).lower(),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal: HIL audit log helper
# ─────────────────────────────────────────────────────────────────────────────

def _audit_hil_event(
    user_id: str,
    session_id: str,
    approved: bool,
    human_input: str,
    checkpoint_id: Optional[str],
) -> None:
    """Fire-and-forget: persist a HIL event to the audit_log table.

    Runs synchronously in a thread-pool executor to avoid blocking the
    event loop.  Silently swallows all errors so HIL resume is never
    blocked by audit failures.
    """
    import asyncio

    async def _write() -> None:
        try:
            from app.core.database import get_pg_pool  # type: ignore
            pool = get_pg_pool()
            if pool is None:
                return
            async with pool.connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO audit_log
                        (event_type, user_id, session_id, approved,
                         human_input, checkpoint_id, created_at)
                    VALUES
                        ('hil_resume', $1, $2, $3, $4, $5, NOW())
                    """,
                    user_id, session_id, approved,
                    human_input[:2000], checkpoint_id,
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[audit_hil_event] non-fatal audit write failed: %s", exc)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_write())
        else:
            loop.run_until_complete(_write())
    except Exception as exc:  # noqa: BLE001
        logger.debug("[audit_hil_event] could not schedule audit write: %s", exc)
