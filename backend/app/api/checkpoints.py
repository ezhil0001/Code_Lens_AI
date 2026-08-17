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


_GRAPH = None


async def _get_graph():
    """Compile the supervisor graph against the *async* Postgres checkpointer.

    The sync builder falls back to an in-memory saver when an event loop is
    running, which makes every persisted checkpoint invisible to replay,
    branch and resume.
    """
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH
    try:
        from app.graph.supervisor_graph import build_supervisor_graph
        saver = await _get_saver()
        if saver is None:
            return None
        _GRAPH = build_supervisor_graph(checkpointer=saver)
        return _GRAPH
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
                # Agent sub-graphs write their own checkpoints under a namespace
                # (e.g. "DocAgent:<uuid>"). Those ids are not addressable from
                # the root thread, so replay/branch on them 404s. Only the root
                # namespace represents a real conversation checkpoint.
                if configurable.get("checkpoint_ns"):
                    continue
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

    graph = await _get_graph()
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supervisor graph unavailable",
        )

    # Time travel must address the checkpoint on the thread that owns it.
    # Pointing at a fresh branch thread leaves LangGraph with no saved state,
    # which surfaces to the client as "Received no input for __start__".
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        },
        "recursion_limit": 25,
    }

    snapshot = await graph.aget_state(config)
    if snapshot is None or not snapshot.values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Checkpoint {checkpoint_id} not found for this session",
        )

    logger.info(
        "[replay] thread=%s from checkpoint=%s",
        thread_id, checkpoint_id
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

    graph = await _get_graph()
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supervisor graph unavailable",
        )

    # Read the historical state from the thread that owns the checkpoint. The
    # thread_id is namespaced to the caller, so another user's checkpoint is
    # simply not found rather than readable.
    source_config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": body.from_checkpoint_id,
        }
    }
    snapshot = await graph.aget_state(source_config)
    if snapshot is None or not snapshot.values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Checkpoint {body.from_checkpoint_id} not found for this session",
        )

    # Copy the state onto the branch thread so it becomes an independent,
    # continuable lineage. Without this the branch thread has no checkpoint and
    # any replay/resume against it fails with "Received no input for __start__".
    values = dict(snapshot.values)
    if body.new_query:
        values["query"] = body.new_query
        values["final_response"] = None
        logger.info(
            "[branch] new_query=%r applied to branch_thread=%s",
            body.new_query[:60], branch_thread_id
        )

    branch_config = {"configurable": {"thread_id": branch_thread_id}}
    await graph.aupdate_state(branch_config, values)

    branch_state = await graph.aget_state(branch_config)
    if branch_state is None or not branch_state.values:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Branch was created but no state could be persisted",
        )

    logger.info(
        "[branch] branch_thread=%s seeded from checkpoint=%s (%d state keys)",
        branch_thread_id, body.from_checkpoint_id, len(branch_state.values),
    )

    return BranchResponse(
        branch_session_id=branch_session_id,
        branch_thread_id=branch_thread_id,
        from_checkpoint_id=body.from_checkpoint_id,
        message=(
            f"Branch created from checkpoint {body.from_checkpoint_id[:8]} "
            f"with {len(branch_state.values)} state keys. "
            f"Use session_id='{branch_session_id}' to continue this branch."
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
    from langgraph.types import Command

    thread_id = build_thread_id(str(current_user.id), session_id)
    graph = await _get_graph()

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

    # Resume MUST run in the root checkpoint namespace. build_config sets
    # checkpoint_ns=<org_id|"default">, but LangGraph reads checkpoint_ns as a
    # SUBGRAPH name, so resuming with it raises "Subgraph default not found"
    # and the human decision is silently discarded.
    configurable = dict(config.get("configurable") or {})
    configurable["checkpoint_ns"] = ""
    config = {**config, "configurable": configurable}

    # A resume is only meaningful against a thread that is actually paused on a
    # human-review interrupt. Without this guard the endpoint answered 200 and
    # *started a fresh graph run* on the caller's own thread for any session
    # name: a second Approve click, an Approve racing a Reject, a resume after
    # the decision was already made, or another user probing someone else's
    # session id all silently burned LLM quota and wrote phantom checkpoints.
    # The thread is namespaced per user, so this also makes an unauthorised
    # attempt fail closed and indistinguishable from a nonexistent session.
    try:
        # Check the thread's LATEST state, not body.checkpoint_id. Pinning the
        # config to a historical checkpoint returns that past snapshot, whose
        # `tasks` are empty — the owner's own legitimate Approve would 409.
        latest_cfg = {**config, "configurable": {
            k: v for k, v in configurable.items() if k != "checkpoint_id"
        }}
        snapshot = await graph.aget_state(latest_cfg)
        pending = [
            i
            for task in (getattr(snapshot, "tasks", None) or [])
            for i in (getattr(task, "interrupts", None) or [])
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("[resume] could not read thread state: %s", exc)
        snapshot, pending = None, []

    if not pending:
        logger.info(
            "[resume] rejected — no pending interrupt for user=%s session=%s",
            current_user.id, session_id,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No pending human-review interrupt for this session",
        )

    # Official LangGraph resume for a dynamic interrupt(): Command(resume=...)
    # hands the payload straight back to the paused hil_check_node, which then
    # decides what happens next (refuse, or release the agents).
    #
    # Do NOT also call aupdate_state(as_node="hil_check_node") here. That
    # *replaces* the node's output instead of resuming it, so it satisfied the
    # interrupt before Command could: the node never ran, never produced its
    # refusal, and response_node emitted final_response of length 0 — the
    # browser showed an empty answer on reject. The human decision is still
    # durably recorded by _audit_hil_event below, and by the node itself
    # writing hil_approved/hil_human_input into the checkpoint on resume.
    resume_input = Command(resume={
        "approved": body.approved,
        "human_input": body.human_input,
    })

    logger.info(
        "[resume] user=%s session=%s approved=%s checkpoint=%s",
        current_user.id, session_id, body.approved, body.checkpoint_id,
    )

    # Continue the ORIGINAL trace instead of minting a new one. The interrupted
    # run persisted its trace id in state, so reusing it makes the reviewer's
    # decision and everything it unblocks land under the same Langfuse trace as
    # the question that triggered the review. Without this the UI showed two
    # unrelated traces and the HIL story was impossible to follow.
    values = getattr(snapshot, "values", None) or {}
    origin_trace_id = values.get("langfuse_trace_id")
    origin_root_span_id = values.get("langfuse_parent_span_id")
    from app.api.chat import _attach_langfuse
    from app.observability.tracing import open_request_root

    root = open_request_root(
        "chat.hil_resume",
        trace_id=origin_trace_id,
        user_id=str(current_user.id),
        session_id=session_id,
        tags=["chat", "hil", "hil-resume"],
        input=body.human_input,
        metadata={
            "request.source": "hil-resume",
            "hil.approved": body.approved,
            "hil.checkpoint_id": body.checkpoint_id,
            "hil.origin_trace_id": origin_trace_id,
        },
        parent_span_id=origin_root_span_id,
    )
    if origin_trace_id:
        configurable["langfuse_trace_id"] = origin_trace_id
        if root.span_id:
            configurable["langfuse_parent_span_id"] = root.span_id
        _attach_langfuse(
            config,
            user_id=str(current_user.id),
            session_id=session_id,
            org_id=getattr(current_user, "org_id", None),
            trace_id=origin_trace_id,
            parent_span_id=root.span_id,
        )
        config = {**config, "configurable": configurable}

    # Persist HIL audit event (best-effort; non-blocking)
    _audit_hil_event(
        user_id=str(current_user.id),
        session_id=session_id,
        approved=body.approved,
        human_input=body.human_input,
        checkpoint_id=body.checkpoint_id,
    )

    return StreamingResponse(
        _resumed_stream(graph, resume_input, config, root, origin_trace_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-HIL-Approved": str(body.approved).lower(),
        },
    )


async def _resumed_stream(graph, resume_input, config, root, trace_id):
    """Stream the resumed graph inside the original trace, closing the root.

    The root observation has to outlive the whole SSE stream, so it is ended in
    a finally here rather than by a context manager around the endpoint — the
    endpoint returns as soon as the response starts.
    """
    from app.graph.streaming import stream_graph_events
    from app.observability.tracing import request_trace

    try:
        with request_trace(trace_id, root.span_id):
            async for chunk in stream_graph_events(graph, resume_input, config):
                yield chunk
    finally:
        try:
            root.end()
        except Exception as exc:  # noqa: BLE001
            logger.debug("[resume] root span end failed: %s", exc)


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
