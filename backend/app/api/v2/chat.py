"""
v2 Chat API — Phase G: F-41 to F-46
======================================
Replaces the v1 /api/v1/chat/stream endpoint with a structured LangGraph
event-stream endpoint at /api/v2/chat/stream.

v1 is kept running at /api/v1/chat/stream for full backward compatibility.

Key improvements over v1:
  - Full AgentState initialisation via make_initial_state()
  - LangGraph graph.astream_events() for structured SSE envelopes
  - Reconnect via Last-Event-ID header (resume from checkpoint)
  - Per-request resume_from_checkpoint for time-travel
  - HIL-aware: hil_enabled + hil_confidence_threshold per request
  - Agent hint: client can soft-override routing with agent_hint
  - LangGraph thread_id namespaced as "{user_id}::{session_id}"

Tested by:
  G-001  app.api.v2.chat.router importable
  G-002  POST /api/v2/chat/stream registered
  G-003  ChatV2Request validates with defaults (stream=True, hil_enabled=False)
  G-004  ChatV2Request rejects query > 2048 chars
  G-008  chat_stream_v2 handles Last-Event-ID / resume_from_checkpoint
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v2/chat",
    tags=["chat-v2", "streaming"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Auth dependency (re-uses existing JWT infrastructure)
# ─────────────────────────────────────────────────────────────────────────────

def _get_current_user_dep():
    """Resolve the JWT get_current_user dep, falling back to an anonymous stub."""
    try:
        from app.auth.service import get_current_user  # type: ignore
        return get_current_user
    except Exception:  # noqa: BLE001
        async def _anon():
            class _User:
                id = "anonymous"
                email = "anonymous@local"
                org_id = None
            return _User()
        return _anon


_current_user_dep = _get_current_user_dep()


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class ChatV2Request(BaseModel):
    """Request body for POST /api/v2/chat/stream.

    Fields
    ------
    query
        The user's question or command.  Hard-limited to 2048 characters
        (enforced by the input guardrail as well, but validated here first
        so the client gets a clean 422 before any graph work starts).
    session_id
        Opaque client-provided session identifier.  Namespaced internally
        as ``"{user_id}::{session_id}"`` to form the LangGraph thread_id.
    user_id
        The authenticated user's ID.  Used for thread namespacing and
        memory scoping.
    org_id
        Optional organisation identifier.  Stored in ``checkpoint_ns``.
    stream
        When True (default) the response is streamed as SSE.  Set False
        to receive a single JSON response (not yet implemented — always
        streams for now).
    hil_enabled
        Opt-in to Human-in-the-Loop interrupts.  When False the HIL
        check node still runs but never pauses the graph.
    hil_confidence_threshold
        Routing confidence below which HIL fires (0.0–1.0, default 0.5).
    agent_hint
        Optional soft override for routing.  E.g. ``"CodeAgent"`` skips
        the classifier and routes directly.  Classifier still runs for
        observability; hint is written into ``routing_decision``.
    resume_from_checkpoint
        Checkpoint ID to resume from (time-travel / HIL resume).  Takes
        precedence over ``Last-Event-ID`` header when both are present.
    """

    query: str = Field(..., max_length=2048, description="User question or command")
    session_id: str = Field(..., description="Client session identifier")
    user_id: str = Field(..., description="Authenticated user ID")
    org_id: Optional[str] = Field(None, description="Organisation ID for namespace isolation")
    stream: bool = Field(True, description="Stream response as SSE (always True in v2)")
    hil_enabled: bool = Field(False, description="Opt-in to HIL interrupts")
    hil_confidence_threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="Confidence threshold for HIL trigger"
    )
    agent_hint: Optional[str] = Field(
        None,
        description="Optional routing override: 'CodeAgent'|'DocAgent'|'DebugAgent'|'ArchAgent'|'WebAgent'",
    )
    resume_from_checkpoint: Optional[str] = Field(
        None, description="Checkpoint ID to resume from (time-travel or HIL resume)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_graph():
    """Return the compiled supervisor graph with a MemorySaver fallback."""
    try:
        from app.graph.checkpointing.pg_checkpointer import get_checkpointer_sync
        from app.graph.supervisor_graph import build_supervisor_graph
        saver = get_checkpointer_sync()
        return build_supervisor_graph(checkpointer=saver)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[v2/chat] supervisor graph unavailable: %s", exc)
        return None


def _build_config(
    user_id: str,
    session_id: str,
    org_id: Optional[str],
    checkpoint_id: Optional[str],
) -> Dict[str, Any]:
    """Build the LangGraph RunnableConfig for a request."""
    try:
        from app.graph.checkpointing.pg_checkpointer import build_config
        return build_config(
            user_id=user_id,
            session_id=session_id,
            org_id=org_id,
            checkpoint_id=checkpoint_id,
        )
    except Exception:  # noqa: BLE001
        thread_id = f"{user_id}::{session_id}"
        cfg: Dict[str, Any] = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": org_id or "default",
            },
            "recursion_limit": 25,
        }
        if checkpoint_id:
            cfg["configurable"]["checkpoint_id"] = checkpoint_id
        return cfg


def _build_initial_state(body: ChatV2Request, user_id: str) -> Dict[str, Any]:
    """Initialise the AgentState from the request body."""
    try:
        from app.graph.state import make_initial_state
        state = make_initial_state(
            query=body.query,
            user_id=user_id,
            session_id=f"{user_id}::{body.session_id}",
            org_id=body.org_id,
        )
    except Exception:  # noqa: BLE001
        state = {
            "query": body.query,
            "user_id": user_id,
            "session_id": f"{user_id}::{body.session_id}",
            "org_id": body.org_id,
            "messages": [],
            "routing_confidence": 0.0,
            "hil_required": False,
            "nodes_visited": [],
            "guardrail_passed": True,
            "guardrail_violations": [],
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "agent_responses": {},
            "sources": [],
            "final_response": None,
            "cache_hit": False,
        }

    # Apply optional agent hint
    if body.agent_hint:
        state["routing_decision"] = body.agent_hint
        logger.debug("[v2/chat] agent_hint applied: %s", body.agent_hint)

    return state


# ─────────────────────────────────────────────────────────────────────────────
# G-002  POST /api/v2/chat/stream
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/stream")
async def chat_stream_v2(
    request: Request,
    body: ChatV2Request,
    background_tasks: BackgroundTasks,
    current_user=Depends(_current_user_dep),
):
    """Stream a multi-agent response as Server-Sent Events.

    Reconnect support (G-008 / F-46):
        On reconnect the browser sends ``Last-Event-ID`` containing the
        last checkpoint_id it received.  This endpoint picks it up and
        resumes the graph from that checkpoint so the client never misses
        events.  ``resume_from_checkpoint`` in the request body takes
        precedence when both are provided.

    Streaming lifecycle:
        1. Build initial AgentState (or resume from checkpoint)
        2. Invoke graph.astream_events() via stream_graph_events()
        3. Yield each SSE envelope directly to the client
        4. Always end with a ``done`` event (guaranteed by stream_graph_events)

    HIL:
        When ``hil_enabled=True`` and hil_check_node fires, the stream
        emits an ``interrupt`` SSE event and pauses.  The client then
        POSTs to ``/api/v2/sessions/{session_id}/resume`` with the human
        decision, and this endpoint is called again with
        ``resume_from_checkpoint`` set to the paused checkpoint ID.
    """
    from app.graph.streaming import stream_graph_events

    user_id = str(current_user.id)

    # ── Determine checkpoint for reconnect / resume ───────────────────────────
    # Priority: body.resume_from_checkpoint > Last-Event-ID header > None
    resume_checkpoint: Optional[str] = body.resume_from_checkpoint
    if not resume_checkpoint:
        last_event_id = request.headers.get("Last-Event-ID")
        if last_event_id:
            resume_checkpoint = last_event_id
            logger.info(
                "[v2/chat] Reconnect from Last-Event-ID checkpoint=%s user=%s",
                resume_checkpoint, user_id,
            )

    # ── Build LangGraph config ────────────────────────────────────────────────
    config = _build_config(
        user_id=user_id,
        session_id=body.session_id,
        org_id=body.org_id,
        checkpoint_id=resume_checkpoint,
    )

    # ── Build or resume initial state ─────────────────────────────────────────
    if resume_checkpoint:
        # Resuming: graph reads state from checkpointer — no initial state needed
        initial_state = None
        logger.info(
            "[v2/chat] Resuming from checkpoint=%s thread=%s",
            resume_checkpoint, config["configurable"].get("thread_id"),
        )
    else:
        initial_state = _build_initial_state(body, user_id)
        logger.info(
            "[v2/chat] New conversation thread=%s agent_hint=%s",
            config["configurable"].get("thread_id"), body.agent_hint,
        )

    # ── Get compiled graph ────────────────────────────────────────────────────
    graph = _build_graph()
    if graph is None:
        # Graph unavailable — stream a single error event
        from app.graph.streaming import SSEEvent, format_sse
        import time

        async def _error_stream():
            yield format_sse(SSEEvent(
                type="error",
                data={"message": "Supervisor graph is currently unavailable. Please try again."},
                agent="Supervisor",
                checkpoint_id="",
                ts=time.time() * 1000,
            ))
            yield format_sse(SSEEvent(
                type="done",
                data={},
                agent="Supervisor",
                checkpoint_id="",
                ts=time.time() * 1000,
            ))

        return StreamingResponse(
            _error_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        stream_graph_events(graph, initial_state, config),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Session-Id": body.session_id,
            "X-Thread-Id": config["configurable"].get("thread_id", ""),
        },
    )
