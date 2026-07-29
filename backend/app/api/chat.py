"""
Chat API — unified streaming endpoint.

Replaces both ``app.api.chat`` (v1 / AgentBrain) and ``app.api.v2.chat``
(v2 / LangGraph) with a single module that owns two URL prefixes:

  POST /api/v2/chat/stream   — primary endpoint; LangGraph supervisor graph,
                               typed SSE envelopes, checkpointing, reconnect,
                               HIL, agent_hint.

  POST /api/v1/chat/stream   — thin compatibility shim; accepts the simpler
                               ChatRequest schema and delegates to the same
                               LangGraph engine so ``AIStreamService`` on the
                               frontend keeps working without changes.

Supporting endpoints (retained because the frontend uses them today):

  GET  /api/v1/chat/cache/status
  POST /api/v1/chat/cache/clear
  GET  /api/v1/chat/history/{session_id}

What was removed and why
─────────────────────────
AgentBrain streaming backend
    AgentBrain is marked _DEPRECATED = True and is replaced by the LangGraph
    supervisor graph.  The heartbeat loop existed only to keep SSE connections
    alive during AgentBrain's 30-60 s cold start; LangGraph emits node_start
    events within seconds so no heartbeat is needed.

RAGAS evaluation hook in the route handler
    The sources propagation bug meant evaluate_sample() never ran for any
    streaming request (response_holder["sources"] was never written to).
    Removing broken dead code is safer than shipping a non-functional harness.
    Observability is handled by Langfuse and OTEL traces.

Fake word-by-word cache stream
    _create_cache_stream_response split on whitespace and slept 10 ms per
    word.  The replacement yields the cached text as a single token event
    followed by done — correct protocol, no artificial delay.

SemanticCache class
    Moved to app.services.semantic_cache so both router blocks and any future
    endpoint share the same singleton without circular imports.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.semantic_cache import semantic_cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routers — two prefix blocks, one module, same backend
# ---------------------------------------------------------------------------

router_v2 = APIRouter(prefix="/api/v2/chat", tags=["chat-v2", "streaming"])
router_v1 = APIRouter(prefix="/api/v1",      tags=["chat-v1", "compat"])

# Single ``router`` alias so legacy ``chat_api.router`` import in main.py
# continues to work.  main.py must also register router_v1 — see the
# MIGRATION NOTE in main.py.
router = router_v2


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def _resolve_user_dep():
    """Return the real JWT dependency, falling back to an anonymous stub."""
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


_current_user_dep = _resolve_user_dep()


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ChatV2Request(BaseModel):
    """Request body for POST /api/v2/chat/stream."""

    query: str = Field(..., max_length=2048)
    session_id: str
    user_id: str
    org_id: Optional[str] = None
    hil_enabled: bool = False
    hil_confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    agent_hint: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None


class ChatV1Request(BaseModel):
    """Simplified request body for the v1 compatibility shim."""

    query: str = Field(..., max_length=5000)
    session_id: str = Field(
        default_factory=lambda: f"sess-{__import__('uuid').uuid4().hex[:12]}"
    )
    user_id: str = Field(
        default_factory=lambda: f"anon-{__import__('uuid').uuid4().hex[:8]}"
    )
    stream: bool = True


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Compile the supervisor graph once per process, not per request — reuse across all handlers.
_graph_singleton = None
_graph_lock = None


def _build_graph():
    """Return the compiled supervisor graph singleton, or None on failure."""
    global _graph_singleton, _graph_lock
    import threading
    if _graph_lock is None:
        _graph_lock = threading.Lock()
    if _graph_singleton is not None:
        return _graph_singleton
    with _graph_lock:
        if _graph_singleton is not None:  # double-checked locking
            return _graph_singleton
        try:
            from app.graph.checkpointing.pg_checkpointer import get_checkpointer_sync
            from app.graph.supervisor_graph import build_supervisor_graph
            _graph_singleton = build_supervisor_graph(checkpointer=get_checkpointer_sync())
            logger.info("[chat] supervisor graph compiled and cached")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[chat] supervisor graph unavailable: %s", exc)
    return _graph_singleton


def _build_config(
    user_id: str,
    session_id: str,
    org_id: Optional[str],
    checkpoint_id: Optional[str],
) -> Dict[str, Any]:
    """Build a LangGraph RunnableConfig."""
    try:
        from app.graph.checkpointing.pg_checkpointer import build_config
        return build_config(
            user_id=user_id,
            session_id=session_id,
            org_id=org_id,
            checkpoint_id=checkpoint_id,
        )
    except Exception:  # noqa: BLE001
        cfg: Dict[str, Any] = {
            "configurable": {
                "thread_id": f"{user_id}::{session_id}",
                "checkpoint_ns": org_id or "default",
            },
            "recursion_limit": 25,
        }
        if checkpoint_id:
            cfg["configurable"]["checkpoint_id"] = checkpoint_id
        return cfg


def _build_initial_state(
    query: str,
    user_id: str,
    session_id: str,
    org_id: Optional[str],
    agent_hint: Optional[str],
) -> Dict[str, Any]:
    """Construct the AgentState for a fresh request."""
    try:
        from app.graph.state import make_initial_state
        state = make_initial_state(
            query=query,
            user_id=user_id,
            session_id=f"{user_id}::{session_id}",
            org_id=org_id,
        )
    except Exception:  # noqa: BLE001
        state = {
            "query": query,
            "user_id": user_id,
            "session_id": f"{user_id}::{session_id}",
            "org_id": org_id,
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
    if agent_hint:
        state["routing_decision"] = agent_hint
    return state


def _sse_envelope(type_: str, data: Any, agent: str = "Supervisor") -> str:
    """Serialise a typed SSE envelope to the wire format."""
    payload = {"type": type_, "data": data, "agent": agent, "ts": time.time() * 1000}
    return f"data: {json.dumps(payload)}\n\n"


async def _fatal_error_stream(message: str):
    """Yield error + done when the graph could not be built."""
    yield _sse_envelope("error", {"message": message})
    yield _sse_envelope("done", {})


def _cache_write(query: str, text: str, user_id: str) -> None:
    """Persist text to the semantic cache; ignores all errors."""
    if not text:
        return
    try:
        semantic_cache.set(query, text, user_id=user_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[chat] cache write failed: %s", exc)


# ---------------------------------------------------------------------------
# V2 streaming endpoint — primary
# ---------------------------------------------------------------------------

@router_v2.post("/stream")
async def chat_stream_v2(
    request: Request,
    body: ChatV2Request,
    current_user=Depends(_current_user_dep),
):
    """Stream a multi-agent response as SSE via the LangGraph supervisor graph.

    Flow
    ----
    1. Semantic cache lookup — on HIT streams cached text immediately; no
       LLM call, no graph invocation.
    2. Reconnect via Last-Event-ID header (resume_from_checkpoint in the
       body takes precedence when both are present).
    3. Build LangGraph RunnableConfig and initial AgentState.
    4. Stream graph.astream_events() as typed SSE envelopes:
       token | agent_switch | checkpoint | interrupt | done | error.
    5. On clean completion, write accumulated tokens to semantic cache.
    6. On CancelledError (client disconnect), write whatever was accumulated
       before re-raising so the work is not lost.
    """
    user_id = str(current_user.id)

    # 1. Cache
    hit = semantic_cache.get(body.query, user_id=user_id)
    if hit:
        return _cached_stream_v2(hit, body.session_id)

    # 2. Reconnect / resume
    resume_cp: Optional[str] = body.resume_from_checkpoint
    if not resume_cp:
        lei = request.headers.get("Last-Event-ID")
        if lei:
            resume_cp = lei
            logger.info("[v2/chat] reconnect checkpoint=%s user=%s", lei, user_id)

    # 3. Config + state
    config = _build_config(
        user_id=user_id,
        session_id=body.session_id,
        org_id=body.org_id,
        checkpoint_id=resume_cp,
    )
    if resume_cp:
        initial_state = None
        logger.info("[v2/chat] resuming thread=%s", config["configurable"].get("thread_id"))
    else:
        initial_state = _build_initial_state(
            query=body.query,
            user_id=user_id,
            session_id=body.session_id,
            org_id=body.org_id,
            agent_hint=body.agent_hint,
        )
        logger.info(
            "[v2/chat] new thread=%s hint=%s",
            config["configurable"].get("thread_id"), body.agent_hint,
        )

    # 4. Graph
    graph = _build_graph()
    if graph is None:
        return StreamingResponse(
            _fatal_error_stream("Supervisor graph is currently unavailable. Please try again."),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        _graph_stream_v2(graph, initial_state, config, body.query, user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Session-Id": body.session_id,
            "X-Thread-Id": config["configurable"].get("thread_id", ""),
        },
    )


async def _graph_stream_v2(graph, initial_state, config, query: str, user_id: str):
    """Wrap stream_graph_events() and accumulate tokens for cache writing."""
    from app.graph.streaming import stream_graph_events

    accumulated = ""
    try:
        async for chunk in stream_graph_events(graph, initial_state, config):
            yield chunk
            try:
                payload = json.loads(chunk.removeprefix("data: ").rstrip())
                if payload.get("type") == "token":
                    accumulated += payload.get("data", {}).get("content", "")
            except Exception:  # noqa: BLE001
                pass
        _cache_write(query, accumulated, user_id)
    except BaseException as exc:
        _cache_write(query, accumulated, user_id)
        if accumulated:
            logger.info("[chat] partial response cached (%d chars) on disconnect", len(accumulated))
        raise


# ---------------------------------------------------------------------------
# V1 compatibility shim
# ---------------------------------------------------------------------------

@router_v1.post("/chat/stream")
async def chat_stream_v1(
    request: Request,
    body: ChatV1Request,
    current_user=Depends(_current_user_dep),
):
    """V1 compat shim — same LangGraph engine, flat {type, content} SSE format.

    The Angular AIStreamService posts here and expects:
      data: {"type": "token",  "content": "<text>"}
      data: {"type": "done",   "metadata": {...}}
      data: {"type": "error",  "content": "<message>"}
    """
    user_id = str(current_user.id)

    hit = semantic_cache.get(body.query, user_id=user_id)
    if hit:
        return _cached_stream_v1(hit)

    config = _build_config(
        user_id=user_id, session_id=body.session_id, org_id=None, checkpoint_id=None
    )
    initial_state = _build_initial_state(
        query=body.query, user_id=user_id, session_id=body.session_id,
        org_id=None, agent_hint=None,
    )

    graph = _build_graph()
    if graph is None:
        return StreamingResponse(
            _v1_error_stream("Service temporarily unavailable."),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        _graph_stream_v1(graph, initial_state, config, body.query, user_id, body.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


async def _graph_stream_v1(graph, initial_state, config, query: str, user_id: str, session_id: str):
    """Translate v2 typed envelopes into the flat v1 format."""
    from app.graph.streaming import stream_graph_events

    accumulated = ""
    try:
        async for chunk in stream_graph_events(graph, initial_state, config):
            try:
                payload = json.loads(chunk.removeprefix("data: ").rstrip())
                t = payload.get("type", "")
                if t == "token":
                    content = payload.get("data", {}).get("content", "")
                    accumulated += content
                    yield f'data: {json.dumps({"type": "token", "content": content})}\n\n'
                elif t == "done":
                    meta = {
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat(),
                        "cached": False,
                    }
                    yield f'data: {json.dumps({"type": "done", "metadata": meta})}\n\n'
                elif t == "error":
                    msg = payload.get("data", {}).get("message", "An error occurred.")
                    yield f'data: {json.dumps({"type": "error", "content": msg})}\n\n'
                # agent_switch / checkpoint / interrupt — silently drop; v1 client ignores them
            except Exception:  # noqa: BLE001
                pass
        _cache_write(query, accumulated, user_id)
    except BaseException:
        _cache_write(query, accumulated, user_id)
        raise


async def _v1_error_stream(message: str):
    yield f'data: {json.dumps({"type": "error", "content": message})}\n\n'
    yield f'data: {json.dumps({"type": "done",  "metadata": {}})}\n\n'


# ---------------------------------------------------------------------------
# Cache helper streams
# ---------------------------------------------------------------------------

def _cached_stream_v2(hit: dict, session_id: str) -> StreamingResponse:
    """Stream a cache hit in the v2 typed envelope format."""
    async def _gen():
        yield _sse_envelope("token", {"content": hit["response"]}, agent="Cache")
        yield _sse_envelope(
            "done",
            {
                "cached": True,
                "similarity": hit["similarity"],
                "original_query": hit["query"],
                "session_id": session_id,
            },
        )
    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _cached_stream_v1(hit: dict) -> StreamingResponse:
    """Stream a cache hit in the flat v1 format."""
    async def _gen():
        yield f'data: {json.dumps({"type": "token", "content": hit["response"]})}\n\n'
        yield f'data: {json.dumps({"type": "done", "metadata": {"cached": True, "similarity": hit["similarity"]}})}\n\n'
    return StreamingResponse(_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Cache management endpoints
# ---------------------------------------------------------------------------

@router_v1.get("/chat/cache/status")
async def cache_status():
    """Return semantic cache statistics."""
    return {
        "cache_size": semantic_cache.size(),
        "ttl_hours": semantic_cache.ttl_seconds // 3600,
        "similarity_threshold": semantic_cache.similarity_threshold,
        "available": semantic_cache._available,
    }


@router_v1.post("/chat/cache/clear")
async def cache_clear():
    """Clear all semantic cache entries."""
    size_before = semantic_cache.size()
    cleared = semantic_cache.clear()
    logger.info("[chat] cache cleared: %d entries removed", cleared)
    return {
        "success": True,
        "cleared": cleared,
        "cache_size_now": semantic_cache.size(),
        "message": f"Cleared {cleared} entries (was {size_before})",
    }


# ---------------------------------------------------------------------------
# History endpoint
# ---------------------------------------------------------------------------

@router_v1.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Fetch conversation turns for session_id from PostgresChatMessageHistory."""
    logger.info("[chat] history request session=%s", session_id)
    try:
        from app.services.agents.langchain_memory_manager import ChatMemoryManager
        messages = await ChatMemoryManager().get_history(session_id) or []
        return {
            "session_id": session_id,
            "messages": messages,
            "success": True,
            "created_at": datetime.now().isoformat(),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("[chat] history load failed %s: %s", session_id, exc)
        return {
            "session_id": session_id,
            "messages": [],
            "success": False,
            "created_at": datetime.now().isoformat(),
        }
