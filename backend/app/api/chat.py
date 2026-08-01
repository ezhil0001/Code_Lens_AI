"""
Chat API — LangGraph streaming endpoint (single, V2-only).

  POST /api/v2/chat/stream    — LangGraph supervisor graph, typed SSE
                                envelopes, checkpointing, reconnect, HIL,
                                agent_hint.

Supporting endpoints (same router / prefix):

  GET  /api/v2/chat/cache/status
  POST /api/v2/chat/cache/clear
  GET  /api/v2/chat/history/{session_id}
  POST /api/v2/chat/feedback
  POST /api/v2/chat/curate

The SemanticCache singleton lives in ``app.services.semantic_cache``.
"""

from __future__ import annotations

import asyncio
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
# Router — single V2 router
# ---------------------------------------------------------------------------

router_v2 = APIRouter(prefix="/api/v2/chat", tags=["chat", "streaming"])

# ``router`` alias kept so ``main.py`` can register a single symbol.
router = router_v2


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def _resolve_user_dep():
    """Return the real JWT dependency, falling back to an anonymous stub.

    The production dependency lives in ``app.routes.auth`` (it validates the
    Bearer token, checks the JTI blacklist, and loads the user). The stub is
    used ONLY when the auth stack cannot be imported (e.g. stripped-down test
    environments without DB/JWT deps).
    """
    try:
        from app.routes.auth import get_current_user  # type: ignore
        return get_current_user
    except Exception:  # noqa: BLE001
        logger.warning("[chat] auth stack unavailable — using anonymous user dependency")
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


class FeedbackRequest(BaseModel):
    """User feedback on a chat response, linked to its Langfuse trace.

    ``trace_id`` comes from the ``done`` SSE event's metadata.
    """

    trace_id: str = Field(..., min_length=8, max_length=64)
    thumbs_up: Optional[bool] = None
    rating: Optional[int] = Field(None, ge=1, le=5)
    comment: Optional[str] = Field(None, max_length=2000)
    session_id: Optional[str] = None


class DatasetCurationRequest(BaseModel):
    """Curate a production interaction into the regression dataset."""

    query: str = Field(..., max_length=5000)
    expected_output: Optional[str] = Field(None, max_length=20000)
    trace_id: Optional[str] = None
    tags: Optional[list] = None


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
    """Build a LangGraph RunnableConfig, with Langfuse tracing attached.

    A per-request Langfuse ``CallbackHandler`` is added to ``callbacks`` so the
    whole supervisor run (intent routing → agents → retrieval → rerank →
    generation → guardrails) is captured as one trace with full parent-child
    spans, token usage, latency, and cost. Reserved ``langfuse_*`` metadata
    keys bind the trace to the user and session. All of this is a safe no-op
    when Langfuse is disabled.
    """
    try:
        from app.graph.checkpointing.pg_checkpointer import build_config
        cfg = build_config(
            user_id=user_id,
            session_id=session_id,
            org_id=org_id,
            checkpoint_id=checkpoint_id,
        )
    except Exception:  # noqa: BLE001
        cfg = {
            "configurable": {
                "thread_id": f"{user_id}::{session_id}",
                "checkpoint_ns": org_id or "default",
            },
            "recursion_limit": 25,
        }
        if checkpoint_id:
            cfg["configurable"]["checkpoint_id"] = checkpoint_id

    trace_id = _attach_langfuse(cfg, user_id=user_id, session_id=session_id, org_id=org_id)
    # Stash the minted trace id on the config so the caller can thread it into
    # the graph state for deterministic evaluation scoring (C-2).
    if trace_id:
        cfg.setdefault("configurable", {})["langfuse_trace_id"] = trace_id
    return cfg


def _attach_langfuse(
    cfg: Dict[str, Any],
    *,
    user_id: str,
    session_id: str,
    org_id: Optional[str],
) -> Optional[str]:
    """Attach a Langfuse callback handler + trace metadata to a RunnableConfig.

    Mints a deterministic trace id up front, pins the callback handler to it,
    and returns it so the caller can store it in the graph state. This makes
    evaluation scoring target the exact originating trace without relying on
    ambient OTEL context inside node execution.

    Returns the trace id (or ``None`` when Langfuse is disabled).
    Never raises — observability failures must not break chat.
    """
    try:
        from app.observability.langfuse_client import (
            get_callback_handler,
            build_trace_metadata,
            create_trace_id,
            should_sample,
        )
        if not should_sample():
            return None

        tags = ["chat", "supervisor-graph"]
        if org_id:
            tags.append(f"org:{org_id}")

        trace_id = create_trace_id()

        # M-4: register trace ownership so only the requesting user can
        # attach feedback scores to this trace.
        try:
            from app.observability.evaluation.feedback import register_trace_owner
            register_trace_owner(trace_id, user_id)
        except Exception:  # noqa: BLE001
            pass

        handler = get_callback_handler(trace_id=trace_id)
        if handler is not None:
            cfg.setdefault("callbacks", []).append(handler)

        md = build_trace_metadata(
            user_id=user_id,
            session_id=session_id,
            trace_name="chat.supervisor",
            tags=tags,
        )
        existing_md = cfg.get("metadata") or {}
        existing_md.update(md)
        cfg["metadata"] = existing_md
        return trace_id
    except Exception as exc:  # noqa: BLE001
        logger.debug("[chat] Langfuse attach skipped: %s", exc)
        return None


def _build_initial_state(
    query: str,
    user_id: str,
    session_id: str,
    org_id: Optional[str],
    agent_hint: Optional[str],
    langfuse_trace_id: Optional[str] = None,
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
    # Thread the Langfuse trace id through state so response_node can score the
    # exact originating trace without relying on ambient OTEL context (C-2).
    if langfuse_trace_id:
        state["langfuse_trace_id"] = langfuse_trace_id
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

    # Build the config FIRST so the per-request sampling decision + trace id
    # exist before any instrumented service (cache) runs. All handler-level
    # spans then join this single request trace (H-1) or no-op when the
    # request is unsampled (H-2).
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
    trace_id = config.get("configurable", {}).get("langfuse_trace_id")

    from app.observability.tracing import request_trace

    # 1. Cache — inside the request trace context so the lookup span nests
    # under this request's trace instead of rooting its own.
    with request_trace(trace_id):
        # Z-2: semantic_cache.get is synchronous (embedding + pgvector query,
        # ~300ms). Run it in a worker thread so the event loop keeps serving
        # other requests. asyncio.to_thread copies contextvars, so the span
        # still nests under this request's trace.
        hit = await asyncio.to_thread(semantic_cache.get, body.query, user_id=user_id)
    if hit:
        return _cached_stream_v2(hit, body.session_id, trace_id=trace_id)

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
            langfuse_trace_id=config.get("configurable", {}).get("langfuse_trace_id"),
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
        _graph_stream_v2(graph, initial_state, config, body.query, user_id, trace_id=trace_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Session-Id": body.session_id,
            "X-Thread-Id": config["configurable"].get("thread_id", ""),
            # H-3: expose the trace id so v2 clients can submit trace-linked
            # feedback (also present in the done event payload).
            "X-Trace-Id": trace_id or "",
        },
    )


async def _graph_stream_v2(graph, initial_state, config, query: str, user_id: str, trace_id: Optional[str] = None):
    """Wrap stream_graph_events() and accumulate tokens for cache writing."""
    from app.graph.streaming import stream_graph_events
    from app.observability.tracing import request_trace

    accumulated = ""
    cache_written = False  # F-3: guarantee exactly one cache write per request
    completed = False      # Z-3: only cache COMPLETE responses (done event seen)
    with request_trace(trace_id):
        try:
            async for chunk in stream_graph_events(graph, initial_state, config):
                try:
                    payload = json.loads(chunk.removeprefix("data: ").rstrip())
                    t = payload.get("type")
                    if t == "token":
                        accumulated += payload.get("data", {}).get("content", "")
                    elif t == "done":
                        # Z-3: mark completion only on a clean done event that
                        # is not itself reporting an upstream error.
                        if not payload.get("data", {}).get("error"):
                            completed = True
                        # H-3: inject the trace id into the done envelope so
                        # v2 clients can link feedback to this trace.
                        payload.setdefault("data", {})["trace_id"] = trace_id
                        yield f"data: {json.dumps(payload)}\n\n"
                        continue
                    elif t == "error":
                        completed = False
                except Exception:  # noqa: BLE001
                    pass
                yield chunk
        except BaseException:
            # Z-3: client disconnect / cancellation / error → do NOT cache the
            # partial response (poisoning risk: truncated answers would be
            # served verbatim to future semantically-similar queries).
            if accumulated and not completed:
                logger.info(
                    "[chat] discarding %d partial chars on disconnect (not cached)",
                    len(accumulated),
                )
            raise
        finally:
            # Normal completion path → write exactly once, only if complete.
            if completed and not cache_written:
                # Z-2: cache write embeds + INSERTs synchronously; offload it.
                await asyncio.to_thread(_cache_write, query, accumulated, user_id)
                cache_written = True


# ---------------------------------------------------------------------------
# Cache helper stream
# ---------------------------------------------------------------------------

def _cached_stream_v2(hit: dict, session_id: str, trace_id: Optional[str] = None) -> StreamingResponse:
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
                "trace_id": trace_id,
            },
        )
    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Trace-Id": trace_id or ""},
    )


# ---------------------------------------------------------------------------
# Cache management endpoints
# ---------------------------------------------------------------------------

@router_v2.get("/cache/status")
async def cache_status(current_user=Depends(_current_user_dep)):
    """Return semantic cache statistics (authenticated)."""
    user_id = str(current_user.id)
    return {
        "cache_size": semantic_cache.size(user_id=user_id),
        "ttl_hours": semantic_cache.ttl_seconds // 3600,
        "similarity_threshold": semantic_cache.similarity_threshold,
        "available": semantic_cache._available,
    }


@router_v2.post("/cache/clear")
async def cache_clear(current_user=Depends(_current_user_dep)):
    """Clear the requesting user's semantic cache entries (never global)."""
    user_id = str(current_user.id)
    size_before = semantic_cache.size(user_id=user_id)
    cleared = semantic_cache.clear(user_id=user_id)
    logger.info("[chat] cache cleared for user=%s: %d entries removed", user_id, cleared)
    return {
        "success": True,
        "cleared": cleared,
        "cache_size_now": semantic_cache.size(user_id=user_id),
        "message": f"Cleared {cleared} entries (was {size_before})",
    }


# ---------------------------------------------------------------------------
# History endpoint
# ---------------------------------------------------------------------------

@router_v2.get("/history/{session_id}")
async def get_chat_history(session_id: str, current_user=Depends(_current_user_dep)):
    """Fetch conversation turns for the caller's session.

    Ownership: history is stored under the composite key
    ``{user_id}::{session_id}`` (see _build_initial_state), so scoping the
    lookup to the authenticated user makes cross-user session enumeration
    impossible (IDOR-safe).
    """
    user_id = str(current_user.id)
    scoped_session = session_id if session_id.startswith(f"{user_id}::") else f"{user_id}::{session_id}"
    logger.info("[chat] history request session=%s user=%s", session_id, user_id)
    try:
        from app.services.agents.langchain_memory_manager import ChatMemoryManager
        messages = await ChatMemoryManager().get_history(scoped_session) or []
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


# ---------------------------------------------------------------------------
# Evaluation endpoints — user feedback + regression-dataset curation
# ---------------------------------------------------------------------------

@router_v2.post("/feedback")
async def submit_feedback(body: FeedbackRequest, current_user=Depends(_current_user_dep)):
    """Record user feedback (thumbs/rating/comment) as trace-linked Langfuse scores.

    The frontend obtains ``trace_id`` from the ``done`` SSE event metadata.
    Safe no-op (recorded=false) when Langfuse is disabled.
    """
    try:
        from app.observability.evaluation import record_user_feedback
        recorded = record_user_feedback(
            trace_id=body.trace_id,
            thumbs_up=body.thumbs_up,
            rating=body.rating,
            comment=body.comment,
            user_id=str(getattr(current_user, "id", "anonymous")),
            session_id=body.session_id,
        )
        return {"success": True, "recorded": recorded}
    except Exception as exc:  # noqa: BLE001
        logger.warning("[chat] feedback failed: %s", exc)
        return {"success": False, "recorded": False}


@router_v2.post("/curate")
async def curate_to_dataset(body: DatasetCurationRequest, current_user=Depends(_current_user_dep)):
    """Add a production interaction to the Langfuse regression dataset.

    Used by reviewers to promote valuable/problematic queries into the offline
    experiment dataset (idempotent upsert, linked to source trace).
    """
    try:
        from app.observability.evaluation.datasets import add_interaction_to_dataset
        added = add_interaction_to_dataset(
            query=body.query,
            expected_output=body.expected_output,
            trace_id=body.trace_id,
            metadata={
                "curated_by": str(getattr(current_user, "id", "anonymous")),
                "curated_at": datetime.now().isoformat(),
                "tags": body.tags or [],
            },
        )
        return {"success": True, "added": added}
    except Exception as exc:  # noqa: BLE001
        logger.warning("[chat] dataset curation failed: %s", exc)
        return {"success": False, "added": False}
