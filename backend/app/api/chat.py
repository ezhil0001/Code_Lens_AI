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
import functools
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
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
    """Return the compiled supervisor graph singleton, or None on failure.

    Sync entry point (used by non-async callers). Prefer ``_build_graph_async``
    from async request handlers so the Postgres checkpointer can be awaited —
    the sync path can only obtain a MemorySaver when an event loop is already
    running.
    """
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


async def _build_graph_async():
    """Async graph builder — awaits the real AsyncPostgresSaver checkpointer.

    This is the correct path for the streaming endpoint: the graph is driven by
    ``astream_events`` (async), so it needs an async checkpointer. Building it
    here (instead of via ``get_checkpointer_sync``) is what makes Postgres-
    backed checkpoints / history / branch / replay actually persist rather than
    silently falling back to an ephemeral in-memory saver.
    """
    global _graph_singleton, _graph_lock
    import threading
    if _graph_lock is None:
        _graph_lock = threading.Lock()
    if _graph_singleton is not None:
        return _graph_singleton
    try:
        from app.graph.checkpointing.pg_checkpointer import get_checkpointer
        from app.graph.supervisor_graph import build_supervisor_graph
        checkpointer = await get_checkpointer()
        with _graph_lock:
            if _graph_singleton is None:
                _graph_singleton = build_supervisor_graph(checkpointer=checkpointer)
                logger.info("[chat] supervisor graph compiled and cached (async checkpointer)")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[chat] supervisor graph unavailable: %s", exc)
    return _graph_singleton


def _build_config(
    user_id: str,
    session_id: str,
    org_id: Optional[str],
    checkpoint_id: Optional[str],
    trace_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a LangGraph RunnableConfig, with Langfuse tracing attached.

    A per-request Langfuse ``CallbackHandler`` is added to ``callbacks`` so the
    whole supervisor run (intent routing → agents → retrieval → rerank →
    generation → guardrails) is captured with full parent-child spans, token
    usage, latency, and cost. ``parent_span_id`` nests that whole tree under
    the request's root observation instead of making it a second root.
    Reserved ``langfuse_*`` metadata keys bind the trace to the user and
    session. All of this is a safe no-op when Langfuse is disabled.
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

    _attach_langfuse(
        cfg,
        user_id=user_id,
        session_id=session_id,
        org_id=org_id,
        trace_id=trace_id,
        parent_span_id=parent_span_id,
    )
    # Stash the trace id on the config so the caller can thread it into the
    # graph state for deterministic evaluation scoring (C-2).
    if trace_id:
        cfg.setdefault("configurable", {})["langfuse_trace_id"] = trace_id
    if parent_span_id:
        cfg.setdefault("configurable", {})["langfuse_parent_span_id"] = parent_span_id
    return cfg


def _mint_trace_id() -> Optional[str]:
    """Take the single per-request sampling decision and mint its trace id.

    Returns ``None`` when the request is not sampled or Langfuse is disabled,
    which switches every downstream observation to a no-op. Never raises.
    """
    try:
        from app.observability.langfuse_client import create_trace_id, should_sample
        if not should_sample():
            return None
        return create_trace_id()
    except Exception as exc:  # noqa: BLE001
        logger.debug("[chat] Langfuse trace id skipped: %s", exc)
        return None


def _attach_langfuse(
    cfg: Dict[str, Any],
    *,
    user_id: str,
    session_id: str,
    org_id: Optional[str],
    trace_id: Optional[str],
    parent_span_id: Optional[str] = None,
) -> None:
    """Attach a Langfuse callback handler + trace metadata to a RunnableConfig.

    The handler is pinned to ``trace_id`` and, when available, to the request
    root observation via ``parent_span_id`` so the LangGraph tree nests under
    ``chat.supervisor`` rather than starting a parallel root observation.

    Never raises — observability failures must not break chat.
    """
    if not trace_id:
        return
    try:
        from app.observability.langfuse_client import (
            get_callback_handler,
            build_trace_metadata,
        )

        tags = ["chat", "supervisor-graph"]
        if org_id:
            tags.append(f"org:{org_id}")

        handler = get_callback_handler(trace_id=trace_id, parent_span_id=parent_span_id)
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
    except Exception as exc:  # noqa: BLE001
        logger.debug("[chat] Langfuse attach skipped: %s", exc)


def _build_initial_state(
    query: str,
    user_id: str,
    session_id: str,
    org_id: Optional[str],
    agent_hint: Optional[str],
    langfuse_trace_id: Optional[str] = None,
    hil_enabled: bool = False,
    hil_confidence_threshold: Optional[float] = None,
    langfuse_parent_span_id: Optional[str] = None,
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
        from app.graph.state import RESET
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
            "retrieved_chunks": RESET,
            "reranked_chunks": RESET,
            "agent_responses": RESET,
            "sources": RESET,
            "final_response": None,
            "cache_hit": False,
        }
    if agent_hint:
        state["routing_decision"] = agent_hint
    # The request's HIL settings were accepted by the schema and then dropped,
    # so the UI's "HIL Review" toggle and threshold had no effect at all.
    state["hil_enabled"] = bool(hil_enabled)
    if hil_confidence_threshold is not None:
        state["hil_confidence_threshold"] = float(hil_confidence_threshold)
    # Thread the Langfuse trace id through state so response_node can score the
    # exact originating trace without relying on ambient OTEL context (C-2).
    if langfuse_trace_id:
        state["langfuse_trace_id"] = langfuse_trace_id
    if langfuse_parent_span_id:
        state["langfuse_parent_span_id"] = langfuse_parent_span_id
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

    # 2. Reconnect / resume
    resume_cp: Optional[str] = body.resume_from_checkpoint
    if not resume_cp:
        lei = request.headers.get("Last-Event-ID")
        if lei:
            resume_cp = lei
            logger.info("[v2/chat] reconnect checkpoint=%s user=%s", lei, user_id)

    # One sampling decision + one trace id + ONE root observation per request,
    # opened before any instrumented service runs. Everything downstream (cache
    # lookup, LangGraph callback tree, background evaluation) parents to this
    # root, so a request is exactly one trace with exactly one root named
    # "chat.supervisor" — on the cache-hit path too.
    from app.observability.tracing import open_request_root, request_trace

    trace_id = _mint_trace_id()
    tags = ["chat", "supervisor-graph"]
    if body.org_id:
        tags.append(f"org:{body.org_id}")
    root = open_request_root(
        "chat.supervisor",
        trace_id=trace_id,
        user_id=user_id,
        session_id=body.session_id,
        tags=tags,
        input=body.query,
        metadata={"request.source": "api", "hil.enabled": body.hil_enabled},
    )

    try:
        # 3. Config + state
        config = _build_config(
            user_id=user_id,
            session_id=body.session_id,
            org_id=body.org_id,
            checkpoint_id=resume_cp,
            trace_id=trace_id,
            parent_span_id=root.span_id,
        )

        # 1. Cache — inside the request trace context so the lookup span nests
        # under this request's root span instead of rooting its own trace.
        with request_trace(trace_id, root.span_id):
            # Z-2: semantic_cache.get is synchronous (embedding + pgvector query,
            # ~300ms). Run it in a worker thread so the event loop keeps serving
            # other requests. run_retrieval copies no contextvars implicitly, so we
            # bind the trace with request_trace above; the span still nests here.
            #
            # It must use the *bounded retrieval* pool, not asyncio's default one:
            # this is model inference, and /api/health also does asyncio.to_thread.
            # With N concurrent requests embedding on the 12-thread default pool,
            # health could not obtain a thread and timed out (000) under load.
            from app.core.database import run_retrieval
            # The cache is an optimisation, not a dependency. It was unguarded, so
            # a Postgres/pgvector outage turned every chat request into a 500.
            try:
                hit = await run_retrieval(
                    functools.partial(semantic_cache.get, body.query, user_id=user_id)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[chat] semantic cache lookup failed (%s) — treating as miss", exc)
                hit = None

        if hit:
            return _cached_stream_v2(
                hit, body.session_id, trace_id=trace_id, user_id=user_id,
                query=body.query, root=root,
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
                langfuse_trace_id=trace_id,
                hil_enabled=body.hil_enabled,
                hil_confidence_threshold=body.hil_confidence_threshold,
                langfuse_parent_span_id=root.span_id,
            )
            logger.info(
                "[v2/chat] new thread=%s hint=%s",
                config["configurable"].get("thread_id"), body.agent_hint,
            )

        # 4. Graph
        graph = await _build_graph_async()
        if graph is None:
            root.end(
                level="ERROR",
                status_message="supervisor graph unavailable",
                output={"error": "graph_unavailable"},
            )
            return StreamingResponse(
                _fatal_error_stream("Supervisor graph is currently unavailable. Please try again."),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
    except BaseException as exc:
        # Never leave the root observation open — it would show as a hung trace.
        root.end(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
        raise

    return StreamingResponse(
        _graph_stream_v2(
            graph, initial_state, config, body.query, user_id,
            trace_id=trace_id, session_id=body.session_id, root=root,
        ),
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


async def _persist_turn(
    user_id: str, session_id: Optional[str], query: str, answer: str
) -> None:
    """Append the completed turn to durable chat history.

    Nothing wrote to ChatMemoryManager, so GET /history always returned an
    empty list and a browser refresh silently lost the conversation. Keyed
    ``{user_id}::{session_id}`` to match the read path and stay IDOR-safe.
    """
    if not session_id or not answer:
        return
    scoped = session_id if session_id.startswith(f"{user_id}::") else f"{user_id}::{session_id}"
    try:
        from app.services.agents.langchain_memory_manager import ChatMemoryManager
        mgr = ChatMemoryManager()
        await mgr.add_message(scoped, user_id, "user", query)
        await mgr.add_message(scoped, user_id, "assistant", answer)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[chat] history persist failed for %s: %s", scoped, exc)


async def _graph_stream_v2(graph, initial_state, config, query: str, user_id: str, trace_id: Optional[str] = None, session_id: Optional[str] = None, root: Optional[Any] = None):
    """Wrap stream_graph_events() and accumulate tokens for cache writing."""
    from app.graph.streaming import stream_graph_events
    from app.observability.tracing import RequestRootSpan, request_trace

    accumulated = ""
    cache_written = False  # F-3: guarantee exactly one cache write per request
    completed = False      # Z-3: only cache COMPLETE responses (done event seen)
    root = root if root is not None else RequestRootSpan(None, trace_id)
    with request_trace(trace_id, root.span_id):
        try:
            async for chunk in stream_graph_events(graph, initial_state, config):
                try:
                    payload = json.loads(chunk.removeprefix("data: ").rstrip())
                    t = payload.get("type")
                    if t == "token":
                        accumulated += payload.get("data", {}).get("content", "")
                    elif t == "token_reset":
                        # Synthesis supersedes the per-agent drafts; cache the
                        # final answer only, never drafts + synthesis.
                        accumulated = ""
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
        except BaseException as exc:
            # Z-3: client disconnect / cancellation / error → do NOT cache the
            # partial response (poisoning risk: truncated answers would be
            # served verbatim to future semantically-similar queries).
            if accumulated and not completed:
                logger.info(
                    "[chat] discarding %d partial chars on disconnect (not cached)",
                    len(accumulated),
                )
            root.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            # Normal completion path → write exactly once, only if complete.
            if completed and not cache_written:
                # Z-2: cache write embeds + INSERTs synchronously; offload it to
                # the bounded retrieval pool (model inference), never the
                # default executor that /api/health shares.
                # Guarded independently: a cache-write failure must not abort
                # the response, and must not skip history persistence.
                from app.core.database import run_retrieval
                cache_written = True
                try:
                    await run_retrieval(
                        functools.partial(_cache_write, query, accumulated, user_id)
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[chat] semantic cache write failed: %s", exc)
                await _persist_turn(user_id, session_id, query, accumulated)
            root.end(output=accumulated[:2000] if accumulated else None)


# ---------------------------------------------------------------------------
# Cache helper stream
# ---------------------------------------------------------------------------

def _cached_stream_v2(
    hit: dict,
    session_id: str,
    trace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    query: Optional[str] = None,
    root: Optional[Any] = None,
) -> StreamingResponse:
    """Stream a cache hit in the v2 typed envelope format."""
    async def _gen():
        try:
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
            # A cache hit is still a conversation turn. This path bypasses
            # _graph_stream_v2, so without this the turn never reached history and
            # a refresh lost every cached answer.
            if user_id and query:
                await _persist_turn(user_id, session_id, query, hit["response"])
        finally:
            if root is not None:
                root.end(
                    output=str(hit.get("response", ""))[:2000],
                    metadata={"cache.hit": True, "cache.similarity": hit.get("similarity")},
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
        messages = await ChatMemoryManager().get_messages(scoped_session) or []
        return {
            "session_id": session_id,
            "messages": messages,
            "success": True,
            "created_at": datetime.now().isoformat(),
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("[chat] history load failed %s: %s", session_id, exc)
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
        from app.observability.tracing import get_trace_owner

        user_id = str(getattr(current_user, "id", "anonymous"))
        # Any authenticated user could previously score ANY trace id, including
        # another user's trace or a fabricated one, poisoning its evaluation
        # data. Only the trace's owner may submit feedback for it.
        owner = get_trace_owner(body.trace_id)
        if owner is None:
            logger.warning(
                "[chat] feedback rejected — unknown/expired trace %s", body.trace_id
            )
            raise HTTPException(status_code=404, detail="Unknown trace")
        if owner != user_id:
            logger.warning(
                "[chat] feedback rejected — user %s does not own trace %s",
                user_id, body.trace_id,
            )
            raise HTTPException(status_code=403, detail="Not your trace")

        recorded = record_user_feedback(
            trace_id=body.trace_id,
            thumbs_up=body.thumbs_up,
            rating=body.rating,
            comment=body.comment,
            user_id=user_id,
            session_id=body.session_id,
        )
        return {"success": True, "recorded": recorded}
    except HTTPException:
        raise
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
