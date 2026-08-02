"""Reusable Langfuse span instrumentation for non-LangChain code paths.

The LangGraph ``CallbackHandler`` (see ``langfuse_client``) already auto-traces
every graph node, LLM call, tool, and retriever with a full parent-child
hierarchy. This module covers **everything else** — service methods, database
queries, cache lookups, embeddings, external API calls, file/storage ops, and
background jobs — with two tiny primitives built on the Langfuse OTEL-based
tracing API (v3/v4):

``span(...)``
    A context manager that opens a child observation under whatever trace/span
    is currently active (OTEL context propagation handles parenting), records
    input/output/metadata/latency, and marks errors automatically.

``observe_span(...)``
    A decorator (sync + async) that wraps a function in ``span(...)`` with
    zero boilerplate. New services inherit tracing by adding one line.

Design guarantees
-----------------
- **Graceful degradation** — every primitive is a safe no-op when Langfuse is
  disabled/uninstalled. Business logic can never fail because of tracing.
- **Correct parenting** — spans nest under the active OTEL context, so a
  service call made inside a LangGraph node appears as a child of that node's
  span in the same trace. No orphan spans, no duplicate traces.
- **Bounded payloads** — inputs/outputs are truncated so huge documents or
  embeddings never bloat the ingestion pipeline.
- **Minimal overhead** — when disabled, the wrapper is a try/except around the
  original call; when enabled, span creation is in-memory and batched by the
  SDK's background thread.

Usage
-----
    from app.observability.tracing import observe_span, span

    @observe_span(name="semantic_cache.get", kind="span")
    def get(self, query: str, user_id: str): ...

    async def call_api(...):
        with span("tavily.search", kind="tool", input={"query": q}) as s:
            resp = client.search(query=q)
            s.update(output={"results": len(resp)})
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

# ── Request-scoped trace context (H-1 trace fragmentation, H-2 sampling) ─────
#
# One sampling decision + one trace id per request, carried across the request
# via a contextvar so every service span joins the SAME trace instead of
# rooting its own, and unsampled requests emit NOTHING.
#
# Sentinel semantics:
#   _UNSET  → no request context (background/startup code): spans allowed,
#             parenting follows ambient OTEL context (or explicit trace_id).
#   None    → request explicitly NOT sampled: all spans no-op.
#   str     → sampled request: spans join this trace id.
_UNSET = object()
_REQUEST_TRACE: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "langfuse_request_trace", default=_UNSET
)


@contextmanager
def request_trace(trace_id: Optional[str]) -> Iterator[None]:
    """Bind the per-request sampling decision + trace id to the current context.

    ``trace_id=None`` marks the request as NOT sampled (all spans inside
    become no-ops). A string joins every nested :func:`span` to that trace.
    """
    token = _REQUEST_TRACE.set(trace_id)
    try:
        yield
    finally:
        # A ``ContextVar`` token can only be reset in the same context it was
        # created in. Streaming responses (SSE) drive their async generator to
        # completion in a *different* context than the one that entered this
        # manager, so ``reset(token)`` raises
        # ``ValueError: Token was created in a different Context``. That error
        # must never surface — it would abort the stream and orphan the trace.
        # Fall back to clearing the var directly when the token is unusable.
        try:
            _REQUEST_TRACE.reset(token)
        except ValueError:
            _REQUEST_TRACE.set(_UNSET)



def current_request_trace_id() -> Optional[str]:
    """Return the bound request trace id, or None when unsampled/unbound."""
    val = _REQUEST_TRACE.get()
    return val if isinstance(val, str) else None

# Observation kinds understood by the Langfuse v3/v4 SDK. Anything else is
# recorded as a plain span.
_VALID_KINDS = {"span", "generation", "event", "embedding", "retriever", "tool", "agent", "chain", "evaluator", "guardrail"}

_MAX_PAYLOAD_CHARS = 4000  # bound serialized input/output size per span


def _truncate(value: Any) -> Any:
    """Bound the size of a payload before attaching it to a span."""
    if value is None:
        return None
    try:
        text = value if isinstance(value, str) else repr(value)
        if len(text) > _MAX_PAYLOAD_CHARS:
            return text[:_MAX_PAYLOAD_CHARS] + f"… [truncated, {len(text)} chars total]"
        return value
    except Exception:  # noqa: BLE001
        return None


class _SpanHandle:
    """Thin wrapper so callers can ``s.update(...)`` without SDK knowledge."""

    __slots__ = ("_obs",)

    def __init__(self, obs: Optional[Any]) -> None:
        self._obs = obs

    def update(
        self,
        *,
        output: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
    ) -> None:
        """Attach output/metadata to the live span. Never raises."""
        if self._obs is None:
            return
        try:
            kwargs: Dict[str, Any] = {}
            if output is not None:
                kwargs["output"] = _truncate(output)
            if metadata:
                kwargs["metadata"] = metadata
            if level:
                kwargs["level"] = level
            if status_message:
                kwargs["status_message"] = status_message
            if kwargs:
                self._obs.update(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[tracing] span.update failed: %s", exc)


@contextmanager
def span(
    name: str,
    *,
    kind: str = "span",
    input: Any = None,  # noqa: A002  (mirrors Langfuse SDK naming)
    metadata: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
) -> Iterator[_SpanHandle]:
    """Open a Langfuse child observation under the active trace.

    Parenting/sampling rules (H-1, H-2):

    1. Inside a request bound via :func:`request_trace`:
       - unsampled request → pure no-op (no orphan spans);
       - sampled request with no ambient span yet → the span joins the
         request's trace via ``trace_context`` instead of rooting a new one;
       - ambient OTEL span already active (e.g. inside a LangGraph node) →
         normal child nesting.
    2. Explicit ``trace_id`` (background jobs) → joins that trace.
    3. No request context at all → legacy behaviour (ambient nesting).

    Yields a :class:`_SpanHandle`; call ``handle.update(output=...)`` to record
    the result. Exceptions are recorded with level=ERROR and re-raised. Safe
    no-op when Langfuse is disabled.
    """
    obs_cm = None
    try:
        req_trace = _REQUEST_TRACE.get()
        if req_trace is None and trace_id is None:
            # Request explicitly not sampled → emit nothing (H-2).
            yield _SpanHandle(None)
            return

        from app.observability.langfuse_client import get_client

        client = get_client()
        if client is not None:
            as_type = kind if kind in _VALID_KINDS else "span"

            # Resolve the trace to join: explicit arg > request context.
            join_trace_id = trace_id or (req_trace if isinstance(req_trace, str) else None)
            trace_context = None
            if join_trace_id:
                try:
                    from opentelemetry import trace as _otel_trace
                    if not _otel_trace.get_current_span().get_span_context().is_valid:
                        # No ambient parent — pin to the request trace so this
                        # span nests under it instead of rooting a new trace.
                        trace_context = {"trace_id": join_trace_id}
                except Exception:  # noqa: BLE001
                    trace_context = {"trace_id": join_trace_id}

            starter = getattr(client, "start_as_current_observation", None)
            if starter is not None:
                kwargs: Dict[str, Any] = dict(
                    name=name, as_type=as_type, input=_truncate(input), metadata=metadata,
                )
                if trace_context:
                    kwargs["trace_context"] = trace_context
                obs_cm = starter(**kwargs)
            else:  # older v3 SDKs only expose start_as_current_span
                starter = getattr(client, "start_as_current_span", None)
                if starter is not None:
                    obs_cm = starter(name=name, input=_truncate(input), metadata=metadata)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[tracing] span start failed (%s): %s", name, exc)
        obs_cm = None

    if obs_cm is None:
        # Disabled path: yield an inert handle so caller code is unchanged.
        yield _SpanHandle(None)
        return

    try:
        with obs_cm as obs:
            handle = _SpanHandle(obs)
            try:
                yield handle
            except Exception as exc:
                handle.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
                raise
    except Exception:
        raise
    finally:
        pass


def observe_span(
    name: Optional[str] = None,
    *,
    kind: str = "span",
    capture_input: bool = True,
    capture_output: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> Callable:
    """Decorator: trace a sync or async function as a Langfuse observation.

    Records input arguments (bounded), output (bounded), wall-clock latency,
    and any exception (level=ERROR, then re-raised). Parenting follows the
    active OTEL context automatically. No-op when Langfuse is disabled.
    """

    def decorator(fn: Callable) -> Callable:
        span_name = name or f"{fn.__module__.rsplit('.', 1)[-1]}.{fn.__qualname__}"

        def _build_input(args: tuple, kwargs: dict) -> Any:
            if not capture_input:
                return None
            try:
                # Skip self/cls for readability.
                shown_args = args[1:] if args and hasattr(args[0], "__class__") and fn.__qualname__ != fn.__name__ else args
                payload: Dict[str, Any] = {}
                if shown_args:
                    payload["args"] = _truncate(shown_args)
                if kwargs:
                    payload["kwargs"] = _truncate(kwargs)
                return payload or None
            except Exception:  # noqa: BLE001
                return None

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                with span(span_name, kind=kind, input=_build_input(args, kwargs), metadata=metadata) as s:
                    result = await fn(*args, **kwargs)
                    s.update(
                        output=result if capture_output else None,
                        metadata={"latency_ms": round((time.perf_counter() - start) * 1000, 2)},
                    )
                    return result
            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            with span(span_name, kind=kind, input=_build_input(args, kwargs), metadata=metadata) as s:
                result = fn(*args, **kwargs)
                s.update(
                    output=result if capture_output else None,
                    metadata={"latency_ms": round((time.perf_counter() - start) * 1000, 2)},
                )
                return result
        return sync_wrapper

    return decorator
