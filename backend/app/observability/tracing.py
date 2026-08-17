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
import threading
import time
from collections import OrderedDict
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

# Observation id of the request's single root span. Every span that has to pin
# itself to the request trace (no ambient OTEL parent) nests under this instead
# of becoming a second root observation.
_REQUEST_ROOT_SPAN: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "langfuse_request_root_span", default=None
)


@contextmanager
def request_trace(
    trace_id: Optional[str], root_span_id: Optional[str] = None
) -> Iterator[None]:
    """Bind the per-request sampling decision + trace id to the current context.

    ``trace_id=None`` marks the request as NOT sampled (all spans inside
    become no-ops). A string joins every nested :func:`span` to that trace,
    parented to ``root_span_id`` when one is supplied.
    """
    token = _REQUEST_TRACE.set(trace_id)
    root_token = _REQUEST_ROOT_SPAN.set(root_span_id)
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
        try:
            _REQUEST_ROOT_SPAN.reset(root_token)
        except ValueError:
            _REQUEST_ROOT_SPAN.set(None)


def current_request_trace_id() -> Optional[str]:
    """Return the bound request trace id, or None when unsampled/unbound."""
    val = _REQUEST_TRACE.get()
    return val if isinstance(val, str) else None


def current_request_root_span_id() -> Optional[str]:
    """Return the observation id of the request's root span, if bound."""
    return _REQUEST_ROOT_SPAN.get()


# Root name per trace id. Background work (RAGAS, cache writes) finishes after
# the request span closed and would otherwise donate its own name to the trace.
# Bounded so a long-running process cannot grow this without limit.
_TRACE_ROOT_NAMES: "OrderedDict[str, str]" = OrderedDict()
_TRACE_ROOT_NAMES_MAX = 2048
_TRACE_ROOT_NAMES_LOCK = threading.Lock()


def register_trace_root_name(trace_id: str, name: str) -> None:
    """Remember the canonical root name for a trace."""
    if not trace_id or not name:
        return
    with _TRACE_ROOT_NAMES_LOCK:
        _TRACE_ROOT_NAMES[trace_id] = name
        _TRACE_ROOT_NAMES.move_to_end(trace_id)
        while len(_TRACE_ROOT_NAMES) > _TRACE_ROOT_NAMES_MAX:
            _TRACE_ROOT_NAMES.popitem(last=False)


def get_trace_root_name(trace_id: Optional[str]) -> Optional[str]:
    """Return the canonical root name previously registered for a trace."""
    if not trace_id:
        return None
    with _TRACE_ROOT_NAMES_LOCK:
        return _TRACE_ROOT_NAMES.get(trace_id)


# Trace ownership. /api/v2/chat/feedback accepted any trace_id, so an
# authenticated user could attach scores to another user's trace (or to a
# fabricated id), poisoning that trace's evaluation data.
_TRACE_OWNERS: "OrderedDict[str, str]" = OrderedDict()
_TRACE_OWNERS_MAX = 4096
_TRACE_OWNERS_LOCK = threading.Lock()


def register_trace_owner(trace_id: Optional[str], user_id: Optional[str]) -> None:
    """Record which user a trace was created for."""
    if not trace_id or not user_id:
        return
    with _TRACE_OWNERS_LOCK:
        _TRACE_OWNERS[trace_id] = str(user_id)
        _TRACE_OWNERS.move_to_end(trace_id)
        while len(_TRACE_OWNERS) > _TRACE_OWNERS_MAX:
            _TRACE_OWNERS.popitem(last=False)


def get_trace_owner(trace_id: Optional[str]) -> Optional[str]:
    """Return the user a trace belongs to, or None if unknown/evicted."""
    if not trace_id:
        return None
    with _TRACE_OWNERS_LOCK:
        return _TRACE_OWNERS.get(trace_id)



class RequestRootSpan:
    """Manual-lifecycle root observation for one request.

    A chat request spans two disjoint execution scopes: the endpoint coroutine
    (auth, cache lookup) and the SSE generator that streams the graph. A
    ``with`` block cannot cover both, so the root is opened explicitly here and
    closed by whichever path finishes the request. ``span_id`` is what makes
    every other observation \u2014 cache lookup, LangGraph callback tree,
    background evaluation \u2014 a *child* rather than a second root.
    """

    __slots__ = ("_obs", "trace_id", "span_id", "_ended", "_name")

    def __init__(self, obs: Optional[Any], trace_id: Optional[str], name: str = "") -> None:
        self._obs = obs
        self.trace_id = trace_id
        self.span_id = getattr(obs, "id", None) if obs is not None else None
        self._ended = False
        self._name = name

    def update(self, **kwargs: Any) -> None:
        """Attach output/metadata/level to the root span. Never raises."""
        if self._obs is None:
            return
        try:
            if "output" in kwargs:
                kwargs["output"] = _truncate(kwargs["output"])
            self._obs.update(**{k: v for k, v in kwargs.items() if v is not None})
        except Exception as exc:  # noqa: BLE001
            logger.debug("[tracing] root span update failed: %s", exc)

    def end(self, **kwargs: Any) -> None:
        """Close the root span exactly once. Never raises.

        Re-asserts the trace-level name and output last. The LangGraph callback
        tree is pinned via ``trace_context``, which makes Langfuse treat it as a
        trace root too, so without this the trace's input/output would show the
        raw graph state instead of the user's query and the answer.
        """
        if self._obs is None or self._ended:
            return
        self._ended = True
        try:
            output = kwargs.get("output")
            if kwargs:
                self.update(**kwargs)
            _stamp_trace_attributes(
                trace_name=self._name or None,
                trace_output=output,
                otel_span=getattr(self._obs, "_otel_span", None),
            )
            self._obs.end()
        except Exception as exc:  # noqa: BLE001
            logger.debug("[tracing] root span end failed: %s", exc)


def open_request_root(
    name: str,
    *,
    trace_id: Optional[str],
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[list] = None,
    input: Any = None,  # noqa: A002
    metadata: Optional[Dict[str, Any]] = None,
    parent_span_id: Optional[str] = None,
) -> RequestRootSpan:
    """Open the single root observation for a request.

    Without an explicit root, whichever inner observation happens to start
    first (``semantic_cache.get``, ``memory.ltm_retrieve``, …) becomes a trace
    root and donates its name to the trace. Returns an inert handle when the
    request is unsampled or Langfuse is disabled.

    ``parent_span_id`` attaches this span to an existing observation instead of
    starting a second top-level branch. A HIL resume is a separate HTTP request
    that continues an earlier one, so nesting it keeps the trace single-rooted
    and preserves the original trace name.
    """
    if not trace_id:
        return RequestRootSpan(None, None, name)
    try:
        from app.observability.langfuse_client import get_client

        client = get_client()
        starter = getattr(client, "start_observation", None) if client else None
        if starter is None:
            return RequestRootSpan(None, trace_id, name)
        trace_context: Dict[str, Any] = {"trace_id": trace_id}
        if parent_span_id:
            trace_context["parent_span_id"] = parent_span_id
        obs = starter(
            name=name,
            as_type="span",
            input=_truncate(input),
            metadata=metadata,
            trace_context=trace_context,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("[tracing] root span start failed (%s): %s", name, exc)
        return RequestRootSpan(None, trace_id, name)

    if parent_span_id:
        # A nested continuation must not rename the trace or re-stamp the
        # trace-level input — the originating request already owns those.
        return RequestRootSpan(obs, trace_id, name)

    register_trace_root_name(trace_id, name)
    register_trace_owner(trace_id, user_id)
    _stamp_trace_attributes(
        trace_name=name,
        user_id=user_id,
        session_id=session_id,
        tags=tags,
        trace_input=input,
        otel_span=getattr(obs, "_otel_span", None),
    )
    return RequestRootSpan(obs, trace_id, name)


@contextmanager
def request_root_span(
    name: str,
    *,
    trace_id: Optional[str],
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[list] = None,
    input: Any = None,  # noqa: A002
    metadata: Optional[Dict[str, Any]] = None,
) -> Iterator["RequestRootSpan"]:
    """Scoped form of :func:`open_request_root` for single-scope callers."""
    register_trace_owner(trace_id, user_id)
    root = open_request_root(
        name,
        trace_id=trace_id,
        user_id=user_id,
        session_id=session_id,
        tags=tags,
        input=input,
        metadata=metadata,
    )
    with request_trace(trace_id, root.span_id):
        try:
            yield root
        except Exception as exc:
            root.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            root.end()


def _stamp_trace_attributes(
    *,
    trace_name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[list] = None,
    trace_input: Any = None,
    trace_output: Any = None,
    otel_span: Any = None,
) -> None:
    """Write Langfuse trace-level fields onto ``otel_span`` (default: current)."""
    try:
        from opentelemetry import trace as _otel_trace
        from langfuse import LangfuseOtelSpanAttributes as _A

        sp = otel_span if otel_span is not None else _otel_trace.get_current_span()
        if not sp or not sp.get_span_context().is_valid:
            return
        if trace_name:
            sp.set_attribute(_A.TRACE_NAME, trace_name)
        if trace_input is not None:
            sp.set_attribute(_A.TRACE_INPUT, _serialise(trace_input))
        if trace_output is not None:
            sp.set_attribute(_A.TRACE_OUTPUT, _serialise(trace_output))
        if user_id:
            sp.set_attribute(_A.TRACE_USER_ID, str(user_id))
        if session_id:
            sp.set_attribute(_A.TRACE_SESSION_ID, str(session_id))
        if tags:
            sp.set_attribute(_A.TRACE_TAGS, [str(t) for t in tags])
        try:
            from app.core.config import get_settings
            env = getattr(get_settings(), "environment", None)
            if env:
                sp.set_attribute(_A.ENVIRONMENT, str(env))
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("[tracing] trace attribute stamp failed: %s", exc)


# Observation kinds understood by the Langfuse v3/v4 SDK. Anything else is
# recorded as a plain span.
_VALID_KINDS = {"span", "generation", "embedding", "retriever", "tool", "agent", "chain", "evaluator", "guardrail"}

_MAX_PAYLOAD_CHARS = 4000  # bound serialized input/output size per span


def _serialise(value: Any) -> str:
    """Render a payload as a bounded string for an OTEL attribute."""
    try:
        import json
        text = value if isinstance(value, str) else json.dumps(value, default=str)
    except Exception:  # noqa: BLE001
        text = repr(value)
    return text[:_MAX_PAYLOAD_CHARS]


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


def _remote_parent_context(trace_id: str, parent_span_id: Optional[str]) -> Optional[Any]:
    """Return a context manager making ``parent_span_id`` the current OTEL span.

    Used instead of the SDK's ``trace_context`` so the resulting observation is
    an ordinary child: spans created from a ``trace_context`` are stamped
    ``as_root``, and Langfuse then re-derives the whole trace's name, input and
    output from them. Returns ``None`` when the ids are unusable, so the caller
    can fall back to ``trace_context``.
    """
    if not parent_span_id:
        return None
    try:
        from opentelemetry import trace as _otel_trace

        ctx = _otel_trace.SpanContext(
            trace_id=int(trace_id, 16),
            span_id=int(parent_span_id, 16),
            is_remote=False,
            trace_flags=_otel_trace.TraceFlags(0x01),
        )
        if not ctx.is_valid:
            return None
        return _otel_trace.use_span(
            _otel_trace.NonRecordingSpan(ctx), end_on_exit=False, record_exception=False
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("[tracing] remote parent context failed: %s", exc)
        return None


@contextmanager
def span(
    name: str,
    *,
    kind: str = "span",
    input: Any = None,  # noqa: A002  (mirrors Langfuse SDK naming)
    metadata: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
) -> Iterator[_SpanHandle]:
    """Open a Langfuse child observation under the active trace.

    Parenting/sampling rules (H-1, H-2):

    1. Inside a request bound via :func:`request_trace`:
       - unsampled request → pure no-op (no orphan spans);
       - sampled request with no ambient span yet → the span joins the
         request's trace via ``trace_context``, parented to the request root
         span so it never becomes a second root;
       - ambient OTEL span already active (e.g. inside a LangGraph node) →
         normal child nesting.
    2. Explicit ``trace_id`` (background jobs) → joins that trace, under
       ``parent_span_id`` when supplied.
    3. No request context at all → legacy behaviour (ambient nesting).

    Yields a :class:`_SpanHandle`; call ``handle.update(output=...)`` to record
    the result. Exceptions are recorded with level=ERROR and re-raised. Safe
    no-op when Langfuse is disabled.
    """
    obs_cm = None
    join_trace_id: Optional[str] = None
    pinned = False
    parent_cm = None
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
            needs_pin = False
            if join_trace_id:
                try:
                    from opentelemetry import trace as _otel_trace
                    needs_pin = not _otel_trace.get_current_span().get_span_context().is_valid
                except Exception:  # noqa: BLE001
                    needs_pin = True

            trace_context = None
            if needs_pin and join_trace_id:
                parent = parent_span_id or _REQUEST_ROOT_SPAN.get()
                # Prefer re-establishing the OTEL parent context over passing
                # trace_context: the SDK stamps `as_root` on every span built
                # from a trace_context, which makes Langfuse re-derive the
                # trace's name/input/output from it. A late child (cache write,
                # RAGAS) would then rename the trace and overwrite its I/O.
                parent_cm = _remote_parent_context(join_trace_id, parent)
                if parent_cm is not None:
                    parent_cm.__enter__()
                else:
                    trace_context = {"trace_id": join_trace_id}
                    if parent:
                        trace_context["parent_span_id"] = parent
                    pinned = True

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
        if parent_cm is not None:
            try:
                parent_cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
        yield _SpanHandle(None)
        return

    try:
        with obs_cm as obs:
            # A span the SDK built from a trace_context carries Langfuse's
            # "as_root" marker, so the server re-derives the trace name from
            # it. Without re-asserting the canonical name here, a late child
            # (semantic_cache.set, RAGAS) renames the whole trace after the
            # request finished.
            if pinned and join_trace_id:
                root_name = get_trace_root_name(join_trace_id)
                if root_name and root_name != name:
                    _stamp_trace_attributes(trace_name=root_name)
            handle = _SpanHandle(obs)
            try:
                yield handle
            except Exception as exc:
                handle.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
                raise
    finally:
        if parent_cm is not None:
            try:
                parent_cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
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
