"""
Node middleware — wraps any async LangGraph node with timeout, retry, and tracing.

Applied at graph-build time so individual node functions stay focused on
their domain logic.  The wrapper handles three concerns that are identical
across every node:

  Timeout: asyncio.wait_for terminates nodes that hang due to a slow
  downstream service.  The default is 30s; retrieval nodes use 15s.

  Retry: exponential backoff on transient errors.  ChromaDB occasionally
  returns a gRPC UNAVAILABLE on cold start; two retries cover that window
  without waiting longer than ~6s total.  The retry set is explicit so a
  programming error (KeyError, AttributeError) is not silently swallowed.

  OTEL: emits a span per node execution with node_name and session_id
  attributes.  The Prometheus histogram gives per-node p95 latency visible
  in Grafana without any additional instrumentation in the node itself.

Usage:
    builder.add_node(
        "code_retrieve_node",
        with_node_middleware(
            code_retrieve_node,
            node_name="code_retrieve",
            max_retries=2,
            retry_on=(ChromaDBError,),
        )
    )
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple, Type

from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

# Type alias for a LangGraph async node function
NodeFn = Callable[..., Any]


def with_node_middleware(
    node_fn: NodeFn,
    *,
    node_name: str,
    enable_retry: bool = True,
    max_retries: int = 3,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
    timeout_seconds: float = 30.0,
    trace: bool = True,
) -> NodeFn:
    """Wrap a LangGraph node with retry, timeout, and optional observability.

    Parameters
    ----------
    node_fn
        The async node function to wrap.  Must accept ``(state, config)``
        and return a partial state dict.
    node_name
        Human-readable name used in logs, OTEL spans, and Prometheus labels.
    enable_retry
        Set to ``False`` to disable retry logic entirely.
    max_retries
        Maximum number of additional attempts after the first failure.
        Total calls = max_retries + 1.
    retry_on
        Tuple of exception types that trigger a retry.  Exceptions not in
        this tuple propagate immediately without retrying.
    timeout_seconds
        Per-attempt wall-clock timeout enforced via ``asyncio.wait_for()``.
        ``asyncio.TimeoutError`` is **not** in ``retry_on`` by default —
        it propagates immediately.
    trace
        If True, attempts to create an OTEL span and record Prometheus
        latency.  Silently skipped when the libraries are unavailable.
    """

    @functools.wraps(node_fn)
    async def wrapped(state: Dict[str, Any], config: RunnableConfig = None) -> dict:
        start_time = time.perf_counter()
        attempt = 0

        while True:
            try:
                # ── Optional OTEL span ────────────────────────────────────
                span_ctx = _optional_span(node_name, state) if trace else _noop_span()

                with span_ctx:
                    result = await asyncio.wait_for(
                        node_fn(state, config),
                        timeout=timeout_seconds,
                    )

                elapsed_ms = (time.perf_counter() - start_time) * 1000
                _record_latency(node_name, elapsed_ms)
                return result

            except asyncio.TimeoutError:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.error(
                    "[%s] Timeout after %.1fs (attempt %d)",
                    node_name, timeout_seconds, attempt + 1,
                )
                raise

            except retry_on as exc:  # type: ignore[misc]
                if not enable_retry or attempt >= max_retries:
                    logger.error(
                        "[%s] Failed after %d attempt(s): %s",
                        node_name, attempt + 1, exc,
                    )
                    raise

                attempt += 1
                wait_s = min(2 ** attempt, 30)
                logger.warning(
                    "[%s] Retry %d/%d in %.1fs — %s: %s",
                    node_name, attempt, max_retries, wait_s,
                    type(exc).__name__, exc,
                )
                await asyncio.sleep(wait_s)

    return wrapped


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

class _noop_span:
    """Context manager that does nothing — used when OTEL is disabled."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


def _optional_span(node_name: str, state: Dict[str, Any]):
    """Return an OTEL span context manager, or a no-op if OTEL is unavailable."""
    try:
        from opentelemetry import trace as otel_trace  # type: ignore
        tracer = otel_trace.get_tracer("langgraph.supervisor")
        ctx = tracer.start_as_current_span(f"langgraph.node.{node_name}")
        try:
            span = otel_trace.get_current_span()
            span.set_attribute("node.name", node_name)
            span.set_attribute("session.id", str(state.get("session_id", "")))
        except Exception:  # noqa: BLE001
            pass
        return ctx
    except Exception:  # noqa: BLE001
        return _noop_span()


def _record_latency(node_name: str, elapsed_ms: float) -> None:
    """Record per-node latency in the Prometheus histogram — best-effort."""
    try:
        from app.observability.quality_metrics import NODE_LATENCY_MS  # type: ignore
        if NODE_LATENCY_MS is not None:
            NODE_LATENCY_MS.labels(
                node_name=node_name,
                agent="unknown",
            ).observe(elapsed_ms)
    except Exception:  # noqa: BLE001
        pass
