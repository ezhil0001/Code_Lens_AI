"""
LangGraph Runtime Instrumentation
===================================
Automatic OTEL span injection for every LangGraph node, without requiring
individual nodes to import this module. Spans surface in Langfuse, which also
handles LLM-level tracing and evaluation.

Two integration paths are provided so callers can choose the right trade-off
between explicitness and convenience:

1. Decorator path — ``instrument_node(node_fn, node_name=...)``
   Wraps any async node function with timing and an OTEL span. Use this when
   building the graph so every node emits consistent telemetry without
   boilerplate in each node function.

2. Callback path — ``LangGraphObservabilityCallback``
   A LangChain/LangGraph callback handler that listens to
   ``on_chain_start`` / ``on_chain_end`` events and records node timing,
   edge traversal, token usage, HIL interrupts, and LTM lookups
   automatically from the event stream — zero node modification required.

Usage at graph build time:
    from app.observability.langgraph_instrumentation import (
        instrument_node, LangGraphObservabilityCallback,
    )

    # Wrap individual nodes
    builder.add_node("code_retrieve_node",
        instrument_node(code_retrieve_node, node_name="code_retrieve"))

    # Attach callback to all invocations
    config = {
        "callbacks": [LangGraphObservabilityCallback()],
        "configurable": {...},
    }

Tested by:
  H-001  app.observability.langgraph_instrumentation importable
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Runtime metric handles (imported from quality_metrics — None sentinels;
# emission is best-effort and no-ops when handles are unset).
# ─────────────────────────────────────────────────────────────────────────────

def _get_metrics():
    """Lazy-load quality_metrics to avoid circular imports at startup."""
    try:
        from app.observability.quality_metrics import (  # type: ignore
            NODE_LATENCY_MS,
            AGENT_TOKENS,
            GRAPH_EDGES_TRAVERSED,
            HIL_INTERRUPTS,
            LTM_LOOKUPS,
            GUARDRAIL_EVENTS,
        )
        return {
            "NODE_LATENCY_MS": NODE_LATENCY_MS,
            "AGENT_TOKENS": AGENT_TOKENS,
            "GRAPH_EDGES_TRAVERSED": GRAPH_EDGES_TRAVERSED,
            "HIL_INTERRUPTS": HIL_INTERRUPTS,
            "LTM_LOOKUPS": LTM_LOOKUPS,
            "GUARDRAIL_EVENTS": GUARDRAIL_EVENTS,
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("[langgraph_instrumentation] metrics unavailable: %s", exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# OTEL tracer
# ─────────────────────────────────────────────────────────────────────────────

def _get_tracer():
    """Return an OTEL tracer, or None if OTEL is not configured."""
    try:
        from opentelemetry import trace  # type: ignore
        return trace.get_tracer("langgraph.codelens_ai")
    except Exception:  # noqa: BLE001
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Decorator path: instrument_node()
# ─────────────────────────────────────────────────────────────────────────────

def instrument_node(
    node_fn: Callable,
    *,
    node_name: str,
    agent_name: str = "unknown",
) -> Callable:
    """Wrap an async LangGraph node with an OTEL span + timing.

    Parameters
    ----------
    node_fn
        The async node function to wrap.  Signature:
        ``async def node(state: dict, config: RunnableConfig) -> dict``
    node_name
        Short human-readable name used as the OTEL span name and the
        ``node_name`` telemetry label value (e.g. ``"code_retrieve"``).
    agent_name
        Which agent sub-graph this node belongs to (e.g. ``"CodeAgent"``).
        Used as the ``agent`` label on ``NODE_LATENCY_MS``.

    Returns
    -------
    Callable
        A wrapped coroutine with the same signature as *node_fn*.
    """
    @functools.wraps(node_fn)
    async def _wrapped(state: dict, config: Any = None) -> dict:
        start_ms = time.perf_counter() * 1000
        tracer = _get_tracer()
        metrics = _get_metrics()

        span_context = None
        if tracer is not None:
            span_context = tracer.start_as_current_span(
                f"langgraph.node.{node_name}",
                attributes={
                    "langgraph.node": node_name,
                    "langgraph.agent": agent_name,
                },
            )

        try:
            if span_context is not None:
                with span_context:
                    result = await node_fn(state, config)
            else:
                result = await node_fn(state, config)
        except Exception as exc:
            logger.error("[instrument_node] %s raised: %s", node_name, exc)
            raise
        finally:
            elapsed_ms = time.perf_counter() * 1000 - start_ms
            _emit_node_latency(metrics, node_name, agent_name, elapsed_ms)
            logger.debug(
                "[langgraph] %s completed in %.1fms", node_name, elapsed_ms
            )

        return result

    return _wrapped


def _emit_node_latency(
    metrics: Dict[str, Any],
    node_name: str,
    agent_name: str,
    elapsed_ms: float,
) -> None:
    """Safely emit NODE_LATENCY_MS — never raises."""
    metric = metrics.get("NODE_LATENCY_MS")
    if metric is None:
        return
    try:
        metric.labels(node_name=node_name, agent=agent_name).observe(elapsed_ms)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[langgraph_instrumentation] emit_node_latency failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Callback path: LangGraphObservabilityCallback
# ─────────────────────────────────────────────────────────────────────────────

class LangGraphObservabilityCallback:
    """LangChain callback handler that records LangGraph runtime metrics.

    Attach to a graph invocation via the ``callbacks`` key in
    ``RunnableConfig``::

        config = {
            "callbacks": [LangGraphObservabilityCallback()],
            "configurable": {"thread_id": "..."},
        }

    Metrics recorded:
      - ``NODE_LATENCY_MS``       on every chain start/end pair
      - ``GRAPH_EDGES_TRAVERSED`` once per graph run (on_chain_end for root)
      - ``AGENT_TOKENS``          on on_llm_end (if token usage is available)
      - ``HIL_INTERRUPTS``        when state contains hil_required=True
      - ``LTM_LOOKUPS``           when state contains ltm lookup result
    """

    def __init__(self) -> None:
        self._node_start_times: Dict[str, float] = {}
        self._edges_this_run: int = 0

    # ── on_chain_start ────────────────────────────────────────────────────────

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: str = "",
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Record the start time for a node and count edges."""
        node_name = (serialized or {}).get("name") or (
            (tags or [""])[0] if tags else ""
        )
        if node_name:
            self._node_start_times[run_id] = time.perf_counter() * 1000
            if parent_run_id:
                self._edges_this_run += 1

        # Check for HIL interrupt in inputs
        metrics = _get_metrics()
        if isinstance(inputs, dict) and inputs.get("hil_required"):
            hil_reason = inputs.get("hil_reason", "unknown")
            hil_metric = metrics.get("HIL_INTERRUPTS")
            if hil_metric is not None:
                try:
                    hil_metric.labels(reason=str(hil_reason)[:64]).inc()
                except Exception:  # noqa: BLE001
                    pass

    # ── on_chain_end ──────────────────────────────────────────────────────────

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: str = "",
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Record node latency and graph edge count on completion."""
        metrics = _get_metrics()
        start_ms = self._node_start_times.pop(run_id, None)
        if start_ms is not None:
            elapsed = time.perf_counter() * 1000 - start_ms
            node_name = (tags or ["unknown"])[0] if tags else "unknown"
            _emit_node_latency(metrics, node_name, "unknown", elapsed)

        # Root chain end → emit total edges for this turn
        if parent_run_id is None:
            edge_metric = metrics.get("GRAPH_EDGES_TRAVERSED")
            if edge_metric is not None and self._edges_this_run > 0:
                try:
                    edge_metric.observe(float(self._edges_this_run))
                except Exception:  # noqa: BLE001
                    pass
            self._edges_this_run = 0

    # ── on_llm_end ────────────────────────────────────────────────────────────

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: str = "",
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Record token usage from LLM response."""
        metrics = _get_metrics()
        token_metric = metrics.get("AGENT_TOKENS")
        if token_metric is None:
            return

        agent_name = (tags or ["unknown"])[0] if tags else "unknown"
        try:
            # LangChain LLMResult usage metadata
            if hasattr(response, "llm_output") and response.llm_output:
                usage = response.llm_output.get("token_usage") or {}
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                if prompt_tokens:
                    token_metric.labels(
                        agent_name=agent_name, token_type="prompt"
                    ).observe(float(prompt_tokens))
                if completion_tokens:
                    token_metric.labels(
                        agent_name=agent_name, token_type="completion"
                    ).observe(float(completion_tokens))
        except Exception as exc:  # noqa: BLE001
            logger.debug("[langgraph_instrumentation] on_llm_end error: %s", exc)

    # ── on_chain_error ────────────────────────────────────────────────────────

    def on_chain_error(
        self,
        error: Exception,
        *,
        run_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Clean up timing state on node errors."""
        self._node_start_times.pop(run_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: record_ltm_lookup — called from long_term_store.py
# ─────────────────────────────────────────────────────────────────────────────

def record_ltm_lookup(result: str) -> None:
    """Emit a LTM_LOOKUPS counter increment.

    Parameters
    ----------
    result
        ``"hit"`` when facts were retrieved, ``"miss"`` when none found,
        ``"error"`` on retrieval failure.
    """
    metrics = _get_metrics()
    metric = metrics.get("LTM_LOOKUPS")
    if metric is None:
        return
    try:
        metric.labels(result=result).inc()
    except Exception as exc:  # noqa: BLE001
        logger.debug("[langgraph_instrumentation] record_ltm_lookup failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: record_hil_interrupt — called from hil_node.py
# ─────────────────────────────────────────────────────────────────────────────

def record_hil_interrupt(reason: str) -> None:
    """Emit a HIL_INTERRUPTS counter increment.

    Parameters
    ----------
    reason
        Short reason string, e.g. ``"low_confidence"`` or
        ``"destructive_intent"``.
    """
    metrics = _get_metrics()
    metric = metrics.get("HIL_INTERRUPTS")
    if metric is None:
        return
    try:
        metric.labels(reason=reason[:64]).inc()
    except Exception as exc:  # noqa: BLE001
        logger.debug("[langgraph_instrumentation] record_hil_interrupt failed: %s", exc)


__all__ = [
    "instrument_node",
    "LangGraphObservabilityCallback",
    "record_ltm_lookup",
    "record_hil_interrupt",
]
