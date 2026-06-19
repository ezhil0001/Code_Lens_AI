"""Quality metrics — RAGAS scores exported as Prometheus gauges.

AUDIT FIX L4
============
The original observability documentation referenced
``rag_faithfulness_score`` / ``rag_context_recall_score`` /
``rag_answer_relevancy_score`` gauges, and ``alert-rules.yml`` shipped
``LowFaithfullnessScore`` / ``LowContextRecall`` / ``LowAnswerRelevancy``
alerts on top of them.

But the gauges were never declared anywhere in the codebase. The alerts
could never fire — quality monitoring was *documented*, not *operational*.

This module:

1. Declares the three RAGAS Prometheus gauges (and a sample counter that
   the alert rules use to detect stale data).
2. Exposes :func:`publish_ragas_scores` as the single sink for any RAGAS
   evaluation result. ``RAGEvaluator`` calls it after each background
   evaluation run.
3. Keeps labels strictly bounded — only ``model`` and
   ``retriever_strategy`` (both enum-valued in practice). Never
   user_id / session_id / query_text — those would create a
   cardinality bomb (see audit finding L1).

Bucket-tuning note
------------------
RAGAS scores are bounded in [0.0, 1.0], so they're modelled as
``Gauge`` (a single point-in-time value), not ``Histogram``. We rely on
PromQL's ``avg_over_time(... [1h])`` to smooth judge variance — see
``alert-rules.yml`` :: ``LowFaithfullnessScore``.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram

    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-only
    _PROMETHEUS_AVAILABLE = False
    Counter = Gauge = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Metric declarations (module scope — declared exactly once at import time).
# ---------------------------------------------------------------------------

if _PROMETHEUS_AVAILABLE:
    RAG_FAITHFULNESS = Gauge(
        "rag_faithfulness_score",
        "Rolling-window faithfulness score from RAGAS (0.0–1.0). "
        "Low values indicate the LLM is hallucinating relative to retrieved context.",
        labelnames=["model", "retriever_strategy"],
    )

    RAG_CONTEXT_RECALL = Gauge(
        "rag_context_recall_score",
        "Rolling-window RAGAS context recall (0.0–1.0). "
        "Low values indicate retrieval is missing relevant chunks.",
        labelnames=["model", "retriever_strategy"],
    )

    RAG_ANSWER_RELEVANCY = Gauge(
        "rag_answer_relevancy_score",
        "Rolling-window RAGAS answer relevancy (0.0–1.0). "
        "Low values indicate the LLM is answering off-topic.",
        labelnames=["model", "retriever_strategy"],
    )

    RAG_CONTEXT_PRECISION = Gauge(
        "rag_context_precision_score",
        "Rolling-window RAGAS context precision (0.0–1.0). "
        "Low values indicate the reranker is leaving noise high in the result list.",
        labelnames=["model", "retriever_strategy"],
    )

    # Stale-data guard: alert rules join against rate(rag_quality_samples_total)
    # so they don't fire while the evaluator is silent.
    RAG_QUALITY_SAMPLES = Counter(
        "rag_quality_samples_total",
        "Number of RAGAS evaluations completed and published.",
        labelnames=["model", "retriever_strategy"],
    )

    # ── Phase F / H: LangGraph runtime metrics ────────────────────────────────

    # Per-node latency histogram (Phase F middleware + Phase H observability)
    NODE_LATENCY_MS = Histogram(
        "langgraph_node_latency_ms",
        "Execution latency per LangGraph node in milliseconds.",
        labelnames=["node_name", "agent"],
        buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000],
    ) if _PROMETHEUS_AVAILABLE else None

    # Guardrail check events counter (Phase F)
    GUARDRAIL_EVENTS = Counter(
        "langgraph_guardrail_events_total",
        "Guardrail check events — passed, blocked, or scrubbed.",
        labelnames=["check_name", "action"],
    ) if _PROMETHEUS_AVAILABLE else None

    # Per-agent token usage histogram (Phase H)
    AGENT_TOKENS = Histogram(
        "langgraph_agent_tokens_total",
        "LLM tokens consumed per agent per turn.",
        labelnames=["agent_name", "token_type"],
        buckets=[100, 500, 1000, 2000, 4000, 8000],
    ) if _PROMETHEUS_AVAILABLE else None

    # Graph edges traversed per turn (Phase H)
    GRAPH_EDGES_TRAVERSED = Histogram(
        "langgraph_edges_per_turn",
        "Number of graph edges traversed per query.",
        buckets=[1, 2, 3, 5, 8, 13, 21],
    ) if _PROMETHEUS_AVAILABLE else None

    # HIL interrupt counter (Phase E / H)
    HIL_INTERRUPTS = Counter(
        "langgraph_hil_interrupts_total",
        "Total HIL interrupt events.",
        labelnames=["reason"],
    ) if _PROMETHEUS_AVAILABLE else None

    # Long-term memory lookup counter (Phase C / H)
    LTM_LOOKUPS = Counter(
        "langgraph_ltm_lookups_total",
        "Long-term memory lookup events.",
        labelnames=["result"],
    ) if _PROMETHEUS_AVAILABLE else None

else:  # pragma: no cover
    RAG_FAITHFULNESS = None
    RAG_CONTEXT_RECALL = None
    RAG_ANSWER_RELEVANCY = None
    RAG_CONTEXT_PRECISION = None
    RAG_QUALITY_SAMPLES = None
    NODE_LATENCY_MS = None
    GUARDRAIL_EVENTS = None
    AGENT_TOKENS = None
    GRAPH_EDGES_TRAVERSED = None
    HIL_INTERRUPTS = None
    LTM_LOOKUPS = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Map RAGAS metric name → Prometheus gauge.
_GAUGE_BY_METRIC: Mapping[str, Any] = (
    {
        "faithfulness": RAG_FAITHFULNESS,
        "context_recall": RAG_CONTEXT_RECALL,
        "answer_relevancy": RAG_ANSWER_RELEVANCY,
        "context_precision": RAG_CONTEXT_PRECISION,
    }
    if _PROMETHEUS_AVAILABLE
    else {}
)


def publish_ragas_scores(
    scores: Mapping[str, Optional[float]],
    *,
    model: str,
    retriever_strategy: str,
) -> None:
    """Publish a RAGAS evaluation result to Prometheus gauges.

    Called from the background evaluation task in ``rag_evaluator.py``.

    Parameters
    ----------
    scores
        Mapping of RAGAS metric name to score in ``[0.0, 1.0]``. Missing or
        ``None`` values are skipped (RAGAS sometimes returns NaN / None for
        an individual metric while others succeed).
    model
        LLM identifier — bounded enum: ``"mistral-7b"``, ``"gpt-4o"``,
        ``"deepseek-chat"``, etc.
    retriever_strategy
        Retrieval strategy — bounded enum: ``"hybrid"``, ``"dense_only"``,
        ``"bm25_only"``, ``"pdr"``.

    Notes
    -----
    Silent on missing dependencies / metrics so this never breaks a request
    path. All emission failures are logged at DEBUG.
    """
    if not _PROMETHEUS_AVAILABLE:
        return

    labels = {"model": model, "retriever_strategy": retriever_strategy}
    published_any = False

    for metric_name, value in scores.items():
        gauge = _GAUGE_BY_METRIC.get(metric_name)
        if gauge is None:
            continue
        if value is None:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            logger.debug("Skipping non-numeric RAGAS score: %s=%r", metric_name, value)
            continue
        # RAGAS scores can occasionally come back NaN — skip those too.
        if score != score:  # NaN check
            continue
        # Clamp to the documented [0, 1] range — defensive against judge
        # models that occasionally return slightly out-of-bounds values.
        score = max(0.0, min(1.0, score))
        try:
            gauge.labels(**labels).set(score)
            published_any = True
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to publish %s: %s", metric_name, exc)

    if published_any and RAG_QUALITY_SAMPLES is not None:
        try:
            RAG_QUALITY_SAMPLES.labels(**labels).inc()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to increment quality samples counter: %s", exc)


__all__ = [
    "RAG_FAITHFULNESS",
    "RAG_CONTEXT_RECALL",
    "RAG_ANSWER_RELEVANCY",
    "RAG_CONTEXT_PRECISION",
    "RAG_QUALITY_SAMPLES",
    "NODE_LATENCY_MS",
    "GUARDRAIL_EVENTS",
    "AGENT_TOKENS",
    "GRAPH_EDGES_TRAVERSED",
    "HIL_INTERRUPTS",
    "LTM_LOOKUPS",
    "publish_ragas_scores",
]
