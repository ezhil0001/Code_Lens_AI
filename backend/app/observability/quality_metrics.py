"""Quality metrics — RAGAS scores and LangGraph runtime signals.

Observability for CodeLens_AI is provided by **Langfuse** (LLM tracing,
span-level latency, token/cost tracking, and online evaluation).

This module previously exported RAGAS scores and LangGraph runtime signals
as in-process metrics. That path has been removed. The metric
handles are retained as ``None`` sentinels and :func:`publish_ragas_scores`
is kept as a stable no-op so the many best-effort call sites across the
graph (``node_middleware``, guardrails, instrumentation helpers) continue
to import and call into this module without change.

RAGAS evaluation results are still computed asynchronously in
``rag_evaluator.py``, persisted to SQLite, and streamed to Langfuse for
trend analysis — see the README observability section.
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric handles — retained as None sentinels for API compatibility.
# All emission call sites already guard with `if <metric> is not None`.
# ---------------------------------------------------------------------------

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

def publish_ragas_scores(
    scores: Mapping[str, Optional[float]],
    *,
    model: str,
    retriever_strategy: str,
) -> None:
    """No-op sink for RAGAS evaluation results.

    Kept for backward compatibility with ``rag_evaluator.py``. RAGAS scores
    are streamed to Langfuse from the evaluator itself; this function no
    longer emits any in-process metrics.
    """
    logger.debug(
        "publish_ragas_scores called (model=%s, strategy=%s) — "
        "scores streamed via Langfuse.",
        model,
        retriever_strategy,
    )
    return None


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
