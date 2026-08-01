"""Online evaluation — score every sampled production trace automatically.

Runs the deterministic code-evaluator suite against a finished request and
publishes trace-linked Langfuse scores. Designed to be called from
``response_node`` right after the RAGAS evaluation is scheduled:

    from app.observability.evaluation import evaluate_response_online
    evaluate_response_online(state)          # sync, sub-millisecond compute

Publishing is delegated to the Langfuse SDK's background batch queue, so this
adds no network latency to the request path. Sampling consistency is inherited
from the tracing layer: a trace id only exists in state when the request was
sampled, so we never emit orphan scores. No-op when Langfuse is disabled.

Deduplication: scores are published exactly once per trace — this is the only
call site, and the RAGAS pipeline publishes a disjoint score set
(faithfulness / context_recall / answer_relevancy / rag_aggregate_score).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.observability.evaluation.code_evaluators import (
    EvalContext,
    EvalScore,
    run_code_evaluators,
)

logger = logging.getLogger(__name__)


def evaluate_response_online(
    state: Dict[str, Any],
    *,
    trace_id_override: Optional[str] = None,
) -> List[EvalScore]:
    """Evaluate a finished request and publish scores onto its trace.

    ``trace_id_override`` lets the caller supply the CURRENT request's trace
    id (config-resolved) — correct on checkpoint resume where state may carry
    a stale id (M-3). Returns the computed scores. Never raises.
    """
    try:
        ctx = EvalContext.from_state(state)
        if trace_id_override:
            ctx.trace_id = trace_id_override
        if not ctx.trace_id:
            # Request wasn't sampled (or Langfuse disabled) — evaluation would
            # produce orphan scores; skip entirely for consistency (M-4).
            return []

        scores = run_code_evaluators(ctx)
        if not scores:
            return []

        _publish(ctx.trace_id, scores, session_id=ctx.session_id)
        logger.debug(
            "[eval] online evaluation published %d scores for trace=%s",
            len(scores), ctx.trace_id,
        )
        return scores
    except Exception as exc:  # noqa: BLE001
        logger.debug("[eval] online evaluation skipped: %s", exc)
        return []


def _publish(trace_id: str, scores: List[EvalScore], *, session_id: Optional[str] = None) -> None:
    """Send scores to Langfuse via the batched SDK client. Never raises."""
    try:
        from app.observability.langfuse_client import get_client

        client = get_client()
        if client is None:
            return
        for s in scores:
            try:
                # BOOLEAN scores: SDK value type is Union[float, str] — cast
                # bools explicitly (True→1.0) for type consistency.
                value = s.value
                if s.data_type == "BOOLEAN" and isinstance(value, bool):
                    value = 1.0 if value else 0.0
                client.create_score(
                    trace_id=trace_id,
                    name=s.name,
                    value=value,
                    data_type=s.data_type,  # type: ignore[arg-type]
                    comment=s.comment,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("[eval] create_score(%s) failed: %s", s.name, exc)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[eval] score publish skipped: %s", exc)
