"""Offline experiments — compare prompts / models / retrieval configs on the
regression dataset before deployment.

Built on the Langfuse v4 ``run_experiment`` API: each run executes a *task*
(the pipeline variant under test) against every dataset item, applies the
deterministic code-evaluator suite plus optional custom evaluators, and links
results to the dataset in the Langfuse UI for side-by-side run comparison.

Usage (e.g. from a CI job or notebook):

    from app.observability.evaluation.experiments import run_rag_experiment

    result = run_rag_experiment(
        run_name="reranker-bge-v2",
        task=my_pipeline_fn,            # (item) -> answer str | dict
        metadata={"reranker": "bge-v2"},
    )

Two runs on the same dataset can then be diffed in Langfuse → Datasets →
Runs, giving objective regression detection before shipping a change.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _default_evaluators() -> List[Callable]:
    """Adapt the online code-evaluator suite to the experiment API.

    ``run_experiment`` evaluators receive keyword args (input, output,
    expected_output, metadata) and return ``Evaluation`` objects.
    """
    from langfuse import Evaluation

    from app.observability.evaluation.code_evaluators import EvalContext, run_code_evaluators

    def code_suite(*, input: Any = None, output: Any = None, expected_output: Any = None, metadata: Any = None, **_: Any) -> List[Evaluation]:  # noqa: A002
        query = input.get("query", "") if isinstance(input, dict) else str(input or "")
        answer = output if isinstance(output, str) else (output or {}).get("answer", "") if isinstance(output, dict) else str(output or "")
        extra = output if isinstance(output, dict) else {}
        ctx = EvalContext(
            query=query,
            answer=answer,
            retrieved_chunks=list(extra.get("retrieved_chunks") or []),
            reranked_chunks=list(extra.get("reranked_chunks") or []),
            rerank_scores=list(extra.get("rerank_scores") or []),
            sources=list(extra.get("sources") or []),
            routing_agents=list(extra.get("routing_agents") or []),
            routing_confidence=float(extra.get("routing_confidence") or 0.0),
            guardrail_passed=bool(extra.get("guardrail_passed", True)),
            nodes_visited=list(extra.get("nodes_visited") or []),
        )
        evals: List[Evaluation] = []
        for s in run_code_evaluators(ctx):
            try:
                evals.append(Evaluation(name=s.name, value=s.value, comment=s.comment))
            except Exception as exc:  # noqa: BLE001
                logger.debug("[eval] experiment evaluation build failed (%s): %s", s.name, exc)
        # Exact-match vs annotated expected output, when present.
        if expected_output and isinstance(expected_output, str) and answer:
            evals.append(Evaluation(
                name="expected_output_overlap",
                value=round(_token_overlap(answer, expected_output), 2),
                comment="lexical overlap with annotated expected output",
            ))
        return evals

    return [code_suite]


def _token_overlap(a: str, b: str) -> float:
    import re
    ta = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{3,}", a.lower()))
    tb = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{3,}", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def run_rag_experiment(
    *,
    run_name: str,
    task: Callable[..., Any],
    dataset_name: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
    extra_evaluators: Optional[List[Callable]] = None,
    max_concurrency: int = 4,
) -> Optional[Any]:
    """Run an offline experiment against the regression dataset.

    Parameters
    ----------
    run_name:
        Unique name of this run (e.g. ``"groq-llama-3.3-70b"`` or
        ``"prompt-v7"``) — compared side-by-side in the Langfuse UI.
    task:
        Callable receiving ``item`` (dataset item; ``item.input["query"]``)
        and returning either the answer string or a dict with ``answer`` plus
        optional pipeline internals (``reranked_chunks``, ``sources``, …) for
        richer evaluation.
    dataset_name:
        Defaults to the managed regression dataset.
    extra_evaluators:
        Additional ``run_experiment``-style evaluator callables (e.g. an
        LLM-as-judge) appended to the deterministic suite.

    Returns the SDK ``ExperimentResult`` or ``None`` when Langfuse is
    disabled / dataset empty. Never raises.
    """
    try:
        from app.observability.langfuse_client import get_client
        from app.observability.evaluation.datasets import DEFAULT_DATASET, get_dataset_items

        client = get_client()
        if client is None:
            logger.info("[eval] experiment skipped — Langfuse disabled")
            return None

        ds_name = dataset_name or DEFAULT_DATASET
        items = get_dataset_items(ds_name)
        if not items:
            logger.warning("[eval] experiment skipped — dataset '%s' is empty", ds_name)
            return None

        evaluators = _default_evaluators() + list(extra_evaluators or [])

        result = client.run_experiment(
            name=f"rag-eval::{ds_name}",
            run_name=run_name,
            description=description or f"Offline RAG evaluation run '{run_name}'",
            data=items,
            task=task,
            evaluators=evaluators,
            metadata=metadata,
            max_concurrency=max_concurrency,
        )
        logger.info("[eval] experiment run '%s' completed on dataset '%s'", run_name, ds_name)
        # Make sure results are delivered before a short-lived CI process exits.
        from app.observability.langfuse_client import flush
        flush()
        return result
    except Exception as exc:  # noqa: BLE001
        logger.error("[eval] experiment run failed: %s", exc)
        return None
