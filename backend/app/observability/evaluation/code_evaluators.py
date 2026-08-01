"""Deterministic code evaluators — objective, zero-cost quality checks.

These run online on every sampled production trace (they are pure Python:
no LLM, no network, sub-millisecond). Semantic evaluation (faithfulness,
context recall, answer relevancy) stays with the RAGAS LLM-as-judge pipeline
in ``app.observability.rag_evaluator`` — do NOT duplicate it here.

Each evaluator receives an :class:`EvalContext` snapshot of the finished
request and returns an :class:`EvalScore` (or ``None`` to abstain when the
check is not applicable). Scores use Langfuse-native data types:
NUMERIC, BOOLEAN, CATEGORICAL.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data contracts
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalContext:
    """Snapshot of a finished request handed to the code evaluators."""

    query: str
    answer: str
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    reranked_chunks: List[Dict[str, Any]] = field(default_factory=list)
    rerank_scores: List[float] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    routing_agents: List[str] = field(default_factory=list)
    routing_confidence: float = 0.0
    guardrail_passed: bool = True
    guardrail_violations: List[Any] = field(default_factory=list)
    cache_hit: bool = False
    nodes_visited: List[str] = field(default_factory=list)
    intent: Optional[str] = None
    trace_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "EvalContext":
        """Build from a LangGraph final AgentState. Never raises."""
        try:
            return cls(
                query=state.get("query", "") or "",
                answer=state.get("final_response", "") or "",
                retrieved_chunks=list(state.get("retrieved_chunks") or []),
                reranked_chunks=list(state.get("reranked_chunks") or []),
                rerank_scores=list(state.get("rerank_scores") or []),
                sources=list(state.get("sources") or []),
                routing_agents=list(state.get("routing_agents") or []),
                routing_confidence=float(state.get("routing_confidence") or 0.0),
                guardrail_passed=bool(state.get("guardrail_passed", True)),
                guardrail_violations=list(state.get("guardrail_violations") or []),
                cache_hit=bool(state.get("cache_hit", False)),
                nodes_visited=list(state.get("nodes_visited") or []),
                intent=state.get("intent") or state.get("routing_decision"),
                trace_id=state.get("langfuse_trace_id"),
                session_id=state.get("session_id"),
                user_id=state.get("user_id"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[eval] EvalContext.from_state failed: %s", exc)
            return cls(query="", answer="")


@dataclass
class EvalScore:
    """One score destined for Langfuse ``create_score``."""

    name: str
    value: Any  # float | bool | str, matching data_type
    data_type: str  # NUMERIC | BOOLEAN | CATEGORICAL
    comment: Optional[str] = None


Evaluator = Callable[[EvalContext], Optional[EvalScore]]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluators
# ─────────────────────────────────────────────────────────────────────────────

def eval_response_structure(ctx: EvalContext) -> Optional[EvalScore]:
    """BOOLEAN — response is non-empty, not an error placeholder, and any
    opened markdown code fence is properly closed."""
    ans = ctx.answer.strip()
    ok = bool(ans) and not ans.startswith("[No response]") and "[unavailable]" not in ans.lower()
    if ok and ans.count("```") % 2 != 0:
        ok = False
        comment = "unterminated code fence"
    else:
        comment = None if ok else "empty or placeholder response"
    return EvalScore("response_structure_valid", ok, "BOOLEAN", comment)


def eval_response_completeness(ctx: EvalContext) -> Optional[EvalScore]:
    """NUMERIC 0–1 — length/shape heuristic for answer completeness.

    Penalises truncated answers (ends mid-sentence) and one-liners for
    non-trivial queries. Cheap proxy; semantic completeness is RAGAS's job.
    """
    ans = ctx.answer.strip()
    if not ans:
        return EvalScore("response_completeness", 0.0, "NUMERIC", "empty answer")
    score = 1.0
    if len(ans) < 40:
        score -= 0.4
    if not re.search(r"[.!?`\)\]]\s*$", ans):
        score -= 0.3  # ends mid-sentence → likely truncated
    return EvalScore("response_completeness", round(max(score, 0.0), 2), "NUMERIC")


def eval_citation_quality(ctx: EvalContext) -> Optional[EvalScore]:
    """NUMERIC 0–1 — did a RAG answer come with sources, and do inline file
    references correspond to retrieved chunks? Abstains on cache hits and
    non-RAG paths (no retrieval performed)."""
    if ctx.cache_hit or (not ctx.retrieved_chunks and not ctx.reranked_chunks):
        return None
    chunks = ctx.reranked_chunks or ctx.retrieved_chunks
    if not chunks:
        return None
    score = 0.0
    if ctx.sources:
        score += 0.6
    # Do any retrieved file names actually appear in the answer?
    fnames = set()
    for c in chunks:
        meta = c.get("metadata", {}) if isinstance(c, dict) else {}
        src = meta.get("source") or meta.get("file_path") or ""
        if src:
            fnames.add(str(src).rsplit("/", 1)[-1])
    if fnames and any(fn in ctx.answer for fn in fnames):
        score += 0.4
    elif not fnames:
        score += 0.2  # no filenames available to cite — partial credit
    return EvalScore("citation_quality", round(min(score, 1.0), 2), "NUMERIC")


def eval_context_utilization(ctx: EvalContext) -> Optional[EvalScore]:
    """NUMERIC 0–1 — lexical overlap between answer and reranked context.

    A grounding proxy: near-zero overlap on a RAG answer signals the model
    ignored retrieval (potential hallucination). Abstains without retrieval.
    """
    chunks = ctx.reranked_chunks or ctx.retrieved_chunks
    if ctx.cache_hit or not chunks or not ctx.answer:
        return None
    context_text = " ".join(
        (c.get("content") or c.get("page_content") or "") for c in chunks if isinstance(c, dict)
    ).lower()
    if not context_text.strip():
        return None
    answer_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{3,}", ctx.answer.lower()))
    if not answer_tokens:
        return None
    context_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{3,}", context_text))
    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    return EvalScore("context_utilization", round(min(overlap, 1.0), 2), "NUMERIC")


def eval_retrieval_quality(ctx: EvalContext) -> Optional[EvalScore]:
    """NUMERIC 0–1 — top rerank score as a retrieval-relevance signal.

    Cross-encoder relevance of the best chunk is a strong objective proxy
    for "did we find anything relevant at all". Abstains without reranking.
    """
    if not ctx.rerank_scores:
        return None
    top = float(ctx.rerank_scores[0])
    # BGE cross-encoder outputs are roughly logits; squash into 0–1 when needed.
    if top < 0.0 or top > 1.0:
        import math
        top = 1.0 / (1.0 + math.exp(-top))
    return EvalScore("retrieval_quality", round(top, 3), "NUMERIC")


def eval_routing_confidence(ctx: EvalContext) -> Optional[EvalScore]:
    """CATEGORICAL — bucketed intent-router confidence for trend analysis."""
    if ctx.cache_hit:
        return None
    c = ctx.routing_confidence
    bucket = "high" if c >= 0.8 else "medium" if c >= 0.5 else "low"
    return EvalScore(
        "routing_confidence_bucket", bucket, "CATEGORICAL",
        f"confidence={c:.2f} agents={','.join(ctx.routing_agents) or 'n/a'}",
    )


def eval_guardrail_compliance(ctx: EvalContext) -> Optional[EvalScore]:
    """BOOLEAN — did the response pass all guardrails (safety constraint)."""
    return EvalScore(
        "guardrail_compliance",
        bool(ctx.guardrail_passed),
        "BOOLEAN",
        f"{len(ctx.guardrail_violations)} violation(s)" if ctx.guardrail_violations else None,
    )


def eval_agent_goal_completion(ctx: EvalContext) -> Optional[EvalScore]:
    """NUMERIC 0–1 — did the agent pipeline execute end-to-end?

    Checks that the graph reached response assembly, at least one agent
    produced output (or cache served), and no fallback stubs fired.
    """
    visited = ctx.nodes_visited
    if not visited:
        return None
    score = 0.0
    if "response_node" in visited:
        score += 0.5
    if ctx.cache_hit or ctx.answer.strip():
        score += 0.3
    if not any("fallback" in n for n in visited):
        score += 0.2
    return EvalScore("agent_goal_completion", round(min(score, 1.0), 2), "NUMERIC")


#: The online evaluator suite, in execution order.
CODE_EVALUATORS: List[Evaluator] = [
    eval_response_structure,
    eval_response_completeness,
    eval_citation_quality,
    eval_context_utilization,
    eval_retrieval_quality,
    eval_routing_confidence,
    eval_guardrail_compliance,
    eval_agent_goal_completion,
]


def run_code_evaluators(ctx: EvalContext) -> List[EvalScore]:
    """Run all code evaluators; collect non-abstained scores. Never raises."""
    scores: List[EvalScore] = []
    for ev in CODE_EVALUATORS:
        try:
            s = ev(ctx)
            if s is not None:
                scores.append(s)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[eval] evaluator %s failed: %s", getattr(ev, "__name__", ev), exc)
    return scores
