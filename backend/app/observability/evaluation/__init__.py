"""Langfuse Evaluation Framework for CodeLens_AI.

Modular, production-grade evaluation built on the Langfuse v4 SDK.

Layout
------
``code_evaluators``   Deterministic (code-based) evaluators — structure,
                      citations, grounding heuristics, guardrail compliance.
                      Run online on every sampled trace: fast, free, objective.
``online_evaluation`` Online scoring pipeline — runs code evaluators against a
                      finished request and publishes trace-linked scores.
                      Complements the RAGAS LLM-as-judge pipeline
                      (``app.observability.rag_evaluator``) which covers
                      faithfulness / context-recall / answer-relevancy.
``feedback``          User-feedback → Langfuse score mapping (thumbs, rating,
                      free-text comment), trace-linked.
``datasets``          Build regression datasets from production interactions.
``experiments``       Offline experiment runner (prompt / model / retrieval
                      comparison) on those datasets via ``run_experiment``.

Every entry point degrades to a safe no-op when Langfuse is disabled.
"""

from app.observability.evaluation.online_evaluation import evaluate_response_online  # noqa: F401
from app.observability.evaluation.feedback import record_user_feedback  # noqa: F401
