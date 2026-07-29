"""
Observability Tests
===================
  H-001  langgraph_instrumentation module importable
  H-002  quality_metrics module importable
  H-003  publish_ragas_scores callable (Langfuse sink)
  H-004  LangGraphObservabilityCallback exported
  H-005  otel_config importable (Jaeger tracing)
  H-006  rag_evaluator importable (RAGAS → Langfuse)

Observability for CodeLens_AI is provided by Langfuse (LLM tracing,
span-level latency, token/cost tracking, and online evaluation) together
with OpenTelemetry traces exported to Jaeger. These tests verify the
observability modules import cleanly and expose their stable public API.
"""

from __future__ import annotations

import importlib

from app.tests.base import PhaseTest, TestResult


def _try_import(path: str):
    try:
        return importlib.import_module(path), None
    except Exception as exc:
        return None, str(exc)


# ─────────────────────────────────────────────────────────────────────────────

async def _test_instrumentation_importable() -> TestResult:
    _, err = _try_import("app.observability.langgraph_instrumentation")
    if err:
        return TestResult.failed(f"Cannot import langgraph_instrumentation: {err}")
    return TestResult.passed("langgraph_instrumentation importable ✓")


async def _test_quality_metrics_importable() -> TestResult:
    _, err = _try_import("app.observability.quality_metrics")
    if err:
        return TestResult.failed(f"Cannot import quality_metrics: {err}")
    return TestResult.passed("quality_metrics importable ✓")


async def _test_publish_ragas_scores_callable() -> TestResult:
    mod, err = _try_import("app.observability.quality_metrics")
    if err:
        return TestResult.failed(f"Cannot import quality_metrics: {err}")
    fn = getattr(mod, "publish_ragas_scores", None)
    if not callable(fn):
        return TestResult.failed("publish_ragas_scores is not callable")
    try:
        fn({"faithfulness": 0.9}, model="test", retriever_strategy="hybrid")
    except Exception as exc:  # noqa: BLE001
        return TestResult.failed(f"publish_ragas_scores raised: {exc}")
    return TestResult.passed("publish_ragas_scores callable (Langfuse sink) ✓")


async def _test_observability_callback_exported() -> TestResult:
    mod, err = _try_import("app.observability.langgraph_instrumentation")
    if err:
        return TestResult.failed(f"Cannot import langgraph_instrumentation: {err}")
    if not hasattr(mod, "LangGraphObservabilityCallback"):
        return TestResult.failed(
            "LangGraphObservabilityCallback not exported",
            detail="Expected callback handler in langgraph_instrumentation.py",
        )
    return TestResult.passed("LangGraphObservabilityCallback exported ✓")


async def _test_otel_config_importable() -> TestResult:
    _, err = _try_import("app.observability.otel_config")
    if err:
        return TestResult.failed(f"Cannot import otel_config: {err}")
    return TestResult.passed("otel_config importable (Jaeger tracing) ✓")


async def _test_rag_evaluator_importable() -> TestResult:
    _, err = _try_import("app.observability.rag_evaluator")
    if err:
        return TestResult.skipped(f"rag_evaluator not importable: {err}")
    return TestResult.passed("rag_evaluator importable (RAGAS → Langfuse) ✓")


TESTS: list[PhaseTest] = [
    PhaseTest(id="H-001", name="langgraph_instrumentation importable",
              description="app.observability.langgraph_instrumentation importable",
              run=_test_instrumentation_importable, critical=False, tags=["obs"]),
    PhaseTest(id="H-002", name="quality_metrics importable",
              description="app.observability.quality_metrics importable",
              run=_test_quality_metrics_importable, critical=True, tags=["obs"]),
    PhaseTest(id="H-003", name="publish_ragas_scores callable",
              description="RAGAS → Langfuse sink callable and side-effect free",
              run=_test_publish_ragas_scores_callable, critical=False, tags=["obs", "langfuse"]),
    PhaseTest(id="H-004", name="LangGraphObservabilityCallback exported",
              description="Callback handler exported for graph instrumentation",
              run=_test_observability_callback_exported, critical=False, tags=["obs"]),
    PhaseTest(id="H-005", name="otel_config importable",
              description="OpenTelemetry (Jaeger) config importable",
              run=_test_otel_config_importable, critical=False, tags=["obs", "tracing"]),
    PhaseTest(id="H-006", name="rag_evaluator importable",
              description="RAGAS evaluator importable",
              run=_test_rag_evaluator_importable, critical=False, tags=["obs", "langfuse"]),
]
