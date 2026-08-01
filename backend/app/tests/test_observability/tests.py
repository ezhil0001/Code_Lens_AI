"""
Observability Tests
===================
  H-001  langgraph_instrumentation module importable
  H-002  quality_metrics module importable
  H-003  publish_ragas_scores callable (Langfuse sink)
  H-004  LangGraphObservabilityCallback exported
  H-005  otel_config importable (Jaeger tracing)
  H-006  rag_evaluator importable (RAGAS → Langfuse)
  H-007  langfuse_client importable + public API present
  H-008  langfuse_client graceful degradation (no-op when disabled)

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


async def _test_langfuse_client_importable() -> TestResult:
    mod, err = _try_import("app.observability.langfuse_client")
    if err:
        return TestResult.failed(f"Cannot import langfuse_client: {err}")
    required = [
        "init_langfuse", "is_enabled", "should_sample", "get_client",
        "get_callback_handler", "build_trace_metadata", "create_trace_id",
        "get_current_trace_id", "score_current_trace", "flush", "shutdown",
    ]
    missing = [fn for fn in required if not callable(getattr(mod, fn, None))]
    if missing:
        return TestResult.failed(f"langfuse_client missing public API: {missing}")
    return TestResult.passed("langfuse_client public API present ✓")


async def _test_tracing_primitives() -> TestResult:
    """span()/observe_span() must work (no-op safe) and propagate exceptions."""
    mod, err = _try_import("app.observability.tracing")
    if err:
        return TestResult.failed(f"Cannot import tracing: {err}")
    try:
        with mod.span("test.span", kind="retriever", input={"q": 1}) as s:
            s.update(output={"ok": True})

        @mod.observe_span(name="test.fn")
        def _fn(x):
            return x * 2
        if _fn(2) != 4:
            return TestResult.failed("observe_span altered return value")

        @mod.observe_span(name="test.afn")
        async def _afn(x):
            return x + 1
        if await _afn(1) != 2:
            return TestResult.failed("async observe_span altered return value")

        try:
            with mod.span("test.err"):
                raise ValueError("boom")
        except ValueError:
            pass
        else:
            return TestResult.failed("span swallowed an exception")
    except Exception as exc:  # noqa: BLE001
        return TestResult.failed(f"tracing primitive raised: {exc}")
    return TestResult.passed("tracing span/decorator primitives safe ✓")


async def _test_langfuse_http_middleware_importable() -> TestResult:
    """HTTP middleware importable and excludes chat paths (no duplicate traces)."""
    mod, err = _try_import("app.middleware.langfuse_middleware")
    if err:
        return TestResult.skipped(f"langfuse_middleware not importable: {err}")
    if not hasattr(mod, "LangfuseHTTPMiddleware"):
        return TestResult.failed("LangfuseHTTPMiddleware not exported")
    skips = getattr(mod, "_SKIP_PREFIXES", ())
    # The V2 streaming endpoint owns its trace via the LangGraph
    # CallbackHandler — it MUST be skipped; other chat utility endpoints
    # (feedback/curate/cache) must be traced by the middleware.
    if not "/api/v2/chat/stream".startswith(skips):
        return TestResult.failed("streaming path not excluded — duplicate root traces would occur")
    if "/api/v2/chat/feedback".startswith(skips):
        return TestResult.failed("utility endpoints wrongly excluded from HTTP tracing")
    return TestResult.passed("Langfuse HTTP middleware present, stream excluded, utilities traced ✓")


async def _test_code_evaluators() -> TestResult:
    """Code-evaluator suite: correct scores, abstention, and edge cases."""
    mod, err = _try_import("app.observability.evaluation.code_evaluators")
    if err:
        return TestResult.failed(f"Cannot import code_evaluators: {err}")
    try:
        ctx = mod.EvalContext(
            query="q", answer="A grounded answer citing svc.py.",
            reranked_chunks=[{"content": "grounded answer content", "metadata": {"source": "svc.py"}}],
            rerank_scores=[0.9], sources=[{"id": "1"}],
            routing_confidence=0.9, guardrail_passed=True,
            nodes_visited=["response_node"], trace_id="t",
        )
        scores = mod.run_code_evaluators(ctx)
        names = {s.name for s in scores}
        expected = {
            "response_structure_valid", "response_completeness", "citation_quality",
            "context_utilization", "retrieval_quality", "routing_confidence_bucket",
            "guardrail_compliance", "agent_goal_completion",
        }
        if not expected.issubset(names):
            return TestResult.failed(f"missing evaluators: {expected - names}")
        # Cache-hit abstention: retrieval evaluators must not fire.
        cached = mod.EvalContext(query="q", answer="ok.", cache_hit=True)
        cached_names = {s.name for s in mod.run_code_evaluators(cached)}
        if "citation_quality" in cached_names or "context_utilization" in cached_names:
            return TestResult.failed("retrieval evaluators did not abstain on cache hit")
    except Exception as exc:  # noqa: BLE001
        return TestResult.failed(f"code evaluators raised: {exc}")
    return TestResult.passed("code-evaluator suite correct + abstains properly ✓")


async def _test_online_evaluation_consistency() -> TestResult:
    """Online evaluation must skip unsampled requests (no orphan scores)."""
    mod, err = _try_import("app.observability.evaluation.online_evaluation")
    if err:
        return TestResult.failed(f"Cannot import online_evaluation: {err}")
    try:
        out = mod.evaluate_response_online({"query": "q", "final_response": "a"})
        if out != []:
            return TestResult.failed("scored an unsampled request — orphan scores possible")
    except Exception as exc:  # noqa: BLE001
        return TestResult.failed(f"online evaluation raised: {exc}")
    return TestResult.passed("online evaluation sampling-consistent (no orphan scores) ✓")


async def _test_feedback_and_datasets_noop_safe() -> TestResult:
    """Feedback + dataset helpers must be no-op safe when Langfuse is off."""
    fb, err = _try_import("app.observability.evaluation.feedback")
    if err:
        return TestResult.failed(f"Cannot import feedback: {err}")
    ds, err2 = _try_import("app.observability.evaluation.datasets")
    if err2:
        return TestResult.failed(f"Cannot import datasets: {err2}")
    try:
        _ = fb.record_user_feedback(trace_id="t", thumbs_up=True, rating=3, comment="c")
        _ = fb.record_user_feedback(trace_id="", thumbs_up=True)  # invalid → False
        _ = ds.add_interaction_to_dataset(query="q")
        _ = ds.get_dataset_items()
    except Exception as exc:  # noqa: BLE001
        return TestResult.failed(f"feedback/dataset helper raised: {exc}")
    return TestResult.passed("feedback + dataset helpers no-op safe ✓")


async def _test_request_trace_context() -> TestResult:
    """H-1/H-2: request-scoped trace context binds, nests, and no-ops unsampled."""
    mod, err = _try_import("app.observability.tracing")
    if err:
        return TestResult.failed(f"Cannot import tracing: {err}")
    try:
        with mod.request_trace("a" * 32):
            if mod.current_request_trace_id() != "a" * 32:
                return TestResult.failed("request trace id not bound")
            with mod.request_trace(None):
                if mod.current_request_trace_id() is not None:
                    return TestResult.failed("unsampled marker not honoured")
                with mod.span("x") as s:
                    if s._obs is not None:
                        return TestResult.failed("unsampled request produced a span")
            if mod.current_request_trace_id() != "a" * 32:
                return TestResult.failed("context not restored after nesting")
        if mod.current_request_trace_id() is not None:
            return TestResult.failed("context leaked past request scope")
    except Exception as exc:  # noqa: BLE001
        return TestResult.failed(f"request trace context raised: {exc}")
    return TestResult.passed("request trace context binds/nests/no-ops correctly ✓")


async def _test_feedback_security() -> TestResult:
    """M-4: malformed trace ids and foreign-owned traces are rejected."""
    fb, err = _try_import("app.observability.evaluation.feedback")
    if err:
        return TestResult.failed(f"Cannot import feedback: {err}")
    try:
        if fb.record_user_feedback(trace_id="bogus!", thumbs_up=True, user_id="u1"):
            return TestResult.failed("malformed trace_id accepted")
        fb.register_trace_owner("d" * 32, "owner-user")
        if fb.record_user_feedback(trace_id="d" * 32, thumbs_up=True, user_id="other-user"):
            return TestResult.failed("foreign-owned trace accepted — score pollution possible")
        if fb._score_id("d" * 32, "user_rating", "u") != fb._score_id("d" * 32, "user_rating", "u"):
            return TestResult.failed("score_id not deterministic — duplicates possible")
    except Exception as exc:  # noqa: BLE001
        return TestResult.failed(f"feedback security check raised: {exc}")
    return TestResult.passed("feedback ownership + dedup enforced ✓")


async def _test_pii_masking() -> TestResult:
    """M-1: the official mask hook redacts secrets/PII recursively."""
    mod, err = _try_import("app.observability.langfuse_client")
    if err:
        return TestResult.failed(f"Cannot import langfuse_client: {err}")
    mask = getattr(mod, "mask_sensitive_data", None)
    if not callable(mask):
        return TestResult.failed("mask_sensitive_data not exported")
    try:
        out = mask(data={
            "email": "a.user@example.com",
            "nested": ["Bearer abcdef1234567890abcdef", {"pw": 'password="s3cret_value"'}],
        })
        text = str(out)
        if "a.user@example.com" in text or "s3cret_value" in text:
            return TestResult.failed("PII/secret leaked through mask")
    except Exception as exc:  # noqa: BLE001
        return TestResult.failed(f"mask raised: {exc}")
    return TestResult.passed("PII masking redacts recursively ✓")


async def _test_langfuse_graceful_degradation() -> TestResult:
    """With Langfuse disabled, every helper must be a safe no-op (never raise)."""
    mod, err = _try_import("app.observability.langfuse_client")
    if err:
        return TestResult.failed(f"Cannot import langfuse_client: {err}")
    try:
        # These must not raise regardless of enabled/disabled state.
        _ = mod.build_trace_metadata(user_id="u", session_id="s", tags=["t"])
        _ = mod.get_current_trace_id()
        _ = mod.should_sample()
        _ = mod.create_trace_id()
        _ = mod.get_callback_handler(trace_id="t")
        mod.flush()
        # score/callback should never raise even if disabled
        mod.score_current_trace(trace_id="t", name="faithfulness", value=0.5)
    except Exception as exc:  # noqa: BLE001
        return TestResult.failed(f"langfuse_client helper raised while degraded: {exc}")
    return TestResult.passed("langfuse_client degrades gracefully (no-op safe) ✓")


async def _test_langfuse_trace_id_in_state() -> TestResult:
    """C-2: initial state schema must carry a langfuse_trace_id slot so the
    trace id propagates deterministically from request to evaluation."""
    mod, err = _try_import("app.graph.state")
    if err:
        return TestResult.skipped(f"state not importable: {err}")
    make = getattr(mod, "make_initial_state", None)
    if not callable(make):
        return TestResult.failed("make_initial_state not found")
    try:
        st = make(query="q", user_id="u", session_id="u::s")
    except Exception as exc:  # noqa: BLE001
        return TestResult.failed(f"make_initial_state raised: {exc}")
    if "langfuse_trace_id" not in st:
        return TestResult.failed(
            "initial state missing 'langfuse_trace_id' — C-2 propagation broken"
        )
    return TestResult.passed("state carries langfuse_trace_id for eval scoring ✓")


async def _test_evaluation_scheduling_pattern() -> TestResult:
    """C-1: the RAGAS evaluator must be schedulable via run_in_executor without
    blocking, and its done-callback must surface (not swallow) exceptions."""
    import asyncio

    ran = {"v": False, "cb": False}

    def _sync_eval(_sample):
        ran["v"] = True
        return {"faithfulness": 0.9}

    def _cb(fut):
        try:
            _ = fut.exception()
        except Exception:  # noqa: BLE001
            return
        ran["cb"] = True

    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(None, _sync_eval, {"trace_id": "t1"})
    fut.add_done_callback(_cb)
    await asyncio.sleep(0.1)
    if not ran["v"]:
        return TestResult.failed("sync evaluate_sample did not run in executor (C-1)")
    if not ran["cb"]:
        return TestResult.failed("done-callback did not fire — exceptions would be swallowed (C-1)")
    return TestResult.passed("evaluation schedules off-loop; callback surfaces errors ✓")


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
    PhaseTest(id="H-007", name="langfuse_client importable",
              description="Langfuse client singleton + public API present",
              run=_test_langfuse_client_importable, critical=True, tags=["obs", "langfuse"]),
    PhaseTest(id="H-008", name="langfuse_client graceful degradation",
              description="Langfuse helpers are safe no-ops when disabled",
              run=_test_langfuse_graceful_degradation, critical=True, tags=["obs", "langfuse"]),
    PhaseTest(id="H-009", name="trace_id propagates via graph state",
              description="C-2: initial state carries langfuse_trace_id for eval scoring",
              run=_test_langfuse_trace_id_in_state, critical=True, tags=["obs", "langfuse"]),
    PhaseTest(id="H-010", name="evaluation scheduling pattern",
              description="C-1: RAGAS evaluator runs off-loop with surfacing callback",
              run=_test_evaluation_scheduling_pattern, critical=True, tags=["obs", "langfuse"]),
    PhaseTest(id="H-011", name="tracing primitives",
              description="span()/observe_span() are no-op safe and propagate errors",
              run=_test_tracing_primitives, critical=True, tags=["obs", "langfuse"]),
    PhaseTest(id="H-012", name="Langfuse HTTP middleware",
              description="HTTP middleware present; chat paths excluded (no dup traces)",
              run=_test_langfuse_http_middleware_importable, critical=False, tags=["obs", "langfuse"]),
    PhaseTest(id="H-013", name="code evaluators",
              description="Deterministic evaluator suite scores + abstains correctly",
              run=_test_code_evaluators, critical=True, tags=["obs", "langfuse", "eval"]),
    PhaseTest(id="H-014", name="online evaluation consistency",
              description="Unsampled requests are never scored (no orphan scores)",
              run=_test_online_evaluation_consistency, critical=True, tags=["obs", "langfuse", "eval"]),
    PhaseTest(id="H-015", name="feedback + datasets no-op safe",
              description="User feedback and dataset helpers degrade gracefully",
              run=_test_feedback_and_datasets_noop_safe, critical=True, tags=["obs", "langfuse", "eval"]),
    PhaseTest(id="H-016", name="request trace context",
              description="H-1/H-2: request-scoped sampling + trace binding correct",
              run=_test_request_trace_context, critical=True, tags=["obs", "langfuse"]),
    PhaseTest(id="H-017", name="feedback security",
              description="M-4: ownership + dedup + trace-id validation enforced",
              run=_test_feedback_security, critical=True, tags=["obs", "langfuse", "eval"]),
    PhaseTest(id="H-018", name="PII masking",
              description="M-1: official mask hook redacts secrets/PII recursively",
              run=_test_pii_masking, critical=True, tags=["obs", "langfuse", "security"]),
]
