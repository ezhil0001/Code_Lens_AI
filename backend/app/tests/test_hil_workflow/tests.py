"""
HIL Workflow Tests
==================
  E-001  hil_node module importable
  E-002  hil_check_node sets hil_required=True on low confidence
  E-003  hil_check_node sets hil_required=True on destructive keywords
  E-004  hil_check_node sets hil_required=False on normal query
  E-005  hil_check_node appends to nodes_visited
  E-006  HIL resume endpoint registered in API router
  E-007  HILResumeRequest schema validates correctly
  E-008  audit_log table exists (HIL events persisted)
  E-009  contains_destructive_intent() detects risky keywords
"""

from __future__ import annotations

import importlib
from typing import Any

from app.tests.base import PhaseTest, TestResult


def _try_import(path: str):
    try:
        return importlib.import_module(path), None
    except Exception as exc:
        return None, str(exc)


def _is_interrupt_signal(exc: BaseException) -> bool:
    """True when the node reached a LangGraph interrupt.

    Inside a graph run this is GraphInterrupt; calling the node directly makes
    interrupt() fail on the missing runnable context, which still proves the
    interrupt path was taken rather than the node returning normally.
    """
    try:
        from langgraph.errors import GraphInterrupt  # type: ignore
        if isinstance(exc, GraphInterrupt):
            return True
    except ImportError:
        pass
    return isinstance(exc, RuntimeError) and "runnable context" in str(exc)


async def _test_hil_node_importable() -> TestResult:
    mod, err = _try_import("app.graph.nodes.hil_node")
    if err:
        return TestResult.failed(f"Cannot import app.graph.nodes.hil_node: {err}")
    if not hasattr(mod, "hil_check_node"):
        return TestResult.failed("hil_check_node not found")
    return TestResult.passed("hil_node.hil_check_node found ✓")


async def _test_hil_triggers_on_low_confidence() -> TestResult:
    """Confidence-based review is opt-in via the request's hil_enabled flag.

    Both directions are asserted: enabled -> review required, disabled -> no
    review. (The destructive-keyword gate is separate and always on; see E-003.)
    """
    mod, err = _try_import("app.graph.nodes.hil_node")
    if err:
        return TestResult.skipped("hil_node not importable")

    def _state(hil_enabled: bool) -> dict[str, Any]:
        return {
            "query": "how does the cache work?",
            "routing_confidence": 0.3,   # below threshold
            "active_agent": "CodeAgent",
            "final_response": "Some response",
            "nodes_visited": [],
            "hil_enabled": hil_enabled,
        }

    # HIL enabled → must require review (or interrupt outright).
    try:
        result = await mod.hil_check_node(_state(True), {})
        if not result.get("hil_required"):
            return TestResult.failed(
                "hil_check_node should set hil_required=True when confidence=0.3 "
                "and hil_enabled=True"
            )
    except Exception as exc:
        if not _is_interrupt_signal(exc):
            return TestResult.error(exc)

    # HIL disabled → the same low confidence must NOT pause the graph.
    try:
        result = await mod.hil_check_node(_state(False), {})
        if result.get("hil_required"):
            return TestResult.failed(
                "hil_check_node required review despite hil_enabled=False"
            )
    except Exception as exc:
        if _is_interrupt_signal(exc):
            return TestResult.failed(
                "hil_check_node interrupted despite hil_enabled=False"
            )
        return TestResult.error(exc)

    return TestResult.passed(
        "low-confidence review honours the request's hil_enabled flag (both ways) ✓"
    )


async def _test_hil_triggers_on_destructive_keyword() -> TestResult:
    mod, err = _try_import("app.graph.nodes.hil_node")
    if err:
        return TestResult.skipped("hil_node not importable")
    fake_state: dict[str, Any] = {
        "query": "drop table users and remove all data",
        "routing_confidence": 0.9,
        "active_agent": "DebugAgent",
        "final_response": "",
        "nodes_visited": [],
    }
    try:
        result = await mod.hil_check_node(fake_state, {})
        if not result.get("hil_required"):
            return TestResult.failed(
                "HIL should trigger on destructive keywords ('drop table', 'remove all')"
            )
        return TestResult.passed("HIL triggered on destructive keyword ✓")
    except Exception as exc:
        if _is_interrupt_signal(exc):
            return TestResult.passed(
                "HIL invoked the LangGraph interrupt on destructive keyword ✓"
            )
        return TestResult.error(exc)


async def _test_hil_passes_on_normal_query() -> TestResult:
    mod, err = _try_import("app.graph.nodes.hil_node")
    if err:
        return TestResult.skipped("hil_node not importable")
    fake_state: dict[str, Any] = {
        "query": "how does authenticate_user handle expired tokens?",
        "routing_confidence": 0.85,
        "active_agent": "CodeAgent",
        "final_response": "The function checks token expiry using ...",
        "nodes_visited": [],
    }
    try:
        result = await mod.hil_check_node(fake_state, {})
        if result.get("hil_required"):
            return TestResult.failed(
                "HIL should NOT trigger on a safe, high-confidence query"
            )
        return TestResult.passed("HIL correctly passes for normal query ✓")
    except Exception as exc:
        return TestResult.error(exc)


async def _test_hil_appends_nodes_visited() -> TestResult:
    mod, err = _try_import("app.graph.nodes.hil_node")
    if err:
        return TestResult.skipped("hil_node not importable")
    fake_state: dict[str, Any] = {
        "query": "safe query",
        "routing_confidence": 0.9,
        "active_agent": "CodeAgent",
        "final_response": "",
        "nodes_visited": ["intent_classifier"],
    }
    try:
        result = await mod.hil_check_node(fake_state, {})
        visited = result.get("nodes_visited", [])
        if not any("hil" in v.lower() for v in visited):
            return TestResult.failed(
                f"hil_check_node did not append to nodes_visited: {visited}"
            )
        return TestResult.passed(f"nodes_visited after HIL: {visited} ✓")
    except Exception as exc:
        return TestResult.error(exc)


async def _test_hil_resume_endpoint_registered() -> TestResult:
    mod, err = _try_import("app.api.checkpoints")
    if err:
        return TestResult.skipped("checkpoints router not importable")
    routes = [str(r.path) for r in getattr(mod.router, "routes", [])]
    matching = [r for r in routes if "resume" in r]
    if not matching:
        return TestResult.failed(f"No /resume route found. Routes: {routes}")
    return TestResult.passed(f"HIL resume route: {matching[0]} ✓")


async def _test_hil_resume_schema() -> TestResult:
    mod, err = _try_import("app.api.checkpoints")
    if err:
        return TestResult.skipped("checkpoints not importable")
    if not hasattr(mod, "HILResumeRequest"):
        return TestResult.failed("HILResumeRequest schema not found in checkpoints.py")
    try:
        req = mod.HILResumeRequest(human_input="Looks good, proceed.", approved=True)
        assert req.approved is True
        assert req.human_input == "Looks good, proceed."
        return TestResult.passed("HILResumeRequest schema validates correctly ✓")
    except Exception as exc:
        return TestResult.error(exc)


async def _test_audit_log_table_exists() -> TestResult:
    try:
        import asyncio

        from app.core.database import get_pg_pool
        pool = get_pg_pool()
        if pool is None:
            return TestResult.skipped("pg_pool not available")

        def _probe():
            # psycopg3 pools are synchronous; the asyncpg-style call used before
            # always raised and masked the real result as a skip.
            with pool.connection() as conn:
                cur = conn.execute(
                    """SELECT 1 FROM information_schema.tables
                       WHERE table_name = 'audit_logs'"""
                )
                return cur.fetchone()

        row = await asyncio.to_thread(_probe)
        if row is None:
            return TestResult.failed(
                "audit_logs table does not exist",
                detail="HIL interrupts must be persisted to audit_logs"
            )
        return TestResult.passed("audit_logs table exists ✓")
    except Exception as exc:
        return TestResult.skipped(f"Postgres unreachable: {exc}")


async def _test_destructive_intent_detector() -> TestResult:
    mod, err = _try_import("app.graph.nodes.hil_node")
    if err:
        return TestResult.skipped("hil_node not importable")
    if not hasattr(mod, "contains_destructive_intent"):
        return TestResult.failed("contains_destructive_intent() not found")
    cases = [
        ("drop table users", True),
        ("remove all files", True),
        ("delete everything", True),
        ("how does caching work?", False),
        ("explain the auth module", False),
    ]
    failures = []
    for query, expected in cases:
        actual = mod.contains_destructive_intent(query)
        if actual != expected:
            failures.append(f"'{query}' → expected {expected}, got {actual}")
    if failures:
        return TestResult.failed(
            f"contains_destructive_intent errors: {'; '.join(failures)}"
        )
    return TestResult.passed(
        f"contains_destructive_intent correct on all {len(cases)} test cases ✓"
    )


async def _real_graph():
    """Compile the real supervisor graph with an in-memory checkpointer."""
    from langgraph.checkpoint.memory import MemorySaver

    from app.graph.supervisor_graph import build_supervisor_graph

    return build_supervisor_graph(checkpointer=MemorySaver())


def _thread_cfg() -> dict:
    import uuid
    return {
        "configurable": {"thread_id": f"t-{uuid.uuid4().hex[:8]}", "checkpoint_ns": ""},
        "recursion_limit": 25,
    }


async def _drive(graph, payload, cfg) -> tuple[list, list]:
    """Run the graph, returning (nodes_started, answer_token_nodes)."""
    from app.graph.streaming import ANSWER_NODES

    nodes: list[str] = []
    tokens: list[str] = []
    async for ev in graph.astream_events(payload, cfg, version="v2"):
        kind = ev.get("event", "")
        if kind == "on_chain_start":
            nodes.append(ev.get("name", ""))
        elif kind == "on_chat_model_stream":
            node = (ev.get("metadata") or {}).get("langgraph_node", "")
            if node in ANSWER_NODES:
                tokens.append(node)
    return nodes, tokens


async def _test_gate_precedes_generation() -> TestResult:
    """R-HIL-001/002/003/007: gate fires before ANY answer node or token.

    The gate used to sit after synthesizer_node, so a "DROP TABLE users"
    request was fully generated and streamed to the browser before the review
    banner appeared, and the output guardrail could only annotate content the
    user had already read.
    """
    from app.graph.state import make_initial_state
    from app.graph.streaming import ANSWER_NODES

    graph = await _real_graph()
    cfg = _thread_cfg()
    st = make_initial_state(query="DROP TABLE users and remove all data",
                            user_id="u1", session_id="u1::s")
    st["hil_enabled"] = False  # R-HIL-012: mandatory gate ignores the client toggle
    nodes, tokens = await _drive(graph, st, cfg)

    snap = await graph.aget_state(cfg)
    interrupts = [i for t in (snap.tasks or []) for i in (t.interrupts or [])]
    if not interrupts:
        return TestResult.failed("destructive query did not interrupt")
    ran = {n for n in nodes if n in ANSWER_NODES}
    if ran:
        return TestResult.failed(f"answer nodes ran before approval: {sorted(ran)}")
    if tokens:
        return TestResult.failed(f"answer tokens streamed before approval: {sorted(set(tokens))}")
    if snap.values.get("final_response"):
        return TestResult.failed("final_response populated before approval")
    if "hil_check_node" not in (snap.next or ()):
        return TestResult.failed(f"checkpoint not paused at hil_check_node: next={snap.next}")
    return TestResult.passed("gate interrupts before any generation or token ✓")


async def _test_reject_runs_no_generation() -> TestResult:
    """R-HIL-005/006/009: reject terminates safely with no generation."""
    from langgraph.types import Command

    from app.graph.state import make_initial_state
    from app.graph.streaming import ANSWER_NODES

    graph = await _real_graph()
    cfg = _thread_cfg()
    st = make_initial_state(query="drop table users", user_id="u1", session_id="u1::s")
    await _drive(graph, st, cfg)

    nodes, tokens = await _drive(
        graph, Command(resume={"approved": False, "human_input": "denied"}), cfg
    )
    ran = {n for n in nodes if n in ANSWER_NODES}
    if ran or tokens:
        return TestResult.failed(f"generation ran after REJECT: nodes={sorted(ran)} tokens={sorted(set(tokens))}")

    snap = await graph.aget_state(cfg)
    final = snap.values.get("final_response") or ""
    if "rejected" not in final.lower():
        return TestResult.failed(f"no refusal produced: {final[:80]!r}")
    if "drop table" in final.lower():
        return TestResult.failed("refusal leaked the destructive statement")
    if snap.next:
        return TestResult.failed(f"pending task remains after reject: {snap.next}")
    return TestResult.passed("reject: zero generation, one safe refusal, no pending task ✓")


async def _test_approve_single_generation() -> TestResult:
    """R-HIL-004/008/015: approve resumes the same thread exactly once.

    Drives a full generation, so it is opt-in (HIL_E2E_TESTS=1) to keep server
    startup from issuing real LLM traffic on every boot.
    """
    import os

    if os.getenv("HIL_E2E_TESTS", "").lower() not in ("1", "true", "yes"):
        return TestResult.skipped("set HIL_E2E_TESTS=1 (drives a real LLM generation)")

    from langgraph.types import Command

    from app.graph.state import make_initial_state
    from app.graph.streaming import ANSWER_NODES

    graph = await _real_graph()
    cfg = _thread_cfg()
    st = make_initial_state(query="drop table users", user_id="u1", session_id="u1::s")
    await _drive(graph, st, cfg)
    thread_before = cfg["configurable"]["thread_id"]

    nodes, _ = await _drive(
        graph, Command(resume={"approved": True, "human_input": "ok"}), cfg
    )
    gen = [n for n in nodes if n in ANSWER_NODES]
    if not gen:
        return TestResult.failed("approve did not run generation")
    if gen.count("synthesizer_node") > 1:
        return TestResult.failed(f"duplicate synthesis after approve: {gen}")

    snap = await graph.aget_state(cfg)
    if cfg["configurable"]["thread_id"] != thread_before:
        return TestResult.failed("approve did not resume the same thread")
    if snap.next:
        return TestResult.failed(f"pending task remains after approve: {snap.next}")
    if not (snap.values.get("final_response") or ""):
        return TestResult.failed("approve produced no final response")
    return TestResult.passed("approve: same thread, exactly one synthesis, no pending task ✓")


async def _test_benign_query_not_gated() -> TestResult:
    """R-HIL-013: normal requests are unaffected by the gate.

    Drives a full generation, so it is opt-in (HIL_E2E_TESTS=1) to keep server
    startup from issuing real LLM traffic on every boot.
    """
    import os

    if os.getenv("HIL_E2E_TESTS", "").lower() not in ("1", "true", "yes"):
        return TestResult.skipped("set HIL_E2E_TESTS=1 (drives a real LLM generation)")

    from app.graph.state import make_initial_state

    graph = await _real_graph()
    cfg = _thread_cfg()
    st = make_initial_state(query="How does the reranker score chunks?",
                            user_id="u1", session_id="u1::s")
    nodes, _ = await _drive(graph, st, cfg)

    snap = await graph.aget_state(cfg)
    if [i for t in (snap.tasks or []) for i in (t.interrupts or [])]:
        return TestResult.failed("benign query was incorrectly gated")
    if snap.next:
        return TestResult.failed(f"benign query left a pending task: {snap.next}")
    if not (snap.values.get("final_response") or ""):
        return TestResult.failed("benign query produced no response")
    if "hil_check_node" not in nodes:
        return TestResult.failed("gate node was skipped entirely")
    return TestResult.passed("benign query passes the gate and still answers ✓")


async def _test_graph_topology_gate_before_agents() -> TestResult:
    """R-HIL-001 (static): the gate must not be downstream of the synthesiser."""
    import inspect

    from app.graph import supervisor_graph as sg

    src = inspect.getsource(sg.build_supervisor_graph)
    if 'add_edge("synthesizer_node", "hil_check_node")' in src:
        return TestResult.failed("synthesizer still feeds hil_check_node — gate is post-generation")
    if 'add_edge("intent_classifier_node", "hil_check_node")' not in src:
        return TestResult.failed("classifier does not feed the gate")
    if 'add_edge("synthesizer_node", "output_guardrail_node")' not in src:
        return TestResult.failed("synthesizer must go straight to the output guardrail")
    if not hasattr(sg, "route_after_hil"):
        return TestResult.failed("route_after_hil router missing")

    # Reject must bypass the agents entirely.
    dest = sg.route_after_hil({"hil_approved": False, "routing_agents": ["CodeAgent"]})
    if dest != "output_guardrail_node":
        return TestResult.failed(f"reject routed to {dest!r} instead of the guardrail")
    sends = sg.route_after_hil({"hil_approved": True, "routing_agents": ["CodeAgent"]})
    if not isinstance(sends, list) or not sends:
        return TestResult.failed("approve did not fan out to agents")
    return TestResult.passed("topology: classifier → gate → agents → synthesiser → guardrail ✓")


async def _test_interrupt_resume_value_consumed() -> TestResult:
    """R-HIL-005: the node must consume interrupt()'s resume payload.

    Ignoring the return value made Command(resume={"approved": False}) fall
    through to the normal path and generate the refused answer.
    """
    import inspect

    from app.graph.nodes.hil_node import hil_check_node

    src = inspect.getsource(hil_check_node)
    if "interrupt(" not in src:
        return TestResult.failed("node no longer uses LangGraph interrupt()")
    if "decision = interrupt(" not in src:
        return TestResult.failed("interrupt() return value is discarded — reject would be ignored")
    if '"hil_approved"' not in src:
        return TestResult.failed("node does not surface hil_approved for routing")
    return TestResult.passed("interrupt() resume payload is consumed and surfaced ✓")


async def _test_resume_requires_pending_interrupt() -> TestResult:
    """R-HIL-010/011: /resume must fail closed without a pending interrupt.

    The endpoint used to answer 200 and *start a fresh graph run* on the
    caller's own thread for any session name. A second Approve click, an
    Approve racing a Reject, a resume after the decision was made, or another
    user probing someone else's session id all silently burned LLM quota and
    wrote phantom checkpoints (a cross-user probe produced 24 of them).
    """
    import inspect

    from app.api import checkpoints as cp

    src = inspect.getsource(cp.resume_after_hil)
    if "aget_state" not in src:
        return TestResult.failed("resume does not inspect thread state for a pending interrupt")
    if "409" not in src and "HTTP_409_CONFLICT" not in src:
        return TestResult.failed("resume does not reject a missing interrupt with 409")
    # The guard must run before the graph is resumed.
    if src.index("HTTP_409_CONFLICT") > src.index("Command(resume="):
        return TestResult.failed("pending-interrupt guard runs after the resume — too late")
    # Thread must stay namespaced per user so the guard is also an authz boundary.
    if "build_config" not in src or "current_user.id" not in src:
        return TestResult.failed("resume thread is not namespaced to the authenticated user")
    return TestResult.passed("resume fails closed (409) unless the caller's thread is paused ✓")


async def _test_resumed_stream_executes() -> TestResult:
    """R-HIL-012: the resume SSE generator must actually run.

    It is a module-level async generator, so any helper it references has to be
    resolvable from *its* scope. `stream_graph_events` was imported inside the
    endpoint instead, and because generators only execute on first iteration the
    resulting NameError surfaced as an HTTP 200 with an empty body — an approve
    silently produced no answer.
    """
    from app.api.checkpoints import _resumed_stream

    class _FakeGraph:
        async def astream(self, *a, **k):  # pragma: no cover - not reached
            yield {}

    class _FakeRoot:
        span_id = None

        def end(self, **kwargs):
            self.ended = True

    root = _FakeRoot()
    chunks = []
    try:
        async for chunk in _resumed_stream(_FakeGraph(), None, {"configurable": {}}, root, None):
            chunks.append(chunk)
    except NameError as exc:
        return TestResult.failed(f"resume generator has an unresolved name: {exc}")
    except Exception:
        # Any other failure comes from the fake graph, not from name resolution.
        pass
    if not getattr(root, "ended", False):
        return TestResult.failed("resume generator did not close its Langfuse root span")
    return TestResult.passed("resume stream generator executes and closes its root span ✓")


async def _test_resume_continues_origin_trace() -> TestResult:
    """R-HIL-013: resume must reuse the interrupted run's Langfuse trace id.

    Otherwise the reviewer's decision and everything it unblocks land on a
    second, unlinked trace and the HIL story cannot be followed in the UI.
    """
    import inspect

    from app.api import checkpoints as cp

    src = inspect.getsource(cp.resume_after_hil)
    if "langfuse_trace_id" not in src:
        return TestResult.failed("resume does not read the origin trace id from state")
    if "open_request_root" not in src or "chat.hil_resume" not in src:
        return TestResult.failed("resume does not open a root observation on the origin trace")
    if "_mint_trace_id" in src:
        return TestResult.failed("resume mints a NEW trace id instead of continuing the original")
    return TestResult.passed("resume continues the originating Langfuse trace ✓")


TESTS: list[PhaseTest] = [
    PhaseTest(id="E-001", name="hil_node importable",
              description="app.graph.nodes.hil_node.hil_check_node found",
              run=_test_hil_node_importable, critical=True, tags=["hil"]),
    PhaseTest(id="E-002", name="HIL triggers on low routing confidence",
              description="hil_required=True when confidence < threshold",
              run=_test_hil_triggers_on_low_confidence, critical=True, tags=["hil"]),
    PhaseTest(id="E-003", name="HIL triggers on destructive keywords",
              description="'drop table', 'remove all' trigger HIL",
              run=_test_hil_triggers_on_destructive_keyword, critical=True, tags=["hil", "security"]),
    PhaseTest(id="E-004", name="HIL passes on safe high-confidence query",
              description="Normal queries do not trigger HIL",
              run=_test_hil_passes_on_normal_query, critical=False, tags=["hil"]),
    PhaseTest(id="E-005", name="hil_check_node appends to nodes_visited",
              description="nodes_visited includes 'hil_check_node:*' after execution",
              run=_test_hil_appends_nodes_visited, critical=False, tags=["hil"]),
    PhaseTest(id="E-006", name="/resume endpoint registered",
              description="POST .../resume route exists in checkpoints router",
              run=_test_hil_resume_endpoint_registered, critical=False, tags=["hil", "api"]),
    PhaseTest(id="E-007", name="HILResumeRequest schema valid",
              description="HILResumeRequest(human_input, approved) validates",
              run=_test_hil_resume_schema, critical=False, tags=["hil", "api"]),
    PhaseTest(id="E-008", name="audit_log table exists",
              description="HIL events persist to audit_log",
              run=_test_audit_log_table_exists, critical=False, tags=["hil", "db"]),
    PhaseTest(id="E-009", name="contains_destructive_intent() correct",
              description="Detects destructive keywords; ignores safe queries",
              run=_test_destructive_intent_detector, critical=True, tags=["hil", "security"]),
    PhaseTest(id="E-010", name="gate precedes generation",
              description="R-HIL-001/002/003/007: interrupt before any answer node or token",
              run=_test_gate_precedes_generation, critical=True, tags=["hil", "security"]),
    PhaseTest(id="E-011", name="reject runs no generation",
              description="R-HIL-005/006/009: safe refusal, zero generation, no pending task",
              run=_test_reject_runs_no_generation, critical=True, tags=["hil", "security"]),
    PhaseTest(id="E-012", name="approve resumes once",
              description="R-HIL-004/008/015: same thread, exactly one synthesis (HIL_E2E_TESTS=1)",
              run=_test_approve_single_generation, critical=False, tags=["hil", "e2e"]),
    PhaseTest(id="E-013", name="benign query not gated",
              description="R-HIL-013: normal requests still stream and answer (HIL_E2E_TESTS=1)",
              run=_test_benign_query_not_gated, critical=False, tags=["hil", "e2e"]),
    PhaseTest(id="E-014", name="gate topology before agents",
              description="R-HIL-001: classifier → gate → agents → synthesiser → guardrail",
              run=_test_graph_topology_gate_before_agents, critical=True, tags=["hil", "security"]),
    PhaseTest(id="E-015", name="interrupt resume value consumed",
              description="R-HIL-005: Command(resume=...) decision is honoured",
              run=_test_interrupt_resume_value_consumed, critical=True, tags=["hil", "security"]),
    PhaseTest(id="E-016", name="resume requires pending interrupt",
              description="R-HIL-010/011: /resume fails closed (409) without a paused thread",
              run=_test_resume_requires_pending_interrupt, critical=True, tags=["hil", "security"]),
    PhaseTest(id="E-017", name="resume stream generator executes",
              description="R-HIL-012: resume SSE generator resolves its names and closes its span",
              run=_test_resumed_stream_executes, critical=True, tags=["hil", "observability"]),
    PhaseTest(id="E-018", name="resume continues origin trace",
              description="R-HIL-013: resume reuses the interrupted run's Langfuse trace",
              run=_test_resume_continues_origin_trace, critical=True, tags=["hil", "observability"]),
]
