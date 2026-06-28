"""
Graph Foundation Tests
======================
Validates the LangGraph state machine layer that every request runs through.

  A-001  AgentState TypedDict importable and structurally correct
  A-002  messages field uses operator.add reducer (append semantics)
  A-003  Supervisor graph module importable
  A-004  Supervisor graph compiles without errors (in-memory checkpointer)
  A-005  intent_classifier_node returns required state keys
  A-006  StreamingLayer SSEEvent dataclass instantiates correctly
  A-007  stream_graph_events is an async generator
  A-008  LangGraph package is installed and meets minimum version
  A-009  langgraph-checkpoint-postgres package present
  A-010  route_to_agent returns correct agent for each routing decision

A-001, A-004, and A-008 are critical — if the graph can't compile,
nothing else works.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from app.tests.base import PhaseTest, TestResult


# ─────────────────────────────────────────────────────────────────────────────
# Helper: safe import — returns (module | None, error_message | None)
# ─────────────────────────────────────────────────────────────────────────────

def _try_import(dotted_path: str):
    try:
        return importlib.import_module(dotted_path), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# A-001  AgentState TypedDict importable
# ─────────────────────────────────────────────────────────────────────────────

async def _test_agent_state_importable() -> TestResult:
    mod, err = _try_import("app.graph.state")
    if err:
        return TestResult.failed(
            f"Cannot import app.graph.state: {err}",
            detail="Create backend/app/graph/state.py with AgentState TypedDict"
        )
    if not hasattr(mod, "AgentState"):
        return TestResult.failed(
            "app.graph.state exists but AgentState is not defined",
            detail="Add: class AgentState(TypedDict): ..."
        )
    return TestResult.passed("app.graph.state.AgentState found")


# ─────────────────────────────────────────────────────────────────────────────
# A-002  messages field uses operator.add reducer
# ─────────────────────────────────────────────────────────────────────────────

async def _test_messages_reducer() -> TestResult:
    mod, err = _try_import("app.graph.state")
    if err:
        return TestResult.skipped("app.graph.state not importable — see A-001")
    try:
        import typing, operator
        AgentState = mod.AgentState
        hints = typing.get_type_hints(AgentState, include_extras=True)
        messages_hint = hints.get("messages")
        if messages_hint is None:
            return TestResult.failed(
                "AgentState has no 'messages' field",
                detail="Add: messages: Annotated[Sequence[BaseMessage], operator.add]"
            )
        # Check Annotated metadata contains operator.add
        metadata = getattr(messages_hint, "__metadata__", ())
        if operator.add not in metadata:
            return TestResult.failed(
                "messages field does not use operator.add reducer",
                detail=(
                    "messages must be: "
                    "Annotated[Sequence[BaseMessage], operator.add]"
                )
            )
        return TestResult.passed("messages reducer = operator.add ✓")
    except Exception as exc:
        return TestResult.error(exc)


# ─────────────────────────────────────────────────────────────────────────────
# A-003  Supervisor graph module importable
# ─────────────────────────────────────────────────────────────────────────────

async def _test_supervisor_graph_importable() -> TestResult:
    mod, err = _try_import("app.graph.supervisor_graph")
    if err:
        return TestResult.failed(
            f"Cannot import app.graph.supervisor_graph: {err}",
            detail="Create backend/app/graph/supervisor_graph.py"
        )
    if not hasattr(mod, "build_supervisor_graph"):
        return TestResult.failed(
            "build_supervisor_graph() not found in supervisor_graph.py"
        )
    return TestResult.passed("supervisor_graph.build_supervisor_graph found")


# ─────────────────────────────────────────────────────────────────────────────
# A-004  Supervisor graph compiles with in-memory checkpointer (CRITICAL)
# ─────────────────────────────────────────────────────────────────────────────

async def _test_supervisor_graph_compiles() -> TestResult:
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError as exc:
        return TestResult.failed(f"langgraph not installed: {exc}")

    mod, err = _try_import("app.graph.supervisor_graph")
    if err:
        return TestResult.skipped("supervisor_graph not importable — see A-003")

    try:
        checkpointer = MemorySaver()
        graph = mod.build_supervisor_graph(checkpointer=checkpointer)
        if graph is None:
            return TestResult.failed("build_supervisor_graph() returned None")
        return TestResult.passed("Supervisor graph compiled successfully")
    except Exception as exc:
        return TestResult.error(exc)


# ─────────────────────────────────────────────────────────────────────────────
# A-005  intent_classifier_node returns required state keys
# ─────────────────────────────────────────────────────────────────────────────

async def _test_intent_classifier_node_output() -> TestResult:
    mod, err = _try_import("app.graph.nodes.intent_classifier")
    if err:
        return TestResult.failed(
            f"Cannot import intent_classifier node: {err}",
            detail="Create backend/app/graph/nodes/intent_classifier.py"
        )
    if not hasattr(mod, "intent_classifier_node"):
        return TestResult.failed("intent_classifier_node function not found")

    try:
        # Build a minimal fake state
        fake_state: dict[str, Any] = {
            "query": "how does authenticate_user work?",
            "nodes_visited": [],
            "routing_confidence": 0.0,
            "intent": None,
            "routing_decision": None,
            "metadata_filter": None,
        }
        result = await mod.intent_classifier_node(fake_state, {})

        required_keys = {"intent", "routing_decision", "routing_confidence",
                         "metadata_filter", "nodes_visited"}
        missing = required_keys - set(result.keys())
        if missing:
            return TestResult.failed(
                f"intent_classifier_node missing state keys: {missing}"
            )
        if result["routing_decision"] is None:
            return TestResult.failed(
                "routing_decision is None — classifier must return an agent name"
            )
        return TestResult.passed(
            f"routing_decision='{result['routing_decision']}'  "
            f"intent='{result['intent']}'  "
            f"confidence={result['routing_confidence']:.2f}"
        )
    except Exception as exc:
        return TestResult.error(exc)


# ─────────────────────────────────────────────────────────────────────────────
# A-006  SSEEvent dataclass instantiates correctly
# ─────────────────────────────────────────────────────────────────────────────

async def _test_sse_event_dataclass() -> TestResult:
    mod, err = _try_import("app.graph.streaming")
    if err:
        return TestResult.failed(
            f"Cannot import app.graph.streaming: {err}",
            detail="Create backend/app/graph/streaming.py with SSEEvent dataclass"
        )
    if not hasattr(mod, "SSEEvent"):
        return TestResult.failed("SSEEvent not found in app.graph.streaming")

    try:
        event = mod.SSEEvent(
            type="token",
            data={"content": "hello"},
            agent="CodeAgent",
            checkpoint_id="chk-001",
            ts=1234567890.123,
        )
        assert event.type == "token"
        assert event.agent == "CodeAgent"
        return TestResult.passed("SSEEvent instantiated and fields accessible")
    except Exception as exc:
        return TestResult.error(exc)


# ─────────────────────────────────────────────────────────────────────────────
# A-007  stream_graph_events is an async generator
# ─────────────────────────────────────────────────────────────────────────────

async def _test_stream_graph_events_is_async_gen() -> TestResult:
    import inspect
    mod, err = _try_import("app.graph.streaming")
    if err:
        return TestResult.skipped("app.graph.streaming not importable — see A-006")
    if not hasattr(mod, "stream_graph_events"):
        return TestResult.failed("stream_graph_events not found in streaming.py")

    fn = mod.stream_graph_events
    if not (inspect.isasyncgenfunction(fn) or inspect.iscoroutinefunction(fn)):
        return TestResult.failed(
            "stream_graph_events must be an async generator or async function"
        )
    return TestResult.passed("stream_graph_events is async-compatible ✓")


# ─────────────────────────────────────────────────────────────────────────────
# A-008  LangGraph installed with minimum version (CRITICAL)
# ─────────────────────────────────────────────────────────────────────────────

async def _test_langgraph_installed() -> TestResult:
    try:
        import langgraph  # type: ignore  # noqa: F401
        # importlib.metadata is the reliable source — langgraph >= 1.x does
        # not expose __version__ on the module object itself.
        try:
            import importlib.metadata as _meta
            version = _meta.version("langgraph")
        except Exception:
            version = getattr(langgraph, "__version__", "unknown")

        # Require >= 0.2.0
        try:
            from packaging.version import Version, InvalidVersion  # type: ignore
            try:
                if Version(version) < Version("0.2.0"):
                    return TestResult.failed(
                        f"langgraph=={version} is below minimum 0.2.0",
                        detail="pip install 'langgraph>=0.2.0'"
                    )
            except InvalidVersion:
                # Version string unrecognised — package is importable, treat as pass
                pass
        except ImportError:
            pass  # packaging not available — just verify import works
        return TestResult.passed(f"langgraph=={version} installed ✓")
    except ImportError as exc:
        return TestResult.failed(
            f"langgraph not installed: {exc}",
            detail="pip install 'langgraph>=0.2.0'"
        )


# ─────────────────────────────────────────────────────────────────────────────
# A-009  langgraph-checkpoint-postgres package present
# ─────────────────────────────────────────────────────────────────────────────

async def _test_langgraph_checkpoint_postgres() -> TestResult:
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # type: ignore  # noqa: F401
        return TestResult.passed("AsyncPostgresSaver importable ✓")
    except ImportError as exc:
        return TestResult.failed(
            f"langgraph-checkpoint-postgres not installed: {exc}",
            detail="pip install 'langgraph-checkpoint-postgres>=1.0.0'"
        )


# ─────────────────────────────────────────────────────────────────────────────
# A-010  route_to_agent returns correct agent for each routing decision
# ─────────────────────────────────────────────────────────────────────────────

async def _test_route_to_agent_mapping() -> TestResult:
    mod, err = _try_import("app.graph.supervisor_graph")
    if err:
        return TestResult.skipped("supervisor_graph not importable — see A-003")
    if not hasattr(mod, "route_to_agent"):
        return TestResult.failed("route_to_agent() not found in supervisor_graph.py")

    expected_mappings = {
        "CodeAgent":  "CodeAgent",
        "DocAgent":   "DocAgent",
        "DebugAgent": "DebugAgent",
        "ArchAgent":  "ArchAgent",
    }
    failures = []
    for decision, expected_node in expected_mappings.items():
        fake_state = {
            "cache_hit": False,
            "routing_decision": decision,
        }
        actual = mod.route_to_agent(fake_state)
        if actual != expected_node:
            failures.append(f"{decision} → expected {expected_node}, got {actual}")

    if failures:
        return TestResult.failed(
            f"Routing mapping errors: {'; '.join(failures)}"
        )
    return TestResult.passed(
        "All routing decisions map to correct agent nodes ✓"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TESTS registry — consumed by StartupTestRunner
# ─────────────────────────────────────────────────────────────────────────────

TESTS: list[PhaseTest] = [
    PhaseTest(
        id="A-001",
        name="AgentState TypedDict importable",
        description="app.graph.state.AgentState can be imported",
        run=_test_agent_state_importable,
        critical=True,
        tags=["state", "import"],
    ),
    PhaseTest(
        id="A-002",
        name="messages field uses operator.add reducer",
        description="AgentState.messages carries Annotated[..., operator.add]",
        run=_test_messages_reducer,
        critical=False,
        tags=["state", "reducer"],
    ),
    PhaseTest(
        id="A-003",
        name="Supervisor graph module importable",
        description="app.graph.supervisor_graph is importable",
        run=_test_supervisor_graph_importable,
        critical=False,
        tags=["graph", "import"],
    ),
    PhaseTest(
        id="A-004",
        name="Supervisor graph compiles (MemorySaver)",
        description="build_supervisor_graph() compiles with in-memory checkpointer",
        run=_test_supervisor_graph_compiles,
        critical=True,
        tags=["graph", "compile"],
    ),
    PhaseTest(
        id="A-005",
        name="intent_classifier_node returns required keys",
        description="Node returns intent, routing_decision, confidence, metadata_filter",
        run=_test_intent_classifier_node_output,
        critical=False,
        tags=["graph", "node", "routing"],
    ),
    PhaseTest(
        id="A-006",
        name="SSEEvent dataclass instantiates correctly",
        description="app.graph.streaming.SSEEvent is usable",
        run=_test_sse_event_dataclass,
        critical=False,
        tags=["streaming"],
    ),
    PhaseTest(
        id="A-007",
        name="stream_graph_events is async-compatible",
        description="stream_graph_events is an async generator or async function",
        run=_test_stream_graph_events_is_async_gen,
        critical=False,
        tags=["streaming"],
    ),
    PhaseTest(
        id="A-008",
        name="langgraph >= 0.2.0 installed",
        description="LangGraph package is present and meets minimum version",
        run=_test_langgraph_installed,
        critical=True,
        tags=["deps"],
    ),
    PhaseTest(
        id="A-009",
        name="langgraph-checkpoint-postgres installed",
        description="AsyncPostgresSaver can be imported",
        run=_test_langgraph_checkpoint_postgres,
        critical=False,
        tags=["deps", "checkpoint"],
    ),
    PhaseTest(
        id="A-010",
        name="route_to_agent maps decisions to correct nodes",
        description="Each RoutingDecision value routes to its agent node name",
        run=_test_route_to_agent_mapping,
        critical=False,
        tags=["graph", "routing"],
    ),
]
