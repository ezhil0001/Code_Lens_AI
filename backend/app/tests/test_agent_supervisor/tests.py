"""
Agent Supervisor Tests
======================
Validates all agent sub-graphs and the supervisor node wiring.

  B-001  CodeAgent sub-graph module importable
  B-002  CodeAgent sub-graph compiles
  B-003  code_search_tool has Pydantic args_schema
  B-004  DocAgent sub-graph module importable
  B-005  DocAgent uses kt_doc metadata filter
  B-006  DebugAgent sub-graph module importable
  B-007  ArchAgent sub-graph module importable
  B-008  WebAgent sub-graph module importable
  B-009  synthesizer_node handles single-agent path (no LLM call)
  B-010  synthesizer_node deduplicates sources by source_id
  B-011  All agents registered as nodes in Supervisor graph
  B-012  agent_brain.py compat shim still callable (backward compat)

B-001, B-002, and B-011 are critical.
"""

from __future__ import annotations

import importlib
from typing import Any

from app.tests.base import PhaseTest, TestResult


def _try_import(path: str):
    try:
        return importlib.import_module(path), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# B-001  CodeAgent importable
# ─────────────────────────────────────────────────────────────────────────────

async def _test_code_agent_importable() -> TestResult:
    mod, err = _try_import("app.graph.agents.code_agent")
    if err:
        return TestResult.failed(f"Cannot import app.graph.agents.code_agent: {err}")
    if not hasattr(mod, "build_code_agent"):
        return TestResult.failed("build_code_agent() not found in code_agent.py")
    return TestResult.passed("code_agent.build_code_agent found ✓")


# ─────────────────────────────────────────────────────────────────────────────
# B-002  CodeAgent compiles (CRITICAL)
# ─────────────────────────────────────────────────────────────────────────────

async def _test_code_agent_compiles() -> TestResult:
    mod, err = _try_import("app.graph.agents.code_agent")
    if err:
        return TestResult.skipped("code_agent not importable — see B-001")
    try:
        graph = mod.build_code_agent()
        if graph is None:
            return TestResult.failed("build_code_agent() returned None")
        return TestResult.passed("CodeAgent sub-graph compiled ✓")
    except Exception as exc:
        return TestResult.error(exc)


# ─────────────────────────────────────────────────────────────────────────────
# B-003  code_search_tool has Pydantic args_schema
# ─────────────────────────────────────────────────────────────────────────────

async def _test_code_search_tool_schema() -> TestResult:
    mod, err = _try_import("app.graph.agents.code_agent")
    if err:
        return TestResult.skipped("code_agent not importable")
    if not hasattr(mod, "code_search_tool"):
        return TestResult.failed("code_search_tool not found in code_agent.py")
    tool = mod.code_search_tool
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return TestResult.failed(
            "code_search_tool.args_schema is None — must set a Pydantic model"
        )
    # Verify it's a Pydantic model class
    try:
        from pydantic import BaseModel  # type: ignore
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            return TestResult.failed(
                f"args_schema={schema!r} is not a Pydantic BaseModel subclass"
            )
    except ImportError:
        pass
    return TestResult.passed(f"code_search_tool.args_schema = {schema.__name__} ✓")


# ─────────────────────────────────────────────────────────────────────────────
# B-004  DocAgent importable
# ─────────────────────────────────────────────────────────────────────────────

async def _test_doc_agent_importable() -> TestResult:
    mod, err = _try_import("app.graph.agents.doc_agent")
    if err:
        return TestResult.failed(f"Cannot import app.graph.agents.doc_agent: {err}")
    if not hasattr(mod, "build_doc_agent"):
        return TestResult.failed("build_doc_agent() not found in doc_agent.py")
    return TestResult.passed("doc_agent.build_doc_agent found ✓")


# ─────────────────────────────────────────────────────────────────────────────
# B-005  DocAgent uses kt_doc metadata filter
# ─────────────────────────────────────────────────────────────────────────────

async def _test_doc_agent_filter() -> TestResult:
    mod, err = _try_import("app.graph.agents.doc_agent")
    if err:
        return TestResult.skipped("doc_agent not importable")
    filter_val = getattr(mod, "DOC_AGENT_METADATA_FILTER", None)
    if filter_val is None:
        return TestResult.failed(
            "DOC_AGENT_METADATA_FILTER constant not found in doc_agent.py",
            detail='Add: DOC_AGENT_METADATA_FILTER = {"file_type": "kt_doc"}'
        )
    if filter_val.get("file_type") != "kt_doc":
        return TestResult.failed(
            f"DOC_AGENT_METADATA_FILTER has wrong file_type: {filter_val}"
        )
    return TestResult.passed('DOC_AGENT_METADATA_FILTER = {"file_type": "kt_doc"} ✓')


# ─────────────────────────────────────────────────────────────────────────────
# B-006  DebugAgent importable
# ─────────────────────────────────────────────────────────────────────────────

async def _test_debug_agent_importable() -> TestResult:
    mod, err = _try_import("app.graph.agents.debug_agent")
    if err:
        return TestResult.failed(f"Cannot import app.graph.agents.debug_agent: {err}")
    if not hasattr(mod, "build_debug_agent"):
        return TestResult.failed("build_debug_agent() not found")
    return TestResult.passed("debug_agent.build_debug_agent found ✓")


# ─────────────────────────────────────────────────────────────────────────────
# B-007  ArchAgent importable
# ─────────────────────────────────────────────────────────────────────────────

async def _test_arch_agent_importable() -> TestResult:
    mod, err = _try_import("app.graph.agents.arch_agent")
    if err:
        return TestResult.failed(f"Cannot import app.graph.agents.arch_agent: {err}")
    if not hasattr(mod, "build_arch_agent"):
        return TestResult.failed("build_arch_agent() not found")
    return TestResult.passed("arch_agent.build_arch_agent found ✓")


# ─────────────────────────────────────────────────────────────────────────────
# B-008  WebAgent importable
# ─────────────────────────────────────────────────────────────────────────────

async def _test_web_agent_importable() -> TestResult:
    mod, err = _try_import("app.graph.agents.web_agent")
    if err:
        return TestResult.failed(f"Cannot import app.graph.agents.web_agent: {err}")
    if not hasattr(mod, "build_web_agent"):
        return TestResult.failed("build_web_agent() not found")
    return TestResult.passed("web_agent.build_web_agent found ✓")


# ─────────────────────────────────────────────────────────────────────────────
# B-009  synthesizer_node — single-agent path: no LLM call
# ─────────────────────────────────────────────────────────────────────────────

async def _test_synthesizer_single_agent() -> TestResult:
    mod, err = _try_import("app.graph.nodes.synthesizer")
    if err:
        return TestResult.failed(f"Cannot import synthesizer node: {err}")
    if not hasattr(mod, "synthesizer_node"):
        return TestResult.failed("synthesizer_node not found in synthesizer.py")

    fake_state: dict[str, Any] = {
        "agent_responses": {"CodeAgent": "The function uses bcrypt."},
        "sources": [{"id": "src-1", "content": "..."}],
        "nodes_visited": [],
        "active_agent": "CodeAgent",
    }
    try:
        result = await mod.synthesizer_node(fake_state, {})
        if "final_response" not in result:
            return TestResult.failed("synthesizer_node did not write final_response")
        if result["final_response"] != "The function uses bcrypt.":
            return TestResult.failed(
                f"Single-agent path should copy agent response verbatim, "
                f"got: {result['final_response']!r}"
            )
        return TestResult.passed(
            "Single-agent synthesizer copies response without LLM call ✓"
        )
    except Exception as exc:
        return TestResult.error(exc)


# ─────────────────────────────────────────────────────────────────────────────
# B-010  synthesizer_node deduplicates sources by source_id
# ─────────────────────────────────────────────────────────────────────────────

async def _test_synthesizer_deduplication() -> TestResult:
    mod, err = _try_import("app.graph.nodes.synthesizer")
    if err:
        return TestResult.skipped("synthesizer not importable — see B-009")
    if not hasattr(mod, "deduplicate_sources"):
        return TestResult.failed("deduplicate_sources() helper not found in synthesizer.py")

    sources = [
        {"id": "src-1", "content": "aaa"},
        {"id": "src-2", "content": "bbb"},
        {"id": "src-1", "content": "aaa"},  # duplicate
    ]
    deduped = mod.deduplicate_sources(sources)
    if len(deduped) != 2:
        return TestResult.failed(
            f"Expected 2 sources after dedup, got {len(deduped)}: {deduped}"
        )
    return TestResult.passed("Source deduplication by id works ✓")


# ─────────────────────────────────────────────────────────────────────────────
# B-011  All agents registered as nodes in Supervisor (CRITICAL)
# ─────────────────────────────────────────────────────────────────────────────

async def _test_all_agents_in_supervisor() -> TestResult:
    try:
        from langgraph.checkpoint.memory import MemorySaver  # type: ignore
    except ImportError:
        return TestResult.skipped("langgraph not installed — see Phase A A-008")

    mod, err = _try_import("app.graph.supervisor_graph")
    if err:
        return TestResult.skipped("supervisor_graph not importable")

    try:
        graph = mod.build_supervisor_graph(checkpointer=MemorySaver())
        node_names = set(graph.nodes.keys())
        required = {"CodeAgent", "DocAgent", "DebugAgent", "ArchAgent", "WebAgent",
                    "synthesizer_node", "hil_check_node", "output_guardrail_node",
                    "response_node"}
        missing = required - node_names
        if missing:
            return TestResult.failed(
                f"Supervisor graph missing nodes: {missing}",
                detail=f"Registered nodes: {node_names}"
            )
        return TestResult.passed(
            f"All {len(required)} required nodes present in Supervisor ✓"
        )
    except Exception as exc:
        return TestResult.error(exc)


# ─────────────────────────────────────────────────────────────────────────────
# B-012  agent_brain.py compat shim still callable
# ─────────────────────────────────────────────────────────────────────────────

async def _test_agent_brain_compat_shim() -> TestResult:
    mod, err = _try_import("app.services.agents.agent_brain")
    if err:
        return TestResult.failed(f"Cannot import agent_brain: {err}")
    if not hasattr(mod, "AgentBrain"):
        return TestResult.failed("AgentBrain class not found — compat shim broken")
    # Check that the deprecated marker is present
    brain_cls = mod.AgentBrain
    has_deprecation = (
        "deprecated" in getattr(brain_cls, "__doc__", "").lower()
        or getattr(brain_cls, "_DEPRECATED", False)
        or hasattr(brain_cls, "_compat_shim")
    )
    if not has_deprecation:
        return TestResult.failed(
            "AgentBrain is not marked as deprecated",
            detail=(
                "Add _DEPRECATED = True or update docstring with 'deprecated' "
                "and ensure process_query delegates to supervisor graph"
            )
        )
    return TestResult.passed("AgentBrain compat shim present and marked deprecated ✓")


# ─────────────────────────────────────────────────────────────────────────────
# TESTS registry
# ─────────────────────────────────────────────────────────────────────────────

TESTS: list[PhaseTest] = [
    PhaseTest(
        id="B-001", name="CodeAgent importable",
        description="app.graph.agents.code_agent is importable",
        run=_test_code_agent_importable, critical=True, tags=["code-agent"],
    ),
    PhaseTest(
        id="B-002", name="CodeAgent compiles",
        description="build_code_agent() returns a compiled graph",
        run=_test_code_agent_compiles, critical=True, tags=["code-agent"],
    ),
    PhaseTest(
        id="B-003", name="code_search_tool has Pydantic args_schema",
        description="Tool arg validation is wired correctly",
        run=_test_code_search_tool_schema, critical=False, tags=["code-agent", "tools"],
    ),
    PhaseTest(
        id="B-004", name="DocAgent importable",
        description="app.graph.agents.doc_agent is importable",
        run=_test_doc_agent_importable, critical=False, tags=["doc-agent"],
    ),
    PhaseTest(
        id="B-005", name="DocAgent uses kt_doc metadata filter",
        description="DOC_AGENT_METADATA_FILTER constant has file_type=kt_doc",
        run=_test_doc_agent_filter, critical=False, tags=["doc-agent"],
    ),
    PhaseTest(
        id="B-006", name="DebugAgent importable",
        description="app.graph.agents.debug_agent is importable",
        run=_test_debug_agent_importable, critical=False, tags=["debug-agent"],
    ),
    PhaseTest(
        id="B-007", name="ArchAgent importable",
        description="app.graph.agents.arch_agent is importable",
        run=_test_arch_agent_importable, critical=False, tags=["arch-agent"],
    ),
    PhaseTest(
        id="B-008", name="WebAgent importable",
        description="app.graph.agents.web_agent is importable",
        run=_test_web_agent_importable, critical=False, tags=["web-agent"],
    ),
    PhaseTest(
        id="B-009", name="synthesizer_node single-agent fast path",
        description="Single-agent path copies response verbatim, no LLM",
        run=_test_synthesizer_single_agent, critical=False, tags=["synthesizer"],
    ),
    PhaseTest(
        id="B-010", name="synthesizer_node deduplicates sources",
        description="deduplicate_sources() removes entries with identical id",
        run=_test_synthesizer_deduplication, critical=False, tags=["synthesizer"],
    ),
    PhaseTest(
        id="B-011", name="All agents registered in Supervisor",
        description="Supervisor graph contains all 5 agent + 4 infrastructure nodes",
        run=_test_all_agents_in_supervisor, critical=True, tags=["supervisor"],
    ),
    PhaseTest(
        id="B-012", name="agent_brain.py compat shim callable",
        description="AgentBrain still exists and is marked deprecated",
        run=_test_agent_brain_compat_shim, critical=False, tags=["compat"],
    ),
]
