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


async def _test_hil_node_importable() -> TestResult:
    mod, err = _try_import("app.graph.nodes.hil_node")
    if err:
        return TestResult.failed(f"Cannot import app.graph.nodes.hil_node: {err}")
    if not hasattr(mod, "hil_check_node"):
        return TestResult.failed("hil_check_node not found")
    return TestResult.passed("hil_node.hil_check_node found ✓")


async def _test_hil_triggers_on_low_confidence() -> TestResult:
    mod, err = _try_import("app.graph.nodes.hil_node")
    if err:
        return TestResult.skipped("hil_node not importable")
    fake_state: dict[str, Any] = {
        "query": "how does the cache work?",
        "routing_confidence": 0.3,   # below threshold
        "active_agent": "CodeAgent",
        "final_response": "Some response",
        "nodes_visited": [],
    }
    try:
        from langgraph.errors import NodeInterrupt
    except Exception:
        NodeInterrupt = None  # type: ignore[assignment]
    try:
        result = await mod.hil_check_node(fake_state, {})
        if not result.get("hil_required"):
            return TestResult.failed(
                "hil_check_node should set hil_required=True when confidence=0.3"
            )
        return TestResult.passed("HIL triggered on low-confidence routing ✓")
    except Exception as exc:
        if NodeInterrupt is not None and isinstance(exc, NodeInterrupt):
            return TestResult.passed(
                "HIL raised NodeInterrupt on low-confidence routing (true interrupt) ✓"
            )
        return TestResult.error(exc)


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
        from langgraph.errors import NodeInterrupt
    except Exception:
        NodeInterrupt = None  # type: ignore[assignment]
    try:
        result = await mod.hil_check_node(fake_state, {})
        if not result.get("hil_required"):
            return TestResult.failed(
                "HIL should trigger on destructive keywords ('drop table', 'remove all')"
            )
        return TestResult.passed("HIL triggered on destructive keyword ✓")
    except Exception as exc:
        if NodeInterrupt is not None and isinstance(exc, NodeInterrupt):
            return TestResult.passed(
                "HIL raised NodeInterrupt on destructive keyword (true interrupt) ✓"
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
        from app.core.database import get_pg_pool
        pool = get_pg_pool()
        if pool is None:
            return TestResult.skipped("pg_pool not available")
        async with pool.connection() as conn:
            row = await conn.fetchrow(
                """SELECT 1 FROM information_schema.tables
                   WHERE table_name = 'audit_log'"""
            )
        if row is None:
            return TestResult.failed(
                "audit_log table does not exist",
                detail="HIL interrupts must be persisted to audit_log"
            )
        return TestResult.passed("audit_log table exists ✓")
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
]
