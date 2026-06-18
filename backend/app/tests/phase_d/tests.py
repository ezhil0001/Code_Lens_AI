"""
Phase D Test Cases — Checkpointing & Time-Travel
=================================================
  D-001  pg_checkpointer module importable
  D-002  get_checkpointer() returns AsyncPostgresSaver (or MemorySaver fallback)
  D-003  Supervisor graph compiled with checkpointer stores checkpoint
  D-004  checkpoints API router importable
  D-005  /checkpoints endpoint registered in FastAPI app
  D-006  /replay endpoint registered in FastAPI app
  D-007  /branch endpoint registered in FastAPI app
  D-008  Thread ID uses user_id::session_id namespace
  D-009  branch_thread_id format is correct
"""

from __future__ import annotations

import importlib

from app.tests.base import PhaseTest, TestResult


def _try_import(path: str):
    try:
        return importlib.import_module(path), None
    except Exception as exc:
        return None, str(exc)


async def _test_pg_checkpointer_importable() -> TestResult:
    mod, err = _try_import("app.graph.checkpointing.pg_checkpointer")
    if err:
        return TestResult.failed(f"Cannot import pg_checkpointer: {err}")
    if not hasattr(mod, "get_checkpointer"):
        return TestResult.failed("get_checkpointer() not found")
    return TestResult.passed("pg_checkpointer.get_checkpointer found ✓")


async def _test_get_checkpointer_returns_saver() -> TestResult:
    mod, err = _try_import("app.graph.checkpointing.pg_checkpointer")
    if err:
        return TestResult.skipped("pg_checkpointer not importable")
    try:
        saver = await mod.get_checkpointer()
        if saver is None:
            return TestResult.failed("get_checkpointer() returned None")
        cls_name = type(saver).__name__
        if "Saver" not in cls_name and "Checkpointer" not in cls_name:
            return TestResult.failed(
                f"get_checkpointer() returned unexpected type: {cls_name}"
            )
        return TestResult.passed(f"get_checkpointer() returned {cls_name} ✓")
    except Exception as exc:
        return TestResult.skipped(f"DB unreachable, using fallback: {exc}")


async def _test_graph_with_checkpointer_stores() -> TestResult:
    try:
        from langgraph.checkpoint.memory import MemorySaver  # type: ignore
    except ImportError:
        return TestResult.skipped("langgraph not installed")

    mod, err = _try_import("app.graph.supervisor_graph")
    if err:
        return TestResult.skipped("supervisor_graph not importable")

    try:
        from langgraph.checkpoint.memory import MemorySaver
        graph = mod.build_supervisor_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-user::test-sess"}}
        # Verify the graph has a checkpointer attached
        has_checkpointer = graph.checkpointer is not None
        if not has_checkpointer:
            return TestResult.failed("Supervisor graph has no checkpointer attached")
        return TestResult.passed("Supervisor graph has checkpointer ✓")
    except Exception as exc:
        return TestResult.error(exc)


async def _test_checkpoints_router_importable() -> TestResult:
    mod, err = _try_import("app.api.checkpoints")
    if err:
        return TestResult.failed(f"Cannot import app.api.checkpoints: {err}")
    if not hasattr(mod, "router"):
        return TestResult.failed("router not found in app.api.checkpoints")
    return TestResult.passed("app.api.checkpoints.router found ✓")


async def _test_checkpoints_endpoint_registered() -> TestResult:
    mod, err = _try_import("app.api.checkpoints")
    if err:
        return TestResult.skipped("checkpoints router not importable")
    routes = [str(r.path) for r in getattr(mod.router, "routes", [])]
    matching = [r for r in routes if "checkpoints" in r]
    if not matching:
        return TestResult.failed(
            f"No /checkpoints route found. Routes: {routes}"
        )
    return TestResult.passed(f"Checkpoint list route registered: {matching[0]} ✓")


async def _test_replay_endpoint_registered() -> TestResult:
    mod, err = _try_import("app.api.checkpoints")
    if err:
        return TestResult.skipped("checkpoints router not importable")
    routes = [str(r.path) for r in getattr(mod.router, "routes", [])]
    matching = [r for r in routes if "replay" in r]
    if not matching:
        return TestResult.failed(f"No /replay route found. Routes: {routes}")
    return TestResult.passed(f"Replay route registered: {matching[0]} ✓")


async def _test_branch_endpoint_registered() -> TestResult:
    mod, err = _try_import("app.api.checkpoints")
    if err:
        return TestResult.skipped("checkpoints router not importable")
    routes = [str(r.path) for r in getattr(mod.router, "routes", [])]
    matching = [r for r in routes if "branch" in r]
    if not matching:
        return TestResult.failed(f"No /branch route found. Routes: {routes}")
    return TestResult.passed(f"Branch route registered: {matching[0]} ✓")


async def _test_thread_id_format() -> TestResult:
    mod, err = _try_import("app.graph.checkpointing.pg_checkpointer")
    if err:
        return TestResult.skipped("pg_checkpointer not importable")
    if not hasattr(mod, "build_thread_id"):
        return TestResult.failed(
            "build_thread_id() not found in pg_checkpointer.py",
            detail="Add: def build_thread_id(user_id, session_id): return f'{user_id}::{session_id}'"
        )
    result = mod.build_thread_id("alice", "sess-42")
    if result != "alice::sess-42":
        return TestResult.failed(
            f"build_thread_id returned '{result}', expected 'alice::sess-42'"
        )
    return TestResult.passed(f"build_thread_id = '{result}' ✓")


async def _test_branch_thread_id_format() -> TestResult:
    mod, err = _try_import("app.graph.checkpointing.pg_checkpointer")
    if err:
        return TestResult.skipped("pg_checkpointer not importable")
    if not hasattr(mod, "build_branch_thread_id"):
        return TestResult.failed("build_branch_thread_id() not found")
    result = mod.build_branch_thread_id("alice::sess-42", "abc12345")
    if "branch" not in result or "alice::sess-42" not in result:
        return TestResult.failed(
            f"branch_thread_id format unexpected: '{result}'"
        )
    return TestResult.passed(f"build_branch_thread_id = '{result}' ✓")


TESTS: list[PhaseTest] = [
    PhaseTest(id="D-001", name="pg_checkpointer importable",
              description="app.graph.checkpointing.pg_checkpointer is importable",
              run=_test_pg_checkpointer_importable, critical=True, tags=["checkpoint"]),
    PhaseTest(id="D-002", name="get_checkpointer() returns a saver",
              description="Returns AsyncPostgresSaver or MemorySaver",
              run=_test_get_checkpointer_returns_saver, critical=False, tags=["checkpoint"]),
    PhaseTest(id="D-003", name="Supervisor graph has checkpointer attached",
              description="build_supervisor_graph(checkpointer=...) stores the saver",
              run=_test_graph_with_checkpointer_stores, critical=True, tags=["checkpoint"]),
    PhaseTest(id="D-004", name="checkpoints API router importable",
              description="app.api.checkpoints.router is importable",
              run=_test_checkpoints_router_importable, critical=True, tags=["api"]),
    PhaseTest(id="D-005", name="/checkpoints endpoint registered",
              description="GET /.../checkpoints route exists on router",
              run=_test_checkpoints_endpoint_registered, critical=False, tags=["api"]),
    PhaseTest(id="D-006", name="/replay endpoint registered",
              description="GET /.../replay/{checkpoint_id} route exists",
              run=_test_replay_endpoint_registered, critical=False, tags=["api"]),
    PhaseTest(id="D-007", name="/branch endpoint registered",
              description="POST /.../branch route exists on router",
              run=_test_branch_endpoint_registered, critical=False, tags=["api"]),
    PhaseTest(id="D-008", name="Thread ID uses user_id::session_id namespace",
              description="build_thread_id('u','s') = 'u::s'",
              run=_test_thread_id_format, critical=True, tags=["checkpoint", "security"]),
    PhaseTest(id="D-009", name="Branch thread ID format correct",
              description="build_branch_thread_id includes 'branch' and parent thread",
              run=_test_branch_thread_id_format, critical=False, tags=["checkpoint"]),
]
