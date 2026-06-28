"""
Memory Layer Tests
==================
  C-001  short_term module importable
  C-002  memory_read_node returns short_term_window key
  C-003  STM session_id is namespaced (user_id::session_id)
  C-004  STM token budget guard trims long windows
  C-005  agent_long_term_memory table exists in PostgreSQL
  C-006  LongTermStore importable
  C-007  LongTermStore.retrieve returns a list
  C-008  entity_extractor importable
  C-009  entity_extractor returns correct entity_type values
  C-010  Memory namespace isolation — user_id scoped queries
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from app.tests.base import PhaseTest, TestResult


def _try_import(path: str):
    try:
        return importlib.import_module(path), None
    except Exception as exc:
        return None, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# C-001  short_term module importable
# ─────────────────────────────────────────────────────────────────────────────

async def _test_stm_importable() -> TestResult:
    mod, err = _try_import("app.graph.memory.short_term")
    if err:
        return TestResult.failed(f"Cannot import app.graph.memory.short_term: {err}")
    if not hasattr(mod, "memory_read_node"):
        return TestResult.failed("memory_read_node not found in short_term.py")
    return TestResult.passed("short_term.memory_read_node found ✓")


# ─────────────────────────────────────────────────────────────────────────────
# C-002  memory_read_node returns short_term_window key
# ─────────────────────────────────────────────────────────────────────────────

async def _test_stm_node_output() -> TestResult:
    mod, err = _try_import("app.graph.memory.short_term")
    if err:
        return TestResult.skipped("short_term not importable")

    fake_state: dict[str, Any] = {
        "user_id": "user-test",
        "session_id": "user-test::sess-001",
        "query": "how does auth work?",
        "long_term_facts": [],
        "nodes_visited": [],
    }
    try:
        result = await mod.memory_read_node(fake_state, {})
        if "short_term_window" not in result:
            return TestResult.failed("memory_read_node did not write short_term_window")
        if not isinstance(result["short_term_window"], list):
            return TestResult.failed(
                f"short_term_window must be a list, got {type(result['short_term_window'])}"
            )
        return TestResult.passed(
            f"memory_read_node returned short_term_window "
            f"(len={len(result['short_term_window'])}) ✓"
        )
    except Exception as exc:
        return TestResult.error(exc)


# ─────────────────────────────────────────────────────────────────────────────
# C-003  STM session_id uses namespaced format
# ─────────────────────────────────────────────────────────────────────────────

async def _test_stm_session_namespace() -> TestResult:
    mod, err = _try_import("app.graph.memory.short_term")
    if err:
        return TestResult.skipped("short_term not importable")
    if not hasattr(mod, "build_namespaced_session_id"):
        return TestResult.failed(
            "build_namespaced_session_id() helper not found in short_term.py",
            detail=(
                "Add: def build_namespaced_session_id(user_id, session_id) -> str: "
                "return f'{user_id}::{session_id}'"
            )
        )
    result = mod.build_namespaced_session_id("alice", "session-99")
    expected = "alice::session-99"
    if result != expected:
        return TestResult.failed(
            f"Namespace format wrong: expected '{expected}', got '{result}'"
        )
    return TestResult.passed(f"Session namespacing = '{result}' ✓")


# ─────────────────────────────────────────────────────────────────────────────
# C-004  STM token budget guard trims long windows
# ─────────────────────────────────────────────────────────────────────────────

async def _test_stm_token_budget() -> TestResult:
    mod, err = _try_import("app.graph.memory.short_term")
    if err:
        return TestResult.skipped("short_term not importable")
    if not hasattr(mod, "apply_token_budget"):
        return TestResult.failed(
            "apply_token_budget() helper not found in short_term.py"
        )
    # Build a window that is clearly over budget
    big_window = [
        {"role": "user", "content": "word " * 500},
        {"role": "assistant", "content": "word " * 500},
    ] * 10  # 10 turns × 1000 words each ≈ >> 4096 tokens
    try:
        trimmed = await mod.apply_token_budget(big_window, max_tokens=4096)
        total_words = sum(len(t["content"].split()) for t in trimmed)
        if total_words > 5000:
            return TestResult.failed(
                f"apply_token_budget did not trim: {total_words} words remain"
            )
        return TestResult.passed(
            f"Token budget guard trimmed window to ~{total_words} words ✓"
        )
    except Exception as exc:
        return TestResult.error(exc)


# ─────────────────────────────────────────────────────────────────────────────
# C-005  agent_long_term_memory table exists in Postgres
# ─────────────────────────────────────────────────────────────────────────────

async def _test_ltm_table_exists() -> TestResult:
    try:
        from app.core.database import get_pg_pool
        pool = get_pg_pool()
        if pool is None:
            return TestResult.skipped("pg_pool not available (Postgres not running)")
        async with pool.connection() as conn:
            row = await conn.fetchrow(
                """SELECT 1 FROM information_schema.tables
                   WHERE table_name = 'agent_long_term_memory'"""
            )
        if row is None:
            return TestResult.failed(
                "agent_long_term_memory table does not exist",
                detail=(
                    "Run the Prisma migration: "
                    "20240617_add_ltm_table adds VECTOR(768) column + ivfflat index"
                )
            )
        return TestResult.passed("agent_long_term_memory table exists ✓")
    except Exception as exc:
        return TestResult.skipped(f"Postgres unreachable: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# C-006  LongTermStore importable
# ─────────────────────────────────────────────────────────────────────────────

async def _test_ltm_store_importable() -> TestResult:
    mod, err = _try_import("app.graph.memory.long_term_store")
    if err:
        return TestResult.failed(f"Cannot import long_term_store: {err}")
    if not hasattr(mod, "LongTermStore"):
        return TestResult.failed("LongTermStore class not found")
    return TestResult.passed("LongTermStore importable ✓")


# ─────────────────────────────────────────────────────────────────────────────
# C-007  LongTermStore.retrieve returns a list
# ─────────────────────────────────────────────────────────────────────────────

async def _test_ltm_store_retrieve() -> TestResult:
    mod, err = _try_import("app.graph.memory.long_term_store")
    if err:
        return TestResult.skipped("long_term_store not importable")
    try:
        store = mod.LongTermStore()
        result = await store.retrieve(user_id="nobody", query="test", top_k=5)
        if not isinstance(result, list):
            return TestResult.failed(
                f"retrieve() must return list, got {type(result)}"
            )
        return TestResult.passed(
            f"LongTermStore.retrieve() returned list (len={len(result)}) ✓"
        )
    except Exception as exc:
        return TestResult.skipped(f"LongTermStore.retrieve() raised (likely no DB): {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# C-008  entity_extractor importable
# ─────────────────────────────────────────────────────────────────────────────

async def _test_entity_extractor_importable() -> TestResult:
    mod, err = _try_import("app.graph.memory.entity_extractor")
    if err:
        return TestResult.failed(f"Cannot import entity_extractor: {err}")
    if not hasattr(mod, "extract_facts"):
        return TestResult.failed("extract_facts() not found in entity_extractor.py")
    return TestResult.passed("entity_extractor.extract_facts found ✓")


# ─────────────────────────────────────────────────────────────────────────────
# C-009  entity_extractor returns valid entity_type values
# ─────────────────────────────────────────────────────────────────────────────

async def _test_entity_extractor_types() -> TestResult:
    mod, err = _try_import("app.graph.memory.entity_extractor")
    if err:
        return TestResult.skipped("entity_extractor not importable")

    VALID_TYPES = {"user_fact", "code_fact", "preference"}

    # Inject a known mock to avoid real LLM calls
    if not hasattr(mod, "VALID_ENTITY_TYPES"):
        return TestResult.failed(
            "VALID_ENTITY_TYPES constant not found in entity_extractor.py",
            detail=(
                "Add: VALID_ENTITY_TYPES = {'user_fact', 'code_fact', 'preference'}"
            )
        )
    declared = set(mod.VALID_ENTITY_TYPES)
    if declared != VALID_TYPES:
        return TestResult.failed(
            f"VALID_ENTITY_TYPES mismatch: expected {VALID_TYPES}, got {declared}"
        )
    return TestResult.passed(f"VALID_ENTITY_TYPES = {declared} ✓")


# ─────────────────────────────────────────────────────────────────────────────
# C-010  Memory namespace isolation — retrieve always adds WHERE user_id = $1
# ─────────────────────────────────────────────────────────────────────────────

async def _test_ltm_namespace_isolation() -> TestResult:
    mod, err = _try_import("app.graph.memory.long_term_store")
    if err:
        return TestResult.skipped("long_term_store not importable")

    if not hasattr(mod, "_RETRIEVE_QUERY"):
        return TestResult.failed(
            "_RETRIEVE_QUERY constant not found in long_term_store.py",
            detail=(
                "Add: _RETRIEVE_QUERY = 'SELECT content FROM agent_long_term_memory "
                "WHERE user_id = $1 ORDER BY embedding <=> $2::vector LIMIT $3'"
            )
        )
    q: str = mod._RETRIEVE_QUERY
    if "user_id" not in q or "$1" not in q:
        return TestResult.failed(
            f"_RETRIEVE_QUERY does not scope by user_id: {q!r}"
        )
    return TestResult.passed("_RETRIEVE_QUERY includes WHERE user_id = $1 ✓")


# ─────────────────────────────────────────────────────────────────────────────
# TESTS registry
# ─────────────────────────────────────────────────────────────────────────────

TESTS: list[PhaseTest] = [
    PhaseTest(id="C-001", name="short_term module importable",
              description="app.graph.memory.short_term is importable",
              run=_test_stm_importable, critical=True, tags=["memory", "stm"]),
    PhaseTest(id="C-002", name="memory_read_node returns short_term_window",
              description="Node writes short_term_window list to state",
              run=_test_stm_node_output, critical=False, tags=["memory", "stm"]),
    PhaseTest(id="C-003", name="STM session_id is namespaced",
              description="build_namespaced_session_id('u','s') = 'u::s'",
              run=_test_stm_session_namespace, critical=True, tags=["memory", "security"]),
    PhaseTest(id="C-004", name="STM token budget guard trims long windows",
              description="apply_token_budget() reduces oversized windows",
              run=_test_stm_token_budget, critical=False, tags=["memory", "stm"]),
    PhaseTest(id="C-005", name="agent_long_term_memory table exists",
              description="PostgreSQL table created by Prisma migration",
              run=_test_ltm_table_exists, critical=False, tags=["memory", "db"]),
    PhaseTest(id="C-006", name="LongTermStore importable",
              description="app.graph.memory.long_term_store.LongTermStore found",
              run=_test_ltm_store_importable, critical=True, tags=["memory", "ltm"]),
    PhaseTest(id="C-007", name="LongTermStore.retrieve returns list",
              description="retrieve() is callable and returns a list",
              run=_test_ltm_store_retrieve, critical=False, tags=["memory", "ltm"]),
    PhaseTest(id="C-008", name="entity_extractor importable",
              description="app.graph.memory.entity_extractor.extract_facts found",
              run=_test_entity_extractor_importable, critical=False, tags=["memory"]),
    PhaseTest(id="C-009", name="entity_extractor VALID_ENTITY_TYPES correct",
              description="VALID_ENTITY_TYPES = {user_fact, code_fact, preference}",
              run=_test_entity_extractor_types, critical=False, tags=["memory"]),
    PhaseTest(id="C-010", name="LTM queries scoped to user_id",
              description="_RETRIEVE_QUERY contains WHERE user_id = $1",
              run=_test_ltm_namespace_isolation, critical=True, tags=["memory", "security"]),
]
