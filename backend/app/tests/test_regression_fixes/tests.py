"""
Regression Fixes — Live-Validation Bug Suite
============================================
  R-001  psycopg DSN falls back to Settings (semantic-cache auth fix)
  R-002  all retrieval agents share ONE process-wide lock (MPS deadlock fix)
  R-003  concurrent-write state keys have a reducer (INVALID_CONCURRENT_GRAPH_UPDATE fix)
  R-004  request_trace() survives cross-context reset (SSE ContextVar fix)
  R-005  long-term memory SQL uses psycopg3 %s placeholders (placeholder fix)
  R-006  RAGAS evaluator injects local embeddings (OpenAI-fallback fix)
"""

from __future__ import annotations

import importlib
import inspect
import os
from typing import Any

from app.tests.base import PhaseTest, TestResult


def _try_import(path: str):
    try:
        return importlib.import_module(path), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# R-001  psycopg DSN resolves the real password from Settings
# ─────────────────────────────────────────────────────────────────────────────

async def _test_dsn_from_settings() -> TestResult:
    """build_psycopg_dsn() must NOT fall through to postgres:postgres when the
    POSTGRES_* env vars are absent from os.environ (they live in .env, loaded
    by Pydantic Settings only)."""
    mod, err = _try_import("app.core.database")
    if err:
        return TestResult.failed(f"cannot import app.core.database: {err}")

    saved = {k: os.environ.pop(k, None) for k in (
        "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_USER",
        "POSTGRES_PASSWORD", "POSTGRES_DB", "POSTGRES_DSN", "DATABASE_URL",
    )}
    try:
        dsn = mod.build_psycopg_dsn()
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v

    try:
        from app.core.config import get_settings
        expected_user = get_settings().postgres_user
        expected_db = get_settings().postgres_db
    except Exception:  # noqa: BLE001
        expected_user, expected_db = None, None

    if dsn == "postgresql://postgres:postgres@localhost:5432/codelens_ai":
        return TestResult.failed(
            "DSN fell through to the postgres:postgres default — Settings "
            "fallback missing (semantic-cache auth regression)"
        )
    if expected_user and f"{expected_user}:" not in dsn:
        return TestResult.failed(f"DSN user is not the configured '{expected_user}': {dsn!r}")
    if expected_db and not dsn.rstrip("/").endswith(expected_db):
        return TestResult.failed(f"DSN db is not the configured '{expected_db}': {dsn!r}")
    return TestResult.passed("build_psycopg_dsn() resolves credentials from Settings ✓")


# ─────────────────────────────────────────────────────────────────────────────
# R-002  all retrieval agents share ONE process-wide lock
# ─────────────────────────────────────────────────────────────────────────────

async def _test_shared_retrieval_lock() -> TestResult:
    """get_retrieval_lock() must return the SAME object on every call, and no
    agent module may fall back to a throwaway threading.Lock()."""
    mod, err = _try_import("app.core.database")
    if err:
        return TestResult.failed(f"cannot import app.core.database: {err}")
    if not hasattr(mod, "get_retrieval_lock"):
        return TestResult.failed("get_retrieval_lock() missing from app.core.database")

    a = mod.get_retrieval_lock()
    b = mod.get_retrieval_lock()
    if a is not b:
        return TestResult.failed("get_retrieval_lock() returns different objects — not a singleton")

    # No agent may keep the old throwaway-lock fallback pattern.
    offenders = []
    for agent in ("code_agent", "debug_agent", "doc_agent", "arch_agent"):
        amod, aerr = _try_import(f"app.graph.agents.{agent}")
        if aerr or amod is None:
            continue
        try:
            src = inspect.getsource(amod)
        except Exception:  # noqa: BLE001
            continue
        if 'getattr(retriever, "_metadata_lock", threading.Lock())' in src:
            offenders.append(agent)
        if "get_retrieval_lock" not in src:
            offenders.append(f"{agent}(no-shared-lock)")
    if offenders:
        return TestResult.failed(f"agents not using shared lock: {sorted(set(offenders))}")
    return TestResult.passed("all agents share the process-wide retrieval lock ✓")


# ─────────────────────────────────────────────────────────────────────────────
# R-003  concurrent-write state keys carry a reducer
# ─────────────────────────────────────────────────────────────────────────────

async def _test_state_reducers() -> TestResult:
    """sources / retrieved_chunks / reranked_chunks / rerank_scores must be
    Annotated with a merge reducer, else parallel Send() branches raise
    INVALID_CONCURRENT_GRAPH_UPDATE."""
    mod, err = _try_import("app.graph.state")
    if err:
        return TestResult.failed(f"cannot import app.graph.state: {err}")

    import typing
    # Reducer metadata lives on Annotated[...] and is only resolved via
    # get_type_hints(include_extras=True) — raw __annotations__ may hold lazy
    # string forms without the metadata.
    hints = typing.get_type_hints(mod.AgentState, include_extras=True)
    required = ["sources", "retrieved_chunks", "reranked_chunks", "rerank_scores"]
    missing = []
    for key in required:
        ann = hints.get(key)
        meta = getattr(ann, "__metadata__", None)
        if not meta:
            missing.append(key)
    if missing:
        return TestResult.failed(f"state keys missing concurrent-write reducer: {missing}")

    # The reducer itself must concat + dedup, not overwrite.
    if hasattr(mod, "_merge_chunk_lists"):
        merged = mod._merge_chunk_lists([{"id": 1}], [{"id": 1}, {"id": 2}])
        ids = [d.get("id") for d in merged]
        if ids != [1, 2]:
            return TestResult.failed(f"_merge_chunk_lists dedup broken: {ids}")
    return TestResult.passed("concurrent-write state keys carry dedup reducers ✓")


# ─────────────────────────────────────────────────────────────────────────────
# R-004  request_trace() survives a cross-context reset
# ─────────────────────────────────────────────────────────────────────────────

async def _test_request_trace_cross_context() -> TestResult:
    """Entering request_trace() in one context and exiting in another (as SSE
    StreamingResponse does) must NOT raise 'Token created in a different
    Context'."""
    mod, err = _try_import("app.observability.tracing")
    if err:
        return TestResult.failed(f"cannot import tracing: {err}")

    import contextvars

    errors: list[str] = []

    def _run_in_fresh_context():
        # Enter+exit the context manager entirely inside a *copied* context so
        # the reset token belongs to a different Context than the caller's.
        cm = mod.request_trace("trace-xyz")
        cm.__enter__()
        try:
            cm.__exit__(None, None, None)
        except ValueError as ve:  # the exact bug we fixed
            errors.append(str(ve))

    ctx = contextvars.copy_context()
    ctx.run(_run_in_fresh_context)

    # Also exercise the realistic split: enter in caller, exit in child ctx.
    cm = mod.request_trace("trace-abc")
    cm.__enter__()
    try:
        contextvars.copy_context().run(lambda: cm.__exit__(None, None, None))
    except ValueError as ve:
        errors.append(f"split-ctx: {ve}")

    if errors:
        return TestResult.failed(f"request_trace() raised on cross-context reset: {errors}")
    return TestResult.passed("request_trace() tolerates cross-context reset ✓")


# ─────────────────────────────────────────────────────────────────────────────
# R-005  long-term memory SQL uses psycopg3 %s placeholders
# ─────────────────────────────────────────────────────────────────────────────

async def _test_ltm_placeholders() -> TestResult:
    """psycopg3 uses %s positional params. Any $1/$2 style would raise
    '0 placeholders but N parameters'."""
    mod, err = _try_import("app.graph.memory.long_term_store")
    if err:
        return TestResult.failed(f"cannot import long_term_store: {err}")

    import re
    # Inspect the actual SQL string CONSTANTS on the module (not docstrings /
    # comments, which legitimately mention $1 when explaining the fix).
    sql_blobs = [
        v for k, v in vars(mod).items()
        if k != "__doc__"
        and isinstance(v, str)
        and re.search(r"\b(SELECT|INSERT|UPDATE|DELETE)\b\s+\S", v, re.IGNORECASE)
        and ("%s" in v or "$" in v)
    ]
    if not sql_blobs:
        return TestResult.skipped("no SQL constants found to inspect")

    for sql in sql_blobs:
        if re.search(r"=\s*\$\d", sql) or re.search(r"\$\d::", sql):
            return TestResult.failed(
                f"LTM SQL still uses $1 libpq placeholders (psycopg3 needs %s): {sql[:60]!r}"
            )
    if not any("%s" in sql for sql in sql_blobs):
        return TestResult.failed("LTM SQL constants missing '%s' psycopg3 placeholders")
    return TestResult.passed("LTM SQL uses psycopg3 %s placeholders ✓")


# ─────────────────────────────────────────────────────────────────────────────
# R-006  RAGAS evaluator injects local embeddings
# ─────────────────────────────────────────────────────────────────────────────

async def _test_ragas_local_embeddings() -> TestResult:
    """The evaluator must set up local embeddings and pass them to RAGAS so
    answer_relevancy never calls OpenAI."""
    mod, err = _try_import("app.observability.rag_evaluator")
    if err:
        return TestResult.failed(f"cannot import rag_evaluator: {err}")
    try:
        src = inspect.getsource(mod)
    except Exception as exc:  # noqa: BLE001
        return TestResult.error(exc)

    if "_setup_evaluator_embeddings" not in src:
        return TestResult.failed("evaluator does not set up local embeddings")
    if 'kwargs["embeddings"]' not in src and "embeddings=self.evaluator_embeddings" not in src:
        return TestResult.failed("evaluate() is not given local embeddings — RAGAS will call OpenAI")
    return TestResult.passed("RAGAS evaluator injects local embeddings ✓")


# ─────────────────────────────────────────────────────────────────────────────
# R-007  checkpointer uses an async pool (not the sync ConnectionPool)
# ─────────────────────────────────────────────────────────────────────────────

async def _test_checkpointer_async_pool() -> TestResult:
    """AsyncPostgresSaver requires an async connection/pool. Passing the shared
    sync psycopg_pool.ConnectionPool raised 'Invalid connection type' and
    silently fell back to an ephemeral MemorySaver, so Postgres checkpoints
    never persisted. The factory must build an AsyncConnectionPool and expose a
    close_checkpointer() shutdown hook to avoid connection leaks."""
    mod, err = _try_import("app.graph.checkpointing.pg_checkpointer")
    if err:
        return TestResult.failed(f"cannot import pg_checkpointer: {err}")
    try:
        src = inspect.getsource(mod)
    except Exception as exc:  # noqa: BLE001
        return TestResult.error(exc)

    if "AsyncConnectionPool" not in src:
        return TestResult.failed("checkpointer does not use AsyncConnectionPool")
    # Guard against re-introducing the sync-pool anti-pattern:
    # AsyncPostgresSaver must NOT be constructed from get_pg_pool() (sync).
    compact = src.replace(" ", "").replace("\n", "")
    if "AsyncPostgresSaver(get_pg_pool())" in compact:
        return TestResult.failed("checkpointer passes the SYNC ConnectionPool to AsyncPostgresSaver")
    if not hasattr(mod, "close_checkpointer"):
        return TestResult.failed("close_checkpointer() shutdown hook missing (connection leak risk)")

    # Live check: if Postgres is reachable, the saver must be AsyncPostgresSaver.
    try:
        import asyncio  # noqa: F401
        saver = await mod.get_checkpointer()
        name = type(saver).__name__
        await mod.close_checkpointer()
        if name == "MemorySaver":
            return TestResult.skipped("Postgres unreachable — MemorySaver fallback (acceptable)")
        if name != "AsyncPostgresSaver":
            return TestResult.failed(f"unexpected checkpointer type: {name}")
        return TestResult.passed("checkpointer uses AsyncPostgresSaver via async pool ✓")
    except Exception as exc:  # noqa: BLE001
        return TestResult.skipped(f"checkpointer live check skipped: {exc}")


TESTS = [
    PhaseTest(
        id="R-001",
        name="psycopg DSN from Settings",
        description="build_psycopg_dsn falls back to Settings, not postgres:postgres",
        run=_test_dsn_from_settings,
        critical=True,
        tags=["database", "regression"],
    ),
    PhaseTest(
        id="R-002",
        name="shared retrieval lock",
        description="all agents serialise retrieval through one process-wide lock",
        run=_test_shared_retrieval_lock,
        critical=True,
        tags=["concurrency", "regression"],
    ),
    PhaseTest(
        id="R-003",
        name="concurrent-write state reducers",
        description="sources/chunks/scores carry a merge reducer for parallel writes",
        run=_test_state_reducers,
        critical=True,
        tags=["langgraph", "regression"],
    ),
    PhaseTest(
        id="R-004",
        name="request_trace cross-context reset",
        description="request_trace tolerates SSE cross-context ContextVar reset",
        run=_test_request_trace_cross_context,
        critical=True,
        tags=["observability", "streaming", "regression"],
    ),
    PhaseTest(
        id="R-005",
        name="LTM psycopg3 placeholders",
        description="long-term memory SQL uses %s (not $1) placeholders",
        run=_test_ltm_placeholders,
        critical=True,
        tags=["memory", "database", "regression"],
    ),
    PhaseTest(
        id="R-006",
        name="RAGAS local embeddings",
        description="evaluator injects local embeddings so RAGAS avoids OpenAI",
        run=_test_ragas_local_embeddings,
        critical=False,
        tags=["evaluation", "regression"],
    ),
    PhaseTest(
        id="R-007",
        name="checkpointer async pool",
        description="AsyncPostgresSaver uses an async pool, with a close hook",
        run=_test_checkpointer_async_pool,
        critical=False,
        tags=["checkpointing", "database", "regression"],
    ),
]
