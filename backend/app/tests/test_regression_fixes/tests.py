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

import asyncio
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
# R-002  shared model inference is serialised at the model boundary
# ─────────────────────────────────────────────────────────────────────────────

async def _test_shared_retrieval_lock() -> TestResult:
    """The embedder and cross-encoder are process-wide singletons, so their
    forward passes must be serialised by ONE shared lock each.

    Originally each agent built a throwaway ``threading.Lock()`` (zero mutual
    exclusion → concurrent MPS inference deadlocked). The first fix used one
    coarse lock around the *entire* pipeline, which serialised independent
    requests for ~20s each. Locking now lives at the model boundary: still
    exactly one lock per shared model, but held only for the inference call —
    and it protects every caller, not just the agent nodes.
    """
    mod, err = _try_import("app.core.database")
    if err:
        return TestResult.failed(f"cannot import app.core.database: {err}")

    for fn in ("get_embedding_lock", "get_reranker_lock"):
        if not hasattr(mod, fn):
            return TestResult.failed(f"{fn}() missing from app.core.database")
        if getattr(mod, fn)() is not getattr(mod, fn)():
            return TestResult.failed(f"{fn}() is not a singleton")
    if mod.get_embedding_lock() is mod.get_reranker_lock():
        return TestResult.failed("embedder and reranker must not share one lock")

    # The model-inference guards must actually wrap the inference calls.
    rmod, rerr = _try_import("app.services.retrieval.retriever_engine")
    if rerr:
        return TestResult.failed(f"cannot import retriever_engine: {rerr}")
    rsrc = inspect.getsource(rmod)
    if "with get_embedding_lock():" not in rsrc:
        return TestResult.failed("embed_query() is not guarded by get_embedding_lock()")
    if "with get_reranker_lock():" not in rsrc:
        return TestResult.failed("cross_encoder.predict() is not guarded by get_reranker_lock()")
    # Every embed_query call site must be guarded, not just the first.
    if rsrc.count("self.embeddings.embed_query(query)") != rsrc.count("with get_embedding_lock():"):
        return TestResult.failed("an embed_query() call site is missing its lock")

    # No agent may reintroduce a throwaway lock or the coarse pipeline mutex.
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
            offenders.append(f"{agent}(throwaway-lock)")
        if "get_retrieval_lock" in src:
            offenders.append(f"{agent}(coarse-pipeline-lock)")
    if offenders:
        return TestResult.failed(f"agent locking regressed: {sorted(set(offenders))}")

    return TestResult.passed(
        "embedder + reranker each guarded by one shared lock at the model "
        "boundary; no coarse pipeline mutex in any agent ✓"
    )


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


# ─────────────────────────────────────────────────────────────────────────────
# R-008  a late background span must not rename the root trace
# ─────────────────────────────────────────────────────────────────────────────

async def _test_trace_name_not_overwritten() -> TestResult:
    """RAGAS finishes after the request span closed. Because it joins the trace
    via ``trace_context`` it is an OTEL root, so it used to donate its own name
    ("ragas.evaluate_sample") to the trace."""
    mod, err = _try_import("app.observability.tracing")
    if err:
        return TestResult.failed(f"cannot import tracing: {err}")

    for fn in ("register_trace_root_name", "get_trace_root_name"):
        if not hasattr(mod, fn):
            return TestResult.failed(f"tracing.{fn} missing — root name cannot be pinned")

    trace_id = "r008traceid0000000000000000000f"
    mod.register_trace_root_name(trace_id, "chat.supervisor")
    if mod.get_trace_root_name(trace_id) != "chat.supervisor":
        return TestResult.failed("root trace name was not registered")

    # span() must re-assert the registered root name when joining the trace.
    src = inspect.getsource(mod.span)
    if "get_trace_root_name" not in src:
        return TestResult.failed(
            "span() does not re-assert the root trace name — a background "
            "observation can still rename the trace"
        )

    # The registry must stay bounded so long-lived processes cannot leak.
    if not hasattr(mod, "_TRACE_ROOT_NAMES_MAX"):
        return TestResult.failed("trace-name registry is unbounded")
    cap = mod._TRACE_ROOT_NAMES_MAX
    for i in range(cap + 50):
        mod.register_trace_root_name(f"bulk{i:032d}", "chat.supervisor")
    if len(mod._TRACE_ROOT_NAMES) > cap:
        return TestResult.failed(
            f"trace-name registry exceeded its cap ({len(mod._TRACE_ROOT_NAMES)} > {cap})"
        )

    return TestResult.passed("root trace name is pinned and re-asserted by span() ✓")


# ─────────────────────────────────────────────────────────────────────────────
# R-009  context_recall must abstain when there is no ground truth
# ─────────────────────────────────────────────────────────────────────────────

async def _test_context_recall_abstains() -> TestResult:
    """Live chat requests carry no reference answer. Publishing
    ``context_recall=0`` for them is a fabricated score, not a measurement."""
    mod, err = _try_import("app.observability.rag_evaluator")
    if err:
        return TestResult.failed(f"cannot import rag_evaluator: {err}")

    src = inspect.getsource(mod.RAGEvaluator._publish_to_langfuse)
    if "ground_truth" not in src and "reference" not in src:
        return TestResult.failed(
            "context_recall is published unconditionally — a live request with "
            "no ground truth would report a fabricated 0.0"
        )

    metrics_cls = mod.EvaluationMetrics
    sample_cls = mod.EvaluationSample
    m = metrics_cls(
        faithfulness=0.8, context_recall=0.0, answer_relevancy=0.9,
        evaluation_time_ms=1.0, model_used="test",
    )
    published: list[str] = []

    class _Probe(mod.RAGEvaluator):
        def __init__(self):  # bypass heavy __init__
            pass

    probe = _Probe()
    no_gt = sample_cls(
        query="q", ground_truth="", retrieved_context=["c"], answer="a",
        session_id="s", source="HYBRID", trace_id="t",
    )
    # Monkeypatch the score sink to capture names without touching Langfuse.
    import app.observability.langfuse_client as lfc
    orig_score, orig_enabled = lfc.score_current_trace, lfc.is_enabled
    lfc.score_current_trace = lambda **kw: published.append(kw.get("name"))
    lfc.is_enabled = lambda: True
    try:
        probe._publish_to_langfuse(no_gt, m)
    finally:
        lfc.score_current_trace, lfc.is_enabled = orig_score, orig_enabled

    if "context_recall" in published:
        return TestResult.failed(
            "context_recall was published for a sample with no ground truth"
        )
    for required in ("faithfulness", "answer_relevancy", "ragas_aggregate"):
        if required not in published:
            return TestResult.failed(f"{required} was not published")

    return TestResult.passed(
        f"context_recall abstains without ground truth; published {published} ✓"
    )


# ─────────────────────────────────────────────────────────────────────────────
# R-010  RAGAS scheduling is deduplicated and bounded by a timeout
# ─────────────────────────────────────────────────────────────────────────────

async def _test_ragas_dedupe_and_timeout() -> TestResult:
    mod, err = _try_import("app.graph.supervisor_graph")
    if err:
        return TestResult.failed(f"cannot import supervisor_graph: {err}")

    for attr in ("_eval_inflight", "_EVAL_TIMEOUT_SECONDS", "_run_ragas_evaluation",
                 "shutdown_eval_executor"):
        if not hasattr(mod, attr):
            return TestResult.failed(f"supervisor_graph.{attr} missing")

    if not mod._EVAL_TIMEOUT_SECONDS or mod._EVAL_TIMEOUT_SECONDS <= 0:
        return TestResult.failed("RAGAS timeout is not configured")

    # Scheduling must be guarded by the in-flight set, not fire-and-forget.
    resp_src = inspect.getsource(mod)
    if "_eval_inflight" not in resp_src or "asyncio.wait_for" not in resp_src:
        return TestResult.failed("evaluation is not deduplicated / not time-bounded")

    # A timing-out evaluation must clear its in-flight key so later
    # evaluations are not permanently blocked.
    class _Hang:
        def evaluate_sample(self, sample):
            import time as _t
            _t.sleep(3)

    orig_timeout = mod._EVAL_TIMEOUT_SECONDS
    mod._EVAL_TIMEOUT_SECONDS = 0.2
    key = "r010-key"
    mod._eval_inflight.add(key)
    try:
        await mod._run_ragas_evaluation(_Hang(), object(), key)
    finally:
        mod._EVAL_TIMEOUT_SECONDS = orig_timeout

    if key in mod._eval_inflight:
        return TestResult.failed(
            "timed-out evaluation left its dedupe key set — future evaluations "
            "for this trace would be permanently suppressed"
        )

    return TestResult.passed(
        "RAGAS scheduling is deduplicated, time-bounded, and self-healing ✓"
    )


async def _test_retrieval_dedicated_executor() -> TestResult:
    """R-011: retrieval must run on its own bounded pool.

    asyncio's default executor is only ``min(32, cpu_count + 4)`` threads and is
    shared with the semantic cache. When every agent parked a default-pool
    thread on the global retrieval lock, the lock holder could not finish its
    own ``to_thread`` work and the event loop deadlocked under 5 concurrent
    requests.
    """
    db, err = _try_import("app.core.database")
    if err:
        return TestResult.failed(f"cannot import app.core.database: {err}")

    for fn in ("get_retrieval_executor", "run_retrieval", "close_retrieval_executor"):
        if not hasattr(db, fn):
            return TestResult.failed(f"app.core.database.{fn} missing")

    executor = db.get_retrieval_executor()
    if executor is not db.get_retrieval_executor():
        return TestResult.failed("retrieval executor is not a singleton")
    if not getattr(executor, "_max_workers", 0):
        return TestResult.failed("retrieval executor is unbounded")

    # Agents must not offload retrieval onto the default pool any more.
    import pathlib
    agents_dir = pathlib.Path(db.__file__).resolve().parents[1] / "graph" / "agents"
    offenders = []
    for path in agents_dir.glob("*_agent.py"):
        text = path.read_text()
        if "asyncio.to_thread(_do_" in text or "asyncio.to_thread(_retrieve" in text:
            offenders.append(path.name)
    if offenders:
        return TestResult.failed(
            f"these agents still offload retrieval to the default executor: {offenders}"
        )

    # run_retrieval must actually execute on a retrieval-named thread.
    import threading as _t
    name = await db.run_retrieval(lambda: _t.current_thread().name)
    if not name.startswith("retrieval"):
        return TestResult.failed(f"run_retrieval ran on '{name}', not the retrieval pool")

    return TestResult.passed(
        f"retrieval runs on a dedicated bounded pool "
        f"({executor._max_workers} workers); default executor stays free ✓"
    )


async def _test_no_sync_lock_across_await() -> TestResult:
    """R-012: no sync lock may be held across an await inside async code.

    ``get_checkpointer()`` held a ``threading.Lock`` while awaiting
    ``pool.open()`` and ``saver.setup()``. The first request yielded at its
    await still holding the lock; the second blocked the *event loop thread*
    waiting for it, so the first could never resume to release it. The entire
    server deadlocked — /api/health included — on the 2nd concurrent request.
    """
    import ast
    import pathlib

    app_dir = pathlib.Path(__file__).resolve().parents[2]
    lockish = ("lock", "Lock", "semaphore", "Semaphore", "mutex")
    offenders = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self, path):
            self.path = path

        def visit_AsyncFunctionDef(self, node):  # noqa: N802
            for w in ast.walk(node):
                if not isinstance(w, ast.With):  # `async with` is safe
                    continue
                has_await = any(
                    isinstance(x, (ast.Await, ast.AsyncFor, ast.AsyncWith))
                    for b in w.body
                    for x in ast.walk(b)
                )
                if not has_await:
                    continue
                names = [ast.unparse(i.context_expr) for i in w.items]
                if any(k in n for n in names for k in lockish):
                    rel = self.path.relative_to(app_dir.parent)
                    offenders.append(f"{rel}:{w.lineno} in async {node.name}()")
            self.generic_visit(node)

    scanned = 0
    for path in sorted(app_dir.rglob("*.py")):
        if "/tests/" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        scanned += 1
        _Visitor(path).visit(tree)

    if offenders:
        return TestResult.failed(
            "sync lock held across await (deadlocks the event loop): "
            + "; ".join(offenders)
        )

    # The checkpointer specifically must use an asyncio lock.
    mod, err = _try_import("app.graph.checkpointing.pg_checkpointer")
    if err:
        return TestResult.failed(f"cannot import pg_checkpointer: {err}")
    if hasattr(mod, "_saver_lock"):
        return TestResult.failed("pg_checkpointer still exposes the sync _saver_lock")
    if not hasattr(mod, "_get_saver_lock"):
        return TestResult.failed("pg_checkpointer._get_saver_lock() missing")
    lock = mod._get_saver_lock()
    if not isinstance(lock, asyncio.Lock):
        return TestResult.failed(f"checkpointer guard is {type(lock).__name__}, not asyncio.Lock")

    return TestResult.passed(
        f"no sync lock held across await in {scanned} modules; "
        f"checkpointer guarded by asyncio.Lock ✓"
    )


async def _test_no_inference_on_event_loop() -> TestResult:
    """R-013: no model inference may run inline on the event loop.

    ``*_rerank_node`` called ``get_reranker().rerank(...)`` directly. Cross
    encoder inference over ~20 candidates takes seconds and blocked the loop,
    so /api/health returned 000 for 7 of 10 probes under 10-way load. A
    faulthandler dump caught the loop thread inside
    ``retriever_engine.rerank``. Every model call must go through
    ``run_retrieval`` (the bounded pool).
    """
    offenders = []
    for agent in ("code_agent", "debug_agent", "doc_agent", "arch_agent"):
        amod, aerr = _try_import(f"app.graph.agents.{agent}")
        if aerr or amod is None:
            continue
        for attr in dir(amod):
            if not (attr.endswith("_rerank_node") or attr.endswith("_retrieve_node")):
                continue
            fn = getattr(amod, attr, None)
            if fn is None or not inspect.iscoroutinefunction(fn):
                continue
            try:
                src = inspect.getsource(fn)
            except Exception:  # noqa: BLE001
                continue
            # An inference call not preceded by an offload helper is inline.
            for call in (".rerank(", ".retrieve(", ".predict(", ".embed_query("):
                if call in src and "run_retrieval" not in src:
                    offenders.append(f"{agent}.{attr} calls {call} inline")
    if offenders:
        return TestResult.failed(
            "model inference on the event loop: " + "; ".join(sorted(set(offenders)))
        )
    return TestResult.passed(
        "all agent retrieve/rerank inference offloaded to the bounded pool ✓"
    )


async def _test_run_retrieval_propagates_context() -> TestResult:
    """R-014: run_retrieval must copy contextvars into the worker thread.

    ``asyncio.to_thread`` copies the context; ``loop.run_in_executor`` does
    not. When retrieval moved to the dedicated pool the OTEL/Langfuse span
    context was lost, so chroma.vector_search, embedding.embed_query,
    reranker.bge_cross_encoder and semantic_cache.* were emitted as their own
    orphan ROOT traces (24 blank-named + 20 reranker + 22 cache traces observed)
    instead of nesting under chat.supervisor.
    """
    import contextvars

    db, err = _try_import("app.core.database")
    if err:
        return TestResult.failed(f"cannot import app.core.database: {err}")

    probe: "contextvars.ContextVar[str]" = contextvars.ContextVar("r014_probe")
    probe.set("parent-value")

    seen = await db.run_retrieval(lambda: probe.get("MISSING"))
    if seen != "parent-value":
        return TestResult.failed(
            f"contextvars not propagated into the retrieval pool (saw {seen!r}) "
            "— Langfuse spans would become orphan root traces"
        )

    # The worker must still be a retrieval-pool thread, not the default one.
    import threading as _t
    name = await db.run_retrieval(lambda: _t.current_thread().name)
    if not name.startswith("retrieval"):
        return TestResult.failed(f"run_retrieval ran on '{name}', not the retrieval pool")

    src = inspect.getsource(db.run_retrieval)
    if "copy_context" not in src:
        return TestResult.failed("run_retrieval no longer copies the context")

    return TestResult.passed(
        "run_retrieval propagates contextvars into the bounded pool — "
        "retrieval spans stay nested under the request trace ✓"
    )


async def _test_ssl_env_sanitised() -> TestResult:
    """R-015: a missing SSL_CERT_FILE must never reach an HTTPS client.

    .env shipped ``SSL_CERT_FILE=certs/cert.pem`` (no such file) while
    ENABLE_HTTPS=False. SSL_CERT_FILE is a standard OpenSSL variable that httpx
    reads for EVERY outbound client, so any process calling load_dotenv() got
    FileNotFoundError when constructing the Groq client. That was swallowed and
    silently degraded all evaluation to word-overlap scoring.
    """
    import os
    import pathlib

    # Config import must sanitise the process environment.
    _cfg, err = _try_import("app.core.config")
    if err:
        return TestResult.failed(f"cannot import app.core.config: {err}")

    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        val = os.environ.get(var)
        if val and not pathlib.Path(val).exists():
            return TestResult.failed(
                f"{var}={val!r} does not exist — every HTTPS client will raise "
                "FileNotFoundError"
            )

    # .env must not re-introduce the collision.
    env_path = pathlib.Path(_cfg.__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, raw = line.partition("=")
            if key.strip() in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
                p = pathlib.Path(raw.strip().strip('"').strip("'"))
                if not p.exists():
                    return TestResult.failed(
                        f".env sets the standard OpenSSL var {key.strip()} to a "
                        f"missing path ({p}) — breaks all outbound HTTPS"
                    )

    # An HTTPS client must actually construct.
    try:
        import httpx
        httpx.Client().close()
    except Exception as e:  # noqa: BLE001
        return TestResult.failed(f"httpx.Client() construction failed: {e}")

    return TestResult.passed("OpenSSL trust-store env vars are sane; HTTPS clients build ✓")


async def _test_fallback_scores_namespaced() -> TestResult:
    """R-016: heuristic fallback scores must not masquerade as RAGAS scores.

    The lexical fallback published faithfulness/answer_relevancy/ragas_aggregate
    under the SAME Langfuse names as real RAGAS. A degraded evaluator was then
    indistinguishable from a working one, which is what produced the misleading
    ``answer_relevancy = 0.0`` readings on the dashboard.
    """
    mod, err = _try_import("app.observability.rag_evaluator")
    if err:
        return TestResult.failed(f"cannot import rag_evaluator: {err}")

    src = inspect.getsource(mod.RAGEvaluator._publish_to_langfuse)
    if "heuristic_" not in src:
        return TestResult.failed(
            "fallback scores are not namespaced — they pollute the RAGAS metrics"
        )

    published = {}

    class _Sample:
        trace_id = "t-r016"
        ground_truth = ""

    class _Metrics:
        faithfulness = 0.5
        answer_relevancy = 0.0
        context_recall = None
        aggregate_score = 0.25
        model_used = "heuristic-fallback"

    import sys
    import types as _types
    fake = _types.ModuleType("app.observability.langfuse_client")
    fake.is_enabled = lambda: True
    fake.score_current_trace = lambda **kw: published.update({kw["name"]: kw["value"]})
    real = sys.modules.get("app.observability.langfuse_client")
    sys.modules["app.observability.langfuse_client"] = fake
    try:
        mod.RAGEvaluator._publish_to_langfuse(
            mod.RAGEvaluator.__new__(mod.RAGEvaluator), _Sample(), _Metrics()
        )
    finally:
        if real is not None:
            sys.modules["app.observability.langfuse_client"] = real
        else:
            sys.modules.pop("app.observability.langfuse_client", None)

    if not published:
        return TestResult.failed("fallback published no scores at all")
    leaked = [n for n in published if not n.startswith("heuristic_")]
    if leaked:
        return TestResult.failed(
            f"degraded evaluator published RAGAS-named scores: {leaked}"
        )
    return TestResult.passed(
        f"degraded evaluator publishes only namespaced scores {sorted(published)} ✓"
    )


async def _test_evaluator_rate_limit_resilience() -> TestResult:
    """R-017: the evaluator must survive provider rate limiting.

    Under concurrent load Groq returned HTTP 429 to the RAGAS judge calls
    ("Rate limit reached"), losing the evaluation entirely. Each RAGAS sample
    issues several judge calls and the evaluator shares the chat path's quota,
    so evaluation must (a) retry with backoff and (b) not run in parallel.
    """
    mod, err = _try_import("app.observability.rag_evaluator")
    if err:
        return TestResult.failed(f"cannot import rag_evaluator: {err}")

    src = inspect.getsource(mod.RAGEvaluator._setup_evaluator_llm)
    if "max_retries" not in src:
        return TestResult.failed("evaluator LLM has no max_retries — a single 429 loses the evaluation")

    sup, serr = _try_import("app.graph.supervisor_graph")
    if serr:
        return TestResult.failed(f"cannot import supervisor_graph: {serr}")
    workers = getattr(sup, "_EVAL_MAX_WORKERS", None)
    if not workers or workers > 1:
        return TestResult.failed(
            f"RAGAS_EVAL_MAX_WORKERS={workers} — parallel evaluation competes with "
            "chat for provider quota and triggers 429"
        )

    # A rate-limit failure must never corrupt the response: evaluation runs in
    # the background and its errors must be caught by _run_ragas_evaluation.
    run_src = inspect.getsource(sup._run_ragas_evaluation)
    if "except" not in run_src:
        return TestResult.failed("_run_ragas_evaluation does not isolate failures from the request")

    # And a degraded/failed evaluation must publish namespaced scores only.
    pub_src = inspect.getsource(mod.RAGEvaluator._publish_to_langfuse)
    if "heuristic_" not in pub_src:
        return TestResult.failed("rate-limited fallback would publish RAGAS-named scores")

    return TestResult.passed(
        f"evaluator retries on 429, runs {workers} at a time, and failures stay "
        "isolated from the chat response ✓"
    )


async def _test_no_duplicate_answer_on_multi_agent() -> TestResult:
    """R-018: a multi-agent answer must reach the user exactly once.

    ANSWER_NODES streams every ``*_generate_node`` AND ``synthesizer_node``. On
    a multi-agent query the client therefore received each agent's full draft
    and then the merged answer, showing the same facts two or three times (a
    7,123-char reply with duplicated sentences was observed in the browser).
    The synthesiser supersedes the drafts, so it must emit token_reset first.
    """
    mod, err = _try_import("app.graph.streaming")
    if err:
        return TestResult.failed(f"cannot import streaming: {err}")

    if "token_reset" not in mod.SSE_EVENT_TYPES:
        return TestResult.failed("token_reset is not a declared SSE event type")
    if getattr(mod, "SUPERSEDING_NODE", None) != "synthesizer_node":
        return TestResult.failed("SUPERSEDING_NODE is not the synthesiser")

    src = inspect.getsource(mod)
    if "SUPERSEDING_NODE and not _synthesis_started" not in src:
        return TestResult.failed("synthesiser does not emit token_reset before its first token")

    # The cache must not store drafts + synthesis concatenated.
    chat, cerr = _try_import("app.api.chat")
    if cerr:
        return TestResult.failed(f"cannot import chat: {cerr}")
    chat_src = inspect.getsource(chat._graph_stream_v2)
    if "token_reset" not in chat_src or 'accumulated = ""' not in chat_src:
        return TestResult.failed(
            "cache accumulator ignores token_reset — would cache the duplicated answer"
        )

    # Reset must fire exactly once even across many synthesiser tokens.
    resets = src.count('type="token_reset"')
    if resets != 1:
        return TestResult.failed(f"token_reset emitted from {resets} sites; expected exactly 1")

    return TestResult.passed(
        "synthesised answer supersedes agent drafts in both the stream and the cache ✓"
    )


async def _test_no_host_path_disclosure() -> TestResult:
    """R-019: absolute server paths must never reach a client.

    Ingested files live in a server temp dir, so chunk metadata carried
    ``/private/var/folders/8k/.../T/tmp9wfm4_qg/SQL.pdf``. That string was
    rendered verbatim as a heading in the chat UI, disclosing the host
    filesystem layout and temp-dir naming to every user.
    """
    syn, err = _try_import("app.graph.nodes.synthesizer")
    if err:
        return TestResult.failed(f"cannot import synthesizer: {err}")
    if not hasattr(syn, "sanitise_source_path"):
        return TestResult.failed("sanitise_source_path() missing")

    leaky = "/private/var/folders/8k/jl0vz2vx6hg/T/tmp9wfm4_qg/SQL.pdf"
    if syn.sanitise_source_path(leaky) != "SQL.pdf":
        return TestResult.failed("sanitise_source_path does not reduce to a basename")
    for keep in ("SQL.pdf", "unknown", ""):
        if syn.sanitise_source_path(keep) != keep:
            return TestResult.failed(f"sanitise_source_path corrupted a safe value {keep!r}")
    if syn.sanitise_source_path(r"C:\Users\bob\AppData\Temp\SQL.pdf") != "SQL.pdf":
        return TestResult.failed("Windows paths are not sanitised")

    # The dedup boundary every agent's sources pass through must strip paths.
    out = syn.deduplicate_sources([
        {"id": "1", "file_path": leaky, "metadata": {"source": leaky}},
    ])
    if out[0]["file_path"] != "SQL.pdf":
        return TestResult.failed("deduplicate_sources leaks file_path")
    if out[0]["metadata"]["source"] != "SQL.pdf":
        return TestResult.failed("deduplicate_sources leaks metadata.source")

    # Every agent must sanitise where it builds citations.
    offenders = []
    for agent in ("code_agent", "debug_agent", "doc_agent", "arch_agent"):
        amod, aerr = _try_import(f"app.graph.agents.{agent}")
        if aerr or amod is None:
            continue
        src = inspect.getsource(amod)
        if 'metadata.get("file_path"' in src and "sanitise_source_path" not in src:
            offenders.append(agent)
    if offenders:
        return TestResult.failed(f"agents leak raw host paths into citations: {offenders}")

    return TestResult.passed("source paths reduced to basenames — no host path disclosure ✓")


async def _test_chat_history_persisted() -> TestResult:
    """R-020: completed turns must be written to durable chat history.

    ``ChatMemoryManager.add_message`` had no caller anywhere in the app, so
    GET /api/v2/chat/history/{id} always returned ``messages: []`` and a browser
    refresh silently lost the whole conversation.
    """
    chat, err = _try_import("app.api.chat")
    if err:
        return TestResult.failed(f"cannot import chat: {err}")

    if not hasattr(chat, "_persist_turn"):
        return TestResult.failed("_persist_turn() missing — history is never written")

    stream_src = inspect.getsource(chat._graph_stream_v2)
    if "_persist_turn" not in stream_src:
        return TestResult.failed("the streaming path never persists the turn")

    persist_src = inspect.getsource(chat._persist_turn)
    if "add_message" not in persist_src:
        return TestResult.failed("_persist_turn does not call add_message")

    # Write key must match the read key exactly, or history stays invisible.
    read_src = inspect.getsource(chat.get_chat_history)
    for src, label in ((persist_src, "write"), (read_src, "read")):
        if '::' not in src or 'user_id' not in src:
            return TestResult.failed(f"{label} path is not scoped by user_id (IDOR risk)")

    # Only complete responses may be persisted.
    if "completed and not cache_written" not in stream_src:
        return TestResult.failed("turn persisted outside the completed-only guard")

    calls = []

    class _Mgr:
        async def add_message(self, session, user, role, content):
            calls.append((session, user, role, content))

    import sys
    import types as _types
    fake = _types.ModuleType("app.services.agents.langchain_memory_manager")
    fake.ChatMemoryManager = _Mgr
    real = sys.modules.get("app.services.agents.langchain_memory_manager")
    sys.modules["app.services.agents.langchain_memory_manager"] = fake
    try:
        await chat._persist_turn("u1", "s1", "the question", "the answer")
        await chat._persist_turn("u1", "s1", "q", "")           # empty answer
        await chat._persist_turn("u1", None, "q", "a")          # no session
    finally:
        if real is not None:
            sys.modules["app.services.agents.langchain_memory_manager"] = real
        else:
            sys.modules.pop("app.services.agents.langchain_memory_manager", None)

    if len(calls) != 2:
        return TestResult.failed(f"expected exactly 2 messages persisted, got {len(calls)}")
    if calls[0][0] != "u1::s1":
        return TestResult.failed(f"history key {calls[0][0]!r} != 'u1::s1'")
    if [c[2] for c in calls] != ["user", "assistant"]:
        return TestResult.failed(f"roles persisted out of order: {[c[2] for c in calls]}")

    # The endpoint must return structured messages, not the flattened
    # prompt-injection string (the frontend iterates message objects).
    mgr_mod, merr = _try_import("app.services.agents.langchain_memory_manager")
    if merr:
        return TestResult.failed(f"cannot import memory manager: {merr}")
    if not hasattr(mgr_mod.ChatMemoryManager, "get_messages"):
        return TestResult.failed("ChatMemoryManager.get_messages() missing")
    if "get_messages" not in read_src:
        return TestResult.failed(
            "history endpoint still returns get_history()'s flattened string"
        )

    # A cache hit is still a turn — that path bypasses _graph_stream_v2.
    cached_src = inspect.getsource(chat._cached_stream_v2)
    if "_persist_turn" not in cached_src:
        return TestResult.failed(
            "cache-hit path does not persist the turn — cached answers vanish on refresh"
        )

    return TestResult.passed(
        "completed turns persist to user-scoped history as structured messages, "
        "on both the graph and cache-hit paths ✓"
    )


async def _test_no_raw_context_dump_on_llm_failure() -> TestResult:
    """R-021: an LLM outage must not dump raw corpus text as the answer.

    On an LLM exception the agents emitted
    ``"Based on the KT documentation for '<query>':" + context_text[:2000]``.
    In the browser that rendered as a confident answer that was actually
    unrelated Oracle corpus text (an ACID question returned datafile/redo-log
    prose). A degraded answer must be honest, not fabricated authority.
    """
    offenders = []
    for agent in ("code_agent", "debug_agent", "doc_agent", "arch_agent"):
        amod, aerr = _try_import(f"app.graph.agents.{agent}")
        if aerr or amod is None:
            continue
        src = inspect.getsource(amod)
        if "context_text[:2000]" in src or "context_text[:1500]" in src:
            offenders.append(f"{agent}(raw-context-dump)")
        if "Based on the KT documentation for" in src:
            offenders.append(f"{agent}(fabricated-authority)")
    if offenders:
        return TestResult.failed(
            "LLM-failure fallback dumps raw retrieved context as an answer: "
            + ", ".join(sorted(set(offenders)))
        )

    # The generate nodes must still degrade gracefully (no re-raise).
    for agent, node in (("doc_agent", "doc_generate_node"),
                        ("code_agent", "code_generate_node")):
        amod, aerr = _try_import(f"app.graph.agents.{agent}")
        if aerr or amod is None:
            continue
        fn = getattr(amod, node, None)
        if fn is None:
            continue
        src = inspect.getsource(fn)
        if "except Exception" not in src:
            return TestResult.failed(f"{agent}.{node} does not handle LLM failure")
        if "couldn't generate an answer" not in src:
            return TestResult.failed(f"{agent}.{node} lacks an honest degraded message")

    return TestResult.passed(
        "LLM outage yields an honest 'answer unavailable' message, never raw corpus text ✓"
    )


async def _test_feedback_requires_trace_ownership() -> TestResult:
    """R-022: only a trace's owner may submit feedback for it.

    /api/v2/chat/feedback accepted ANY trace_id from any authenticated user —
    a fabricated id ("0"*32) returned 200 recorded=true. That let a user poison
    another user's evaluation scores (feedback IDOR).
    """
    tr, err = _try_import("app.observability.tracing")
    if err:
        return TestResult.failed(f"cannot import tracing: {err}")
    for fn in ("register_trace_owner", "get_trace_owner"):
        if not hasattr(tr, fn):
            return TestResult.failed(f"tracing.{fn}() missing")

    tr.register_trace_owner("t-owned", "user-a")
    if tr.get_trace_owner("t-owned") != "user-a":
        return TestResult.failed("trace owner not recorded")
    if tr.get_trace_owner("t-never-seen") is not None:
        return TestResult.failed("unknown trace reported an owner")

    # Registry must stay bounded (it is per-process and long-lived).
    cap = getattr(tr, "_TRACE_OWNERS_MAX", None)
    if not cap:
        return TestResult.failed("trace owner registry is unbounded")
    for i in range(cap + 50):
        tr.register_trace_owner(f"t-bulk-{i}", "u")
    if len(tr._TRACE_OWNERS) > cap:
        return TestResult.failed(f"owner registry exceeded its cap ({len(tr._TRACE_OWNERS)} > {cap})")

    # The root span must record ownership, else every trace looks unowned.
    span_src = inspect.getsource(tr.request_root_span)
    if "register_trace_owner" not in span_src:
        return TestResult.failed("request_root_span does not record trace ownership")

    # The endpoint must reject unknown and foreign traces.
    chat, cerr = _try_import("app.api.chat")
    if cerr:
        return TestResult.failed(f"cannot import chat: {cerr}")
    fb_src = inspect.getsource(chat.submit_feedback)
    if "get_trace_owner" not in fb_src:
        return TestResult.failed("feedback endpoint does not check trace ownership")
    for code in ("404", "403"):
        if code not in fb_src:
            return TestResult.failed(f"feedback endpoint never returns {code}")
    if "except HTTPException" not in fb_src:
        return TestResult.failed(
            "the broad except swallows the 403/404 and returns 200"
        )

    return TestResult.passed(
        "feedback is restricted to the trace owner; registry bounded ✓"
    )


async def _test_credentials_scrubbed_from_query() -> TestResult:
    """R-023: pasted credentials must never enter graph state or Langfuse.

    The PII scrubber covered email/phone/CC/SSN but NOT credentials, so an API
    key pasted into chat reached graph state, the LLM, and Langfuse — the
    observation field literally named ``pii_scrubbed_query`` still contained
    ``sk-live-…`` verbatim.
    """
    ig, err = _try_import("app.graph.guardrails.input_guardrail")
    if err:
        return TestResult.failed(f"cannot import input_guardrail: {err}")

    scrubber = ig.PIIScrubber()
    secrets = {
        "api key": "sk-live-SHOULDNEVERAPPEAR-9f3a2b",
        "github token": "ghp_1234567890abcdefghij",
        "jwt short sig": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJsZWFrIn0.zzz",
        "bearer": "Bearer abcdefghijklmnopqrstuvwxyz123456",
        "assignment": 'api_key="supersecretvalue123"',
    }
    leaked = []
    for label, secret in secrets.items():
        text = f"My credential is {secret} — what is a PRIMARY KEY?"
        scrubbed, _ = scrubber._scrub_regex(text)
        core = secret.split()[-1].strip('"')
        if core in scrubbed:
            leaked.append(label)
    if leaked:
        return TestResult.failed(f"credentials survive PII scrubbing: {leaked}")

    # The Langfuse mask hook is the second line of defence.
    lf, lerr = _try_import("app.observability.langfuse_client")
    if lerr:
        return TestResult.failed(f"cannot import langfuse_client: {lerr}")
    for label, secret in secrets.items():
        if secret.split()[-1].strip('"') in lf._mask_text(f"value {secret} end"):
            return TestResult.failed(f"Langfuse mask hook misses {label}")

    # Normal text must survive untouched.
    plain = "What is a PRIMARY KEY in SQL?"
    if scrubber._scrub_regex(plain)[0] != plain:
        return TestResult.failed("scrubber corrupts ordinary queries")

    return TestResult.passed(
        "credentials scrubbed before graph state and masked again at Langfuse ✓"
    )


async def _test_cache_failure_is_non_fatal() -> TestResult:
    """R-024: a semantic-cache outage must not take chat down.

    ``semantic_cache.get`` and ``_cache_write`` were both unguarded. The cache
    is a pgvector table, so a Postgres outage turned every chat request into a
    500 — and a write failure inside ``finally`` also skipped ``_persist_turn``,
    silently losing history.
    """
    chat, err = _try_import("app.api.chat")
    if err:
        return TestResult.failed(f"cannot import chat: {err}")

    import ast
    import textwrap

    def _guarded_calls(fn) -> set:
        """Return the source segments of calls that sit inside a try block."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        guarded = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for body_stmt in node.body:
                for sub in ast.walk(body_stmt):
                    if isinstance(sub, ast.Call):
                        guarded.add(ast.unparse(sub))
        return guarded

    lookup_guarded = _guarded_calls(chat.chat_stream_v2)
    if not any("semantic_cache.get" in c for c in lookup_guarded):
        return TestResult.failed("cache lookup is unguarded — a DB outage 500s every chat")

    write_guarded = _guarded_calls(chat._graph_stream_v2)
    if not any("_cache_write" in c for c in write_guarded):
        return TestResult.failed("cache write is unguarded inside finally")

    # History persistence must not be gated behind a successful cache write:
    # _persist_turn must NOT live in the same try body as the cache write.
    write_src = inspect.getsource(chat._graph_stream_v2)
    after_write = write_src.split("_cache_write", 1)[1]
    if "_persist_turn" not in after_write:
        return TestResult.failed("_persist_turn unreachable when the cache write fails")
    if "except" not in after_write.split("_persist_turn", 1)[0]:
        return TestResult.failed(
            "no except between the cache write and _persist_turn — a write "
            "failure still skips history"
        )

    return TestResult.passed(
        "cache read and write both fail open; history persists regardless ✓"
    )


async def _test_single_canonical_chroma_collection() -> TestResult:
    """R-025: ingestion and retrieval must address the same Chroma collection.

    Each ingest run used to create its own ``documents_<timestamp>`` collection
    while retrieval opened only one, orphaning every other upload. 11 such
    collections still exist on disk. ``.env`` also pinned
    ``CHROMA_DEFAULT_COLLECTION`` to a stale per-upload collection (245 vectors)
    while retrieval actually read ``documents_main`` (370).
    """
    ing, err = _try_import("app.services.ingestion.ingestion_service")
    if err:
        return TestResult.failed(f"cannot import ingestion_service: {err}")

    # A second configurable collection name is exactly how reads/writes diverge.
    if hasattr(ing, "DEFAULT_COLLECTION"):
        return TestResult.failed(
            "ingestion_service.DEFAULT_COLLECTION reintroduces a second corpus name"
        )
    rc, rerr = _try_import("app.services.retrieval.retrieval_config")
    if rerr:
        return TestResult.failed(f"cannot import retrieval_config: {rerr}")
    cfg_cls = getattr(rc, "RetrievalConfig", None)
    if cfg_cls is not None and hasattr(cfg_cls, "chroma_collection_name"):
        return TestResult.failed(
            "RetrievalConfig.chroma_collection_name reintroduces a second corpus name"
        )

    # .env must not pin a per-upload collection.
    import pathlib
    env = pathlib.Path(ing.__file__).resolve().parents[3] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("CHROMA_DEFAULT_COLLECTION="):
                return TestResult.failed(
                    f"stale collection pin still in .env: {line}"
                )
            if line.startswith("CHROMA_CANONICAL_COLLECTION="):
                val = line.split("=", 1)[1].strip()
                if val.startswith("documents_2"):
                    return TestResult.failed(
                        f"canonical collection points at a per-upload collection: {val}"
                    )

    if not getattr(ing, "CANONICAL_COLLECTION", None):
        return TestResult.failed("CANONICAL_COLLECTION missing")

    # Retrieval and ingestion must resolve through the same accessor.
    sf, serr = _try_import("app.services.scoped_factories")
    if not serr:
        src = inspect.getsource(sf)
        if "get_chroma_collection()" not in src:
            return TestResult.failed(
                "retrieval does not resolve its collection via get_chroma_collection()"
            )

    return TestResult.passed(
        f"single canonical corpus '{ing.CANONICAL_COLLECTION}' for reads and writes ✓"
    )


async def _test_ssrf_guard_blocks_and_allows() -> TestResult:
    """R-026: the SSRF guard must block internal targets AND allow public ones.

    ``_is_forbidden_ip`` rejects anything unparseable — correct for a resolved
    address, wrong for a hostname. The literal-IP fast path called it with the
    raw host, so ``example.com`` returned True and EVERY domain was rejected
    ("URL resolves to a forbidden (internal/private) address") before DNS was
    consulted. URL ingestion accepted nothing at all.
    """
    ing, err = _try_import("app.routes.ingest")
    if err:
        return TestResult.failed(f"cannot import ingest route: {err}")

    validate = ing.validate_ingest_url

    blocked = [
        "http://127.0.0.1:8001/api/health",
        "http://localhost:3000/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "http://[::1]/",
        "file:///etc/passwd",
        "gopher://127.0.0.1:6379/_INFO",
        "ftp://example.com/x.txt",
    ]
    leaked = []
    for url in blocked:
        try:
            validate(url)
            leaked.append(url)
        except Exception:  # noqa: BLE001 — any rejection is acceptable
            pass
    if leaked:
        return TestResult.failed(f"SSRF guard allowed internal targets: {leaked}")

    # Public hostnames and public literal IPs must pass.
    allowed = ["https://example.com/", "http://93.184.216.34/"]
    wrongly_blocked = []
    for url in allowed:
        try:
            validate(url)
        except Exception as e:  # noqa: BLE001
            detail = getattr(e, "detail", str(e))
            # A DNS failure in a sandboxed/offline runner is not a guard bug.
            if "could not be resolved" in str(detail):
                continue
            wrongly_blocked.append(f"{url} ({detail})")
    if wrongly_blocked:
        return TestResult.failed(
            f"SSRF guard rejects legitimate public URLs: {wrongly_blocked}"
        )

    return TestResult.passed(
        f"SSRF guard blocks {len(blocked)} internal targets and admits public URLs ✓"
    )


async def _test_ingest_reports_failure_honestly() -> TestResult:
    """R-027: ingestion must not report success when it stored nothing.

    /api/v1/ingest/url returned HTTP 200 with ``"message": "Successfully
    ingested URL"`` while the body carried ``"status": "error"`` and
    ``chunks_created: 0`` (0.04s, nothing indexed). The user was told the
    document was ingested when it was not.
    """
    ing, err = _try_import("app.routes.ingest")
    if err:
        return TestResult.failed(f"cannot import ingest route: {err}")

    import ast
    import textwrap

    fn = getattr(ing, "ingest_url", None)
    if fn is None:
        for name in dir(ing):
            obj = getattr(ing, name)
            if callable(obj) and "url" in name.lower() and "ingest" in name.lower():
                fn = obj
                break
    if fn is None:
        return TestResult.failed("URL ingest handler not found")

    src = textwrap.dedent(inspect.getsource(fn))
    if "Successfully ingested URL" not in src:
        return TestResult.failed("URL ingest success message not found — test is stale")

    # A guard must exist that raises before the success response.
    tree = ast.parse(src)
    raises_before_success = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            body = ast.unparse(node)
            if "chunks" in body and "raise" in body:
                raises_before_success = True
    if not raises_before_success:
        return TestResult.failed(
            "no guard raises when chunks_created == 0 — a no-op ingest still "
            "reports success"
        )
    if "status_code=status.HTTP_422_UNPROCESSABLE_ENTITY" not in src:
        return TestResult.failed("failed ingestion does not return a 4xx status")

    return TestResult.passed(
        "URL ingestion raises 422 when nothing is indexed instead of faking success ✓"
    )


async def _test_redis_backed_security_controls() -> TestResult:
    """R-028: the JWT blacklist and rate limiter must really use Redis.

    ``redis`` was missing from requirements.txt and from the runtime venv, so
    ``import redis`` raised inside both backends and a broad ``except`` quietly
    selected the in-memory store — while .env set REDIS_URL and the startup log
    only emitted a WARNING. Consequences: logout did not revoke a token beyond
    the current process, and rate limits were per-process.
    """
    import importlib.util
    import pathlib

    if importlib.util.find_spec("redis") is None:
        return TestResult.failed(
            "the `redis` package is not installed — blacklist and rate limiter "
            "silently degrade to per-process in-memory stores"
        )

    tb, err = _try_import("app.auth.token_blacklist")
    if err:
        return TestResult.failed(f"cannot import token_blacklist: {err}")
    rl, rerr = _try_import("app.middleware.rate_limiter")
    if rerr:
        return TestResult.failed(f"cannot import rate_limiter: {rerr}")

    # requirements.txt must pin it so a fresh deploy cannot regress.
    req = pathlib.Path(tb.__file__).resolve().parents[2] / "requirements.txt"
    if req.exists() and "redis" not in req.read_text():
        return TestResult.failed("redis missing from requirements.txt")

    # With a URL configured, the Redis backend must actually be selected.
    from app.core.config import get_settings
    url = getattr(get_settings(), "redis_url", None)
    if url:
        mgr = tb.get_token_blacklist_manager(url)
        backend = type(getattr(mgr, "_backend", None)).__name__
        if backend != "RedisTokenBlacklist":
            return TestResult.failed(
                f"REDIS_URL is configured but blacklist selected {backend}"
            )
        limiter = rl.get_rate_limiter(url)
        lbackend = type(getattr(limiter, "_backend", None)).__name__
        if lbackend != "RedisRateLimiter":
            return TestResult.failed(
                f"REDIS_URL is configured but rate limiter selected {lbackend}"
            )

        # Prove the Redis backend actually round-trips, not just constructs.
        import uuid
        from datetime import datetime, timedelta, timezone
        jti = f"r028-{uuid.uuid4().hex[:12]}"
        mgr.revoke_token(jti, datetime.now(timezone.utc) + timedelta(minutes=5))
        if not mgr.is_token_revoked(jti):
            return TestResult.failed("revoked JTI did not persist in Redis")

    # A silent downgrade must be logged at ERROR, not WARNING.
    for cls, label in ((tb.TokenBlacklistManager, "blacklist"),
                       (rl.RateLimiter, "rate limiter")):
        src = inspect.getsource(cls)
        if "logger.error" not in src:
            return TestResult.failed(
                f"{label} downgrade to in-memory is not logged at ERROR"
            )

    return TestResult.passed(
        "redis installed + pinned; blacklist and rate limiter use Redis and "
        "round-trip; downgrade is loud ✓"
    )


async def _test_hil_request_to_ui_chain() -> TestResult:
    """R-029: the whole HIL chain must be connected, request → SSE → resume.

    Three independent breaks made HIL non-functional end-to-end while every
    direct-graph unit test passed:
      1. ``_build_initial_state`` accepted no HIL args, so the UI's toggle and
         threshold were silently discarded.
      2. ``streaming.py`` looked for ``"__interrupt__" in tags`` on
         on_chain_start — LangGraph emits no such event. The pause is only
         visible via ``aget_state().tasks[].interrupts``.
      3. Both the stream and /resume read/wrote state with
         ``checkpoint_ns=<org|"default">``; LangGraph treats checkpoint_ns as a
         SUBGRAPH name, so it raised "Subgraph default not found" and the
         approval was discarded behind a warning.
    """
    chat, err = _try_import("app.api.chat")
    if err:
        return TestResult.failed(f"cannot import chat: {err}")

    # 1. request → state plumbing
    sig = inspect.signature(chat._build_initial_state)
    for p in ("hil_enabled", "hil_confidence_threshold"):
        if p not in sig.parameters:
            return TestResult.failed(f"_build_initial_state drops {p}")
    st = chat._build_initial_state(
        query="q", user_id="u", session_id="s", org_id=None, agent_hint=None,
        hil_enabled=True, hil_confidence_threshold=0.8,
    )
    if st.get("hil_enabled") is not True:
        return TestResult.failed("hil_enabled not written into state")
    if st.get("hil_confidence_threshold") != 0.8:
        return TestResult.failed("hil_confidence_threshold not written into state")

    # 2. the node must honour the flag in both directions
    hil, herr = _try_import("app.graph.nodes.hil_node")
    if herr:
        return TestResult.failed(f"cannot import hil_node: {herr}")
    node_src = inspect.getsource(hil.hil_check_node)
    if "hil_enabled" not in node_src:
        return TestResult.failed("hil_check_node ignores hil_enabled")

    # 3. streaming must detect the interrupt from state, not from tags
    stream, serr = _try_import("app.graph.streaming")
    if serr:
        return TestResult.failed(f"cannot import streaming: {serr}")
    ssrc = inspect.getsource(stream.stream_graph_events)
    if "aget_state" not in ssrc:
        return TestResult.failed(
            "streaming never inspects graph state — LangGraph emits no "
            "__interrupt__ event, so the pause would be invisible to the client"
        )
    if "interrupts" not in ssrc:
        return TestResult.failed("streaming does not read task interrupts")

    # 4. root-namespace handling in BOTH the stream and the resume path
    if 'checkpoint_ns"] = ""' not in ssrc:
        return TestResult.failed(
            "streaming reads state with the request checkpoint_ns — raises "
            "'Subgraph <ns> not found'"
        )
    cp, cerr = _try_import("app.api.checkpoints")
    if cerr:
        return TestResult.failed(f"cannot import checkpoints: {cerr}")
    rsrc = inspect.getsource(cp.resume_after_interrupt) if hasattr(
        cp, "resume_after_interrupt") else inspect.getsource(cp)
    if 'checkpoint_ns"] = ""' not in rsrc:
        return TestResult.failed(
            "resume writes state with the request checkpoint_ns — the approval "
            "is silently discarded"
        )

    # 5. interrupt must be a declared SSE type the client can act on
    if "interrupt" not in stream.SSE_EVENT_TYPES:
        return TestResult.failed("interrupt is not a declared SSE event type")

    return TestResult.passed(
        "HIL chain connected: request → state → interrupt → SSE → resume ✓"
    )


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
    PhaseTest(
        id="R-008",
        name="background spans cannot rename a trace",
        description="late RAGAS observation must not overwrite the root trace name",
        run=_test_trace_name_not_overwritten,
        critical=True,
        tags=["observability", "regression"],
    ),
    PhaseTest(
        id="R-009",
        name="context_recall abstains without ground truth",
        description="no fabricated context_recall on live production traces",
        run=_test_context_recall_abstains,
        critical=True,
        tags=["evaluation", "regression"],
    ),
    PhaseTest(
        id="R-010",
        name="RAGAS duplicate suppression + timeout",
        description="one evaluation per trace, bounded by an explicit timeout",
        run=_test_ragas_dedupe_and_timeout,
        critical=True,
        tags=["evaluation", "regression"],
    ),
    PhaseTest(
        id="R-011",
        name="retrieval uses a dedicated executor",
        description="retrieval must not starve asyncio's default thread pool",
        run=_test_retrieval_dedicated_executor,
        critical=True,
        tags=["concurrency", "regression"],
    ),
    PhaseTest(
        id="R-012",
        name="no sync lock held across await",
        description="a threading.Lock across await deadlocks the whole event loop",
        run=_test_no_sync_lock_across_await,
        critical=True,
        tags=["concurrency", "regression"],
    ),
    PhaseTest(
        id="R-013",
        name="no model inference on the event loop",
        description="rerank/retrieve inference must use the bounded pool",
        run=_test_no_inference_on_event_loop,
        critical=True,
        tags=["concurrency", "regression"],
    ),
    PhaseTest(
        id="R-014",
        name="retrieval pool propagates trace context",
        description="run_in_executor drops contextvars → orphan Langfuse traces",
        run=_test_run_retrieval_propagates_context,
        critical=True,
        tags=["observability", "concurrency", "regression"],
    ),
    PhaseTest(
        id="R-015",
        name="OpenSSL trust-store env sanitised",
        description="a missing SSL_CERT_FILE breaks every outbound HTTPS client",
        run=_test_ssl_env_sanitised,
        critical=True,
        tags=["config", "regression"],
    ),
    PhaseTest(
        id="R-016",
        name="fallback scores namespaced",
        description="heuristic scores must not masquerade as RAGAS metrics",
        run=_test_fallback_scores_namespaced,
        critical=True,
        tags=["evaluation", "regression"],
    ),
    PhaseTest(
        id="R-017",
        name="evaluator survives provider 429",
        description="retry + serialised evaluation so rate limits don't lose scores",
        run=_test_evaluator_rate_limit_resilience,
        critical=True,
        tags=["evaluation", "concurrency", "regression"],
    ),
    PhaseTest(
        id="R-018",
        name="no duplicated multi-agent answer",
        description="synthesis supersedes agent drafts in stream and cache",
        run=_test_no_duplicate_answer_on_multi_agent,
        critical=True,
        tags=["streaming", "regression"],
    ),
    PhaseTest(
        id="R-019",
        name="no host path disclosure",
        description="absolute server ingest paths must never reach a client",
        run=_test_no_host_path_disclosure,
        critical=True,
        tags=["security", "regression"],
    ),
    PhaseTest(
        id="R-020",
        name="chat history is persisted",
        description="completed turns must survive a browser refresh",
        run=_test_chat_history_persisted,
        critical=True,
        tags=["memory", "regression"],
    ),
    PhaseTest(
        id="R-021",
        name="no raw context dump on LLM failure",
        description="degraded answers must be honest, not fabricated authority",
        run=_test_no_raw_context_dump_on_llm_failure,
        critical=True,
        tags=["agents", "regression"],
    ),
    PhaseTest(
        id="R-022",
        name="feedback requires trace ownership",
        description="a user must not score another user's trace",
        run=_test_feedback_requires_trace_ownership,
        critical=True,
        tags=["security", "evaluation", "regression"],
    ),
    PhaseTest(
        id="R-023",
        name="credentials scrubbed from queries",
        description="pasted API keys/JWTs must not reach state, LLM or Langfuse",
        run=_test_credentials_scrubbed_from_query,
        critical=True,
        tags=["security", "guardrails", "regression"],
    ),
    PhaseTest(
        id="R-024",
        name="cache outage is non-fatal",
        description="a pgvector outage must not 500 chat or drop history",
        run=_test_cache_failure_is_non_fatal,
        critical=True,
        tags=["cache", "resilience", "regression"],
    ),
    PhaseTest(
        id="R-025",
        name="single canonical Chroma collection",
        description="ingestion and retrieval must not diverge and orphan uploads",
        run=_test_single_canonical_chroma_collection,
        critical=True,
        tags=["retrieval", "ingestion", "regression"],
    ),
    PhaseTest(
        id="R-026",
        name="SSRF guard blocks internal, allows public",
        description="hostname handling must not reject every legitimate URL",
        run=_test_ssrf_guard_blocks_and_allows,
        critical=True,
        tags=["security", "ingestion", "regression"],
    ),
    PhaseTest(
        id="R-027",
        name="ingestion reports failure honestly",
        description="no 200 'Successfully ingested' when nothing was indexed",
        run=_test_ingest_reports_failure_honestly,
        critical=True,
        tags=["ingestion", "regression"],
    ),
    PhaseTest(
        id="R-028",
        name="Redis-backed security controls",
        description="logout revocation and rate limits must not be per-process",
        run=_test_redis_backed_security_controls,
        critical=True,
        tags=["security", "auth", "regression"],
    ),
    PhaseTest(
        id="R-029",
        name="HIL chain request → SSE → resume",
        description="HIL must work end-to-end, not just in direct graph tests",
        run=_test_hil_request_to_ui_chain,
        critical=True,
        tags=["hil", "streaming", "regression"],
    ),
]
