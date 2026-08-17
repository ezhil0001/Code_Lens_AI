"""Shared infrastructure for low-latency Postgres + embeddings access.

Centralizes two singletons that were previously re-created on every request:

1. `get_pg_pool()`  — `psycopg_pool.ConnectionPool` for raw psycopg work
   (semantic cache, RAG evaluator, pgvector queries).
2. `get_embedder()` — A single `HuggingFaceEmbeddings` instance reused
   across the app. The model load (~500 MB, ~3-5 s warmup) was previously
   paid PER REQUEST inside `SemanticCache`.

Why connection pooling matters here:
    Without a pool, every cache `get`/`set` paid ~10-30 ms on TCP+auth
    handshake. The <20 ms semantic-cache target is unreachable without
    pooled connections. With `psycopg_pool` the handshake is amortized
    across all callers.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# DSN helpers                                                                 #
# --------------------------------------------------------------------------- #
def build_psycopg_dsn() -> str:
    """Return a psycopg-compatible DSN.

    Priority order:
    1. Individual POSTGRES_* vars — special chars in passwords are
       automatically percent-encoded so '@', '#', '!' etc. work correctly.
    2. POSTGRES_DSN explicit override (caller is responsible for encoding).
    3. DATABASE_URL fallback — strips SQLAlchemy prefixes.
    """
    from urllib.parse import quote_plus

    # Pydantic Settings loads values from `.env` into the Settings object, NOT
    # into os.environ. Relying on os.getenv() alone therefore silently falls
    # through to defaults (postgres:postgres) whenever the app is launched
    # without those vars exported into the shell — which disables the semantic
    # cache with an auth failure. Fall back to Settings so the DSN always
    # matches the SQLAlchemy engine's credentials.
    def _cfg(env_key: str, attr: str, default: Optional[str] = None) -> Optional[str]:
        val = os.getenv(env_key)
        if val:
            return val
        try:
            from app.core.config import get_settings

            return getattr(get_settings(), attr, default)
        except Exception:  # noqa: BLE001
            return default

    # 1. Prefer individual vars — encode password to handle special chars
    host = _cfg("POSTGRES_HOST", "postgres_host")
    port = str(_cfg("POSTGRES_PORT", "postgres_port", "5432"))
    user = _cfg("POSTGRES_USER", "postgres_user")
    password = _cfg("POSTGRES_PASSWORD", "postgres_password")
    db = _cfg("POSTGRES_DB", "postgres_db")
    if host and user and password and db:
        return f"postgresql://{quote_plus(str(user))}:{quote_plus(str(password))}@{host}:{port}/{db}"

    # 2. Explicit DSN override
    dsn = os.getenv("POSTGRES_DSN")
    if dsn:
        return dsn

    # 3. DATABASE_URL — strip SQLAlchemy driver prefixes
    url = os.getenv("DATABASE_URL") or _cfg("DATABASE_URL", "database_url")
    if url:
        return (
            url.replace("postgresql+psycopg2://", "postgresql://")
               .replace("postgresql+psycopg://", "postgresql://")
        )

    return "postgresql://postgres:postgres@localhost:5432/codelens_ai"


# --------------------------------------------------------------------------- #
# Connection pool singleton                                                   #
# --------------------------------------------------------------------------- #
_pool_lock = threading.Lock()
_pool = None  # type: ignore


# --------------------------------------------------------------------------- #
# Process-wide retrieval lock                                                 #
# --------------------------------------------------------------------------- #
# Sentence-transformer inference on Apple MPS (and, more generally, a single
# shared embedding/reranker model) is NOT thread-safe. When the supervisor
# dispatches multiple agents in parallel (e.g. CodeAgent + DebugAgent via
# Send()), each agent's retrieve node runs in its own worker thread and would
# otherwise call the same model concurrently — which deadlocks the MPS backend
# and hangs the whole request. Every agent MUST serialise its retrieval through
# this single, process-wide lock. Previously each node fell back to a throwaway
# ``threading.Lock()`` when the retriever lacked a lock attribute, providing no
# mutual exclusion at all.
_RETRIEVAL_LOCK = threading.RLock()


def get_retrieval_lock() -> "threading.RLock":
    """Return the shared, process-wide lock guarding model-backed retrieval.

    DEPRECATED as a coarse pipeline guard. It used to wrap the *entire*
    ``retriever.retrieve()`` call — query expansion (a network LLM call),
    Chroma/BM25 search, reranking and parent-context assembly — which
    serialised every independent request behind one mutex for ~20s.

    Thread-safety is now enforced at the model boundary instead
    (:func:`get_embedding_lock` / :func:`get_reranker_lock`), which is both
    narrower and stronger: it protects every caller of the shared models, not
    just the agent nodes. Retained for callers that genuinely need whole-
    pipeline exclusion.
    """
    return _RETRIEVAL_LOCK


# Fine-grained model-inference guards. The embedder and the cross-encoder are
# process-wide singletons, so concurrent forward passes must be serialised
# (concurrent SentenceTransformer inference on Apple MPS is known to deadlock).
# These are held for milliseconds around the inference call only, so
# independent requests overlap on everything else: query expansion, Chroma and
# BM25 search, parent-context assembly and LLM generation.
_EMBED_LOCK = threading.RLock()
_RERANK_LOCK = threading.RLock()


def get_embedding_lock() -> "threading.RLock":
    """Serialise forward passes through the shared embedding model."""
    return _EMBED_LOCK


def get_reranker_lock() -> "threading.RLock":
    """Serialise forward passes through the shared cross-encoder reranker."""
    return _RERANK_LOCK


# Retrieval must NOT run on asyncio's default executor. That pool is only
# ``min(32, cpu_count + 4)`` threads (12 here) and is also used by the semantic
# cache and other ``asyncio.to_thread`` callers. Under concurrency every pool
# thread ends up parked on _RETRIEVAL_LOCK, so the request holding the lock can
# never complete its own to_thread work — a thread-pool starvation deadlock that
# wedged the whole event loop. A dedicated pool bounds retrieval and keeps the
# default executor free.
_RETRIEVAL_EXECUTOR: Optional[ThreadPoolExecutor] = None
_RETRIEVAL_EXECUTOR_LOCK = threading.Lock()
# Retrieval is CPU-bound (embedding + cross-encoder forward passes) and each
# worker fans out further across torch's intra-op threads. Left unbounded that
# oversubscribes the CPU and starves the event loop of GIL time, so /api/health
# stalls for >10s under load. Keep workers x torch-threads under the core count
# (see configure_inference_threads).
RETRIEVAL_MAX_WORKERS = int(os.getenv("RETRIEVAL_MAX_WORKERS", "2"))


def get_retrieval_executor() -> ThreadPoolExecutor:
    """Return the dedicated pool used for blocking retrieval work."""
    global _RETRIEVAL_EXECUTOR
    if _RETRIEVAL_EXECUTOR is None:
        with _RETRIEVAL_EXECUTOR_LOCK:
            if _RETRIEVAL_EXECUTOR is None:
                _RETRIEVAL_EXECUTOR = ThreadPoolExecutor(
                    max_workers=RETRIEVAL_MAX_WORKERS,
                    thread_name_prefix="retrieval",
                )
    return _RETRIEVAL_EXECUTOR


async def run_retrieval(fn, *args):
    """Run blocking retrieval on the dedicated pool, never the default one.

    The current context is copied into the worker exactly like
    ``asyncio.to_thread`` does. ``loop.run_in_executor`` does NOT propagate
    contextvars, so without this the OpenTelemetry/Langfuse span context is
    lost in the worker and every retrieval, rerank and cache span is emitted as
    its own orphan root trace instead of nesting under chat.supervisor.
    """
    import asyncio
    import contextvars

    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    return await loop.run_in_executor(
        get_retrieval_executor(), lambda: ctx.run(fn, *args)
    )


def close_retrieval_executor(wait: bool = False) -> None:
    """Release retrieval workers at process shutdown."""
    global _RETRIEVAL_EXECUTOR
    if _RETRIEVAL_EXECUTOR is not None:
        _RETRIEVAL_EXECUTOR.shutdown(wait=wait, cancel_futures=not wait)
        _RETRIEVAL_EXECUTOR = None


def get_pg_pool():
    """Return the process-wide `psycopg_pool.ConnectionPool` singleton.

    Pool sizing is conservative-by-default; tune via env:
      - `PG_POOL_MIN_SIZE` (default 2)
      - `PG_POOL_MAX_SIZE` (default 10)
      - `PG_POOL_TIMEOUT`  (default 10s)
    """
    global _pool
    if _pool is not None:
        return _pool

    with _pool_lock:
        if _pool is not None:
            return _pool

        try:
            from psycopg_pool import ConnectionPool  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "psycopg_pool is required. Install with `pip install psycopg-pool`."
            ) from e

        dsn = build_psycopg_dsn()
        min_size = int(os.getenv("PG_POOL_MIN_SIZE", "2"))
        max_size = int(os.getenv("PG_POOL_MAX_SIZE", "10"))
        timeout = float(os.getenv("PG_POOL_TIMEOUT", "10"))

        _pool = ConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            timeout=timeout,
            kwargs={"autocommit": False},
        )
        logger.info(
            f"✅ psycopg ConnectionPool ready (min={min_size}, max={max_size})"
        )
        return _pool


@contextmanager
def pg_connection(register_pgvector: bool = False) -> Iterator:
    """Context manager that lends a pooled connection.

    Usage:
        with pg_connection(register_pgvector=True) as conn:
            with conn.cursor() as cur:
                cur.execute(...)
            conn.commit()
    """
    pool = get_pg_pool()
    with pool.connection() as conn:
        if register_pgvector:
            try:
                from pgvector.psycopg import register_vector  # type: ignore
                register_vector(conn)
            except Exception as e:
                logger.debug(f"register_vector skipped: {e}")
        yield conn


def close_pg_pool() -> None:
    """Dispose the pool (call on application shutdown)."""
    global _pool
    if _pool is not None:
        try:
            _pool.close()
        finally:
            _pool = None
            logger.info("psycopg ConnectionPool closed")


# --------------------------------------------------------------------------- #
# Embedding model singleton                                                   #
# --------------------------------------------------------------------------- #
_embedder_lock = threading.Lock()
_embedder = None  # type: ignore
_DEFAULT_EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2"
)
_EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))


def get_embedder():
    """Return the process-wide `HuggingFaceEmbeddings` singleton.

    The previous `SemanticCache` instantiated a fresh embedder on construction
    — fine — but had no shared handle, so each component (cache, evaluator,
    example selector) was loading the model independently. This consolidates.
    """
    global _embedder
    if _embedder is not None:
        return _embedder

    with _embedder_lock:
        if _embedder is not None:
            return _embedder

        try:
            from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "langchain_huggingface is required. Install with "
                "`pip install langchain-huggingface`."
            ) from e

        _embedder = HuggingFaceEmbeddings(model_name=_DEFAULT_EMBED_MODEL)
        logger.info(f"✅ Singleton embedder loaded: {_DEFAULT_EMBED_MODEL}")
        return _embedder


def get_embed_dim() -> int:
    return _EMBED_DIM


def get_embed_model_name() -> str:
    return _DEFAULT_EMBED_MODEL


# --------------------------------------------------------------------------- #
# Code-specialized embedding model singleton                                  #
# --------------------------------------------------------------------------- #
_code_embedder_lock = threading.Lock()
_code_embedder = None  # type: ignore

# Default: a 768-dim model fine-tuned on code search.
# Must be the same dimension as the general embedder (768) so existing
# ChromaDB collections and pgvector tables are compatible.
# Override via env var: CODE_EMBED_MODEL
_DEFAULT_CODE_EMBED_MODEL = os.getenv(
    "CODE_EMBED_MODEL",
    "flax-sentence-embeddings/st-codesearch-distilroberta-base",
)
_CODE_EMBED_DIM = int(os.getenv("CODE_EMBED_DIM", "768"))


def get_code_embedder():
    """Return the process-wide code-specialized ``HuggingFaceEmbeddings`` singleton.

    Uses ``flax-sentence-embeddings/st-codesearch-distilroberta-base`` by
    default — a 768-dim model trained on CodeSearchNet that significantly
    out-ranks general-purpose text models on code lookup tasks.

    Override via ``CODE_EMBED_MODEL`` env var (must emit 768-dim vectors to
    stay compatible with existing ChromaDB collections; adjust ``CODE_EMBED_DIM``
    if you switch to a different-dim model and create a separate collection).

    Only ``code_retrieve_node`` and the code-chunk embedding step in the
    ingestion pipeline use this embedder.  Doc retrieval, semantic cache,
    and long-term memory all keep ``get_embedder()``.
    """
    global _code_embedder
    if _code_embedder is not None:
        return _code_embedder

    with _code_embedder_lock:
        if _code_embedder is not None:
            return _code_embedder

        try:
            from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "langchain_huggingface is required. Install with "
                "`pip install langchain-huggingface`."
            ) from e

        _code_embedder = HuggingFaceEmbeddings(model_name=_DEFAULT_CODE_EMBED_MODEL)
        logger.info(f"✅ Code-specialized embedder loaded: {_DEFAULT_CODE_EMBED_MODEL}")
        return _code_embedder


def get_code_embed_dim() -> int:
    """Return the vector dimension of the code-specialized embedder."""
    return _CODE_EMBED_DIM


def get_code_embed_model_name() -> str:
    """Return the model name used by the code-specialized embedder."""
    return _DEFAULT_CODE_EMBED_MODEL
