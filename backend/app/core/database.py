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

    # 1. Prefer individual vars — encode password to handle special chars
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    db = os.getenv("POSTGRES_DB")
    if host and user and password and db:
        return f"postgresql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{db}"

    # 2. Explicit DSN override
    dsn = os.getenv("POSTGRES_DSN")
    if dsn:
        return dsn

    # 3. DATABASE_URL — strip SQLAlchemy driver prefixes
    url = os.getenv("DATABASE_URL")
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
