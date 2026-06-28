"""
Semantic cache — pgvector-backed query deduplication.

Extracted from app.api.chat so both the v1 compat endpoint and the v2
LangGraph streaming endpoint can share the same cache instance.

Usage::

    from app.services.semantic_cache import semantic_cache
    hit = semantic_cache.get(query, user_id=user_id)

Architecture notes:

  Embedder
    The shared ``get_embedder()`` singleton from ``app.core.database`` is used
    so the 500 MB model is loaded once at startup rather than once per cache
    constructor call.

  Connection pooling
    Pooled psycopg connections via ``pg_connection()`` amortize the TCP+auth
    handshake (~10–30 ms each).  With a pool the <20 ms GET target is
    achievable.

  Multi-tenant safety
    All lookups and writes include a ``WHERE user_id = %s`` clause so
    cross-user cache hits are impossible by construction.

  TTL
    Entries older than ``DEFAULT_TTL_SECONDS`` (24 h) are excluded from
    lookup without requiring a DELETE sweep.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SemanticCache:
    """pgvector semantic cache scoped to *user_id*."""

    DEFAULT_TTL_SECONDS = 86_400  # 24 hours

    def __init__(self, similarity_threshold: float = 0.95) -> None:
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = self.DEFAULT_TTL_SECONDS
        self._available = False
        try:
            self._init_backend()
            self._available = True
            logger.info(
                "✅ SemanticCache (pgvector + pool) ready — threshold=%.2f",
                similarity_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "⚠️  SemanticCache unavailable (%s); cache disabled for this process.",
                exc,
            )

    # ── Bootstrap ──────────────────────────────────────────────────────────────

    def _init_backend(self) -> None:
        from app.core.database import get_embed_dim, pg_connection

        dim = get_embed_dim()
        with pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS semantic_cache (
                        id         BIGSERIAL PRIMARY KEY,
                        user_id    TEXT NOT NULL DEFAULT 'anonymous',
                        query      TEXT NOT NULL,
                        response   JSONB NOT NULL,
                        embedding  VECTOR({dim}) NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS semantic_cache_user_idx "
                    "ON semantic_cache (user_id);"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS semantic_cache_embedding_idx "
                    "ON semantic_cache USING hnsw (embedding vector_cosine_ops) "
                    "WITH (m = 16, ef_construction = 64);"
                )
                # Idempotent column backfill for upgrades from the un-scoped schema
                cur.execute(
                    """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'semantic_cache'
                              AND column_name = 'user_id'
                        ) THEN
                            ALTER TABLE semantic_cache
                                ADD COLUMN user_id TEXT NOT NULL DEFAULT 'anonymous';
                        END IF;
                    END$$;
                    """
                )
            conn.commit()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(
        self,
        query: str,
        user_id: str = "anonymous",
        similarity_threshold: Optional[float] = None,
    ) -> Optional[dict]:
        """Return a cached response dict for *query* scoped to *user_id*, or None.

        The dict has keys ``response`` (str), ``query`` (str), ``similarity`` (float).
        Returns None on cache MISS, pgvector unavailability, or any error.
        """
        if not self._available:
            return None
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.similarity_threshold
        )
        try:
            from app.core.database import get_embedder, pg_connection
            import numpy as np

            embedding = np.array(get_embedder().embed_query(query), dtype=np.float32)
            with pg_connection(register_pgvector=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT query, response,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM   semantic_cache
                        WHERE  user_id = %s
                          AND  created_at > NOW() - (%s || ' seconds')::interval
                        ORDER  BY embedding <=> %s::vector
                        LIMIT  1;
                        """,
                        (embedding, user_id, str(self.ttl_seconds), embedding),
                    )
                    row = cur.fetchone()
            if row is None:
                return None
            cached_query, cached_response, similarity = row
            if float(similarity) < threshold:
                return None
            response_text = (
                cached_response.get("response")
                if isinstance(cached_response, dict)
                else cached_response
            )
            logger.info(
                "✅ Cache HIT [user=%s] cosine=%.4f ≥ %.2f  query='%s...'",
                user_id, float(similarity), threshold, query[:60],
            )
            return {
                "response": response_text,
                "query": cached_query,
                "similarity": float(similarity),
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("SemanticCache GET failed: %s", exc)
            return None

    def set(self, query: str, response: str, user_id: str = "anonymous") -> None:
        """Store *response* for *query* scoped to *user_id*. Silent on failure."""
        if not self._available or not response:
            return
        try:
            from app.core.database import get_embedder, pg_connection
            import numpy as np

            embedding = np.array(get_embedder().embed_query(query), dtype=np.float32)
            payload = json.dumps({"response": response})
            with pg_connection(register_pgvector=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO semantic_cache (user_id, query, response, embedding) "
                        "VALUES (%s, %s, %s::jsonb, %s);",
                        (user_id, query, payload, embedding),
                    )
                conn.commit()
            logger.info("📝 Cache SET [user=%s]  query='%s...'", user_id, query[:60])
        except Exception as exc:  # noqa: BLE001
            logger.error("SemanticCache SET failed: %s", exc)

    def size(self, user_id: Optional[str] = None) -> int:
        """Return entry count, optionally scoped to *user_id*."""
        if not self._available:
            return 0
        try:
            from app.core.database import pg_connection
            with pg_connection() as conn:
                with conn.cursor() as cur:
                    if user_id is not None:
                        cur.execute(
                            "SELECT COUNT(*) FROM semantic_cache WHERE user_id = %s;",
                            (user_id,),
                        )
                    else:
                        cur.execute("SELECT COUNT(*) FROM semantic_cache;")
                    row = cur.fetchone()
                    return int(row[0]) if row else 0
        except Exception:  # noqa: BLE001
            return 0

    def clear(self, user_id: Optional[str] = None) -> int:
        """Delete cache entries, returning the number deleted."""
        if not self._available:
            return 0
        try:
            from app.core.database import pg_connection
            with pg_connection() as conn:
                with conn.cursor() as cur:
                    if user_id is not None:
                        cur.execute(
                            "DELETE FROM semantic_cache WHERE user_id = %s;",
                            (user_id,),
                        )
                    else:
                        cur.execute("DELETE FROM semantic_cache;")
                    deleted = cur.rowcount
                conn.commit()
            return int(deleted or 0)
        except Exception as exc:  # noqa: BLE001
            logger.error("SemanticCache clear failed: %s", exc)
            return 0

    # Backwards-compat shim for code that accessed .cache as a dict
    @property
    def cache(self) -> dict:
        return {}


# ── Module-level singleton ─────────────────────────────────────────────────────
# Import this instance directly; never construct a second SemanticCache.
semantic_cache = SemanticCache()
