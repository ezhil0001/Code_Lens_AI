"""
pgvector-backed document chunk store — async-safe vector retrieval via
the pooled Postgres connection instead of the global threading.Lock() used
for ChromaDB access.

Usage (retrieval)
-----------------
    store = PgVectorDocumentStore()
    docs  = await store.query_similar(query_embedding, top_k=20,
                                      metadata_filter={"file_type": "code"})

Usage (ingestion)
-----------------
    store = PgVectorDocumentStore()
    await store.ensure_table()
    await store.insert_chunks(chunks, embeddings)

Gating
------
Both paths are activated only when the ``VECTOR_STORE_BACKEND`` env var
contains the string ``"pgvector"`` (e.g. ``"chroma,pgvector"`` for parallel
operation or ``"pgvector"`` to switch fully).  The retriever engine reads this
value via :func:`pgvector_enabled`.

Schema
------
    document_chunks (
        id          BIGSERIAL PRIMARY KEY,
        chunk_id    TEXT UNIQUE NOT NULL,   -- stable hash used by PDR
        content     TEXT NOT NULL,
        metadata    JSONB NOT NULL DEFAULT '{}',
        embedding   VECTOR(768) NOT NULL,
        file_type   TEXT,                  -- denormalized for fast WHERE
        language    TEXT,
        source      TEXT,
        created_at  TIMESTAMPTZ DEFAULT NOW()
    )

The table is created idempotently via ``ensure_table()`` which is called
once at ingestion time and once at server startup (inside RetrieverEngine).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gate check
# ---------------------------------------------------------------------------

def pgvector_enabled() -> bool:
    """Return True when the VECTOR_STORE_BACKEND env var includes 'pgvector'."""
    backend = os.getenv("VECTOR_STORE_BACKEND", "chroma").lower()
    return "pgvector" in backend


# ---------------------------------------------------------------------------
# Table bootstrap
# ---------------------------------------------------------------------------

_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS document_chunks (
    id          BIGSERIAL PRIMARY KEY,
    chunk_id    TEXT UNIQUE NOT NULL,
    content     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}',
    embedding   VECTOR({dim}) NOT NULL,
    file_type   TEXT,
    language    TEXT,
    source      TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
"""

_INDEX_SQL = [
    # ANN index — HNSW works from 0 rows unlike IVFFlat which needs ~300+
    (
        "CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx "
        "ON document_chunks USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64);"
    ),
    (
        "CREATE INDEX IF NOT EXISTS document_chunks_file_type_idx "
        "ON document_chunks (file_type);"
    ),
    (
        "CREATE INDEX IF NOT EXISTS document_chunks_source_idx "
        "ON document_chunks (source);"
    ),
]


def ensure_document_chunks_table() -> None:
    """Create the document_chunks table + indexes if they don't exist.

    Called once at ingestion time and once at server startup.  Safe to call
    many times — all DDL is idempotent.

    Raises:
        RuntimeError: propagated if Postgres or pgvector extension unavailable.
    """
    from app.core.database import get_embed_dim, pg_connection

    dim = get_embed_dim()
    with pg_connection(register_pgvector=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        with conn.cursor() as cur:
            cur.execute(_TABLE_SQL.format(dim=dim))
            for idx_sql in _INDEX_SQL:
                cur.execute(idx_sql)
        conn.commit()
    logger.info(
        "[PgVectorStore] document_chunks table ready (dim=%d, indexes=hnsw+file_type+source)",
        dim,
    )


# ---------------------------------------------------------------------------
# Store class
# ---------------------------------------------------------------------------

class PgVectorDocumentStore:
    """Wraps the document_chunks table with insert + query operations.

    All database I/O uses the pooled ``pg_connection()`` context manager from
    ``app.core.database`` — no per-call connection setup, no threading lock.

    The query path is a single parameterised SQL call using the ``<=>``
    (cosine distance) operator from pgvector.  Result ordering mirrors what
    ``_ChromaCollectionRetriever`` returns so the calling code can treat both
    paths identically.
    """

    def __init__(self) -> None:
        self._available = False
        try:
            ensure_document_chunks_table()
            self._available = True
            logger.info("✅ PgVectorDocumentStore ready")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[PgVectorStore] Unavailable (%s) — falling back to Chroma for vector leg.",
                exc,
            )

    # ------------------------------------------------------------------ #
    # Ingestion                                                           #
    # ------------------------------------------------------------------ #

    def insert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        """Upsert chunks and their embeddings into document_chunks.

        Uses ``ON CONFLICT (chunk_id) DO UPDATE`` so re-ingesting the same
        file is idempotent — only the embedding and content are refreshed.

        Parameters
        ----------
        chunks:
            Each element is the same enriched-chunk dict that goes into
            ChromaDB (keys: id, content, metadata, …).
        embeddings:
            Parallel list of 768-dim float vectors.

        Returns
        -------
        int
            Number of rows upserted.
        """
        if not self._available:
            return 0
        if not chunks:
            return 0

        from app.core.database import pg_connection
        try:
            from pgvector.psycopg import register_vector  # noqa: F401
        except ImportError:
            pass

        rows_upserted = 0
        _SQL = """
            INSERT INTO document_chunks
                (chunk_id, content, metadata, embedding, file_type, language, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                content   = EXCLUDED.content,
                metadata  = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding,
                file_type = EXCLUDED.file_type,
                language  = EXCLUDED.language,
                source    = EXCLUDED.source;
        """

        with pg_connection(register_pgvector=True) as conn:
            with conn.cursor() as cur:
                for chunk, emb in zip(chunks, embeddings):
                    meta = chunk.get("metadata", {})
                    chunk_id = (
                        chunk.get("id")
                        or meta.get("chunk_id")
                        or meta.get("id")
                        or f"chunk_{hash(chunk.get('content', ''))}"
                    )
                    file_type = meta.get("file_type")
                    language  = meta.get("language")
                    source    = meta.get("source")

                    cur.execute(
                        _SQL,
                        (
                            chunk_id,
                            chunk.get("content", ""),
                            json.dumps(meta),
                            emb,          # pgvector accepts a plain Python list
                            file_type,
                            language,
                            source,
                        ),
                    )
                    rows_upserted += 1
            conn.commit()

        logger.info("[PgVectorStore] upserted %d chunks", rows_upserted)
        return rows_upserted

    # ------------------------------------------------------------------ #
    # Retrieval                                                           #
    # ------------------------------------------------------------------ #

    def query_similar(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return the top-K most similar chunks by cosine distance.

        Parameters
        ----------
        query_embedding:
            768-dim float vector (from the shared embedder singleton).
        top_k:
            Maximum number of results.
        metadata_filter:
            Optional ``{field: value}`` or ``{field: {"$eq": value}}``
            filter applied as SQL WHERE clauses on the denormalised columns
            (file_type, language, source) and, for arbitrary keys, via JSONB
            ``metadata @> '{"key":"value"}'::jsonb``.

        Returns
        -------
        list of dicts with keys:
            content, metadata (dict), score (float 0–1), retrieval_method
        """
        if not self._available:
            return []

        from app.core.database import pg_connection

        where_clauses: List[str] = []
        params: List[Any] = [query_embedding]

        if metadata_filter:
            for k, v in metadata_filter.items():
                # Unwrap Chroma-style {"$eq": value} filter syntax
                if isinstance(v, dict) and "$eq" in v:
                    v = v["$eq"]
                # Fast path: denormalised columns
                if k in ("file_type", "language", "source"):
                    where_clauses.append(f"{k} = %s")
                    params.append(v)
                else:
                    # Generic JSONB containment for any other metadata key
                    where_clauses.append("metadata @> %s::jsonb")
                    params.append(json.dumps({k: v}))

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        params.append(top_k)
        _SQL = f"""
            SELECT
                chunk_id,
                content,
                metadata,
                1 - (embedding <=> %s::vector) AS score
            FROM document_chunks
            {where_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        # The query embedding appears twice: once in SELECT for score, once in ORDER BY.
        # Build the final params list: [emb, ...filter_params..., emb, top_k]
        filter_params = params[1:-1]          # strip leading emb and trailing top_k
        final_params  = [query_embedding] + filter_params + [query_embedding, top_k]

        results: List[Dict[str, Any]] = []
        try:
            with pg_connection(register_pgvector=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(_SQL, final_params)
                    rows = cur.fetchall()
            for row in rows:
                chunk_id, content, meta_raw, score = row
                if isinstance(meta_raw, str):
                    try:
                        meta = json.loads(meta_raw)
                    except Exception:
                        meta = {}
                else:
                    meta = meta_raw or {}
                meta["chunk_id"] = chunk_id
                meta["score"]    = float(score)
                meta["retrieval_method"] = "pgvector"
                results.append(
                    {
                        "content": content,
                        "metadata": meta,
                        "score": float(score),
                        "retrieval_method": "pgvector",
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("[PgVectorStore] query_similar failed: %s", exc, exc_info=True)

        logger.debug(
            "[PgVectorStore] query returned %d results (top_k=%d, filter=%s)",
            len(results),
            top_k,
            metadata_filter,
        )
        return results
