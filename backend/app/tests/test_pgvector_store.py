"""Verification tests for the pgvector-backed document store.

Tests:
  1. Top-5 overlap — pgvector and ChromaDB surface similar results for
     the same queries (Jaccard similarity ≥ 40 %).
  2. Concurrency latency — 10 parallel pgvector requests must complete
     without serialisation-style linear scaling.

Run manually:
    VECTOR_STORE_BACKEND=pgvector pytest backend/app/tests/test_pgvector_store.py -v
"""

import os
import time
import random
import threading
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunks(n: int = 5) -> List[Dict[str, Any]]:
    """Generate synthetic Document-like dicts that PgVectorDocumentStore accepts."""
    return [
        {
            "page_content": f"Chunk {i}: the quick brown fox jumps over the lazy dog.",
            "metadata": {
                "chunk_id": f"test-chunk-{i}",
                "file_type": "python",
                "language": "python",
                "source": f"test_file_{i}.py",
            },
        }
        for i in range(n)
    ]


def _make_embeddings(n: int = 5, dim: int = 768) -> List[List[float]]:
    """Generate random unit embeddings."""
    rng = random.Random(42)
    embs = []
    for _ in range(n):
        vec = [rng.gauss(0, 1) for _ in range(dim)]
        norm = sum(v ** 2 for v in vec) ** 0.5 or 1.0
        embs.append([v / norm for v in vec])
    return embs


# ---------------------------------------------------------------------------
# Unit tests — gating
# ---------------------------------------------------------------------------

class TestPgvectorEnabled:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("VECTOR_STORE_BACKEND", raising=False)
        from app.services.retrieval.pgvector_store import pgvector_enabled
        assert pgvector_enabled() is False

    def test_enabled_by_env_var(self, monkeypatch):
        monkeypatch.setenv("VECTOR_STORE_BACKEND", "pgvector")
        # Reload the function to pick up the new env var
        from app.services.retrieval.pgvector_store import pgvector_enabled
        assert pgvector_enabled() is True

    def test_enabled_by_mixed_string(self, monkeypatch):
        monkeypatch.setenv("VECTOR_STORE_BACKEND", "chroma,pgvector")
        from app.services.retrieval.pgvector_store import pgvector_enabled
        assert pgvector_enabled() is True

    def test_disabled_for_chroma_only(self, monkeypatch):
        monkeypatch.setenv("VECTOR_STORE_BACKEND", "chroma")
        from app.services.retrieval.pgvector_store import pgvector_enabled
        assert pgvector_enabled() is False


# ---------------------------------------------------------------------------
# Integration test 1 — Top-5 overlap comparison
# ---------------------------------------------------------------------------

SAMPLE_QUERIES = [
    "authentication middleware JWT token",
    "database connection pool timeout",
    "vector similarity cosine distance",
    "LangGraph state reducer parallel",
    "ChromaDB collection ingestion",
    "FastAPI route handler dependency injection",
    "embedding model sentence transformer",
    "GraphQL resolver query mutation",
    "async task background celery",
    "Docker compose service health check",
]


def _chroma_top5(query: str, collection) -> List[str]:
    """Return top-5 chunk IDs from ChromaDB."""
    from app.core.database import get_embedder
    embedder = get_embedder()
    qv = embedder.embed_query(query)
    res = collection.query(query_embeddings=[qv], n_results=5, include=["metadatas"])
    metas = (res.get("metadatas") or [[]])[0]
    return [m.get("chunk_id", "") for m in metas if m]


def _pgvector_top5(query: str) -> List[str]:
    """Return top-5 chunk IDs from pgvector."""
    from app.services.retrieval.pgvector_store import PgVectorDocumentStore
    from app.core.database import get_embedder
    embedder = get_embedder()
    qv = embedder.embed_query(query)
    store = PgVectorDocumentStore()
    results = store.query_similar(query_embedding=qv, top_k=5)
    return [r["metadata"].get("chunk_id", "") for r in results]


@pytest.mark.integration
@pytest.mark.skipif(
    "pgvector" not in os.getenv("VECTOR_STORE_BACKEND", ""),
    reason="VECTOR_STORE_BACKEND must include 'pgvector' to run integration tests",
)
def test_top5_overlap_with_chroma():
    """pgvector and ChromaDB top-5 results must share ≥40 % of chunk IDs."""
    import chromadb
    from app.core.config import settings

    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
    try:
        collection = client.get_collection("documents")
    except Exception:
        pytest.skip("ChromaDB 'documents' collection not found — run ingestion first")

    overlaps = []
    for q in SAMPLE_QUERIES:
        chroma_ids = set(_chroma_top5(q, collection))
        pg_ids = set(_pgvector_top5(q))
        if not chroma_ids and not pg_ids:
            continue
        union = chroma_ids | pg_ids
        overlap = len(chroma_ids & pg_ids) / len(union) if union else 0.0
        overlaps.append(overlap)
        print(f"  Query '{q[:40]}': Jaccard={overlap:.2f}  chroma={chroma_ids}  pg={pg_ids}")

    if overlaps:
        avg = sum(overlaps) / len(overlaps)
        print(f"\nAverage Jaccard similarity: {avg:.2f}")
        assert avg >= 0.40, f"Top-5 overlap too low: {avg:.2f} < 0.40"


# ---------------------------------------------------------------------------
# Integration test 2 — Concurrency latency
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(
    "pgvector" not in os.getenv("VECTOR_STORE_BACKEND", ""),
    reason="VECTOR_STORE_BACKEND must include 'pgvector'",
)
def test_pgvector_concurrency_no_linear_scaling():
    """10 parallel pgvector queries must finish faster than 10× the serial median.

    ChromaDB with a threading.Lock scales linearly; the pool-backed pgvector
    path should exhibit sub-linear growth (concurrent requests are actually
    served in parallel by PostgreSQL workers).
    """
    from app.services.retrieval.pgvector_store import PgVectorDocumentStore
    from app.core.database import get_embedder

    embedder = get_embedder()
    queries = SAMPLE_QUERIES[:10]
    query_vectors = [embedder.embed_query(q) for q in queries]
    store = PgVectorDocumentStore()

    # --- serial baseline ---
    serial_times = []
    for qv in query_vectors:
        t0 = time.perf_counter()
        store.query_similar(qv, top_k=5)
        serial_times.append(time.perf_counter() - t0)
    serial_median = sorted(serial_times)[len(serial_times) // 2]
    serial_total_expected = serial_median * len(query_vectors)

    # --- parallel run ---
    results: List[float] = [0.0] * len(query_vectors)
    errors: List[Exception] = []

    def _run(idx: int, qv: List[float]) -> None:
        try:
            t0 = time.perf_counter()
            store.query_similar(qv, top_k=5)
            results[idx] = time.perf_counter() - t0
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_run, args=(i, qv)) for i, qv in enumerate(query_vectors)]
    parallel_t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    parallel_wall = time.perf_counter() - parallel_t0

    assert not errors, f"Errors during parallel run: {errors}"
    print(
        f"\nSerial median={serial_median*1000:.1f}ms  "
        f"Serial-total-expected={serial_total_expected*1000:.1f}ms  "
        f"Parallel-wall={parallel_wall*1000:.1f}ms"
    )
    # Parallel wall time must be meaningfully less than serial total
    assert parallel_wall < serial_total_expected * 0.75, (
        f"pgvector concurrency not effective: wall={parallel_wall:.3f}s  "
        f"serial-total={serial_total_expected:.3f}s"
    )


# ---------------------------------------------------------------------------
# Unit tests — insert_chunks / query_similar (mocked DB)
# ---------------------------------------------------------------------------

class TestPgVectorDocumentStore:
    """Fast unit tests that mock the postgres connection."""

    def _make_store(self):
        from app.services.retrieval.pgvector_store import PgVectorDocumentStore
        return PgVectorDocumentStore()

    @patch("app.services.retrieval.pgvector_store.pg_connection")
    def test_insert_chunks_returns_count(self, mock_pg):
        """insert_chunks should return the number of chunks upserted."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pg.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_cur.rowcount = 3

        chunks = _make_chunks(3)
        embeddings = _make_embeddings(3)
        store = self._make_store()
        count = store.insert_chunks(chunks, embeddings)
        assert count == 3

    @patch("app.services.retrieval.pgvector_store.pg_connection")
    def test_insert_chunks_mismatch_raises(self, mock_pg):
        """insert_chunks must raise ValueError when len(chunks) != len(embeddings)."""
        store = self._make_store()
        with pytest.raises(ValueError, match="mismatch"):
            store.insert_chunks(_make_chunks(3), _make_embeddings(2))

    @patch("app.services.retrieval.pgvector_store.pg_connection")
    def test_query_similar_returns_list(self, mock_pg):
        """query_similar should return a list of result dicts."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pg.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_cur.fetchall.return_value = [
            ("chunk-1", "hello world", '{"source": "a.py"}', 0.05),
            ("chunk-2", "foo bar",     '{"source": "b.py"}', 0.12),
        ]

        store = self._make_store()
        qv = _make_embeddings(1)[0]
        results = store.query_similar(qv, top_k=2)
        assert len(results) == 2
        assert results[0]["content"] == "hello world"
        assert results[0]["retrieval_method"] == "pgvector"
        # cosine score = 1 - distance
        assert abs(results[0]["score"] - 0.95) < 1e-6
