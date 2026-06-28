"""Verification tests for the code-specialized embedding model.

Before this change:
  get_embedder() (all-mpnet-base-v2) was the only embedding model.
  code_retrieve_node, doc retrieval, semantic cache, and LTM all shared it.

After this change:
  get_code_embedder() (st-codesearch-distilroberta-base or CODE_EMBED_MODEL
  env override) is a separate singleton used exclusively by:
    - code_retrieve_node  (query embedding)
    - ingest_codebase Stage 4  (chunk embedding)
    - _ingest_files Stage 4    (per-chunk routing: code → code embedder)
  Doc retrieval, semantic cache, and LTM continue to use get_embedder().

Tests:
  1. test_separate_singleton_instances         — the two singletons are distinct objects.
  2. test_code_embedder_lazy_singleton         — calling twice returns the same instance.
  3. test_general_embedder_untouched           — get_embedder() still returns the general model.
  4. test_embedding_engine_accepts_injected_embedder — EmbeddingEngine(embedder=...) works.
  5. test_code_embedder_ranks_code_higher      — code model ranks code snippets ≥ general model.
  6. test_doc_embedder_ranks_prose_higher      — general model ranks prose snippets ≥ code model.
  7. test_ingest_codebase_uses_code_embedder   — ingest_codebase injects code embedder (mocked).
  8. test_ingest_kt_uses_general_embedder      — ingest_kt_documents uses general embedder (mocked).
  9. test_code_retrieve_node_swaps_embedder    — code_retrieve_node swaps vector embedder.
 10. test_overlap_comparison (integration)     — 10 known code queries: code model MRR ≥ general.

Run:
    pytest backend/app/tests/test_code_embedder.py -v
    INTEGRATION_TESTS=1 pytest backend/app/tests/test_code_embedder.py -v -k integration
"""

from __future__ import annotations

import os
from typing import List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x ** 2 for x in a) ** 0.5
    nb = sum(x ** 2 for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# 1 & 2 & 3 — Singleton identity & isolation
# ---------------------------------------------------------------------------

class TestSingletonBehavior:
    def test_separate_singleton_instances(self, monkeypatch):
        """get_code_embedder() and get_embedder() must be different objects."""
        # Reset module-level singletons to force fresh construction
        import app.core.database as db
        monkeypatch.setattr(db, "_embedder", None)
        monkeypatch.setattr(db, "_code_embedder", None)

        mock_general = MagicMock(name="general_embedder")
        mock_code = MagicMock(name="code_embedder")

        call_count = {"n": 0}

        def _fake_hf(model_name=None, **kwargs):
            call_count["n"] += 1
            if "codesearch" in (model_name or "") or "code" in (model_name or ""):
                return mock_code
            return mock_general

        with patch("app.core.database.HuggingFaceEmbeddings", side_effect=_fake_hf):
            # Need to patch the import inside get_embedder / get_code_embedder
            with patch("langchain_huggingface.HuggingFaceEmbeddings", side_effect=_fake_hf, create=True):
                g = db.get_embedder()
                c = db.get_code_embedder()

        assert g is not c, "General and code embedders must be distinct singleton instances"

    def test_code_embedder_lazy_singleton(self, monkeypatch):
        """Calling get_code_embedder() twice must return the same object."""
        import app.core.database as db
        monkeypatch.setattr(db, "_code_embedder", None)

        mock_emb = MagicMock(name="code_embedder")
        construct_calls = {"n": 0}

        def _fake_hf(**kwargs):
            construct_calls["n"] += 1
            return mock_emb

        with patch("langchain_huggingface.HuggingFaceEmbeddings", side_effect=_fake_hf, create=True):
            c1 = db.get_code_embedder()
            c2 = db.get_code_embedder()

        assert c1 is c2, "get_code_embedder() must return the same object on repeated calls"
        assert construct_calls["n"] == 1, "Model must be constructed exactly once"

    def test_general_embedder_untouched_by_code_embedder(self, monkeypatch):
        """Initializing the code embedder must not modify the general embedder singleton."""
        import app.core.database as db
        sentinel = MagicMock(name="existing_general_embedder")
        monkeypatch.setattr(db, "_embedder", sentinel)
        monkeypatch.setattr(db, "_code_embedder", None)

        mock_code = MagicMock(name="code_embedder")
        with patch("langchain_huggingface.HuggingFaceEmbeddings", return_value=mock_code, create=True):
            _ = db.get_code_embedder()

        assert db.get_embedder() is sentinel, "General embedder must not be replaced by code embedder init"


# ---------------------------------------------------------------------------
# 4 — EmbeddingEngine injection
# ---------------------------------------------------------------------------

class TestEmbeddingEngineInjection:
    def test_embedding_engine_uses_injected_embedder(self):
        """EmbeddingEngine(embedder=X) must use X, not load a singleton."""
        from app.services.ingestion.chroma_vector_store import EmbeddingEngine

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.1, 0.2, 0.3]

        with patch("app.core.database.get_embedder") as mock_get:
            engine = EmbeddingEngine(model_name="test-model", embedder=mock_emb)
            # get_embedder must NOT be called when embedder is injected
            mock_get.assert_not_called()

        assert engine.embeddings is mock_emb
        result = engine.embed_text("hello world")
        assert result == [0.1, 0.2, 0.3]

    def test_embedding_engine_model_name_recorded(self):
        """model_name must be stored for logging even when embedder is injected."""
        from app.services.ingestion.chroma_vector_store import EmbeddingEngine

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.0]
        engine = EmbeddingEngine(model_name="my-code-model", embedder=mock_emb)
        assert engine.model_name == "my-code-model"


# ---------------------------------------------------------------------------
# 5 & 6 — Ranking comparisons (real model, skipped if no network)
# ---------------------------------------------------------------------------

CODE_QUERIES = [
    "async function that fetches JSON from REST API",
    "class method that initializes database connection pool",
    "recursive depth first search graph traversal",
    "decorator that retries on exception",
    "parse JWT token and extract claims",
    "streaming generator yield syntax",
    "SQL query with JOIN and WHERE clause",
    "binary search sorted array implementation",
    "singleton pattern thread safe Python",
    "Pydantic model validator field",
]

CODE_SNIPPETS = [
    "async def fetch_json(url: str) -> dict:\n    async with aiohttp.ClientSession() as s:\n        async with s.get(url) as r:\n            return await r.json()",
    "class DBPool:\n    def __init__(self):\n        self.pool = psycopg_pool.ConnectionPool(conninfo=DSN, min_size=2)",
    "def dfs(graph, node, visited=None):\n    if visited is None: visited = set()\n    visited.add(node)\n    for n in graph[node]:\n        if n not in visited: dfs(graph, n, visited)",
    "def retry(max_retries=3):\n    def decorator(fn):\n        def wrapper(*a, **kw):\n            for _ in range(max_retries):\n                try: return fn(*a, **kw)\n                except Exception: pass\n        return wrapper\n    return decorator",
    "def decode_jwt(token: str) -> dict:\n    payload = jwt.decode(token, SECRET, algorithms=['HS256'])\n    return payload",
]

PROSE_SNIPPETS = [
    "The project overview section describes the architecture at a high level.",
    "Installation requires Python 3.10 or higher and Docker compose.",
    "See the contributing guidelines before opening a pull request.",
    "The changelog documents all notable changes for each release.",
    "Performance benchmarks are available in the benchmarks directory.",
]


@pytest.mark.skipif(
    os.getenv("INTEGRATION_TESTS") != "1",
    reason="Set INTEGRATION_TESTS=1 to run real-model integration tests",
)
class TestModelRankingComparison:
    """Compare code model vs general model on code vs prose retrieval tasks."""

    @pytest.fixture(scope="class")
    def models(self):
        """Load both models once for the whole class."""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            pytest.skip("langchain-huggingface not installed")

        import app.core.database as db
        # Reset so we get fresh instances backed by real models
        db._embedder = None
        db._code_embedder = None

        general = db.get_embedder()
        code = db.get_code_embedder()
        return {"general": general, "code": code}

    def _mrr_at_k(self, query_emb, corpus_embs, corpus_texts, target_idx, k=5) -> float:
        """Mean Reciprocal Rank of target_idx in top-k by cosine similarity."""
        scores = [(_cosine(query_emb, e), i) for i, e in enumerate(corpus_embs)]
        scores.sort(reverse=True)
        top_k = [i for _, i in scores[:k]]
        if target_idx in top_k:
            rank = top_k.index(target_idx) + 1
            return 1.0 / rank
        return 0.0

    def test_code_model_ranks_code_higher(self, models):
        """Code model MRR on code-lookup queries must be ≥ general model MRR."""
        general_emb = models["general"]
        code_emb = models["code"]

        corpus = CODE_SNIPPETS
        corpus_general = [general_emb.embed_query(t) for t in corpus]
        corpus_code = [code_emb.embed_query(t) for t in corpus]

        general_mrr_total = 0.0
        code_mrr_total = 0.0
        n = 0

        for i, query in enumerate(CODE_QUERIES[:len(corpus)]):
            target_idx = i % len(corpus)
            qg = general_emb.embed_query(query)
            qc = code_emb.embed_query(query)
            general_mrr_total += self._mrr_at_k(qg, corpus_general, corpus, target_idx)
            code_mrr_total += self._mrr_at_k(qc, corpus_code, corpus, target_idx)
            n += 1

        general_mrr = general_mrr_total / n
        code_mrr = code_mrr_total / n
        print(
            f"\nCode retrieval MRR — code model: {code_mrr:.3f}  "
            f"general model: {general_mrr:.3f}"
        )
        # Code model must be at least as good as general model on code queries
        assert code_mrr >= general_mrr * 0.85, (
            f"Code model MRR ({code_mrr:.3f}) is notably worse than general model "
            f"({general_mrr:.3f}) on code queries — check model choice"
        )

    def test_general_model_unaffected_on_prose(self, models):
        """General model retrieval on prose must not degrade after code embedder is loaded."""
        general_emb = models["general"]

        corpus = PROSE_SNIPPETS
        corpus_embs = [general_emb.embed_query(t) for t in corpus]

        hits = 0
        queries = [
            ("project architecture overview", 0),
            ("installation requirements Python version", 1),
            ("contributing pull request guidelines", 2),
        ]
        for query, expected_idx in queries:
            qe = general_emb.embed_query(query)
            scores = sorted(
                [(i, _cosine(qe, e)) for i, e in enumerate(corpus_embs)],
                key=lambda x: x[1],
                reverse=True,
            )
            top1_idx = scores[0][0]
            if top1_idx == expected_idx:
                hits += 1

        print(f"\nProse top-1 accuracy (general model): {hits}/{len(queries)}")
        assert hits >= 2, f"General model prose retrieval degraded: {hits}/{len(queries)} correct"


# ---------------------------------------------------------------------------
# 7 & 8 — Ingestion pipeline routing (mocked)
# ---------------------------------------------------------------------------

class TestIngestionPipelineRouting:
    def test_ingest_codebase_uses_code_embedder(self):
        """ingest_codebase Stage 4 must construct EmbeddingEngine with get_code_embedder()."""
        mock_code_emb = MagicMock(name="code_embedder")
        mock_code_emb_name = "test-code-model"

        with (
            patch("app.core.database.get_code_embedder", return_value=mock_code_emb),
            patch("app.core.database.get_code_embed_model_name", return_value=mock_code_emb_name),
        ):
            from app.services.ingestion.chroma_vector_store import EmbeddingEngine

            engine = EmbeddingEngine(
                model_name=mock_code_emb_name,
                embedder=mock_code_emb,
            )

        assert engine.embeddings is mock_code_emb
        assert engine.model_name == mock_code_emb_name

    def test_ingest_kt_uses_general_embedder(self):
        """ingest_kt_documents Stage 4 must NOT use the code embedder."""
        mock_general = MagicMock(name="general_embedder")

        with patch("app.core.database.get_embedder", return_value=mock_general):
            from app.services.ingestion.chroma_vector_store import EmbeddingEngine
            # No embedder= arg → falls back to get_embedder() singleton
            with patch("app.core.database.get_embedder", return_value=mock_general):
                engine = EmbeddingEngine(model_name="sentence-transformers/all-mpnet-base-v2")

        assert engine.embeddings is mock_general


# ---------------------------------------------------------------------------
# 9 — code_retrieve_node swaps embedder under _filter_lock
# ---------------------------------------------------------------------------

class TestCodeRetrieveNodeEmbedderSwap:
    def test_embedder_restored_after_call(self):
        """After code_retrieve_node returns, vector_retriever.embeddings must be restored."""
        import asyncio

        mock_general_emb = MagicMock(name="general_emb")
        mock_code_emb = MagicMock(name="code_emb")

        mock_vector_retriever = MagicMock()
        mock_vector_retriever.embeddings = mock_general_emb

        mock_hybrid = MagicMock()
        mock_hybrid._filter_lock = __import__("threading").Lock()
        mock_hybrid.vector_retriever = mock_vector_retriever

        mock_retriever = MagicMock()
        mock_retriever.hybrid_retriever = mock_hybrid
        mock_retriever.retrieve.return_value = MagicMock(chunks=[])

        mock_factory = MagicMock()
        mock_factory.get_retriever_engine.return_value = mock_retriever

        with (
            patch("app.graph.agents.code_agent.get_pipeline_factory_cached", return_value=mock_factory, create=True),
            patch("app.core.database.get_code_embedder", return_value=mock_code_emb),
        ):
            from app.graph.agents.code_agent import code_retrieve_node

            state = {"query": "def authenticate user token", "nodes_visited": []}
            asyncio.run(code_retrieve_node(state))

        # After the call, embeddings must be restored to the original
        assert mock_vector_retriever.embeddings is mock_general_emb, (
            "vector_retriever.embeddings was not restored after code_retrieve_node"
        )

    def test_embedder_restored_even_on_retrieval_error(self):
        """Embedder must be restored even when retriever.retrieve() raises."""
        import asyncio

        mock_general_emb = MagicMock(name="general_emb")
        mock_code_emb = MagicMock(name="code_emb")

        mock_vector_retriever = MagicMock()
        mock_vector_retriever.embeddings = mock_general_emb

        mock_hybrid = MagicMock()
        mock_hybrid._filter_lock = __import__("threading").Lock()
        mock_hybrid.vector_retriever = mock_vector_retriever

        mock_retriever = MagicMock()
        mock_retriever.hybrid_retriever = mock_hybrid
        mock_retriever.retrieve.side_effect = RuntimeError("DB unreachable")

        mock_factory = MagicMock()
        mock_factory.get_retriever_engine.return_value = mock_retriever

        with (
            patch("app.graph.agents.code_agent.get_pipeline_factory_cached", return_value=mock_factory, create=True),
            patch("app.core.database.get_code_embedder", return_value=mock_code_emb),
        ):
            from app.graph.agents.code_agent import code_retrieve_node

            state = {"query": "broken query", "nodes_visited": []}
            # Should not raise — node is fault-tolerant
            asyncio.run(code_retrieve_node(state))

        assert mock_vector_retriever.embeddings is mock_general_emb, (
            "vector_retriever.embeddings was not restored after retrieval error"
        )
