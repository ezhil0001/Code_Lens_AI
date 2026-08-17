"""Tests for the scoped DI factory refactor.

Verifies:
  1. Each scoped factory initialises and exposes the correct API.
  2. All existing get_pipeline_factory_cached() call-sites still work
     (backward-compatibility shim).
  3. code_rerank_node can be unit-tested by mocking ONLY RetrievalFactory —
     no knowledge of LLMClientFactory / MemoryFactory / PromptFactory required.
  4. Singleton behaviour and reset() helpers work correctly.
  5. Dependency function wiring (get_retriever_engine_dep, get_llm_dep, …).

Run:
    pytest backend/app/tests/test_scoped_factories.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock
from typing import Any
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mock_retrieval_factory() -> MagicMock:
    """Return a MagicMock that satisfies the RetrievalFactory interface."""
    f = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = MagicMock(chunks=[], parent_contexts={})
    mock_retriever.hybrid_retriever = MagicMock()
    mock_retriever.hybrid_retriever._filter_lock = __import__("threading").Lock()
    mock_retriever.hybrid_retriever.vector_retriever = MagicMock()
    f.get_retriever_engine.return_value = mock_retriever
    f.get_reranker.return_value = None
    return f


def _mock_llm_factory() -> MagicMock:
    f = MagicMock()
    mock_llm = MagicMock()
    mock_llm.astream = MagicMock()
    f.get_llm.return_value = mock_llm
    return f


def _mock_memory_factory() -> MagicMock:
    f = MagicMock()
    f.get_memory_manager.return_value = None
    return f


def _mock_prompt_factory() -> MagicMock:
    f = MagicMock()
    f.get_example_selector.return_value = MagicMock()
    f.get_prompt_builder.return_value = MagicMock()
    return f


# ─────────────────────────────────────────────────────────────────────────────
# 1. Scoped factory public API
# ─────────────────────────────────────────────────────────────────────────────

class TestScopedFactoryAPI:
    """Verify each factory exposes exactly the right methods and singletons."""

    def test_retrieval_factory_interface(self):
        """RetrievalFactory must expose get_retriever_engine, get_reranker, refresh_bm25_index."""
        from app.services.scoped_factories import RetrievalFactory
        assert hasattr(RetrievalFactory, "get_instance")
        assert hasattr(RetrievalFactory, "reset")
        instance = MagicMock(spec=RetrievalFactory)
        assert hasattr(instance, "get_retriever_engine")
        assert hasattr(instance, "get_reranker")
        assert hasattr(instance, "refresh_bm25_index")

    def test_llm_client_factory_interface(self):
        from app.services.scoped_factories import LLMClientFactory
        assert hasattr(LLMClientFactory, "get_instance")
        instance = MagicMock(spec=LLMClientFactory)
        assert hasattr(instance, "get_llm")

    def test_memory_factory_interface(self):
        from app.services.scoped_factories import MemoryFactory
        assert hasattr(MemoryFactory, "get_instance")
        instance = MagicMock(spec=MemoryFactory)
        assert hasattr(instance, "get_memory_manager")

    def test_prompt_factory_interface(self):
        from app.services.scoped_factories import PromptFactory
        assert hasattr(PromptFactory, "get_instance")
        instance = MagicMock(spec=PromptFactory)
        assert hasattr(instance, "get_example_selector")
        assert hasattr(instance, "get_prompt_builder")

    def test_dependency_functions_exported(self):
        """All FastAPI dependency functions must be importable from scoped_factories."""
        from app.services.scoped_factories import (
            get_retrieval_factory,
            get_llm_client_factory,
            get_memory_factory,
            get_prompt_factory,
            get_retriever_engine_dep,
            get_reranker_dep,
            get_llm_dep,
            get_memory_manager_dep,
            get_prompt_builder_dep,
            get_example_selector_dep,
            reset_all_scoped_factories,
        )
        for fn in (
            get_retrieval_factory, get_llm_client_factory, get_memory_factory,
            get_prompt_factory, get_retriever_engine_dep, get_reranker_dep,
            get_llm_dep, get_memory_manager_dep, get_prompt_builder_dep,
            get_example_selector_dep, reset_all_scoped_factories,
        ):
            assert callable(fn), f"{fn} is not callable"

    def test_pipeline_factory_re_exports_scoped_names(self):
        """pipeline_factory must re-export all scoped names for backward compat."""
        import app.services.pipeline_factory as pf
        for name in (
            "RetrievalFactory", "LLMClientFactory", "MemoryFactory", "PromptFactory",
            "get_retrieval_factory", "get_llm_client_factory",
            "get_memory_factory", "get_prompt_factory",
            "get_retriever_engine_dep", "get_llm_dep",
        ):
            assert hasattr(pf, name), f"pipeline_factory missing re-export: {name}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Singleton behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestSingletonBehaviour:
    """Each factory must return the same instance on repeated get_instance() calls."""

    def _reset_all(self):
        try:
            from app.services.scoped_factories import reset_all_scoped_factories
            reset_all_scoped_factories()
        except Exception:
            pass

    def test_llm_factory_singleton(self):
        """LLMClientFactory.get_instance() must return the same object twice."""
        from app.services.scoped_factories import LLMClientFactory
        LLMClientFactory.reset()
        try:
            with patch.object(LLMClientFactory, "_build_llm", return_value=MagicMock()):
                a = LLMClientFactory.get_instance()
                b = LLMClientFactory.get_instance()
                assert a is b, "get_instance() must be idempotent"
        finally:
            LLMClientFactory.reset()

    def test_memory_factory_singleton(self):
        from app.services.scoped_factories import MemoryFactory
        MemoryFactory.reset()
        try:
            with patch.object(MemoryFactory, "_build_chat_memory", return_value=None):
                a = MemoryFactory.get_instance()
                b = MemoryFactory.get_instance()
                assert a is b
        finally:
            MemoryFactory.reset()

    def test_reset_clears_singleton(self):
        from app.services.scoped_factories import MemoryFactory
        MemoryFactory.reset()
        try:
            with patch.object(MemoryFactory, "_build_chat_memory", return_value=None):
                a = MemoryFactory.get_instance()
                MemoryFactory.reset()
                b = MemoryFactory.get_instance()
                assert a is not b, "reset() must allow a new instance to be created"
        finally:
            MemoryFactory.reset()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Backward-compat shim: get_pipeline_factory_cached() still works
# ─────────────────────────────────────────────────────────────────────────────

class TestBackwardCompatShim:
    """RAGPipelineFactory must still satisfy all existing call-site contracts."""

    def _build_shim(self):
        """Construct a RAGPipelineFactory with all scoped singletons mocked."""
        from app.services import pipeline_factory as pf

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = MagicMock(chunks=[], parent_contexts={})
        mock_retriever.reranking_engine = MagicMock()
        mock_retriever.hybrid_retriever = MagicMock()

        mock_rf = MagicMock()
        mock_rf.get_retriever_engine.return_value = mock_retriever
        mock_rf.get_reranker.return_value = mock_retriever.reranking_engine

        mock_llmf = MagicMock()
        mock_llmf.get_llm.return_value = MagicMock()

        mock_memf = MagicMock()
        mock_memf.get_memory_manager.return_value = None

        mock_pf = MagicMock()
        mock_pf.get_example_selector.return_value = MagicMock()
        mock_pf.get_prompt_builder.return_value = MagicMock()

        pf.RAGPipelineFactory.reset()
        pf.get_pipeline_factory_cached.cache_clear()

        with (
            patch("app.services.pipeline_factory.RetrievalFactory.get_instance", return_value=mock_rf),
            patch("app.services.pipeline_factory.LLMClientFactory.get_instance", return_value=mock_llmf),
            patch("app.services.pipeline_factory.MemoryFactory.get_instance", return_value=mock_memf),
            patch("app.services.pipeline_factory.PromptFactory.get_instance", return_value=mock_pf),
            # Suppress AgenticRouter + AgentBrain — not under test here
            patch("app.services.agents.agentic_router.AgenticRouter", MagicMock(), create=True),
            patch("app.services.agents.agent_brain.AgentBrain", MagicMock(), create=True),
        ):
            factory = pf.RAGPipelineFactory()

        return factory, mock_retriever

    def test_get_retriever_engine_returns_retriever(self):
        factory, mock_retriever = self._build_shim()
        assert factory.get_retriever_engine() is mock_retriever

    def test_get_reranker_returns_reranker(self):
        factory, mock_retriever = self._build_shim()
        assert factory.get_reranker() is not None  # mocked non-None

    def test_get_llm_returns_llm(self):
        factory, _ = self._build_shim()
        assert factory.get_llm() is not None

    def test_get_memory_manager_returns_none_gracefully(self):
        factory, _ = self._build_shim()
        assert factory.get_memory_manager() is None  # mocked to None

    def test_refresh_bm25_delegates_to_retrieval_factory(self):
        factory, _ = self._build_shim()
        factory._retrieval_factory.refresh_bm25_index = MagicMock()
        factory.refresh_bm25_index()
        factory._retrieval_factory.refresh_bm25_index.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Focused unit test: code_rerank_node needs only RetrievalFactory
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeRerankNodeIsolation:
    """Demonstrate that code_rerank_node can be tested by mocking
    ONLY the RetrievalFactory — no knowledge of LLM/Memory/Prompt required."""

    @pytest.mark.asyncio
    async def test_code_rerank_node_with_mocked_retrieval_factory(self):
        """code_rerank_node must call get_reranker() and rerank the chunks."""
        from app.graph.agents.code_agent import code_rerank_node

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = (
            [{"content": "def foo(): pass", "metadata": {}, "score": 0.9}],
            [0.9],
        )

        mock_factory = MagicMock()
        mock_factory.get_reranker.return_value = mock_reranker

        state = {
            "query": "how does foo work",
            "retrieved_chunks": [
                {"content": "def foo(): pass", "metadata": {}, "score": 0.5},
                {"content": "def bar(): pass", "metadata": {}, "score": 0.3},
            ],
            "nodes_visited": [],
        }

        # Patch ONLY the pipeline factory — everything else is irrelevant
        with patch(
            "app.graph.agents.code_agent.get_pipeline_factory_cached",
            return_value=mock_factory,
        ):
            result = await code_rerank_node(state)

        assert "reranked_chunks" in result
        assert len(result["reranked_chunks"]) >= 1
        # Confirm reranker was called — not LLM, not memory, not prompts
        mock_reranker.rerank.assert_called_once()
        # Confirm no LLMClientFactory / MemoryFactory methods were called
        mock_factory.get_llm.assert_not_called()
        mock_factory.get_memory_manager.assert_not_called()

    @pytest.mark.asyncio
    async def test_code_rerank_node_falls_back_when_reranker_none(self):
        """When get_reranker() returns None, node must return top-5 pass-through."""
        from app.graph.agents.code_agent import code_rerank_node

        mock_factory = MagicMock()
        mock_factory.get_reranker.return_value = None

        chunks = [
            {"content": f"chunk {i}", "metadata": {}, "score": float(i)}
            for i in range(8)
        ]
        state = {"query": "test", "retrieved_chunks": chunks, "nodes_visited": []}

        with patch(
            "app.graph.agents.code_agent.get_pipeline_factory_cached",
            return_value=mock_factory,
        ):
            result = await code_rerank_node(state)

        assert len(result["reranked_chunks"]) == 5  # top-5 pass-through


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dependency function wiring
# ─────────────────────────────────────────────────────────────────────────────

class TestDependencyFunctions:
    """Verify each dep function returns the right object from its factory."""

    def test_get_retriever_engine_dep(self):
        from app.services.scoped_factories import get_retriever_engine_dep, RetrievalFactory
        RetrievalFactory.reset()
        try:
            mock_retriever = MagicMock()
            mock_rf = MagicMock()
            mock_rf.get_retriever_engine.return_value = mock_retriever
            with patch.object(RetrievalFactory, "get_instance", return_value=mock_rf):
                result = get_retriever_engine_dep()
            assert result is mock_retriever
        finally:
            RetrievalFactory.reset()

    def test_get_llm_dep(self):
        from app.services.scoped_factories import get_llm_dep, LLMClientFactory
        LLMClientFactory.reset()
        try:
            mock_llm = MagicMock()
            mock_llmf = MagicMock()
            mock_llmf.get_llm.return_value = mock_llm
            with patch.object(LLMClientFactory, "get_instance", return_value=mock_llmf):
                result = get_llm_dep()
            assert result is mock_llm
        finally:
            LLMClientFactory.reset()

    def test_get_memory_manager_dep_returns_none_safely(self):
        from app.services.scoped_factories import get_memory_manager_dep, MemoryFactory
        MemoryFactory.reset()
        try:
            mock_memf = MagicMock()
            mock_memf.get_memory_manager.return_value = None
            with patch.object(MemoryFactory, "get_instance", return_value=mock_memf):
                result = get_memory_manager_dep()
            assert result is None
        finally:
            MemoryFactory.reset()
