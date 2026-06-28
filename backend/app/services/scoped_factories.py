"""Scoped DI factories — each exposes exactly one responsibility.

These replace the monolithic RAGPipelineFactory "God object" for new code.
RAGPipelineFactory is kept as a compatibility shim that delegates to these.

Four factories:
  RetrievalFactory   — RetrieverEngine + reranker
  LLMClientFactory   — LLM client (Groq / OpenAI / Ollama)
  MemoryFactory      — STM / LTM / ChatMemoryManager
  PromptFactory      — FewShotPromptBuilder + SemanticExampleSelector

FastAPI dependency functions (one per factory):
  get_retrieval_factory()   → RetrievalFactory singleton
  get_llm_client_factory()  → LLMClientFactory singleton
  get_memory_factory()      → MemoryFactory singleton
  get_prompt_factory()      → PromptFactory singleton

  Convenience pass-throughs (mirrors of the factory getters):
  get_retriever_engine_dep()  → RetrieverEngine
  get_reranker_dep()          → reranking_engine
  get_llm_dep()               → LLM client
  get_memory_manager_dep()    → ChatMemoryManager (may be None)
  get_prompt_builder_dep()    → FewShotPromptBuilder
  get_example_selector_dep()  → SemanticExampleSelector
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RetrievalFactory
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalFactory:
    """Owns the RetrieverEngine and reranker — nothing else.

    Thread-safe lazy singleton via double-checked locking.
    """

    _instance: Optional["RetrievalFactory"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        from app.core.logger import timed, log_step, log_success, logger as flow_logger
        from app.services.retrieval.retriever_engine import RetrieverEngine
        from app.services.ingestion.ingestion_service import IngestionService

        log_step("[RETRIEVAL_FACTORY]", "Building RetrieverEngine…")
        with timed("[RETRIEVAL_FACTORY]") as ctx:
            chroma_collection = IngestionService.get_chroma_collection()
            documents = IngestionService.load_documents_for_bm25()
            parent_store = IngestionService.load_parent_store()

            if not documents:
                flow_logger.bind(tag="[RETRIEVAL_FACTORY]").warning(
                    "No documents in ChromaDB — retriever returns empty results until ingestion runs."
                )

            self._retriever = RetrieverEngine(
                chroma_collection=chroma_collection,
                documents_for_bm25=documents,
                parent_store=parent_store,
                enable_query_expansion=True,
                enable_reranking=True,
                vector_weight=0.6,
                bm25_weight=0.4,
            )
            ctx["bm25_docs"] = len(documents) if documents else 0

        log_success("[RETRIEVAL_FACTORY]", "RetrieverEngine ready (hybrid + rerank + PDR)")
        logger.info("✅ RetrievalFactory initialised")

    # ── public API ──────────────────────────────────────────────────────────

    def get_retriever_engine(self):
        """Return the RetrieverEngine instance."""
        return self._retriever

    def get_reranker(self):
        """Return the BGE reranking engine (may be None if unavailable)."""
        return getattr(self._retriever, "reranking_engine", None)

    def refresh_bm25_index(self) -> None:
        """Rebuild the in-memory BM25 index from the current ChromaDB corpus."""
        try:
            from app.services.ingestion.ingestion_service import IngestionService
            docs = IngestionService.load_documents_for_bm25()
            logger.info(f"[BM25_REBUILD] Rebuilding with {len(docs)} docs…")
            self._retriever.refresh_bm25_index(docs)
            logger.info("[BM25_REBUILD] ✅ done")
        except Exception as err:
            logger.error(f"[BM25_REBUILD] ❌ {err}", exc_info=True)

    # ── singleton ────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "RetrievalFactory":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton — for use in tests only."""
        with cls._lock:
            cls._instance = None


# ─────────────────────────────────────────────────────────────────────────────
# LLMClientFactory
# ─────────────────────────────────────────────────────────────────────────────

class LLMClientFactory:
    """Owns the LLM client singleton — nothing else.

    Provider selected via LLM_PROVIDER env var (groq / openai / ollama).
    """

    _instance: Optional["LLMClientFactory"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._llm = self._build_llm()
        logger.info("✅ LLMClientFactory initialised")

    # ── build ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_llm():
        provider = os.getenv("LLM_PROVIDER", "groq").lower()
        logger.info(f"[LLM_FACTORY] provider={provider}")

        if provider == "groq":
            from langchain_groq import ChatGroq  # type: ignore
            model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
            llm = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY", ""),
                model=model,
                temperature=0.7,
                max_tokens=2048,
            )
            logger.info(f"✅ Groq LLM ready — model={model}")
            return llm

        if provider == "openai":
            from langchain_openai import ChatOpenAI  # type: ignore
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            llm = ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model=model,
                temperature=0.7,
                max_tokens=2048,
            )
            logger.info(f"✅ OpenAI LLM ready — model={model}")
            return llm

        if provider == "ollama":
            from langchain_ollama import ChatOllama  # type: ignore
            model = os.getenv("OLLAMA_MODEL", "llama3.1")
            llm = ChatOllama(
                model=model,
                base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
                temperature=0.7,
            )
            logger.info(f"✅ Ollama LLM ready — model={model}")
            return llm

        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider}'. Supported: groq, openai, ollama"
        )

    # ── public API ──────────────────────────────────────────────────────────

    def get_llm(self):
        """Return the configured LLM client."""
        return self._llm

    # ── singleton ────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "LLMClientFactory":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None


# ─────────────────────────────────────────────────────────────────────────────
# MemoryFactory
# ─────────────────────────────────────────────────────────────────────────────

class MemoryFactory:
    """Owns all memory components: STM window, LTM store, ChatMemoryManager.

    ChatMemoryManager may be None when Postgres is unavailable; callers must
    handle that case gracefully.
    """

    _instance: Optional["MemoryFactory"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._chat_memory_manager = self._build_chat_memory()
        logger.info("✅ MemoryFactory initialised")

    @staticmethod
    def _build_chat_memory():
        try:
            from app.services.agents.langchain_memory_manager import ChatMemoryManager  # type: ignore
            mgr = ChatMemoryManager()
            logger.info("✅ ChatMemoryManager (LangChain Postgres) ready")
            return mgr
        except Exception as err:
            logger.warning(
                f"ChatMemoryManager unavailable ({err}) — continuing without persistent history"
            )
            return None

    # ── public API ──────────────────────────────────────────────────────────

    def get_memory_manager(self):
        """Return the ChatMemoryManager instance (may be None)."""
        return self._chat_memory_manager

    # ── singleton ────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "MemoryFactory":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None


# ─────────────────────────────────────────────────────────────────────────────
# PromptFactory
# ─────────────────────────────────────────────────────────────────────────────

class PromptFactory:
    """Owns FewShotPromptBuilder and SemanticExampleSelector."""

    _instance: Optional["PromptFactory"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        from app.services.agents.few_shot_prompt import FewShotPromptBuilder  # type: ignore
        from app.services.agents.semantic_example_selector import SemanticExampleSelector  # type: ignore

        # Reuse the shared singleton embedder so we don't load a second model copy.
        embedding_model = None
        try:
            from app.core.database import get_embedder
            embedding_model = get_embedder()
            logger.info("✅ Singleton embedder injected into SemanticExampleSelector")
        except Exception as err:
            logger.warning(f"Could not inject embedder into example selector: {err}")

        self._example_selector = SemanticExampleSelector(embedding_model=embedding_model)
        self._prompt_builder = FewShotPromptBuilder()
        logger.info("✅ PromptFactory initialised")

    # ── public API ──────────────────────────────────────────────────────────

    def get_example_selector(self):
        """Return the SemanticExampleSelector instance."""
        return self._example_selector

    def get_prompt_builder(self):
        """Return the FewShotPromptBuilder instance."""
        return self._prompt_builder

    # ── singleton ────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "PromptFactory":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI dependency functions — one per factory
# ─────────────────────────────────────────────────────────────────────────────

def get_retrieval_factory() -> RetrievalFactory:
    """FastAPI ``Depends`` target → RetrievalFactory singleton."""
    return RetrievalFactory.get_instance()


def get_llm_client_factory() -> LLMClientFactory:
    """FastAPI ``Depends`` target → LLMClientFactory singleton."""
    return LLMClientFactory.get_instance()


def get_memory_factory() -> MemoryFactory:
    """FastAPI ``Depends`` target → MemoryFactory singleton."""
    return MemoryFactory.get_instance()


def get_prompt_factory() -> PromptFactory:
    """FastAPI ``Depends`` target → PromptFactory singleton."""
    return PromptFactory.get_instance()


# ── Convenience pass-throughs (skip instantiating the factory in the caller) ─

def get_retriever_engine_dep():
    """FastAPI dependency → RetrieverEngine directly."""
    return get_retrieval_factory().get_retriever_engine()


def get_reranker_dep():
    """FastAPI dependency → reranker (may be None)."""
    return get_retrieval_factory().get_reranker()


def get_llm_dep():
    """FastAPI dependency → LLM client."""
    return get_llm_client_factory().get_llm()


def get_memory_manager_dep():
    """FastAPI dependency → ChatMemoryManager (may be None)."""
    return get_memory_factory().get_memory_manager()


def get_prompt_builder_dep():
    """FastAPI dependency → FewShotPromptBuilder."""
    return get_prompt_factory().get_prompt_builder()


def get_example_selector_dep():
    """FastAPI dependency → SemanticExampleSelector."""
    return get_prompt_factory().get_example_selector()


# ─────────────────────────────────────────────────────────────────────────────
# reset_all — test helper
# ─────────────────────────────────────────────────────────────────────────────

def reset_all_scoped_factories() -> None:
    """Reset every scoped factory singleton.  Use in test teardown only."""
    RetrievalFactory.reset()
    LLMClientFactory.reset()
    MemoryFactory.reset()
    PromptFactory.reset()
    logger.info("[TEST] All scoped factory singletons reset")
