"""Production-grade RAG Pipeline Factory - Single Source of Truth.

Centralizes all component initialization and dependency injection.
Ensures all phases are properly wired and configured.

Refactor note (2026-06-28):
    RAGPipelineFactory has been split into four scoped factories in
    ``app.services.scoped_factories``:

      RetrievalFactory  — RetrieverEngine + reranker
      LLMClientFactory  — LLM client
      MemoryFactory     — ChatMemoryManager / STM / LTM
      PromptFactory     — FewShotPromptBuilder + SemanticExampleSelector

    New code should import from ``app.services.scoped_factories`` and use
    ``Depends(get_retrieval_factory)`` etc. in FastAPI route handlers.

    RAGPipelineFactory is kept here as a **compatibility shim** — it
    delegates every accessor to the scoped singletons so the 15+
    call-sites that do ``get_pipeline_factory_cached().get_retriever_engine()``
    continue to work without modification.

    Nothing in this module should prevent RAGPipelineFactory from being
    removed once all callers have been migrated to the scoped factories.
"""

import logging
from typing import Optional, Dict, Any
from functools import lru_cache
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Fix SSL certificate path — Anaconda sometimes sets SSL_CERT_FILE to a non-existent path
if "SSL_CERT_FILE" in os.environ and os.environ.get("SSL_CERT_FILE"):
    cert_path = os.environ["SSL_CERT_FILE"]
    if not Path(cert_path).exists():
        anaconda_cert = "/opt/anaconda3/ssl/cacert.pem"
        if Path(anaconda_cert).exists():
            os.environ["SSL_CERT_FILE"] = anaconda_cert
        else:
            del os.environ["SSL_CERT_FILE"]

logger = logging.getLogger(__name__)

from app.core.logger import logger as flow_logger, timed, log_step, log_success

# Re-export scoped factories so new code can import from either module.
from app.services.scoped_factories import (  # noqa: E402
    RetrievalFactory,
    LLMClientFactory,
    MemoryFactory,
    PromptFactory,
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


class RAGPipelineFactory:
    """Compatibility shim — delegates every accessor to the four scoped factories.

    All four sub-factories (RetrievalFactory, LLMClientFactory, MemoryFactory,
    PromptFactory) are singletons; this class simply holds references to their
    products so legacy call-sites continue to work without modification.
    """

    _instance: Optional["RAGPipelineFactory"] = None
    _lock = None

    def __init__(self) -> None:
        logger.info("🏗️  Initializing RAG Pipeline Factory (compatibility shim)…")

        # ── Retrieval (delegates to RetrievalFactory) ────────────────────────
        logger.info("  ├─ Retrieval Engine…")
        self._retrieval_factory = RetrievalFactory.get_instance()
        self.retriever = self._retrieval_factory.get_retriever_engine()
        logger.info("  │  ✅ Retriever ready")

        # ── LLM (delegates to LLMClientFactory) ─────────────────────────────
        logger.info("  ├─ LLM Client…")
        self._llm_factory = LLMClientFactory.get_instance()
        self.llm_client = self._llm_factory.get_llm()
        logger.info("  │  ✅ LLM Client ready")

        # ── Memory (delegates to MemoryFactory) ──────────────────────────────
        logger.info("  ├─ Memory Manager…")
        self._memory_factory = MemoryFactory.get_instance()
        self.memory_manager = self._memory_factory.get_memory_manager()
        logger.info("  │  ✅ Memory Manager ready")

        # ── Prompt (delegates to PromptFactory) ──────────────────────────────
        logger.info("  ├─ Prompt / Example Selector…")
        self._prompt_factory = PromptFactory.get_instance()
        self.example_selector = self._prompt_factory.get_example_selector()
        self.prompt_builder = self._prompt_factory.get_prompt_builder()
        logger.info("  │  ✅ Prompt components ready")

        logger.info("✅ RAGPipelineFactory (shim) initialised\n")

    # ── Accessor API (unchanged surface for legacy callers) ─────────────────

    def get_retriever(self):
        return self.retriever

    def get_retriever_engine(self):
        return self.retriever

    def get_reranker(self):
        return self._retrieval_factory.get_reranker()

    def get_llm(self):
        return self.llm_client

    def get_memory_manager(self):
        return self.memory_manager

    def refresh_bm25_index(self) -> None:
        """Delegate BM25 rebuild to RetrievalFactory."""
        self._retrieval_factory.refresh_bm25_index()

    # ── Singleton ────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "RAGPipelineFactory":
        if cls._instance is None:
            if cls._lock is None:
                import threading
                cls._lock = threading.Lock()
            with cls._lock:
                if cls._instance is None:
                    cls._instance = RAGPipelineFactory()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
        logger.info("RAGPipelineFactory (shim) reset")


# ── Module-level helpers (unchanged public API) ──────────────────────────────

_pipeline_factory: Optional[RAGPipelineFactory] = None


def get_pipeline_factory() -> RAGPipelineFactory:
    """Get or create the pipeline factory singleton (FastAPI dependency)."""
    global _pipeline_factory
    if _pipeline_factory is None:
        _pipeline_factory = RAGPipelineFactory.get_instance()
    return _pipeline_factory


@lru_cache(maxsize=1)
def get_pipeline_factory_cached() -> RAGPipelineFactory:
    """Cached pipeline factory — one instance per process."""
    return RAGPipelineFactory.get_instance()


def get_agent_brain_dependency():
    """FastAPI dependency for AgentBrain."""
    return get_pipeline_factory_cached().get_agent_brain()

    
