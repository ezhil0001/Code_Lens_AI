"""Production-grade RAG Pipeline Factory - Single Source of Truth.

Centralizes all component initialization and dependency injection.
Ensures all phases are properly wired and configured.
"""

import logging
from typing import Optional, Dict, Any
from functools import lru_cache
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file so os.getenv() works correctly
load_dotenv()

# Fix SSL certificate path — Anaconda sometimes sets SSL_CERT_FILE to a non-existent path
# This causes httpx (used by HuggingFace) to fail when downloading models
if "SSL_CERT_FILE" in os.environ and os.environ.get("SSL_CERT_FILE"):
    cert_path = os.environ["SSL_CERT_FILE"]
    if not Path(cert_path).exists():
        # Use anaconda's bundled certificate
        anaconda_cert = "/opt/anaconda3/ssl/cacert.pem"
        if Path(anaconda_cert).exists():
            os.environ["SSL_CERT_FILE"] = anaconda_cert
        else:
            # Last resort: unset it and let httpx find certificates via system paths
            del os.environ["SSL_CERT_FILE"]

logger = logging.getLogger(__name__)

# Centralized debug logger — emits [RETRIEVER_START] / init traces.
from app.core.logger import logger as flow_logger, timed, log_step, log_success

class RAGPipelineFactory:
    """Factory for creating fully-configured RAG pipeline with all phases integrated.
    
    Responsibilities:
    1. Initialize Phase 2 Retriever Engine with dynamic weights
    2. Initialize Phase 3 Agent Components (Router, Memory, Prompt Builder)
    3. Wire all components together
    4. Manage singleton lifecycle
    5. Provide dependency injection for FastAPI
    """
    
    _instance: Optional['RAGPipelineFactory'] = None
    _lock = None
    
    def __init__(self):
        """Initialize all pipeline components."""
        logger.info("🏗️  Initializing RAG Pipeline Factory...")
        
        # Phase 1-2: Retrieval Engine
        logger.info("  ├─ Initializing Phase 2: Retrieval Engine...")
        self.retriever = self._init_retriever()
        logger.info("  │  ✅ Retriever initialized with dynamic weights enabled")
        
        # Phase 3: Orchestration Components
        logger.info("  ├─ Initializing Phase 3: Agent Brain Components...")
        
        # Initialize Router
        from app.services.agents.agentic_router import AgenticRouter, RoutingConfig
        self.router = AgenticRouter(config=RoutingConfig())
        logger.info("  │  ✅ Agentic Router initialized")
        
        # Initialize Few-Shot Example Selector
        from app.services.agents.semantic_example_selector import SemanticExampleSelector
        # P1 FIX: inject the same HuggingFaceEmbeddings used by retrieval so the
        # selector uses TRUE cosine similarity instead of falling back to TF-IDF.
        # P2 #7 FIX: reuse the SHARED singleton embedder from app.core.database
        # so we don't load the model multiple times.
        embedding_model = None
        try:
            from app.core.database import get_embedder
            embedding_model = get_embedder()
            logger.info("  │  ✅ Singleton embedding model injected into example selector")
        except Exception as e:
            logger.warning(f"  │  ⚠️ Could not load embeddings for selector: {e}")

        self.example_selector = SemanticExampleSelector(embedding_model=embedding_model)
        logger.info("  │  ✅ Semantic Example Selector initialized")
        
        # Initialize Prompt Builder
        from app.services.agents.few_shot_prompt import FewShotPromptBuilder
        self.prompt_builder = FewShotPromptBuilder()
        logger.info("  │  ✅ Few-Shot Prompt Builder initialized")
        
        # Initialize Memory Manager (LangChain PostgresChatMessageHistory)
        try:
            from app.services.agents.langchain_memory_manager import ChatMemoryManager
            self.memory_manager = ChatMemoryManager()
            logger.info("  │  ✅ Chat Memory Manager (LangChain Postgres) initialized")
        except Exception as _mem_err:
            logger.warning(
                f"  │  ⚠️  Chat Memory Manager unavailable ({_mem_err}) — "
                "continuing without persistent chat history"
            )
            self.memory_manager = None
        
        # Initialize LLM Client
        logger.info("  ├─ Initializing LLM Client...")
        self.llm_client = self._init_llm_client()
        logger.info("  │  ✅ LLM Client initialized")
        
        # Initialize Agent Brain with all components
        from app.services.agents.agent_brain import AgentBrain, AgentConfig
        
        config = AgentConfig(
            enable_retrieval=True,
            retrieve_k=5,
            use_dynamic_weights=True,
            enable_few_shot=True,
            enable_memory=True,
            enable_streaming=True,
        )
        
        self.agent_brain = AgentBrain(
            config=config,
            retriever_engine=self.retriever,
            router=self.router,
            example_selector=self.example_selector,
            prompt_builder=self.prompt_builder,
            memory_manager=self.memory_manager,
            llm_client=self.llm_client,
        )
        logger.info("  └─ ✅ Agent Brain fully initialized with all components")
        
        # Phase 5: Observability
        logger.info("  ├─ Initializing Phase 5: Observability...")
        self._init_observability()
        logger.info("  │  ✅ OpenTelemetry configured")
        
        logger.info("✅ RAG Pipeline Factory initialized successfully\n")
    
    @staticmethod
    def _init_retriever():
        """Initialize Phase 2 Retrieval Engine with all components.

        Fail-loud: never falls back to a mock — a broken retriever silently
        defeats the entire RAG pipeline.
        """
        from app.services.retrieval.retriever_engine import RetrieverEngine
        from app.services.ingestion.ingestion_service import IngestionService

        log_step("[RETRIEVER_START]", "Building EnsembleRetriever (vector_w=0.6, bm25_w=0.4)")

        with timed("[RETRIEVER_START]") as ctx:
            flow_logger.bind(tag="[RETRIEVER_START]").debug("loading ChromaDB collection…")
            chroma_collection = IngestionService.get_chroma_collection()

            flow_logger.bind(tag="[RETRIEVER_START]").debug("loading documents for BM25…")
            documents = IngestionService.load_documents_for_bm25()

            flow_logger.bind(tag="[RETRIEVER_START]").debug("loading parent store (PDR)…")
            parent_store = IngestionService.load_parent_store()

            if not documents:
                flow_logger.bind(tag="[RETRIEVER_START]").warning(
                    "No documents in ChromaDB — retriever will return empty results until ingestion runs."
                )

            retriever = RetrieverEngine(
                chroma_collection=chroma_collection,
                documents_for_bm25=documents,
                parent_store=parent_store,
                enable_query_expansion=True,
                enable_reranking=True,
                vector_weight=0.6,
                bm25_weight=0.4,
            )
            ctx["bm25_docs"] = len(documents) if documents else 0

        log_success("[RETRIEVER_START]", "Phase 2 retriever fully configured (hybrid + rerank + PDR)")
        return retriever
    
    @staticmethod
    def _init_llm_client():
        """Initialize LLM client — switch via .env only, no code changes needed.

        LLM_PROVIDER=groq   → GROQ_MODEL=llama-3.1-70b-versatile
        LLM_PROVIDER=openai → OPENAI_MODEL=gpt-4o
        LLM_PROVIDER=ollama → OLLAMA_MODEL=llama3.1
        """
        llm_provider = os.getenv("LLM_PROVIDER", "groq").lower()
        logger.info(f"  ├─ LLM Provider: {llm_provider}")

        # ── GROQ ─────────────────────────────────────────────
        if llm_provider == "groq":
            try:
                from langchain_groq import ChatGroq
                model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
                llm = ChatGroq(
                    api_key=os.getenv("GROQ_API_KEY", ""),
                    model=model,
                    temperature=0.7,
                    max_tokens=2048,
                )
                logger.info(f"✅ Groq ready — model={model}")
                return llm
            except Exception as e:
                logger.error(f"❌ Groq init failed: {e}")
                raise

        # ── OPENAI ────────────────────────────────────────────
        if llm_provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
                model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                llm = ChatOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY", ""),
                    model=model,
                    temperature=0.7,
                    max_tokens=2048,
                )
                logger.info(f"✅ OpenAI ready — model={model}")
                return llm
            except Exception as e:
                logger.error(f"❌ OpenAI init failed: {e}")
                raise

        # ── OLLAMA ────────────────────────────────────────────
        if llm_provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
                model = os.getenv("OLLAMA_MODEL", "llama3.1")
                llm = ChatOllama(
                    model=model,
                    base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
                    temperature=0.7,
                )
                logger.info(f"✅ Ollama ready — model={model}")
                return llm
            except Exception as e:
                logger.error(f"❌ Ollama init failed: {e}")
                raise

        raise ValueError(
            f"❌ Unknown LLM_PROVIDER='{llm_provider}'. "
            f"Supported: groq, openai, ollama"
        )
    
    @staticmethod
    def _init_observability():
        """Initialize OpenTelemetry for tracing and metrics."""
        try:
            from app.observability.otel_config import (
                setup_tracer_provider,
                setup_meter_provider,
                setup_instrumentation,
            )
            
            logger.debug("    Setting up Jaeger tracer...")
            setup_tracer_provider()
            
            logger.debug("    Setting up Prometheus metrics...")
            setup_meter_provider()
            
            logger.debug("    Setting up auto-instrumentation...")
            setup_instrumentation()
            
            logger.debug("    ✅ OpenTelemetry fully configured")
        
        except ImportError as e:
            logger.warning(f"OpenTelemetry not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize observability: {e}")
    
    def get_agent_brain(self):
        """Get the configured Agent Brain instance."""
        if self.agent_brain is None:
            raise RuntimeError("Agent Brain not initialized")
        return self.agent_brain
    
    def get_retriever(self):
        """Get the Retrieval Engine."""
        return self.retriever

    def get_retriever_engine(self):
        """Alias for get_retriever() — returns the RetrieverEngine instance."""
        return self.retriever

    def get_reranker(self):
        """Get the reranker component from the retriever (if available)."""
        return getattr(self.retriever, 'reranker', None)

    def get_llm(self):
        """Get the configured LLM client."""
        return self.llm_client

    def get_router(self):
        """Get the Agentic Router."""
        return self.router
    
    def get_memory_manager(self):
        """Get the Chat Memory Manager."""
        return self.memory_manager
    
    @classmethod
    def get_instance(cls) -> 'RAGPipelineFactory':
        """Get or create singleton instance (thread-safe)."""
        if cls._instance is None:
            if cls._lock is None:
                import threading
                cls._lock = threading.Lock()
            
            with cls._lock:
                if cls._instance is None:
                    cls._instance = RAGPipelineFactory()
        
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        cls._instance = None
        logger.info("Pipeline factory reset")


# Module-level factory instance (lazy initialized)
_pipeline_factory: Optional[RAGPipelineFactory] = None


def get_pipeline_factory() -> RAGPipelineFactory:
    """Get or create pipeline factory instance (FastAPI dependency).
    
    This is the primary injection point for all API endpoints.
    """
    global _pipeline_factory
    if _pipeline_factory is None:
        _pipeline_factory = RAGPipelineFactory.get_instance()
    return _pipeline_factory


@lru_cache(maxsize=1)
def get_pipeline_factory_cached() -> RAGPipelineFactory:
    """Cached pipeline factory (created once per application instance).
    
    Use this in FastAPI dependencies for better performance.
    """
    return RAGPipelineFactory.get_instance()


def get_agent_brain_dependency():
    """FastAPI dependency for Agent Brain.
    
    Usage in routes:
        @router.post("/chat")
        async def chat(agent: AgentBrain = Depends(get_agent_brain_dependency)):
            ...
    """
    factory = get_pipeline_factory_cached()
    return factory.get_agent_brain()
