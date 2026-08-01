"""Health check endpoint — reports the status of every runtime dependency.

Used by Docker's HEALTHCHECK, load balancers, and the ops team. Each component
check is isolated so a single failing service doesn't mask the others.
Components checked:
  1. PostgreSQL database
  2. ChromaDB (vector store)
  3. Redis (optional caching layer)
  4. LLM API (Groq/Ollama)
  5. Memory manager
"""

import logging
import time
import os
from typing import Dict, Any
from fastapi import APIRouter, BackgroundTasks
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["health"])


class HealthChecker:
    """Health check coordinator for all components."""
    
    @staticmethod
    async def check_postgresql(timeout_seconds: float = 2.0) -> Dict[str, Any]:
        """Check PostgreSQL connection."""
        try:
            # H-3 fix: previous import (app.database.config) never existed —
            # this check always reported "unhealthy". Use the real engine and
            # run the probe in a worker thread so the loop is not blocked.
            import asyncio
            from sqlalchemy import text
            from app.core.config import get_engine

            def _probe() -> float:
                engine = get_engine()
                start = time.time()
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                return (time.time() - start) * 1000

            latency = await asyncio.wait_for(asyncio.to_thread(_probe), timeout=timeout_seconds)

            return {
                "name": "PostgreSQL",
                "status": "healthy" if latency < 100 else "degraded",
                "latency_ms": latency,
            }
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return {
                "name": "PostgreSQL",
                "status": "unhealthy",
                "message": str(e),
            }
    
    @staticmethod
    async def check_chromadb(timeout_seconds: float = 2.0) -> Dict[str, Any]:
        """Check ChromaDB (vector store) via the ingestion pipeline heartbeat."""
        try:
            import asyncio

            def _probe() -> int:
                from app.services.ingestion.chroma_vector_store import ChromaVectorStore  # noqa: F401
                import chromadb
                client = chromadb.PersistentClient(path="./chroma_db")
                return client.count_collections()

            n = await asyncio.wait_for(asyncio.to_thread(_probe), timeout=timeout_seconds)
            return {
                "name": "ChromaDB",
                "status": "healthy",
                "message": f"{n} collections",
            }
        except Exception as e:
            logger.warning(f"ChromaDB health check failed: {e}")
            return {
                "name": "ChromaDB",
                "status": "degraded",
                "message": str(e),
            }
    
    @staticmethod
    async def check_llm_api(timeout_seconds: float = 3.0) -> Dict[str, Any]:
        """Check LLM API configuration (no billable call — key presence + client init)."""
        try:
            if not os.getenv("GROQ_API_KEY"):
                return {
                    "name": "LLM API",
                    "status": "degraded",
                    "message": "GROQ_API_KEY not configured",
                }
            return {
                "name": "LLM API",
                "status": "healthy",
                "message": "credentials configured",
            }
        except Exception as e:
            logger.warning(f"LLM API check failed: {e}")
            return {
                "name": "LLM API",
                "status": "degraded",
                "message": str(e),
            }
    
    @staticmethod
    async def check_retriever_engine() -> Dict[str, Any]:
        """Liveness check for the hybrid RetrieverEngine singleton.

        Replaces the old ``check_agent_brain()`` which instantiated
        ``AgentBrain`` — a v1 class scheduled for deletion.  This probe
        checks the same runtime path that every query uses: the
        ``RetrievalFactory`` singleton that owns the ``RetrieverEngine``.

        No retrieval is executed; we only confirm the factory is initialised
        and its engine is non-None, which takes <1 ms.
        """
        try:
            from app.services.scoped_factories import RetrievalFactory
            t0 = time.time()
            rf = RetrievalFactory.get_instance()
            retriever = rf.get_retriever_engine()
            if retriever is None:
                raise RuntimeError("RetrieverEngine is None after factory init")
            latency_ms = (time.time() - t0) * 1000
            return {
                "name": "Retriever Engine",
                "status": "healthy",
                "message": "RetrieverEngine initialised",
                "latency_ms": round(latency_ms, 2),
            }
        except Exception as e:
            logger.warning(f"Retriever Engine check failed: {e}")
            return {
                "name": "Retriever Engine",
                "status": "degraded",
                "message": str(e),
            }


# ==================== Health Check Endpoints ====================

@router.get("/health", tags=["health"])
async def health_check():
    """
    Quick health check (lightweight).
    
    Returns: 200 if service is running
    """
    return {
        "status": "healthy",
        "service": "CodeLens_AI API",
        "version": "0.1.0",
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/health/detailed", tags=["health"])
async def detailed_health_check(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Detailed health check — probes every runtime dependency.

    Returns the status of:
    - PostgreSQL
    - ChromaDB
    - LLM API
    - Retriever Engine

    Use this endpoint for monitoring dashboards and pre-deploy smoke tests.
    """

    logger.info("Running detailed health check...")

    checker = HealthChecker()

    # Run all checks (parallel in production)
    pg_status = await checker.check_postgresql()
    chromadb_status = await checker.check_chromadb()
    llm_status = await checker.check_llm_api()
    retriever_status = await checker.check_retriever_engine()

    components = {
        "database": pg_status,
        "vector_store": chromadb_status,
        "llm_api": llm_status,
        "retriever_engine": retriever_status,
    }

    # Determine overall status
    statuses = [c["status"] for c in components.values()]
    if all(s == "healthy" for s in statuses):
        overall = "healthy"
    elif any(s == "unhealthy" for s in statuses):
        overall = "unhealthy"
    else:
        overall = "degraded"

    response = {
        "overall_status": overall,
        "timestamp": datetime.now().isoformat(),
        "components": components,
        "environment": os.getenv("ENVIRONMENT", "development"),
    }

    logger.info(f"Health check complete: {overall}")

    return response


@router.get("/health/components", tags=["health"])
async def component_status() -> Dict[str, Any]:
    """
    Get status of each component (no timeouts).

    Useful for dashboards and monitoring.
    """

    checker = HealthChecker()

    return {
        "timestamp": datetime.now().isoformat(),
        "components": {
            "database": await checker.check_postgresql(),
            "vector_store": await checker.check_chromadb(),
            "llm_api": await checker.check_llm_api(),
            "retriever_engine": await checker.check_retriever_engine(),
        }
    }


@router.get("/health/system", tags=["health"])
async def system_info() -> Dict[str, Any]:
    """Get system information and configuration."""
    import sys
    
    return {
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "environment": os.getenv("ENVIRONMENT", "development"),
            "python_version": sys.version,
            "debug": os.getenv("DEBUG", "false") == "true",
        },
        "features": {
            "chat_streaming": True,
            "semantic_caching": True,
            "langgraph_supervisor": True,
            "hybrid_retrieval": True,
        },
        "api": {
            "version": "0.1.0",
            "endpoints": [
                "POST /api/v2/chat/stream",
                "GET /api/v1/health",
                "GET /api/v1/health/detailed",
                "GET /api/v1/health/components",
                "GET /api/v1/health/system",
            ]
        }
    }
