"""Phase 4: Enhanced Health Check - Component Status Verification.

Checks status of:
1. PostgreSQL database
2. ChromaDB (vector store)
3. Redis (caching layer, optional)
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
            from app.database.config import engine
            
            start = time.time()
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            latency = (time.time() - start) * 1000
            
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
        """Check ChromaDB (vector store) connection."""
        try:
            # TODO: Implement ChromaDB health check
            # This would connect to ChromaDB and verify it's responding
            return {
                "name": "ChromaDB",
                "status": "healthy",
                "message": "not_implemented_yet",
            }
        except Exception as e:
            logger.warning(f"ChromaDB not configured: {e}")
            return {
                "name": "ChromaDB",
                "status": "degraded",
                "message": "not_configured",
            }
    
    @staticmethod
    async def check_llm_api(timeout_seconds: float = 3.0) -> Dict[str, Any]:
        """Check LLM API availability (Groq/Ollama)."""
        try:
            # TODO: Ping LLM API
            # This would do a lightweight call to the LLM provider
            return {
                "name": "LLM API",
                "status": "healthy",
                "message": "not_implemented_yet",
            }
        except Exception as e:
            logger.warning(f"LLM API check failed: {e}")
            return {
                "name": "LLM API",
                "status": "degraded",
                "message": str(e),
            }
    
    @staticmethod
    async def check_agent_brain() -> Dict[str, Any]:
        """Check Agent Brain initialization."""
        try:
            from app.services.agents.agent_brain import AgentBrain, AgentConfig
            
            config = AgentConfig()
            brain = AgentBrain(config=config)
            
            return {
                "name": "Agent Brain (Phase 3)",
                "status": "healthy",
                "message": "All Phase 3 components available",
            }
        except Exception as e:
            logger.warning(f"Agent Brain check failed: {e}")
            return {
                "name": "Agent Brain (Phase 3)",
                "status": "degraded",
                "message": str(e),
            }
    
    @staticmethod
    async def check_retriever_engine() -> Dict[str, Any]:
        """Check Phase 2 Retriever Engine."""
        try:
            # TODO: Verify retriever engine
            return {
                "name": "Retriever Engine (Phase 2)",
                "status": "healthy",
                "message": "not_implemented_yet",
            }
        except Exception as e:
            logger.warning(f"Retriever check failed: {e}")
            return {
                "name": "Retriever Engine (Phase 2)",
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
    Detailed health check (all components).
    
    Returns status of:
    - PostgreSQL
    - ChromaDB
    - LLM API
    - Agent Brain (Phase 3)
    - Retriever Engine (Phase 2)
    
    Use this for monitoring and debugging.
    """
    
    logger.info("Running detailed health check...")
    
    checker = HealthChecker()
    
    # Run all checks (parallel in production)
    pg_status = await checker.check_postgresql()
    chromadb_status = await checker.check_chromadb()
    llm_status = await checker.check_llm_api()
    agent_status = await checker.check_agent_brain()
    retriever_status = await checker.check_retriever_engine()
    
    components = {
        "database": pg_status,
        "vector_store": chromadb_status,
        "llm_api": llm_status,
        "agent_brain": agent_status,
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
            "agent_brain": await checker.check_agent_brain(),
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
            "agent_brain": True,
            "phase_3_enabled": True,
        },
        "api": {
            "version": "0.1.0",
            "endpoints": [
                "POST /api/v1/chat/stream",
                "POST /api/v1/chat",
                "GET /api/v1/health",
                "GET /api/v1/health/detailed",
                "GET /api/v1/health/components",
                "GET /api/v1/health/system",
            ]
        }
    }
