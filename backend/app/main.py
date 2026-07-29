"""
CodeLens_AI FastAPI Application Entry Point

Main application setup and configuration.
Equivalent to NestJS main.ts with app initialization.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

# Centralized debug logger — installs the loguru sink + stdlib bridge as a
# side-effect of import, so every subsequent ``logging.getLogger(__name__)``
# call across the codebase gets the colourised, session-id-aware format.
from app.core.logger import logger as _flow_logger  # noqa: F401  (side-effect import)

# Import database config from consolidated location
from app.core.config import get_engine, get_session_local, close_db, get_settings
from app.models.database import Base
from app.services.startup_service import StartupService
from app.routes import auth, ingest

# Hardening: Import token blacklist & rate limiter
from app.auth.token_blacklist import get_token_blacklist_manager
from app.middleware.rate_limiter import get_rate_limiter
from app.middleware.exception_handler import (
    LoggingMiddleware,
    GlobalExceptionHandler,
    RateLimitExceptionHandler,
    AuthenticationExceptionHandler,
    TokenRevocationExceptionHandler
)

# Import chat API — v1 compat + v2 LangGraph streaming endpoints
try:
    from app.api import chat as chat_api
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Chat API not available — check app/api/chat.py for import errors")
    chat_api = None


# ==================== Logging ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Silence noisy psycopg connection-pool retries when Postgres isn't
# reachable locally (expected in dev — the pool operates in disabled mode).
logging.getLogger("psycopg.pool").setLevel(logging.ERROR)
logging.getLogger("psycopg").setLevel(logging.ERROR)


# ==================== Startup/Shutdown Events ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    
    Startup:
    - Initialize database tables
    - Create default roles and permissions
    - Create superadmin user
    - Initialize token blacklist manager (for JWT revocation)
    - Initialize rate limiter (for brute-force protection)
    
    Shutdown:
    - Close database connections
    - Cleanup managers
    """
    # ===== STARTUP =====
    logger.info("🚀 Starting CodeLens_AI Backend...")
    
    # Get settings
    settings = get_settings()
    
    # Initialize enterprise security managers
    logger.info("🔐 Initializing security managers...")
    try:
        # Initialize token blacklist manager (JWT revocation system)
        app.state.token_blacklist = get_token_blacklist_manager(settings.redis_url)
        logger.info("✓ Token blacklist manager initialized")
        
        # Initialize rate limiter (brute-force protection)
        app.state.rate_limiter = get_rate_limiter(settings.redis_url)
        logger.info("✓ Rate limiter initialized")
    except Exception as e:
        logger.error(f"✗ Security manager initialization failed: {str(e)}")
        raise
    
    # Initialize database
    logger.info("📦 Initializing database...")
    try:
        # Create all tables
        engine = get_engine()
        Base.metadata.create_all(bind=engine)
        logger.info("✓ Database tables created/verified")
        
        # Initialize default data
        SessionLocal = get_session_local()
        db = SessionLocal()
        try:
            result = StartupService.initialize_database(db)
            StartupService.log_initialization_result(result)
        finally:
            db.close()
        
        logger.info("✓ Database initialization completed")
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {str(e)}")
        raise
    
    logger.info("✓ Backend startup completed successfully")

    # Pre-warm the RAG pipeline so the first chat request is instant.
    # This loads ChromaDB, BM25 index, embedding model, and reranker model
    # once at startup rather than on the first HTTP request.
    logger.info("🔥 Pre-warming RAG pipeline (reranker + embeddings)...")
    try:
        from app.services.pipeline_factory import get_pipeline_factory_cached
        get_pipeline_factory_cached()
        logger.info("✓ RAG pipeline pre-warmed — first chat will be fast\n")
    except Exception as e:
        logger.warning(f"⚠️  RAG pipeline pre-warm failed (will init on first request): {e}\n")

    # ── Startup sanity checks ─────────────────────────────────────────────
    # Runs all test_* suites under app/tests/ and logs a structured report.
    # Controlled by STARTUP_TESTS_ENABLED env var (default: true).
    # Failures are logged but never crash the server — a broken optional
    # component shouldn't take the whole service down on startup.
    try:
        from app.tests.runner import StartupTestRunner
        await StartupTestRunner.run_all()
    except Exception as e:
        logger.warning(f"⚠️  Startup test runner failed unexpectedly: {e}")

    yield
    
    # ===== SHUTDOWN =====
    logger.info("\n🛑 Shutting down CodeLens_AI Backend...")
    try:
        close_db()
        logger.info("✓ Database connections closed")
    except Exception as e:
        logger.error(f"✗ Error during shutdown: {str(e)}")

    # P2 #7: dispose the shared psycopg connection pool
    try:
        from app.core.database import close_pg_pool
        close_pg_pool()
    except Exception as e:
        logger.warning(f"psycopg pool close failed: {e}")

    logger.info("✓ Backend shutdown completed\n")


# ==================== FastAPI Application ====================

app = FastAPI(
    title="CodeLens_AI API",
    description="Production-grade RAG system for code documentation with AI streaming",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)


# ==================== Middleware Stack ====================

# Observability is handled by Langfuse (LLM tracing, span-level latency,
# token/cost tracking, and online evaluation) plus OpenTelemetry traces
# exported to Jaeger. No in-process metrics middleware is required.

# Add logging middleware (tracks all requests/responses)
app.add_middleware(LoggingMiddleware)

# ==================== CORS Middleware ====================

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:4200").split(",")
cors_origins = [origin.strip() for origin in cors_origins]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["content-type"],
    max_age=600,
)

logger.info(f"✓ CORS configured for: {', '.join(cors_origins)}")


# ==================== Exception Handlers ====================

# Register enterprise exception handlers
app.add_exception_handler(HTTPException, GlobalExceptionHandler.http_exception_handler)
app.add_exception_handler(Exception, GlobalExceptionHandler.general_exception_handler)

# Optional: Register specialized handlers for rate limits and auth errors
# (These would catch custom exceptions thrown from dependencies)
try:
    from app.middleware.rate_limiter import RateLimitExceeded
    app.add_exception_handler(RateLimitExceeded, RateLimitExceptionHandler.handle_rate_limit_exceeded)
except ImportError:
    pass


# ==================== Health Check ====================

@app.get("/api/health", tags=["health"])
async def health_check():
    """
    Health check endpoint
    
    Used by load balancers and monitoring to verify API is running
    """
    return {
        "status": "healthy",
        "service": "CodeLens_AI API",
        "version": "0.1.0"
    }


@app.get("/api/v1/health", tags=["health"])
async def health_check_v1():
    """Health check endpoint (v1 namespace)"""
    return {
        "status": "healthy",
        "service": "CodeLens_AI API",
        "version": "0.1.0",
        "environment": os.getenv("ENVIRONMENT", "development")
    }


# ==================== Routes ====================

# Include authentication routes
app.include_router(auth.router)

# Include document ingestion routes
app.include_router(ingest.router)
logger.info("✓ Document Ingestion routes registered:")
logger.info("  - POST   /api/v1/ingest/documents")
logger.info("  - POST   /api/v1/ingest/url")
logger.info("  - GET    /api/v1/ingest/status")
logger.info("  - DELETE /api/v1/ingest/clear")

# Unified Chat API — registers both /api/v2/chat/stream (primary) and
# /api/v1/chat/stream (compat shim) from a single module.
if chat_api:
    app.include_router(chat_api.router_v2)
    app.include_router(chat_api.router_v1)
    logger.info("✓ Chat API registered:")
    logger.info("  - POST   /api/v2/chat/stream (LangGraph SSE, primary)")
    logger.info("  - POST   /api/v1/chat/stream (v1 compat shim)")
    logger.info("  - GET    /api/v1/chat/cache/status")
    logger.info("  - POST   /api/v1/chat/cache/clear")
    logger.info("  - GET    /api/v1/chat/history/{session_id}")

try:
    from app.api import checkpoints as checkpoints_api
    app.include_router(checkpoints_api.router)
    logger.info("✓ Checkpoints API registered:")
    logger.info("  - GET    /api/v2/sessions/{id}/checkpoints")
    logger.info("  - GET    /api/v2/sessions/{id}/state/{cp_id}")
    logger.info("  - GET    /api/v2/sessions/{id}/replay/{cp_id}")
    logger.info("  - POST   /api/v2/sessions/{id}/branch")
    logger.info("  - POST   /api/v2/sessions/{id}/resume")
except Exception as _cp_err:  # noqa: BLE001
    logger.warning("Checkpoints API not registered: %s", _cp_err)

logger.info("✓ Routes registered:")
logger.info("  - POST   /api/v1/auth/login")
logger.info("  - POST   /api/v1/auth/register")
logger.info("  - POST   /api/v1/auth/refresh")
logger.info("  - POST   /api/v1/auth/logout")
logger.info("  - POST   /api/v1/auth/change-password")
logger.info("  - GET    /api/v1/auth/me")
logger.info("  - GET    /api/health")
logger.info("  - GET    /api/v1/health")


# ==================== Root Endpoint ====================

@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint - API information
    """
    return {
        "name": "CodeLens_AI API",
        "version": "0.1.0",
        "description": "Production RAG system for code documentation",
        "docs": "/api/docs",
        "redoc": "/api/redoc",
        "health": "/api/health"
    }


# ==================== Not Found Handler ====================

@app.get("/api/{path:path}", tags=["api"])
async def api_not_found(path: str):
    """Catch-all for undefined API routes"""
    return JSONResponse(
        status_code=404,
        content={"error": f"Endpoint /api/{path} not found"}
    )


# ==================== Application Info ====================

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    environment = os.getenv("ENVIRONMENT", "development")
    reload = environment == "development"
    
    logger.info(f"Starting Uvicorn server on {host}:{port}")
    logger.info(f"Environment: {environment}")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
