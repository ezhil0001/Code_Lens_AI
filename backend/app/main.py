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

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CollectorRegistry, REGISTRY
    from prometheus_client import start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("prometheus-client not installed. Metrics will be unavailable.")

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

# Phase 4: Import chat API
try:
    from app.api import chat as chat_api
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Chat API not available (Phase 4)")
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

# Add Prometheus middleware first (if available)
#
# AUDIT FIX L1 + L3:
#   L1 — Use the route TEMPLATE (e.g. "/api/v1/repo/{repo_id}") instead of
#        request.url.path. Path parameters create unbounded label cardinality
#        and will OOM the Prometheus server.
#   L3 — asyncio.CancelledError (client disconnect, common with SSE) was being
#        recorded as status_code=500 → false-positive 5xx alerts. We now record
#        it as status_code="499" (nginx convention: client closed request) so
#        SLO calculations can exclude it cleanly.
#
# Also splits TTFB (time-to-first-byte; user-perceived latency) from total
# request duration (dominated by streaming-body length on SSE endpoints).
if PROMETHEUS_AVAILABLE:
    import asyncio as _asyncio
    from time import perf_counter
    from starlette.middleware.base import BaseHTTPMiddleware

    def _resolve_route_template(request) -> str:
        """Return the matched route template, falling back to a sentinel.

        Critical for label cardinality — `request.url.path` would explode the
        series count on any URL with path parameters.
        """
        route = request.scope.get("route")
        if route is not None and getattr(route, "path", None):
            return route.path
        # Unmatched routes (404s, OPTIONS, etc.) bucket into one series.
        return "unmatched"

    class PrometheusMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            method = request.method
            route_template = _resolve_route_template(request)

            active_connections.inc()
            start = perf_counter()
            ttfb_seconds: Optional[float] = None
            status_code = "200"
            outcome = "success"

            try:
                response = await call_next(request)
                # First-byte signal: response headers are ready, even if body
                # is still streaming. This is what users actually feel.
                ttfb_seconds = perf_counter() - start
                status_code = str(response.status_code)
                return response
            except _asyncio.CancelledError:
                # Client disconnected mid-stream (typical for SSE). Not a 5xx.
                status_code = "499"
                outcome = "client_disconnect"
                http_streams_cancelled.labels(endpoint=route_template).inc()
                raise
            except Exception:
                status_code = "500"
                outcome = "server_error"
                raise
            finally:
                duration = perf_counter() - start
                request_count.labels(
                    method=method,
                    endpoint=route_template,
                    status_code=status_code,
                    outcome=outcome,
                ).inc()
                request_duration.labels(
                    method=method,
                    endpoint=route_template,
                ).observe(duration)
                if ttfb_seconds is not None:
                    request_ttfb.labels(
                        method=method,
                        endpoint=route_template,
                    ).observe(ttfb_seconds)
                active_connections.dec()

    app.add_middleware(PrometheusMiddleware)

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


# ==================== Prometheus Metrics ====================

if PROMETHEUS_AVAILABLE:
    # ----- HTTP / transport metrics -----
    # AUDIT FIX L1: `endpoint` carries the route TEMPLATE, never raw paths.
    # AUDIT FIX L3: `outcome` distinguishes success / client_disconnect / server_error
    #               so SLO PromQL can exclude client-side cancels.
    request_count = Counter(
        'http_requests_total',
        'Total HTTP requests',
        ['method', 'endpoint', 'status_code', 'outcome']
    )

    # AUDIT BUCKET-TUNING: covers the realistic latency range for an SSE-heavy
    # backend (50ms health-checks → 120s long-streaming chats). Default buckets
    # bunch under 1s and make p95/p99 calculation lossy at RAG scale.
    request_duration = Histogram(
        'http_request_duration_seconds',
        'HTTP request latency (total — includes SSE body streaming)',
        ['method', 'endpoint'],
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )

    # NEW: Time-to-first-byte. This is what users actually feel for SSE.
    request_ttfb = Histogram(
        'http_request_ttfb_seconds',
        'Time to first byte (user-perceived latency, SSE-aware)',
        ['method', 'endpoint'],
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    )

    # NEW: Counter for SSE client disconnects — keep this separate from 5xx.
    http_streams_cancelled = Counter(
        'http_streams_cancelled_total',
        'HTTP streams cancelled by client disconnect (asyncio.CancelledError)',
        ['endpoint'],
    )

    active_connections = Gauge(
        'http_active_connections',
        'Active HTTP connections'
    )

    # ----- Chat / RAG pipeline metrics -----
    chat_requests = Counter(
        'chat_requests_total',
        'Total chat requests',
        ['status']
    )

    # AUDIT BUCKET-TUNING: retrieval is fast (vector DB hits in 50–500ms).
    rag_retrieval_duration = Histogram(
        'rag_retrieval_duration_seconds',
        'RAG retrieval duration (vector + BM25 fusion)',
        ['retriever_type'],
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    )

    vector_db_queries = Counter(
        'vector_db_queries_total',
        'Vector database queries',
        ['operation', 'status']
    )

    # ----- Database metrics -----
    db_connection_pool = Gauge(
        'db_connection_pool_size',
        'Database connection pool size'
    )

    db_query_duration = Histogram(
        'db_query_duration_seconds',
        'Database query duration',
        ['operation'],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )


@app.get("/metrics", tags=["monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint
    
    Used by Prometheus server to scrape application metrics.
    Access at: http://localhost:8000/metrics
    """
    if not PROMETHEUS_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Prometheus client not installed"}
        )
    
    return generate_latest(REGISTRY)


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

# Phase 4: Include chat API routes
if chat_api:
    app.include_router(chat_api.router)
    logger.info("✓ Phase 4 Chat API registered:")
    logger.info("  - POST   /api/v1/chat/stream (SSE streaming)")
    logger.info("  - POST   /api/v1/chat (non-streaming)")
    logger.info("  - GET    /api/v1/chat/cache/status")
    logger.info("  - POST   /api/v1/chat/cache/clear")
    logger.info("  - GET    /api/v1/chat/history/{session_id}")

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
