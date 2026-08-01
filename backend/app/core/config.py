"""Application configuration — centralised Pydantic settings for the whole app.

Unified Pydantic settings management for ALL application configuration:
✅ Application settings
✅ API configuration  
✅ JWT & Security settings
✅ Database configuration (consolidated from app/database/config.py)
✅ Vector database settings
✅ API keys & secrets
✅ Rate limiting
✅ OpenTelemetry
✅ CORS
✅ Logging
✅ Compliance & audit

Features:
- Centralized configuration management (single source of truth)
- Type-safe settings with validation
- LRU cache for performance
- Secrets management best practices
- SQLAlchemy engine & session management
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional, Generator
from functools import lru_cache
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import os

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    
    Enterprise-grade configuration management with:
    - Type validation
    - Centralized secrets
    - Environment-specific settings
    - Rate limiting configuration
    - Security settings
    """
    
    # ==================== Application Settings ====================
    app_name: str = "CodeLens_AI"
    debug: bool = False
    environment: str = "development"  # development, staging, production
    
    # ==================== API Configuration ====================
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_version: str = "v1"
    
    # ==================== JWT & Security ====================
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_days: int = 1  # Alhena pattern: 1 day
    refresh_token_expire_days: int = 7
    
    # JWT Blacklist (Redis or in-memory)
    redis_url: Optional[str] = None  # redis://localhost:6379
    enable_jwt_blacklist: bool = True
    jwt_blacklist_cache_ttl: int = 86400  # 24 hours
    
    # ==================== Database ====================
    # Can use DATABASE_URL directly OR set individual components
    database_url: Optional[str] = None  # Will be set as placeholder
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "codelens_ai"
    database_pool_size: int = 20
    database_max_overflow: int = 10
    database_echo: bool = False  # Set to True for SQL debugging
    
    @property
    def get_database_url(self) -> str:
        """Get database URL, using individual POSTGRES_* env vars if available"""
        # URL-encode password to handle special characters like @ safely
        from urllib.parse import quote_plus
        encoded_password = quote_plus(self.postgres_password)
        return f"postgresql+psycopg2://{self.postgres_user}:{encoded_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    # ==================== Vector Database ====================
    chroma_url: str = "http://localhost:8000"
    chroma_api_key: Optional[str] = None
    
    # ==================== API Keys & Secrets ====================
    groq_api_key: str  # Required for LLM
    huggingface_api_key: str  # Required for embeddings
    openai_api_key: Optional[str] = None  # Optional fallback
    
    # ==================== Rate Limiting ====================
    # Brute-force protection (Security hardening)
    rate_limit_enabled: bool = True
    login_rate_limit: int = 5  # 5 attempts per minute
    login_rate_limit_window: int = 60  # seconds
    general_rate_limit: int = 100  # 100 requests per minute
    general_rate_limit_window: int = 60  # seconds
    
    # ==================== OpenTelemetry ====================
    otel_enabled: bool = False
    otel_exporter_endpoint: str = "http://localhost:4317"  # Jaeger gRPC
    jaeger_host: str = "localhost"
    jaeger_port: int = 6831

    # ==================== Langfuse (LLM Observability & Evaluation) ====================
    # Primary LLM observability platform. Self-hosted via
    # docker-compose.langfuse.yml. Traces every LLM/agent/retrieval span with
    # token usage, cost, latency, prompts/completions, and evaluation scores.
    langfuse_enabled: bool = False
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: Optional[str] = None   # pk-lf-...
    langfuse_secret_key: Optional[str] = None   # sk-lf-...
    langfuse_release: Optional[str] = None       # e.g. git SHA / app version
    langfuse_environment: Optional[str] = None   # overrides `environment` in traces
    langfuse_sample_rate: float = 1.0            # 0.0–1.0 trace sampling
    langfuse_debug: bool = False
    langfuse_flush_timeout_seconds: int = 5
    
    # ==================== CORS ====================
    cors_origins: str = "http://localhost:4200,http://localhost:8001"  # Comma-separated
    cors_allow_credentials: bool = True
    cors_allow_methods: list = ["*"]
    cors_allow_headers: list = ["*"]
    
    # ==================== Logging ====================
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # ==================== Audit & Compliance ====================
    enable_audit_logging: bool = True
    audit_log_retention_days: int = 90  # GDPR compliance
    
    # ==================== Feature Flags ====================
    enable_soft_delete: bool = True  # Soft delete for audit trail
    enable_api_versioning: bool = True
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"
    
    @property
    def cors_origins_list(self) -> list:
        """Parse CORS origins from comma-separated string"""
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    def model_post_init(self, __context) -> None:  # noqa: D105
        """C-4: refuse to boot in production with insecure defaults.

        Fails fast with a clear error so a misconfigured deployment can
        never issue forgeable JWTs or run against default DB credentials.
        Development/staging behavior is unchanged.
        """
        if self.environment == "production":
            problems = []
            if self.secret_key == "your-secret-key-change-in-production" or len(self.secret_key) < 32:
                problems.append(
                    "SECRET_KEY is the insecure default or too short (<32 chars)"
                )
            if self.postgres_password in ("postgres", "codelens_password"):
                problems.append("POSTGRES_PASSWORD is a well-known default")
            if problems:
                raise ValueError(
                    "Refusing to start in production with insecure configuration:\n  - "
                    + "\n  - ".join(problems)
                    + "\nSet strong values via environment variables."
                )


# ==================== Database Engine & Session Management ====================
# (Consolidated from app/database/config.py)

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    
    Uses LRU cache to prevent re-reading environment variables on every request.
    This is the recommended FastAPI dependency injection pattern.
    
    Returns:
        Cached Settings instance
    """
    return Settings()


# Module-level singleton — importable as `from app.core.config import settings`
settings = get_settings()


def _get_engine():
    """Create SQLAlchemy engine with settings from configuration"""
    settings = get_settings()
    return create_engine(
        settings.get_database_url,
        echo=settings.database_echo,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_pre_ping=True,  # Test connections before using
    )


# Global engine instance
_engine = None

def get_engine():
    """Get or create the SQLAlchemy engine"""
    global _engine
    if _engine is None:
        _engine = _get_engine()
    return _engine


# Session factory
_SessionLocal = None

def get_session_local():
    """Get or create the session factory"""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine()
        )
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency to get database session
    
    Usage:
        def my_endpoint(db: Session = Depends(get_db)):
            # Use db session
    """
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def init_db():
    """
    Initialize database tables and extensions
    Call this during app startup
    """
    from sqlalchemy import text
    
    engine = get_engine()
    
    try:
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            logger.info(f"✅ Connected to PostgreSQL: {version}")
            
            # Enable pgVector extension if needed
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                logger.info("✅ pgVector extension enabled")
            except Exception as e:
                logger.warning(f"pgVector extension not available: {e}")
        
        # Import models and create tables
        from app.db.base import Base
        Base.metadata.create_all(bind=engine)
        logger.info("✅ All database tables created/verified")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise


def close_db():
    """
    Close database connections (H-4: sync — engine.dispose() does no async
    work, and the previous ``async def`` was called un-awaited from the
    lifespan shutdown, so the engine was never actually disposed).
    Call this during app shutdown.
    """
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None
        logger.info("✅ Database connections closed")

