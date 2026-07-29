"""OpenTelemetry Configuration — distributed tracing setup.

Provides centralised OpenTelemetry initialisation for the full request
pipeline. Traces flow from the FastAPI endpoint through the LangGraph
nodes and down into ChromaDB and Postgres so latency hotspots are visible
in Jaeger without adding per-function instrumentation everywhere.

- Agent Brain orchestration
- Retriever Engine (vector search, BM25, reranking)
- LLM API calls
- Database operations

Exports traces to:
1. Jaeger (tracing backend) - port 4317 (gRPC), 14268 (HTTP)

LLM-level tracing, token/cost tracking, and online evaluation are handled
by Langfuse (see ``rag_evaluator.py`` and the README observability section).
"""

import logging
import os
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ==================== OpenTelemetry Imports ====================

try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.api.trace import get_tracer_provider
    
    HAS_OTEL = True
except ImportError as e:
    HAS_OTEL = False
    JaegerExporter = None  # Define as None if import fails to avoid NameError
    logger.warning(f"OpenTelemetry not installed: {e}. Observability features disabled.")


# ==================== Configuration ====================

class OTelConfig:
    """OpenTelemetry Configuration."""
    
    # Jaeger configuration
    JAEGER_HOST: str = os.getenv("JAEGER_HOST", "localhost")
    JAEGER_PORT: int = int(os.getenv("JAEGER_PORT", "6831"))  # Thrift UDP
    JAEGER_GRPC_PORT: int = int(os.getenv("JAEGER_GRPC_PORT", "4317"))
    JAEGER_HTTP_PORT: int = int(os.getenv("JAEGER_HTTP_PORT", "14268"))
    
    # Service metadata
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "codelens-rag-agent")
    SERVICE_VERSION: str = os.getenv("SERVICE_VERSION", "5.0.0")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # Sampling
    TRACE_SAMPLE_RATE: float = float(os.getenv("TRACE_SAMPLE_RATE", "1.0"))  # 100% sampling
    
    # Export batch settings
    BATCH_SPAN_SIZE: int = int(os.getenv("BATCH_SPAN_SIZE", "512"))
    BATCH_SCHEDULE_DELAY_MILLIS: int = int(os.getenv("BATCH_SCHEDULE_DELAY", "5000"))


# ==================== Jaeger Exporter Setup ====================

def setup_jaeger_exporter() -> Optional[Any]:
    """Setup Jaeger exporter for traces.
    
    Sends traces to Jaeger using gRPC protocol (port 4317).
    Falls back to Thrift UDP if gRPC unavailable.
    
    Returns:
        JaegerExporter instance or None if OpenTelemetry not available
    """
    if not HAS_OTEL:
        return None
    
    try:
        exporter = JaegerExporter(
            agent_host_name=OTelConfig.JAEGER_HOST,
            agent_port=OTelConfig.JAEGER_PORT,
        )
        logger.info(
            f"✓ Jaeger exporter configured: {OTelConfig.JAEGER_HOST}:{OTelConfig.JAEGER_PORT}"
        )
        return exporter
    except Exception as e:
        logger.error(f"Failed to setup Jaeger exporter: {e}")
        return None


# ==================== Tracer Provider Setup ====================

def setup_tracer_provider() -> Optional[TracerProvider]:
    """Setup OpenTelemetry Tracer Provider.
    
    Creates a tracer provider with:
    1. Resource metadata (service name, version, environment)
    2. Batch span processor (exports to Jaeger)
    3. Sampler configuration
    
    Returns:
        Configured TracerProvider instance
    """
    if not HAS_OTEL:
        return None
    
    try:
        # Create resource with metadata
        resource = Resource.create({
            "service.name": OTelConfig.SERVICE_NAME,
            "service.version": OTelConfig.SERVICE_VERSION,
            "deployment.environment": OTelConfig.ENVIRONMENT,
            "host.name": os.getenv("HOSTNAME", "unknown"),
        })
        
        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        
        # Setup Jaeger exporter
        jaeger_exporter = setup_jaeger_exporter()
        if jaeger_exporter:
            tracer_provider.add_span_processor(
                BatchSpanProcessor(
                    jaeger_exporter,
                    max_queue_size=OTelConfig.BATCH_SPAN_SIZE,
                    schedule_delay_millis=OTelConfig.BATCH_SCHEDULE_DELAY_MILLIS,
                )
            )
        
        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)
        logger.info("✓ Tracer provider configured")
        
        return tracer_provider
    
    except Exception as e:
        logger.error(f"Failed to setup tracer provider: {e}")
        return None


# ==================== Instrumentation ====================

def setup_instrumentation(app=None):
    """Setup automatic instrumentation for common libraries.
    
    Instruments:
    1. FastAPI - HTTP routes, middleware
    2. SQLAlchemy - Database queries
    3. Psycopg2 - PostgreSQL calls
    4. Requests/HTTPX - External API calls
    
    Args:
        app: FastAPI app instance (optional, for FastAPI instrumentation)
    """
    if not HAS_OTEL:
        logger.warning("Instrumentation skipped - OpenTelemetry not available")
        return
    
    try:
        # FastAPI instrumentation
        if app:
            FastAPIInstrumentor.instrument_app(app)
            logger.info("✓ FastAPI instrumented")
        
        # Database instrumentation
        try:
            SQLAlchemyInstrumentor().instrument(
                engine_hook=None,  # Will auto-detect
                service=OTelConfig.SERVICE_NAME,
            )
            logger.info("✓ SQLAlchemy instrumented")
        except Exception as e:
            logger.warning(f"SQLAlchemy instrumentation failed: {e}")
        
        # PostgreSQL instrumentation
        try:
            Psycopg2Instrumentor().instrument()
            logger.info("✓ Psycopg2 instrumented")
        except Exception as e:
            logger.warning(f"Psycopg2 instrumentation failed: {e}")
        
        # HTTP client instrumentation
        try:
            HTTPXClientInstrumentor().instrument()
            logger.info("✓ HTTPX instrumented")
        except Exception as e:
            logger.warning(f"HTTPX instrumentation failed: {e}")
        
        try:
            RequestsInstrumentor().instrument()
            logger.info("✓ Requests instrumented")
        except Exception as e:
            logger.warning(f"Requests instrumentation failed: {e}")
    
    except Exception as e:
        logger.error(f"Failed to setup instrumentation: {e}")


# ==================== Tracer Interface ====================

class RAGTracer:
    """Unified tracer for RAG pipeline operations.
    
    Provides convenient methods for tracing RAG-specific operations:
    - Query expansion
    - Vector search
    - Reranking
    - LLM generation
    - Memory operations
    """
    
    def __init__(self, name: str = "codelens-rag"):
        """Initialize RAG tracer.
        
        Args:
            name: Tracer name (for OpenTelemetry)
        """
        self.name = name
        try:
            self.tracer = trace.get_tracer(__name__) if HAS_OTEL else None
        except Exception as e:
            logger.warning(f"Failed to get tracer: {e}")
            self.tracer = None
    
    @contextmanager
    def trace_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for creating a span.
        
        Usage:
            with tracer.trace_span("vector_search", {"k": 5}):
                results = retriever.search(query)
        
        Args:
            name: Span name
            attributes: Optional span attributes (metadata)
            
        Yields:
            Span object (or None if tracing disabled)
        """
        if not self.tracer:
            yield None
            return
        
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    try:
                        span.set_attribute(key, value)
                    except Exception as e:
                        logger.warning(f"Failed to set span attribute {key}: {e}")
            
            yield span
    
    def record_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Record an event within current span.
        
        Args:
            name: Event name
            attributes: Optional event attributes
        """
        if not self.tracer:
            return
        
        try:
            span = trace.get_current_span()
            if span:
                span.add_event(name, attributes=attributes or {})
        except Exception as e:
            logger.warning(f"Failed to record event: {e}")
    
    def set_span_attribute(self, key: str, value: Any):
        """Set attribute on current span.
        
        Args:
            key: Attribute key
            value: Attribute value
        """
        if not self.tracer:
            return
        
        try:
            span = trace.get_current_span()
            if span:
                span.set_attribute(key, value)
        except Exception as e:
            logger.warning(f"Failed to set span attribute: {e}")


# ==================== Metrics Interface ====================

class RAGMetrics:
    """Unified metrics for RAG pipeline monitoring.
    
    Tracks:
    - Latency histograms (retrieval, reranking, generation)
    - Operation counters (queries, cache hits/misses)
    - Gauge metrics (cache size, queue depth)
    """
    
    def __init__(self, name: str = "codelens-rag"):
        """Initialize metrics.
        
        Args:
            name: Meter name (for OpenTelemetry)
        """
        self.name = name
        try:
            self.meter = metrics.get_meter(__name__) if HAS_OTEL else None
        except Exception as e:
            logger.warning(f"Failed to get meter: {e}")
            self.meter = None
        
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup common RAG metrics."""
        if not self.meter:
            return
        
        try:
            # Latency histograms (milliseconds)
            self.retrieval_latency = self.meter.create_histogram(
                name="rag.retrieval.latency",
                description="Vector search + BM25 retrieval latency (ms)",
                unit="ms",
            )
            
            self.reranking_latency = self.meter.create_histogram(
                name="rag.reranking.latency",
                description="Reranker latency (ms)",
                unit="ms",
            )
            
            self.generation_latency = self.meter.create_histogram(
                name="rag.generation.latency",
                description="LLM generation latency (ms)",
                unit="ms",
            )
            
            # Operation counters
            self.query_counter = self.meter.create_counter(
                name="rag.queries.total",
                description="Total queries processed",
                unit="1",
            )
            
            self.cache_hits = self.meter.create_counter(
                name="rag.cache.hits",
                description="Semantic cache hits",
                unit="1",
            )
            
            self.cache_misses = self.meter.create_counter(
                name="rag.cache.misses",
                description="Semantic cache misses",
                unit="1",
            )
            
            self.error_counter = self.meter.create_counter(
                name="rag.errors.total",
                description="Total errors",
                unit="1",
            )
            
            # Gauge metrics
            self.cache_size = self.meter.create_up_down_counter(
                name="rag.cache.size",
                description="Current cache size (entries)",
                unit="1",
            )
            
            logger.info("✓ RAG metrics initialized")
        
        except Exception as e:
            logger.error(f"Failed to setup metrics: {e}")
    
    def record_retrieval_latency(self, latency_ms: float, attributes: Optional[Dict] = None):
        """Record retrieval latency.
        
        Args:
            latency_ms: Latency in milliseconds
            attributes: Optional attributes (retrieval_type, k, etc.)
        """
        if self.retrieval_latency:
            self.retrieval_latency.record(latency_ms, attributes)
    
    def record_reranking_latency(self, latency_ms: float, attributes: Optional[Dict] = None):
        """Record reranking latency.
        
        Args:
            latency_ms: Latency in milliseconds
            attributes: Optional attributes (reranker_type, etc.)
        """
        if self.reranking_latency:
            self.reranking_latency.record(latency_ms, attributes)
    
    def record_generation_latency(self, latency_ms: float, attributes: Optional[Dict] = None):
        """Record LLM generation latency.
        
        Args:
            latency_ms: Latency in milliseconds
            attributes: Optional attributes (model, tokens, etc.)
        """
        if self.generation_latency:
            self.generation_latency.record(latency_ms, attributes)
    
    def increment_query_counter(self, attributes: Optional[Dict] = None):
        """Increment query counter.
        
        Args:
            attributes: Optional attributes (router_path, etc.)
        """
        if self.query_counter:
            self.query_counter.add(1, attributes)
    
    def increment_cache_hit(self):
        """Increment cache hit counter."""
        if self.cache_hits:
            self.cache_hits.add(1)
    
    def increment_cache_miss(self):
        """Increment cache miss counter."""
        if self.cache_misses:
            self.cache_misses.add(1)
    
    def increment_error_counter(self, error_type: str = "unknown"):
        """Increment error counter.
        
        Args:
            error_type: Type of error (retrieval, generation, etc.)
        """
        if self.error_counter:
            self.error_counter.add(1, {"error.type": error_type})
    
    def set_cache_size(self, size: int):
        """Set current cache size.
        
        Args:
            size: Number of entries in cache
        """
        if self.cache_size:
            self.cache_size.add(size)


# ==================== Global Instances ====================

_tracer: Optional[RAGTracer] = None
_metrics: Optional[RAGMetrics] = None


def get_tracer() -> RAGTracer:
    """Get global RAG tracer instance.
    
    Returns:
        RAGTracer instance
    """
    global _tracer
    if _tracer is None:
        _tracer = RAGTracer()
    return _tracer


def get_metrics() -> RAGMetrics:
    """Get global RAG metrics instance.
    
    Returns:
        RAGMetrics instance
    """
    global _metrics
    if _metrics is None:
        _metrics = RAGMetrics()
    return _metrics


def initialize_observability(app=None) -> bool:
    """Initialize complete observability stack.
    
    This is the main entry point for Phase 5 observability setup.
    Should be called once during application startup.
    
    Args:
        app: FastAPI app instance (optional)
        
    Returns:
        True if initialization successful, False otherwise
    """
    if not HAS_OTEL:
        logger.warning(
            "OpenTelemetry not installed. Install with: "
            "pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-jaeger "
            "opentelemetry-instrumentation-fastapi "
            "opentelemetry-instrumentation-sqlalchemy"
        )
        return False
    
    try:
        logger.info("Initializing OpenTelemetry observability stack...")
        
        # Setup providers
        setup_tracer_provider()
        
        # Setup auto-instrumentation
        setup_instrumentation(app)
        
        logger.info("✓ Observability stack initialized successfully")
        return True
    
    except Exception as e:
        logger.error(f"Failed to initialize observability: {e}")
        return False
