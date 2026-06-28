"""Chat request and response schemas used by both the v1 and v2 streaming endpoints."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid


# ── Stream-ready schemas — all responses are designed for SSE streaming ──────

class ChatRequest(BaseModel):
    """Production chat request with session tracking and org scoping."""
    
    query: str = Field(..., description="User's question/query", min_length=1, max_length=5000)
    session_id: str = Field(
        default_factory=lambda: f"sess-{uuid.uuid4().hex[:12]}",
        description="Session identifier for conversation tracking",
    )
    user_id: str = Field(
        default_factory=lambda: f"anon-{uuid.uuid4().hex[:8]}",
        description="User identifier",
    )
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context data")
    stream: bool = Field(True, description="Enable streaming response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "How does the authentication middleware work?",
                "session_id": "sess-abc123",
                "user_id": "user-xyz789",
                "stream": True,
            }
        }


class ChatStreamResponse(BaseModel):
    """Streaming response chunk with sources and routing metadata."""
    
    content: str = Field(..., description="Generated response content")
    session_id: str = Field(..., description="Session identifier")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved context sources")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class StreamToken(BaseModel):
    """Individual token in SSE stream."""
    type: str = Field(..., description="Token type: 'token', 'done', 'error'")
    content: Optional[str] = Field(None, description="Token content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata for 'done' type")


class AnswerSchema(BaseModel):
    """Structured LLM output enforced by PydanticOutputParser.

    The LLM is instructed to emit JSON conforming to this schema so we get
    grounded citations and a self-reported confidence score for downstream
    evaluation/cache decisions.
    """
    answer: str = Field(..., description="The grounded natural-language answer to the user's query.")
    sources: List[str] = Field(
        default_factory=list,
        description="List of source identifiers (file paths, doc IDs, or chunk IDs) used.",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Self-reported confidence in [0.0, 1.0]. Use 0 if answer is not grounded.",
    )


# ==================== Legacy Schemas (Backward Compatibility) ====================

class ChatResponse(BaseModel):
    """Legacy: Response body for chat endpoint"""
    answer: str
    context_used: Optional[str]
    conversation_id: int
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    """Legacy: Chat history item response"""
    id: int
    question: str
    answer: str
    created_at: datetime


# ==================== Health Check Schemas ====================

class ComponentHealth(BaseModel):
    """Individual component health status."""
    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Health status: 'healthy', 'degraded', 'unhealthy'")
    latency_ms: Optional[float] = Field(None, description="Component latency")
    message: Optional[str] = Field(None, description="Additional info")


class FullHealthStatus(BaseModel):
    """Complete system health status."""
    overall_status: str = Field(..., description="Overall status: 'healthy', 'degraded', 'unhealthy'")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    components: Dict[str, ComponentHealth] = Field(default_factory=dict, description="Component status")
    uptime_seconds: Optional[float] = Field(None, description="Uptime in seconds")


# ==================== Cache Schemas ====================

class CacheStatus(BaseModel):
    """Semantic cache status."""
    cache_size: int = Field(..., description="Number of cached queries")
    ttl_hours: int = Field(..., description="Cache TTL in hours")
    similarity_threshold: float = Field(..., description="Similarity threshold")
    cached_queries: List[str] = Field(default_factory=list, description="Sample queries")


# ==================== Session Schemas ====================

class ChatHistoryMessage(BaseModel):
    """Individual message in chat history."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class ChatHistory(BaseModel):
    """Complete chat session history."""
    session_id: str = Field(..., description="Session identifier")
    user_id: str = Field(..., description="User identifier")
    messages: List[ChatHistoryMessage] = Field(default_factory=list, description="Chat messages")
    created_at: datetime = Field(default_factory=datetime.now, description="Session creation")
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")
