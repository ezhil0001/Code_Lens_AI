"""
Fallback response strings for AgentBrain when no LLM client is configured.

Isolated here so the mock strings don't bloat agent_brain.py and so they
can never accidentally escape to production — AgentBrain._mock_response()
is only called when settings.DEBUG is True or the LLM client is None.
In production the caller raises rather than returning these strings.
"""

from __future__ import annotations

from typing import Tuple


_AUTH_MOCK = """The authenticate function validates user credentials against stored tokens.

Key Points:
1. **Token Extraction**: Retrieves JWT token from request headers
2. **Token Decoding**: Decodes JWT using app secret key
3. **Signature Verification**: Validates JWT signature integrity
4. **Expiration Check**: Ensures token hasn't expired
5. **Database Lookup**: Queries database for user record
6. **Error Handling**: Returns user object if valid, raises exception if invalid

Implementation Details:
- Uses JWT for stateless, scalable authentication
- Supports refresh tokens for extended sessions
- Implements exponential backoff for failed attempts
- Logs all authentication events for audit trail
- Handles edge cases (expired tokens, invalid signatures, etc.)

For actual intelligent responses, connect an LLM client (Ollama, Groq, or OpenAI)."""


_ARCH_MOCK = """The system architecture follows a layered design pattern:

**Layers**:
1. **API Layer**: Handles HTTP requests and WebSocket connections
2. **Service Layer**: Business logic and orchestration
3. **Data Layer**: Database operations and caching
4. **External APIs**: Integration with external services

**Key Components**:
- Phase 2 Retriever: Hybrid search (vector + BM25)
- Phase 3 Agent Brain: Reasoning and orchestration
- Message Memory: PostgreSQL-based conversation history
- Tools System: Specialized operations (analysis, linting, etc.)

**Design Principles**:
- Modularity: Independent, testable components
- Scalability: Async/await throughout
- Resilience: Error handling and fallbacks
- Observability: Comprehensive logging and metrics

For production, connect real LLM and database backends."""


_GENERIC_MOCK = """Based on the provided context, here's a comprehensive answer:

The system includes several well-designed components:

1. **Ingestion Pipeline** (Phase 1): Processes and indexes documents
2. **Retrieval Engine** (Phase 2): Hybrid search with semantic similarity
3. **Agent Brain** (Phase 3): Reasoning and response generation
4. **Memory Manager**: Persistent conversation history
5. **Tools System**: Specialized operations

Each component follows best practices for:
- Code organization and modularity
- Error handling and resilience
- Performance optimization
- Type safety and validation

For actual LLM responses, configure a real LLM client (Ollama for local, Groq/OpenAI for cloud).
This is mock response is for development and testing purposes."""


def get_mock_response(prompt: str) -> Tuple[str, int]:
    """Return a hand-crafted mock response keyed off the prompt content.

    Returns: (text, approximate_token_count)
    """
    lower = (prompt or "").lower()
    if "authenticate" in lower or "auth" in lower:
        text = _AUTH_MOCK
    elif "architecture" in lower or "design" in lower:
        text = _ARCH_MOCK
    else:
        text = _GENERIC_MOCK
    return text, len(text.split())
