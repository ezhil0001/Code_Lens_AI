"""Langfuse HTTP tracing middleware — one trace per inbound request.

The streaming chat endpoint (``/api/v2/chat/stream``) is **excluded**: it opens
its own ``chat.supervisor`` root observation and attaches the LangGraph
``CallbackHandler`` to it, so wrapping it here would produce a duplicate root
trace. Every other HTTP request (auth, history, cache, sessions, curate,
feedback) is traced here.

For those requests this middleware opens a root span capturing method, path,
status code, latency, user context (when resolvable from request.state),
environment, and error details. It degrades to a pure pass-through when
Langfuse is disabled. Never breaks request handling.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Paths whose traces are owned by the LangGraph callback handler — the
# streaming chat endpoint mints its own trace via _attach_langfuse(); wrapping
# it here would create a duplicate root trace (one request = one trace).
_SKIP_PREFIXES = ("/api/v2/chat/stream",)
# The HIL resume endpoint continues the *originating* request's trace instead of
# minting one, so tracing it here produced a stray "HTTP POST .../resume" trace
# alongside the real one. The session id is dynamic, so match on the suffix.
_SKIP_SUFFIXES = ("/resume",)
# Noise endpoints not worth tracing.
_IGNORE_PREFIXES = ("/health", "/docs", "/openapi.json", "/redoc", "/favicon")


class LangfuseHTTPMiddleware(BaseHTTPMiddleware):
    """Trace every non-chat HTTP request as a Langfuse root span."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        if (
            path.startswith(_SKIP_PREFIXES)
            or path.startswith(_IGNORE_PREFIXES)
            or path.endswith(_SKIP_SUFFIXES)
        ):
            return await call_next(request)

        try:
            from app.observability.langfuse_client import should_sample, create_trace_id
            # H-2: single per-request sampling decision, shared with every
            # nested service span via request_trace().
            if not should_sample():
                # Bind "not sampled" so nested service spans no-op too.
                from app.observability.tracing import request_trace
                with request_trace(None):
                    return await call_next(request)
        except Exception:  # noqa: BLE001
            return await call_next(request)

        from app.observability.tracing import request_trace, span

        trace_id = None
        try:
            trace_id = create_trace_id()
        except Exception:  # noqa: BLE001
            pass

        start = time.perf_counter()
        name = f"HTTP {request.method} {path}"
        meta: dict[str, Any] = {
            "http.method": request.method,
            "http.path": path,
            "http.query": str(request.url.query) if request.url.query else None,
            "request.source": "api",
            "client.host": request.client.host if request.client else None,
        }

        try:
            with request_trace(trace_id):
                with span(name, kind="span", input={"method": request.method, "path": path}, metadata=meta) as s:
                    response = await call_next(request)
                    s.update(
                        output={"status_code": response.status_code},
                        metadata={
                            "http.status_code": response.status_code,
                            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                        },
                        level="ERROR" if response.status_code >= 500 else None,
                    )
                    return response
        except Exception:
            # span() already tagged the error; propagate to the exception handlers.
            raise
