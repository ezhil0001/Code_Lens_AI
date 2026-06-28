"""
Tests for the SSE streaming API layer.

Checks that both v1 and v2 chat endpoints are registered, that the
ChatV2Request schema enforces length limits, and that the SSE formatter
produces spec-compliant event frames. The v1 route is kept to avoid
breaking any clients that haven't upgraded yet.
"""

SUITE_ID   = "G"
SUITE_NAME = "Streaming API"

from app.tests.test_streaming_api.tests import TESTS  # noqa: E402, F401
