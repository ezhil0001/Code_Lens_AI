"""
Tests for input/output guardrails and node middleware.

Input side: prompt injection detection, token budget enforcement, PII scrubbing.
Output side: dangerous code pattern detection, secret redaction.
Middleware: retry logic and the guardrail-event telemetry that tracks blocked events.
"""

SUITE_ID   = "F"
SUITE_NAME = "Guardrails"

from app.tests.test_guardrails.tests import TESTS  # noqa: E402, F401
