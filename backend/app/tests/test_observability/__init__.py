"""
Tests for runtime observability — Langfuse tracing and OpenTelemetry (Jaeger).

The observability modules are import-checked at startup so a broken public
API surface (e.g. a renamed callback or an unimportable evaluator) is caught
immediately rather than silently degrading LLM tracing and evaluation.
"""

SUITE_ID   = "H"
SUITE_NAME = "Observability"

from app.tests.test_observability.tests import TESTS  # noqa: E402, F401
