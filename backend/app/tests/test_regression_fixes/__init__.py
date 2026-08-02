"""
Regression tests for the six production bugs found during live end-to-end
validation. Each test pins one previously-broken behaviour so it cannot
silently regress:

  R-001  psycopg DSN resolves the real password from Settings (not the
         postgres:postgres default) — semantic cache auth bug.
  R-002  All retrieval agents share ONE process-wide lock — parallel-agent
         MPS deadlock bug.
  R-003  Concurrent-write state keys carry a reducer — LangGraph
         INVALID_CONCURRENT_GRAPH_UPDATE bug.
  R-004  request_trace() survives a cross-context reset — SSE ContextVar
         "Token created in a different Context" bug.
  R-005  Long-term memory SQL uses psycopg3 %s placeholders (not $1) —
         "0 placeholders but N parameters" bug.
  R-006  RAGAS evaluator injects local embeddings — silent OpenAI fallback
         bug.
"""

SUITE_ID = "R"
SUITE_NAME = "Regression Fixes"

from app.tests.test_regression_fixes.tests import TESTS  # noqa: E402, F401
