"""
Tests for the LangGraph state machine foundation.

Covers AgentState structure, graph compilation, intent routing,
and the streaming layer — things that break silently if the graph
schema drifts from what the nodes actually write.
"""

SUITE_ID   = "A"
SUITE_NAME = "Graph Foundation"

from app.tests.test_graph_foundation.tests import TESTS  # noqa: E402, F401
