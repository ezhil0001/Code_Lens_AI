"""
Tests for the multi-agent supervisor system.

Each agent sub-graph (CodeAgent, DocAgent, DebugAgent, ArchAgent, WebAgent)
gets its own import check and compile check. The supervisor wiring test
catches cases where an agent is defined but never registered as a graph node.
"""

SUITE_ID   = "B"
SUITE_NAME = "Agent Supervisor"

from app.tests.test_agent_supervisor.tests import TESTS  # noqa: E402, F401
