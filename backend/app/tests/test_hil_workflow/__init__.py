"""
Tests for the human-in-the-loop interrupt workflow.

Verifies that the hil_check_node correctly triggers on low-confidence
routing and destructive query keywords, and that the resume endpoint
exists so interrupted graph runs can be continued after human review.
"""

SUITE_ID   = "E"
SUITE_NAME = "HIL Workflow"

from app.tests.test_hil_workflow.tests import TESTS  # noqa: E402, F401
