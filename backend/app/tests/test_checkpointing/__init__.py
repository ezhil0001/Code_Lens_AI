"""
Tests for the PostgreSQL checkpointing and time-travel API.

Validates that the AsyncPostgresSaver is wired correctly, thread IDs
are properly namespaced, and all three checkpoint API routes
(/checkpoints, /replay, /branch) are registered in FastAPI.
"""

SUITE_ID   = "D"
SUITE_NAME = "Checkpointing"

from app.tests.test_checkpointing.tests import TESTS  # noqa: E402, F401
