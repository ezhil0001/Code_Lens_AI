"""
Tests for the short-term and long-term memory layer.

Checks namespace isolation (user_id scoping), token budget trimming,
LongTermStore retrieval, and entity extraction. These tests caught the
bug where long_term_facts was always empty because retrieve() was never called.
"""

SUITE_ID   = "C"
SUITE_NAME = "Memory Layer"

from app.tests.test_memory_layer.tests import TESTS  # noqa: E402, F401
