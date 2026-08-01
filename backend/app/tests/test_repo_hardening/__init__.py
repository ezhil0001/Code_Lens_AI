"""
Repository-wide hardening regression suite (R-001…R-010).

Locks in the fixes for the Critical/High findings of the Aug-2026
repository-wide forensic audit.
"""

SUITE_NAME = "Repository Hardening"

from app.tests.test_repo_hardening.tests import TESTS  # noqa: E402, F401
