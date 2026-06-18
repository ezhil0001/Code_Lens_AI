"""
Phase Test Base Classes & Types
================================
Defines the contract every phase test module must follow.

Usage
-----
    from app.tests.base import PhaseTest, TestResult, TestStatus

    async def my_check() -> TestResult:
        try:
            assert 1 + 1 == 2
            return TestResult.passed("arithmetic works")
        except AssertionError as e:
            return TestResult.failed(str(e))

    TESTS = [
        PhaseTest(
            id="PHASEXY-001",
            name="Basic arithmetic",
            description="Verifies that 1+1=2",
            run=my_check,
            critical=True,       # if True, phase is marked FAILED on this test's failure
            tags=["sanity"],
        )
    ]
"""

from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, List, Optional


# ── Status ────────────────────────────────────────────────────────────────────

class TestStatus(str, Enum):
    PASSED  = "PASSED"
    FAILED  = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR   = "ERROR"   # unexpected exception in the test harness itself


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    status:   TestStatus
    message:  str
    detail:   Optional[str] = None      # extended failure detail / stack trace
    duration_ms: float = 0.0

    # ── factories ─────────────────────────────────────────────────────────────

    @classmethod
    def passed(cls, message: str = "OK") -> "TestResult":
        return cls(status=TestStatus.PASSED, message=message)

    @classmethod
    def failed(cls, message: str, detail: Optional[str] = None) -> "TestResult":
        return cls(status=TestStatus.FAILED, message=message, detail=detail)

    @classmethod
    def skipped(cls, reason: str) -> "TestResult":
        return cls(status=TestStatus.SKIPPED, message=reason)

    @classmethod
    def error(cls, exc: Exception) -> "TestResult":
        return cls(
            status=TestStatus.ERROR,
            message=f"{type(exc).__name__}: {exc}",
            detail=traceback.format_exc(),
        )


# ── Single test definition ────────────────────────────────────────────────────

TestFn = Callable[[], Awaitable[TestResult]]


@dataclass
class PhaseTest:
    """A single test case belonging to one phase."""

    id:          str              # e.g. "A-001"
    name:        str              # short human label
    description: str              # what this test validates
    run:         TestFn           # async callable → TestResult
    critical:    bool  = True     # failure blocks phase from being PASSED?
    tags:        List[str] = field(default_factory=list)

    async def execute(self) -> "PhaseTestRun":
        """Execute the test, capturing timing and any unexpected exceptions."""
        t0 = time.perf_counter()
        try:
            result = await self.run()
        except Exception as exc:
            result = TestResult.error(exc)
        result.duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        return PhaseTestRun(test=self, result=result)


# ── Run record (test + its result) ───────────────────────────────────────────

@dataclass
class PhaseTestRun:
    test:   PhaseTest
    result: TestResult

    @property
    def passed(self) -> bool:
        return self.result.status == TestStatus.PASSED

    @property
    def failed(self) -> bool:
        return self.result.status in (TestStatus.FAILED, TestStatus.ERROR)


# ── Phase report ──────────────────────────────────────────────────────────────

@dataclass
class PhaseReport:
    phase_id:   str        # "A", "B", …
    phase_name: str
    runs:       List[PhaseTestRun]

    @property
    def total(self) -> int:
        return len(self.runs)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.runs if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.runs if r.failed)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.runs
                   if r.result.status == TestStatus.SKIPPED)

    @property
    def phase_passed(self) -> bool:
        """Phase passes when no *critical* test failed."""
        return all(
            r.passed or not r.test.critical
            for r in self.runs
        )

    @property
    def total_duration_ms(self) -> float:
        return round(sum(r.result.duration_ms for r in self.runs), 2)
