"""
Startup test runner — runs sanity checks on critical components every time
the server boots, before the first real request is served.

Tests are discovered automatically from sub-packages under app/tests/ that
start with "test_". Each package exposes a TESTS list that the runner
collects and executes in alphabetical order.

Usage (called from main.py lifespan hook):

    from app.tests.runner import StartupTestRunner
    await StartupTestRunner.run_all()

Environment variables:
    STARTUP_TESTS_ENABLED   — default "true". Set "false" to skip in production
                               if startup time is critical and tests ran in CI.
    STARTUP_TESTS_FAIL_FAST — default "false". Set "true" to abort on the first
                               critical failure instead of running all suites.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import pkgutil
import time
from typing import List

from app.core.logger import logger as flow_logger
from app.tests.base import PhaseReport, PhaseTest, PhaseTestRun, TestStatus


# ── ANSI colour helpers (loguru strips these if the sink has no TTY) ──────────

_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"
_DIM    = "\033[2m"

_ICON = {
    TestStatus.PASSED:  f"{_GREEN}✓{_RESET}",
    TestStatus.FAILED:  f"{_RED}✗{_RESET}",
    TestStatus.SKIPPED: f"{_YELLOW}─{_RESET}",
    TestStatus.ERROR:   f"{_RED}⚡{_RESET}",
}

_STATUS_LABEL = {
    TestStatus.PASSED:  f"{_GREEN}PASSED {_RESET}",
    TestStatus.FAILED:  f"{_RED}FAILED {_RESET}",
    TestStatus.SKIPPED: f"{_YELLOW}SKIPPED{_RESET}",
    TestStatus.ERROR:   f"{_RED}ERROR  {_RESET}",
}


# ── Test suite discovery ──────────────────────────────────────────────────────

def _discover_test_suites() -> List[str]:
    """Return sorted module paths for every test_* sub-package under app/tests/.

    Running them alphabetically gives a consistent ordering that matches
    the directory names (test_agent_supervisor, test_checkpointing, …).
    """
    import app.tests as _tests_pkg
    pkg_path = _tests_pkg.__path__
    modules = []
    for _, mod_name, is_pkg in pkgutil.iter_modules(pkg_path):
        if mod_name.startswith("test_") and is_pkg:
            modules.append(f"app.tests.{mod_name}")
    return sorted(modules)


def _load_suite(pkg_path: str) -> tuple[str, str, List[PhaseTest]]:
    """Import a test suite package and return (suite_id, suite_name, tests).

    Falls back to the package name if the module doesn't declare SUITE_ID/NAME,
    so old PHASE_ID/PHASE_NAME keys still work during migration.
    """
    mod = importlib.import_module(pkg_path)
    short_name = pkg_path.split(".")[-1]
    suite_id   = getattr(mod, "SUITE_ID",   getattr(mod, "PHASE_ID",   short_name))
    suite_name = getattr(mod, "SUITE_NAME", getattr(mod, "PHASE_NAME", pkg_path))
    tests      = getattr(mod, "TESTS",      [])
    return suite_id, suite_name, tests


# ── Report rendering ─────────────────────────────────────────────────────────

_SEP  = "─" * 72
_SEP2 = "═" * 72


def _render_report(reports: List[PhaseReport], total_ms: float) -> None:
    log = flow_logger.bind(tag="[STARTUP_TESTS]")

    log.info(f"\n{_BOLD}{_CYAN}{_SEP2}{_RESET}")
    log.info(f"{_BOLD}{_CYAN}  CodeLens AI — Startup Test Report{_RESET}")
    log.info(f"{_BOLD}{_CYAN}{_SEP2}{_RESET}\n")

    all_passed = True

    for report in reports:
        suite_status = (
            f"{_GREEN}{_BOLD}PASSED{_RESET}"
            if report.phase_passed
            else f"{_RED}{_BOLD}FAILED{_RESET}"
        )
        if not report.phase_passed:
            all_passed = False

        log.info(
            f"{_BOLD}[{report.phase_id}] {report.phase_name}{_RESET}  "
            f"[{suite_status}]  "
            f"{_DIM}{report.passed}/{report.total} passed  "
            f"{report.total_duration_ms}ms{_RESET}"
        )
        log.info(f"{_DIM}{_SEP}{_RESET}")

        for run in report.runs:
            icon  = _ICON[run.result.status]
            label = _STATUS_LABEL[run.result.status]
            crit  = f"{_RED}[CRITICAL]{_RESET}" if run.test.critical and run.failed else ""
            log.info(
                f"  {icon}  {label}  {_BOLD}{run.test.id}{_RESET}  "
                f"{run.test.name}  "
                f"{_DIM}({run.result.duration_ms}ms){_RESET}  {crit}"
            )
            if run.result.message and run.result.status != TestStatus.PASSED:
                log.info(f"       {_DIM}↳ {run.result.message}{_RESET}")
            if run.result.detail and run.failed:
                # Print first 5 lines of traceback only to avoid log flooding
                detail_lines = (run.result.detail or "").splitlines()[:5]
                for dl in detail_lines:
                    log.info(f"         {_DIM}{_RED}{dl}{_RESET}")

        log.info("")

    # ── Summary row ────────────────────────────────────────────────────────
    total_tests   = sum(r.total   for r in reports)
    total_passed  = sum(r.passed  for r in reports)
    total_failed  = sum(r.failed  for r in reports)
    total_skipped = sum(r.skipped for r in reports)

    log.info(f"{_BOLD}{_CYAN}{_SEP2}{_RESET}")
    if all_passed:
        log.info(
            f"{_GREEN}{_BOLD}  ✅  ALL SUITES PASSED — "
            f"{total_passed}/{total_tests} tests  "
            f"({round(total_ms, 1)}ms total){_RESET}"
        )
    else:
        log.info(
            f"{_RED}{_BOLD}  ❌  SOME SUITES FAILED — "
            f"{total_passed} passed  {total_failed} failed  "
            f"{total_skipped} skipped  "
            f"({round(total_ms, 1)}ms total){_RESET}"
        )
    log.info(f"{_BOLD}{_CYAN}{_SEP2}{_RESET}\n")


# ── Main runner ──────────────────────────────────────────────────────────────

class StartupTestRunner:
    """Discovers, runs, and reports all phase tests at server startup."""

    @staticmethod
    async def run_all(fail_fast: bool = False) -> bool:
        """
        Run all phase tests.

        Parameters
        ----------
        fail_fast:
            If True, stop running subsequent phases after the first critical
            test failure.  Controlled by env var ``STARTUP_TESTS_FAIL_FAST``.

        Returns
        -------
        bool  —  True if every phase passed, False otherwise.
        """
        enabled = os.getenv("STARTUP_TESTS_ENABLED", "true").lower()
        if enabled not in ("true", "1", "yes"):
            flow_logger.bind(tag="[STARTUP_TESTS]").info(
                "⏭  Startup tests DISABLED (STARTUP_TESTS_ENABLED=false)"
            )
            return True

        fail_fast = fail_fast or (
            os.getenv("STARTUP_TESTS_FAIL_FAST", "false").lower()
            in ("true", "1", "yes")
        )

        flow_logger.bind(tag="[STARTUP_TESTS]").info(
            "🧪  Discovering and running startup tests…"
        )

        packages = _discover_test_suites()
        if not packages:
            flow_logger.bind(tag="[STARTUP_TESTS]").warning(
                "⚠  No test suites found under app/tests/test_*"
            )
            return True

        reports: List[PhaseReport] = []
        t_global_start = time.perf_counter()

        for pkg_path in packages:
            try:
                phase_id, phase_name, tests = _load_suite(pkg_path)
            except Exception as exc:
                flow_logger.bind(tag="[STARTUP_TESTS]").error(
                    f"  ⚡  Failed to load suite {pkg_path}: {exc}"
                )
                continue

            if not tests:
                flow_logger.bind(tag="[STARTUP_TESTS]").warning(
                    f"  ─  [{phase_id}] {phase_name}: no tests defined"
                )
                continue

            runs: List[PhaseTestRun] = []
            for test in tests:
                run = await test.execute()
                runs.append(run)
                if fail_fast and run.failed and run.test.critical:
                    flow_logger.bind(tag="[STARTUP_TESTS]").error(
                        f"  FAIL_FAST triggered on {test.id} — aborting remaining tests"
                    )
                    break

            report = PhaseReport(
                phase_id=phase_id,
                phase_name=phase_name,
                runs=runs,
            )
            reports.append(report)

            if fail_fast and not report.phase_passed:
                break

        total_ms = (time.perf_counter() - t_global_start) * 1000
        _render_report(reports, total_ms)

        return all(r.phase_passed for r in reports)
