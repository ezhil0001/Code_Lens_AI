"""
Startup test runner — validates critical application components at boot time.

Discovers all test packages under app/tests/, runs them sequentially, and
writes a structured pass/fail report to the application logger.  Failures
are visible in the terminal immediately after startup so broken deployments
are caught before the first real request is served.

Called from app/main.py inside the lifespan context manager after the RAG
pipeline is pre-warmed:

    from app.tests.runner import StartupTestRunner
    await StartupTestRunner.run_all()

Environment:
    STARTUP_TESTS_ENABLED   (default "true")  — set "false" to skip in prod.
    STARTUP_TESTS_FAIL_FAST (default "false") — abort on first critical failure.
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


# ── Phase package discovery ──────────────────────────────────────────────────

def _discover_phase_packages() -> List[str]:
    """Return sorted list of phase sub-package module paths.

    Looks for packages named  app.tests.phase_*  in alphabetical order so
    phases always execute in declaration order (phase_a → phase_b → …).
    """
    import app.tests as _tests_pkg
    pkg_path = _tests_pkg.__path__
    modules = []
    for _, mod_name, is_pkg in pkgutil.iter_modules(pkg_path):
        if mod_name.startswith("phase_") and is_pkg:
            modules.append(f"app.tests.{mod_name}")
    return sorted(modules)


def _load_tests_from_package(pkg_path: str) -> tuple[str, str, List[PhaseTest]]:
    """Import a phase package and return (phase_id, phase_name, tests)."""
    mod = importlib.import_module(pkg_path)
    phase_id   = getattr(mod, "PHASE_ID",   pkg_path.split(".")[-1])
    phase_name = getattr(mod, "PHASE_NAME", pkg_path)
    tests      = getattr(mod, "TESTS",      [])
    return phase_id, phase_name, tests


# ── Report rendering ─────────────────────────────────────────────────────────

_SEP  = "─" * 72
_SEP2 = "═" * 72


def _render_report(reports: List[PhaseReport], total_ms: float) -> None:
    log = flow_logger.bind(tag="[STARTUP_TESTS]")

    log.info(f"\n{_BOLD}{_CYAN}{_SEP2}{_RESET}")
    log.info(f"{_BOLD}{_CYAN}  CodeLens AI — Startup Phase Test Report{_RESET}")
    log.info(f"{_BOLD}{_CYAN}{_SEP2}{_RESET}\n")

    all_passed = True

    for report in reports:
        phase_status = (
            f"{_GREEN}{_BOLD}PHASE PASSED{_RESET}"
            if report.phase_passed
            else f"{_RED}{_BOLD}PHASE FAILED{_RESET}"
        )
        if not report.phase_passed:
            all_passed = False

        log.info(
            f"{_BOLD}Phase {report.phase_id}  —  {report.phase_name}{_RESET}  "
            f"[{phase_status}]  "
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
            f"{_GREEN}{_BOLD}  ✅  ALL PHASES PASSED — "
            f"{total_passed}/{total_tests} tests  "
            f"({round(total_ms, 1)}ms total){_RESET}"
        )
    else:
        log.info(
            f"{_RED}{_BOLD}  ❌  SOME PHASES FAILED — "
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
            "🧪  Discovering and running phase tests…"
        )

        packages = _discover_phase_packages()
        if not packages:
            flow_logger.bind(tag="[STARTUP_TESTS]").warning(
                "⚠  No phase test packages found under app/tests/phase_*"
            )
            return True

        reports: List[PhaseReport] = []
        t_global_start = time.perf_counter()

        for pkg_path in packages:
            try:
                phase_id, phase_name, tests = _load_tests_from_package(pkg_path)
            except Exception as exc:
                flow_logger.bind(tag="[STARTUP_TESTS]").error(
                    f"  ⚡  Failed to load package {pkg_path}: {exc}"
                )
                continue

            if not tests:
                flow_logger.bind(tag="[STARTUP_TESTS]").warning(
                    f"  ─  Phase {phase_id} ({phase_name}): no tests defined"
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
