"""
Observability Tests
===================
  H-001  langgraph_instrumentation module importable
  H-002  NODE_LATENCY_MS histogram declared
  H-003  AGENT_TOKENS histogram declared
  H-004  GRAPH_EDGES_TRAVERSED histogram declared
  H-005  HIL_INTERRUPTS counter declared
  H-006  LTM_LOOKUPS counter declared
  H-007  GUARDRAIL_EVENTS counter declared
  H-008  All histogram label names include 'node_name' or 'agent_name'
  H-009  alert-rules.yml contains AgentDeadlock rule
  H-010  alert-rules.yml contains HILBacklog rule
  H-011  alert-rules.yml contains GuardrailBlockSurge rule
  H-012  agent-activity-dashboard.json exists in grafana/dashboards/
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

from app.tests.base import PhaseTest, TestResult


def _try_import(path: str):
    try:
        return importlib.import_module(path), None
    except Exception as exc:
        return None, str(exc)


# ─────────────────────────────────────────────────────────────────────────────

async def _test_instrumentation_importable() -> TestResult:
    mod, err = _try_import("app.observability.langgraph_instrumentation")
    if err:
        return TestResult.failed(f"Cannot import langgraph_instrumentation: {err}")
    return TestResult.passed("langgraph_instrumentation importable ✓")


def _check_metric(metric_name: str) -> TestResult:
    mod, err = _try_import("app.observability.quality_metrics")
    if err:
        return TestResult.failed(f"Cannot import quality_metrics: {err}")
    if not hasattr(mod, metric_name):
        return TestResult.failed(
            f"{metric_name} not declared in quality_metrics.py",
            detail=f"Add at module scope: {metric_name} = Histogram/Counter(...)"
        )
    return TestResult.passed(f"{metric_name} declared ✓")


async def _test_node_latency_histogram() -> TestResult:
    return _check_metric("NODE_LATENCY_MS")

async def _test_agent_tokens_histogram() -> TestResult:
    return _check_metric("AGENT_TOKENS")

async def _test_graph_edges_histogram() -> TestResult:
    return _check_metric("GRAPH_EDGES_TRAVERSED")

async def _test_hil_interrupts_counter() -> TestResult:
    return _check_metric("HIL_INTERRUPTS")

async def _test_ltm_lookups_counter() -> TestResult:
    return _check_metric("LTM_LOOKUPS")

async def _test_guardrail_events_counter() -> TestResult:
    return _check_metric("GUARDRAIL_EVENTS")


async def _test_histogram_label_names() -> TestResult:
    mod, err = _try_import("app.observability.quality_metrics")
    if err:
        return TestResult.skipped("quality_metrics not importable")

    failures = []
    for attr_name, expected_label in [
        ("NODE_LATENCY_MS",    "node_name"),
        ("AGENT_TOKENS",       "agent_name"),
        ("HIL_INTERRUPTS",     "reason"),
        ("LTM_LOOKUPS",        "result"),
        ("GUARDRAIL_EVENTS",   "check_name"),
    ]:
        metric = getattr(mod, attr_name, None)
        if metric is None:
            failures.append(f"{attr_name} not found")
            continue
        # Prometheus client stores labelnames on _labelnames or labelnames
        labels = (
            getattr(metric, "_labelnames", None)
            or getattr(metric, "labelnames", None)
            or []
        )
        if expected_label not in labels:
            failures.append(
                f"{attr_name} missing label '{expected_label}' (has: {list(labels)})"
            )
    if failures:
        return TestResult.failed(
            f"Label name check failures: {'; '.join(failures)}"
        )
    return TestResult.passed("All metric label names correct ✓")


def _alert_rules_path() -> Path:
    base = Path(__file__).parent.parent.parent.parent.parent  # backend/../
    return base / "alert-rules.yml"


async def _test_alert_agent_deadlock() -> TestResult:
    path = _alert_rules_path()
    if not path.exists():
        return TestResult.failed(f"alert-rules.yml not found at {path}")
    content = path.read_text()
    if "AgentDeadlock" not in content:
        return TestResult.failed(
            "AgentDeadlock alert not found in alert-rules.yml"
        )
    return TestResult.passed("AgentDeadlock alert rule present ✓")


async def _test_alert_hil_backlog() -> TestResult:
    path = _alert_rules_path()
    if not path.exists():
        return TestResult.failed(f"alert-rules.yml not found at {path}")
    content = path.read_text()
    if "HILBacklog" not in content:
        return TestResult.failed(
            "HILBacklog alert not found in alert-rules.yml"
        )
    return TestResult.passed("HILBacklog alert rule present ✓")


async def _test_alert_guardrail_block_surge() -> TestResult:
    path = _alert_rules_path()
    if not path.exists():
        return TestResult.failed(f"alert-rules.yml not found at {path}")
    content = path.read_text()
    if "GuardrailBlockSurge" not in content:
        return TestResult.failed(
            "GuardrailBlockSurge alert not found in alert-rules.yml"
        )
    return TestResult.passed("GuardrailBlockSurge alert rule present ✓")


async def _test_agent_activity_dashboard_exists() -> TestResult:
    base = Path(__file__).parent.parent.parent.parent.parent
    dashboard_path = base / "grafana" / "dashboards" / "agent-activity-dashboard.json"
    if not dashboard_path.exists():
        return TestResult.failed(
            f"agent-activity-dashboard.json not found at {dashboard_path}",
            detail="Create grafana/dashboards/agent-activity-dashboard.json"
        )
    content = dashboard_path.read_text()
    if "langgraph" not in content.lower():
        return TestResult.failed(
            "agent-activity-dashboard.json does not reference langgraph metrics"
        )
    return TestResult.passed("agent-activity-dashboard.json exists with LangGraph panels ✓")


TESTS: list[PhaseTest] = [
    PhaseTest(id="H-001", name="langgraph_instrumentation importable",
              description="app.observability.langgraph_instrumentation importable",
              run=_test_instrumentation_importable, critical=False, tags=["obs"]),
    PhaseTest(id="H-002", name="NODE_LATENCY_MS histogram declared",
              description="Module-scope Prometheus Histogram for per-node latency",
              run=_test_node_latency_histogram, critical=True, tags=["obs", "metrics"]),
    PhaseTest(id="H-003", name="AGENT_TOKENS histogram declared",
              description="Module-scope Prometheus Histogram for token usage",
              run=_test_agent_tokens_histogram, critical=False, tags=["obs", "metrics"]),
    PhaseTest(id="H-004", name="GRAPH_EDGES_TRAVERSED histogram declared",
              description="Module-scope Prometheus Histogram for edge count",
              run=_test_graph_edges_histogram, critical=False, tags=["obs", "metrics"]),
    PhaseTest(id="H-005", name="HIL_INTERRUPTS counter declared",
              description="Module-scope Prometheus Counter for HIL events",
              run=_test_hil_interrupts_counter, critical=False, tags=["obs", "metrics"]),
    PhaseTest(id="H-006", name="LTM_LOOKUPS counter declared",
              description="Module-scope Prometheus Counter for LTM hit/miss",
              run=_test_ltm_lookups_counter, critical=False, tags=["obs", "metrics"]),
    PhaseTest(id="H-007", name="GUARDRAIL_EVENTS counter declared",
              description="Module-scope Prometheus Counter for guardrail actions",
              run=_test_guardrail_events_counter, critical=False, tags=["obs", "metrics"]),
    PhaseTest(id="H-008", name="Histogram/Counter label names correct",
              description="Each metric has its required label names",
              run=_test_histogram_label_names, critical=True, tags=["obs", "metrics"]),
    PhaseTest(id="H-009", name="AgentDeadlock alert in alert-rules.yml",
              description="alert-rules.yml contains AgentDeadlock rule",
              run=_test_alert_agent_deadlock, critical=False, tags=["obs", "alerts"]),
    PhaseTest(id="H-010", name="HILBacklog alert in alert-rules.yml",
              description="alert-rules.yml contains HILBacklog rule",
              run=_test_alert_hil_backlog, critical=False, tags=["obs", "alerts"]),
    PhaseTest(id="H-011", name="GuardrailBlockSurge alert in alert-rules.yml",
              description="alert-rules.yml contains GuardrailBlockSurge rule",
              run=_test_alert_guardrail_block_surge, critical=False, tags=["obs", "alerts"]),
    PhaseTest(id="H-012", name="agent-activity-dashboard.json exists",
              description="Grafana dashboard file present with LangGraph panels",
              run=_test_agent_activity_dashboard_exists, critical=False, tags=["obs", "grafana"]),
]
