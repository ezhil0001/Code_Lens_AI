"""
Tests for runtime observability — Prometheus metrics and Grafana dashboards.

Every Prometheus instrument (histograms, counters) is checked at startup
so a typo in a metric name is caught immediately rather than silently
producing a flat line in Grafana for days before someone notices.
"""

SUITE_ID   = "H"
SUITE_NAME = "Observability"

from app.tests.test_observability.tests import TESTS  # noqa: E402, F401
