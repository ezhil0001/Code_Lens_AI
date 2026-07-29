"""
CodeLens AI — Startup test suites.

These run automatically at server boot via StartupTestRunner.run_all().
Each sub-package covers a specific area of the system and exposes a TESTS
list that the runner collects. Adding a new test is just adding a new
PhaseTest entry — no registration needed anywhere else.

    test_graph_foundation  → state schema, graph compilation, intent routing
    test_agent_supervisor  → agent sub-graphs, supervisor node wiring
    test_memory_layer      → STM window, LTM retrieval, entity extraction
    test_checkpointing     → PostgresSaver, thread IDs, time-travel API routes
    test_hil_workflow      → HIL trigger conditions, resume endpoint
    test_guardrails        → injection detection, PII scrubbing, code safety scan
    test_streaming_api     → SSE endpoints, schema validation, event format
    test_observability     → Langfuse tracing + OpenTelemetry (Jaeger) modules

Each module must export:
    SUITE_NAME  : str              — shown in the startup report header
    TESTS       : list[PhaseTest]  — test definitions (see base.py)
"""
