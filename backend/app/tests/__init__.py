"""
CodeLens AI — Phase Test Suite
================================
Auto-discovered by StartupTestRunner at server startup.

Every sub-package here maps 1-to-1 to a LangGraph modernisation phase:
    phase_a  → LangGraph Foundation
    phase_b  → Multi-Agent Supervisor System
    phase_c  → Memory Architecture
    phase_d  → Checkpointing & Time-Travel
    phase_e  → Human-in-the-Loop
    phase_f  → Middleware & Guardrails
    phase_g  → Streaming & API Layer
    phase_h  → Runtime Observability

Each module inside a phase package must expose:
    PHASE_NAME  : str   — human-readable phase name
    TESTS       : list[PhaseTest]  — test definitions (from base.py)
"""
