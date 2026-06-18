"""
CodeLens AI — LangGraph Multi-Agent Graph Package
==================================================
Phase A: LangGraph Foundation

This package contains:
  state.py             — AgentState TypedDict (the graph's shared state)
  supervisor_graph.py  — Root StateGraph wiring all agents together
  streaming.py         — SSEEvent, stream_graph_events, format_sse
  nodes/               — Pure async node functions
  agents/              — Compiled agent sub-graphs (Phase B)
  memory/              — Short-term and long-term memory nodes (Phase C)
  checkpointing/       — PostgresSaver configuration (Phase D)
  guardrails/          — Input/output guardrail nodes (Phase F)
  middleware/          — Node middleware hooks (Phase F)
"""
