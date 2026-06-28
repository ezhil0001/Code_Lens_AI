"""
CodeLens AI — LangGraph orchestration layer.

The supervisor graph is the single entry point for every v2 chat request.
It routes queries through the right agent, applies guardrails on both ends,
loads conversation memory, and streams the final response back via SSE.

Layout:
  state.py             — AgentState TypedDict shared across all nodes
  supervisor_graph.py  — Root StateGraph that wires everything together
  streaming.py         — SSEEvent, stream_graph_events, format_sse

  nodes/               — Shared stateless nodes (classifier, synthesizer, HIL)
  agents/              — One compiled sub-graph per agent type
  memory/              — Short-term window loader + long-term pgvector store
  checkpointing/       — PostgresSaver setup and thread-ID helpers
  guardrails/          — Input and output guardrail nodes
  middleware/          — Per-node retry and logging middleware hooks
"""
