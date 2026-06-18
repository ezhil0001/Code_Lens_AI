"""
Supervisor Graph — Root LangGraph StateGraph (Phase A: F-02, F-04)
===================================================================
Wires all node stubs and agent sub-graphs together into the Supervisor
StateGraph.  Phase A ships stubs for nodes that will be implemented in
later phases — each stub is a valid async node that logs and returns
an unchanged (empty) state delta so the graph compiles and runs.

Node execution sequence (full pipeline):
  __start__
      │
      ▼
  input_guardrail_node       (Phase F — stub for now)
      │
      ▼
  cache_check_node           (Phase G — stub for now)
      │  HIT ────────────────────────────────────────► response_node
      │  MISS
      ▼
  memory_read_node           (Phase C — stub for now)
      │
      ▼
  intent_classifier_node     (Phase A — IMPLEMENTED)
      │
      ├── "CodeAgent"  ──────► code_agent_node   (Phase B — stub)
      ├── "DocAgent"   ──────► doc_agent_node    (Phase B — stub)
      ├── "DebugAgent" ──────► debug_agent_node  (Phase B — stub)
      ├── "ArchAgent"  ──────► arch_agent_node   (Phase B — stub)
      └── "WebAgent"   ──────► web_agent_node    (Phase B — stub)
              │
              ▼
      synthesizer_node       (Phase B — stub)
              │
              ▼
      hil_check_node         (Phase E — stub)
              │
              ▼
      output_guardrail_node  (Phase F — stub)
              │
              ▼
      response_node          (Phase G — stub)
              │
              ▼
          __end__

IMPORTANT DESIGN RULES (preserved across all phases)
-----------------------------------------------------
1. Every node is a pure async function: (state: dict, config: dict) -> dict.
2. Nodes return ONLY the fields they changed — never the full state.
3. `query` is set at entry and never modified by downstream nodes.
4. `session_id` is always the namespaced "{user_id}::{raw_session_id}" form.
5. CancelledError must propagate — never catch bare `except Exception` in
   streaming contexts.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from app.graph.nodes.intent_classifier import intent_classifier_node
from app.graph.state import AgentState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Phase A stubs — minimal async nodes that pass state unchanged.
# Each stub will be replaced with a real implementation in the phase
# documented in its docstring.
# ─────────────────────────────────────────────────────────────────────────────

async def _stub_node(name: str, state: dict, config: RunnableConfig) -> dict:
    """Generic stub: logs, appends to nodes_visited, returns empty delta."""
    logger.debug("[STUB] %s executed (not yet implemented)", name)
    visited = list(state.get("nodes_visited", []))
    visited.append(f"{name}:stub")
    return {"nodes_visited": visited}


async def input_guardrail_node(state: dict, config: RunnableConfig = None) -> dict:
    """Phase F stub — passes all inputs through."""
    return await _stub_node("input_guardrail_node", state, config or {})


async def cache_check_node(state: dict, config: RunnableConfig = None) -> dict:
    """Phase G stub — always reports cache miss."""
    visited = list(state.get("nodes_visited", []))
    visited.append("cache_check_node:stub")
    return {"cache_hit": False, "nodes_visited": visited}


async def memory_read_node(state: dict, config: RunnableConfig = None) -> dict:
    """Phase C stub — returns empty memory fields."""
    return await _stub_node("memory_read_node", state, config or {})


async def code_agent_node(state: dict, config: RunnableConfig = None) -> dict:
    """Phase B stub — CodeAgent sub-graph placeholder."""
    visited = list(state.get("nodes_visited", []))
    visited.append("CodeAgent:stub")
    agent_responses = dict(state.get("agent_responses", {}))
    agent_responses["CodeAgent"] = "[CodeAgent stub — Phase B not yet implemented]"
    return {
        "active_agent": "CodeAgent",
        "agent_responses": agent_responses,
        "nodes_visited": visited,
    }


async def doc_agent_node(state: dict, config: RunnableConfig = None) -> dict:
    """Phase B stub — DocAgent sub-graph placeholder."""
    visited = list(state.get("nodes_visited", []))
    visited.append("DocAgent:stub")
    agent_responses = dict(state.get("agent_responses", {}))
    agent_responses["DocAgent"] = "[DocAgent stub — Phase B not yet implemented]"
    return {
        "active_agent": "DocAgent",
        "agent_responses": agent_responses,
        "nodes_visited": visited,
    }


async def debug_agent_node(state: dict, config: RunnableConfig = None) -> dict:
    """Phase B stub — DebugAgent sub-graph placeholder."""
    visited = list(state.get("nodes_visited", []))
    visited.append("DebugAgent:stub")
    agent_responses = dict(state.get("agent_responses", {}))
    agent_responses["DebugAgent"] = "[DebugAgent stub — Phase B not yet implemented]"
    return {
        "active_agent": "DebugAgent",
        "agent_responses": agent_responses,
        "nodes_visited": visited,
    }


async def arch_agent_node(state: dict, config: RunnableConfig = None) -> dict:
    """Phase B stub — ArchAgent sub-graph placeholder."""
    visited = list(state.get("nodes_visited", []))
    visited.append("ArchAgent:stub")
    agent_responses = dict(state.get("agent_responses", {}))
    agent_responses["ArchAgent"] = "[ArchAgent stub — Phase B not yet implemented]"
    return {
        "active_agent": "ArchAgent",
        "agent_responses": agent_responses,
        "nodes_visited": visited,
    }


async def web_agent_node(state: dict, config: RunnableConfig = None) -> dict:
    """Phase B stub — WebAgent sub-graph placeholder."""
    visited = list(state.get("nodes_visited", []))
    visited.append("WebAgent:stub")
    agent_responses = dict(state.get("agent_responses", {}))
    agent_responses["WebAgent"] = "[WebAgent stub — Phase B not yet implemented]"
    return {
        "active_agent": "WebAgent",
        "agent_responses": agent_responses,
        "nodes_visited": visited,
    }


async def synthesizer_node(state: dict, config: RunnableConfig = None) -> dict:
    """Phase B stub — merges agent responses (single-agent path: direct copy)."""
    agent_responses: dict = state.get("agent_responses", {})
    active_agent: Optional[str] = state.get("active_agent")
    visited = list(state.get("nodes_visited", []))
    visited.append("synthesizer_node:stub")

    # Single-agent shortcut: copy the one response without an LLM call
    if active_agent and active_agent in agent_responses:
        final = agent_responses[active_agent]
    elif agent_responses:
        # Fallback: join all responses
        final = "\n\n---\n\n".join(agent_responses.values())
    else:
        final = "[No response generated]"

    return {"final_response": final, "nodes_visited": visited}


async def hil_check_node(state: dict, config: RunnableConfig = None) -> dict:
    """Phase E stub — never interrupts."""
    return await _stub_node("hil_check_node", state, config or {})


async def output_guardrail_node(state: dict, config: RunnableConfig = None) -> dict:
    """Phase F stub — passes all outputs through."""
    return await _stub_node("output_guardrail_node", state, config or {})


async def response_node(state: dict, config: RunnableConfig = None) -> dict:
    """Phase G stub — logs the final response and marks evaluation queued."""
    final = state.get("final_response", "[No response]")
    logger.info("[RESPONSE_NODE] final_response length=%d chars", len(final))
    visited = list(state.get("nodes_visited", []))
    visited.append("response_node:stub")
    return {"evaluation_queued": True, "nodes_visited": visited}


# ─────────────────────────────────────────────────────────────────────────────
# Routing functions (conditional edges)
# ─────────────────────────────────────────────────────────────────────────────

def route_cache(state: dict) -> str:
    """After cache_check_node: hit → response_node, miss → memory_read_node."""
    if state.get("cache_hit", False):
        return "response_node"
    return "memory_read_node"


def route_to_agent(state: dict) -> str:
    """
    After intent_classifier_node: dispatch to the correct agent node.

    Falls back to CodeAgent if routing_decision is unrecognised — this
    matches the existing AgenticRouter fallback behaviour.
    """
    if state.get("cache_hit", False):
        return "response_node"

    decision = state.get("routing_decision", "CodeAgent")

    _valid = {"CodeAgent", "DocAgent", "DebugAgent", "ArchAgent", "WebAgent"}
    if decision not in _valid:
        logger.warning(
            "[SUPERVISOR] Unknown routing_decision=%r — falling back to CodeAgent",
            decision,
        )
        return "CodeAgent"

    return decision


def route_hil(state: dict) -> str:
    """After hil_check_node: interrupt → (graph pauses), else → output_guardrail."""
    if state.get("hil_required", False):
        # LangGraph handles the actual interrupt via interrupt_before at compile time.
        # This edge is only reached when hil_required is False (normal path).
        return "output_guardrail_node"
    return "output_guardrail_node"


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_supervisor_graph(
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> Any:
    """
    Build and compile the Supervisor StateGraph.

    Parameters
    ----------
    checkpointer:
        A LangGraph checkpoint saver.  Pass a ``MemorySaver`` for tests or
        an ``AsyncPostgresSaver`` for production.  If ``None`` is passed
        the graph is compiled without checkpointing (stateless).

    Returns
    -------
    CompiledGraph
        A compiled LangGraph graph ready to call ``.astream_events()``,
        ``.ainvoke()``, or ``.astream()`` on.
    """
    builder = StateGraph(AgentState)

    # ── Register all nodes ────────────────────────────────────────────────────
    builder.add_node("input_guardrail_node", input_guardrail_node)
    builder.add_node("cache_check_node",     cache_check_node)
    builder.add_node("memory_read_node",     memory_read_node)
    builder.add_node("intent_classifier_node", intent_classifier_node)

    # Agent sub-graphs (stubs — replaced Phase B)
    builder.add_node("CodeAgent",  code_agent_node)
    builder.add_node("DocAgent",   doc_agent_node)
    builder.add_node("DebugAgent", debug_agent_node)
    builder.add_node("ArchAgent",  arch_agent_node)
    builder.add_node("WebAgent",   web_agent_node)

    # Post-agent nodes
    builder.add_node("synthesizer_node",      synthesizer_node)
    builder.add_node("hil_check_node",        hil_check_node)
    builder.add_node("output_guardrail_node", output_guardrail_node)
    builder.add_node("response_node",         response_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.set_entry_point("input_guardrail_node")

    # ── Edges ─────────────────────────────────────────────────────────────────
    builder.add_edge("input_guardrail_node", "cache_check_node")

    # Cache hit → skip all retrieval
    builder.add_conditional_edges(
        "cache_check_node",
        route_cache,
        {
            "memory_read_node": "memory_read_node",
            "response_node":    "response_node",
        },
    )

    builder.add_edge("memory_read_node", "intent_classifier_node")

    # Dispatch to agent
    builder.add_conditional_edges(
        "intent_classifier_node",
        route_to_agent,
        {
            "CodeAgent":  "CodeAgent",
            "DocAgent":   "DocAgent",
            "DebugAgent": "DebugAgent",
            "ArchAgent":  "ArchAgent",
            "WebAgent":   "WebAgent",
            "response_node": "response_node",
        },
    )

    # All agents → synthesizer
    for _agent in ("CodeAgent", "DocAgent", "DebugAgent", "ArchAgent", "WebAgent"):
        builder.add_edge(_agent, "synthesizer_node")

    builder.add_edge("synthesizer_node", "hil_check_node")

    builder.add_conditional_edges(
        "hil_check_node",
        route_hil,
        {"output_guardrail_node": "output_guardrail_node"},
    )

    builder.add_edge("output_guardrail_node", "response_node")
    builder.add_edge("response_node", END)

    # ── Compile ───────────────────────────────────────────────────────────────
    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    graph = builder.compile(**compile_kwargs)

    logger.info(
        "[SUPERVISOR] Graph compiled with %d nodes, checkpointer=%s",
        len(builder.nodes),
        type(checkpointer).__name__ if checkpointer else "None",
    )
    return graph
