"""
Supervisor graph — root StateGraph that orchestrates the full query pipeline.

Entry point for every v2 chat request.  Builds the graph once at startup and
reuses the compiled instance across all requests via get_supervisor_graph().

Node execution order:
  input_guardrail_node  → safety checks (injection, PII, token budget)
  cache_check_node      → semantic cache lookup; hit → skip to response_node
  memory_read_node      → load STM window + LTM facts
  intent_classifier_node → classify query, set routing_decision + metadata_filter
  [agent nodes]         → CodeAgent / DocAgent / DebugAgent / ArchAgent / WebAgent
  synthesizer_node      → merge multi-agent outputs; single-agent is a pass-through
  hil_check_node        → pause for human review if conditions are met
  output_guardrail_node → code safety scan + PII leak scan + citation warnings
  response_node         → assemble final SSE payload, write to cache + memory

Conditional routing at intent_classifier_node dispatches to the correct
agent based on routing_decision.  HYBRID queries go through CodeAgent first;
the supervisor can be extended to run agents in parallel via the Send API
when that latency becomes a bottleneck.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from app.graph.nodes.intent_classifier import intent_classifier_node
from app.graph.nodes.synthesizer import synthesizer_node as _real_synthesizer_node
from app.graph.state import AgentState

# ── Phase B: import real agent sub-graphs ────────────────────────────────────
try:
    from app.graph.agents.code_agent import build_code_agent as _build_code_agent
    from app.graph.agents.doc_agent import build_doc_agent as _build_doc_agent
    from app.graph.agents.debug_agent import build_debug_agent as _build_debug_agent
    from app.graph.agents.arch_agent import build_arch_agent as _build_arch_agent
    from app.graph.agents.web_agent import build_web_agent as _build_web_agent
    _AGENTS_AVAILABLE = True
except Exception as _agent_import_err:  # noqa: BLE001
    logger = logging.getLogger(__name__)
    logger.warning("[SUPERVISOR] Agent sub-graphs not available: %s", _agent_import_err)
    _AGENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight pass-through nodes for pipeline stages that have not yet been
# wired to a real implementation.  Each returns an empty dict so the graph
# compiles and produces valid checkpoints — useful for testing routing logic
# without requiring every service to be running.
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
    """Phase B — CodeAgent: delegates to compiled sub-graph."""
    if _AGENTS_AVAILABLE:
        try:
            graph = _build_code_agent()
            result_state = await graph.ainvoke(state, config or {})
            keys = ("active_agent", "agent_responses", "sources",
                    "retrieved_chunks", "reranked_chunks", "nodes_visited", "tool_calls")
            return {k: result_state[k] for k in keys if k in result_state}
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CodeAgent] sub-graph failed: %s", exc)
    visited = list(state.get("nodes_visited", []))
    visited.append("CodeAgent:fallback")
    agent_responses = dict(state.get("agent_responses", {}))
    agent_responses["CodeAgent"] = "[CodeAgent unavailable]"
    return {"active_agent": "CodeAgent", "agent_responses": agent_responses,
            "nodes_visited": visited}


async def doc_agent_node(state: dict, config: RunnableConfig = None) -> dict:
    """Phase B — DocAgent: delegates to compiled sub-graph."""
    if _AGENTS_AVAILABLE:
        try:
            graph = _build_doc_agent()
            result_state = await graph.ainvoke(state, config or {})
            keys = ("active_agent", "agent_responses", "sources",
                    "retrieved_chunks", "reranked_chunks", "nodes_visited")
            return {k: result_state[k] for k in keys if k in result_state}
        except Exception as exc:  # noqa: BLE001
            logger.warning("[DocAgent] sub-graph failed: %s", exc)
    visited = list(state.get("nodes_visited", []))
    visited.append("DocAgent:fallback")
    agent_responses = dict(state.get("agent_responses", {}))
    agent_responses["DocAgent"] = "[DocAgent unavailable]"
    return {"active_agent": "DocAgent", "agent_responses": agent_responses,
            "nodes_visited": visited}


async def debug_agent_node(state: dict, config: RunnableConfig = None) -> dict:
    """Phase B — DebugAgent: delegates to compiled sub-graph."""
    if _AGENTS_AVAILABLE:
        try:
            graph = _build_debug_agent()
            result_state = await graph.ainvoke(state, config or {})
            keys = ("active_agent", "agent_responses", "sources",
                    "retrieved_chunks", "nodes_visited", "tool_calls")
            return {k: result_state[k] for k in keys if k in result_state}
        except Exception as exc:  # noqa: BLE001
            logger.warning("[DebugAgent] sub-graph failed: %s", exc)
    visited = list(state.get("nodes_visited", []))
    visited.append("DebugAgent:fallback")
    agent_responses = dict(state.get("agent_responses", {}))
    agent_responses["DebugAgent"] = "[DebugAgent unavailable]"
    return {"active_agent": "DebugAgent", "agent_responses": agent_responses,
            "nodes_visited": visited}


async def arch_agent_node(state: dict, config: RunnableConfig = None) -> dict:
    """Phase B — ArchAgent: delegates to compiled sub-graph."""
    if _AGENTS_AVAILABLE:
        try:
            graph = _build_arch_agent()
            result_state = await graph.ainvoke(state, config or {})
            keys = ("active_agent", "agent_responses", "sources",
                    "retrieved_chunks", "nodes_visited")
            return {k: result_state[k] for k in keys if k in result_state}
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ArchAgent] sub-graph failed: %s", exc)
    visited = list(state.get("nodes_visited", []))
    visited.append("ArchAgent:fallback")
    agent_responses = dict(state.get("agent_responses", {}))
    agent_responses["ArchAgent"] = "[ArchAgent unavailable]"
    return {"active_agent": "ArchAgent", "agent_responses": agent_responses,
            "nodes_visited": visited}


async def web_agent_node(state: dict, config: RunnableConfig = None) -> dict:
    """Phase B — WebAgent: delegates to compiled sub-graph."""
    if _AGENTS_AVAILABLE:
        try:
            graph = _build_web_agent()
            result_state = await graph.ainvoke(state, config or {})
            keys = ("active_agent", "agent_responses", "sources",
                    "tool_results", "nodes_visited")
            return {k: result_state[k] for k in keys if k in result_state}
        except Exception as exc:  # noqa: BLE001
            logger.warning("[WebAgent] sub-graph failed: %s", exc)
    visited = list(state.get("nodes_visited", []))
    visited.append("WebAgent:fallback")
    agent_responses = dict(state.get("agent_responses", {}))
    agent_responses["WebAgent"] = "[WebAgent unavailable]"
    return {"active_agent": "WebAgent", "agent_responses": agent_responses,
            "nodes_visited": visited}


async def synthesizer_node(state: dict, config: RunnableConfig = None) -> dict:
    """Phase B — delegates to app.graph.nodes.synthesizer.synthesizer_node."""
    return await _real_synthesizer_node(state, config)


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
