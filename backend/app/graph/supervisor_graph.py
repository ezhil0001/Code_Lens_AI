"""
Supervisor graph — root StateGraph that orchestrates the full query pipeline.

Entry point for every v2 chat request.  Builds the graph once at startup and
reuses the compiled instance across all requests via get_supervisor_graph().

Node execution order:
  input_guardrail_node    → safety checks (injection, PII, token budget)
  cache_check_node        → semantic cache lookup; hit → skip to response_node
  memory_read_node        → load STM window + LTM facts
  intent_classifier_node  → LLM routing; returns 1–2 agent names
  [agent nodes]           → dispatched in parallel via Send() for compound queries
  synthesizer_node        → merge outputs; single-agent is a pass-through
  hil_check_node          → pause for human review if confidence is low
  output_guardrail_node   → code safety scan + PII leak scan + citation warnings
  response_node           → assemble final SSE payload, write to cache + memory

Parallel fan-out
----------------
intent_classifier_node populates state["routing_agents"] with 1–2 agent names.
dispatch_agents() reads that list and returns a list of Send() objects, one per
agent.  LangGraph executes all Send()s in the same superstep so two agents run
concurrently.  Each agent writes its answer into agent_responses under its own
key; the _merge_agent_responses reducer in AgentState combines the dicts so no
answer is lost.  The synthesizer then receives the full set.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from app.graph.nodes.intent_classifier import intent_classifier_node
from app.graph.nodes.synthesizer import synthesizer_node as _real_synthesizer_node
from app.graph.state import AgentState

# ── Real implementations for guardrail / HIL / memory-write nodes ─────────────
try:
    from app.graph.guardrails.input_guardrail import input_guardrail_node as _real_input_guardrail
    _INPUT_GUARDRAIL_AVAILABLE = True
except Exception as _e:
    logger_tmp = logging.getLogger(__name__)
    logger_tmp.warning("[SUPERVISOR] input_guardrail not available: %s", _e)
    _INPUT_GUARDRAIL_AVAILABLE = False

try:
    from app.graph.nodes.hil_node import hil_check_node as _real_hil_check
    _HIL_AVAILABLE = True
except Exception as _e:
    logger_tmp = logging.getLogger(__name__)
    logger_tmp.warning("[SUPERVISOR] hil_node not available: %s", _e)
    _HIL_AVAILABLE = False

try:
    from app.graph.guardrails.output_guardrail import output_guardrail_node as _real_output_guardrail
    _OUTPUT_GUARDRAIL_AVAILABLE = True
except Exception as _e:
    logger_tmp = logging.getLogger(__name__)
    logger_tmp.warning("[SUPERVISOR] output_guardrail not available: %s", _e)
    _OUTPUT_GUARDRAIL_AVAILABLE = False

try:
    from app.graph.memory.entity_extractor import memory_write_node as _real_memory_write
    _MEMORY_WRITE_AVAILABLE = True
except Exception as _e:
    logger_tmp = logging.getLogger(__name__)
    logger_tmp.warning("[SUPERVISOR] memory_write_node not available: %s", _e)
    _MEMORY_WRITE_AVAILABLE = False

try:
    from app.graph.memory.short_term import memory_read_node as _real_memory_read
    _MEMORY_READ_AVAILABLE = True
except Exception as _e:
    logger_tmp = logging.getLogger(__name__)
    logger_tmp.warning("[SUPERVISOR] short_term.memory_read_node not available: %s", _e)
    _MEMORY_READ_AVAILABLE = False

# ── Import real agent sub-graphs — failures are soft so the server still boots
# even if an optional dependency (e.g. web-search library) is missing.
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


# ── H-5: memoized agent sub-graph compilation ────────────────────────────────
# Sub-graphs were previously rebuilt (StateGraph.compile) on EVERY request in
# each agent node. Compiled LangGraph graphs are immutable and thread-safe to
# reuse, so compile each one once per process.
import threading as _threading

_subgraph_cache: dict[str, Any] = {}
_subgraph_lock = _threading.Lock()


def _get_cached_subgraph(name: str, builder) -> Any:
    """Return the compiled sub-graph for *name*, compiling at most once."""
    g = _subgraph_cache.get(name)
    if g is not None:
        return g
    with _subgraph_lock:
        g = _subgraph_cache.get(name)
        if g is None:
            g = builder()
            _subgraph_cache[name] = g
            logger.info("[SUPERVISOR] %s sub-graph compiled and cached", name)
    return g


def _current_langfuse_trace_id() -> Optional[str]:
    """Return the active Langfuse trace ID, or None. Never raises."""
    try:
        from app.observability.langfuse_client import get_current_trace_id
        return get_current_trace_id()
    except Exception:  # noqa: BLE001
        return None


def _log_eval_future_result(fut: "Any") -> None:
    """Done-callback for the background RAGAS evaluation future.

    Ensures exceptions raised inside the executor are logged instead of being
    silently swallowed (C-1). Never raises.
    """
    try:
        exc = fut.exception()
    except Exception:  # noqa: BLE001  (future cancelled)
        return
    if exc is not None:
        logger.warning("[RESPONSE_NODE] RAGAS evaluation failed in background: %s", exc)
    else:
        logger.debug("[RESPONSE_NODE] RAGAS evaluation completed")



# ─────────────────────────────────────────────────────────────────────────────
# Thin wrapper nodes — each one tries the real implementation first and falls
# back to a no-op stub so the graph compiles and checkpoints correctly even
# when an optional service (Postgres, external API) is unavailable at boot.
# ─────────────────────────────────────────────────────────────────────────────

async def _stub_node(name: str, state: dict, config: RunnableConfig) -> dict:
    """No-op fallback used when the real node implementation couldn't be imported."""
    logger.debug("[STUB] %s — real impl not available, returning empty delta", name)
    visited = list(state.get("nodes_visited", []))
    visited.append(f"{name}:stub")
    return {"nodes_visited": visited}


async def input_guardrail_node(state: dict, config: RunnableConfig = None) -> dict:
    """Delegate to real input_guardrail_node or stub."""
    if _INPUT_GUARDRAIL_AVAILABLE:
        return await _real_input_guardrail(state, config)
    return await _stub_node("input_guardrail_node", state, config or {})


async def cache_check_node(state: dict, config: RunnableConfig = None) -> dict:
    """Semantic cache lookup — always reports a miss until the cache service wires in."""
    visited = list(state.get("nodes_visited", []))
    visited.append("cache_check_node:stub")
    return {"cache_hit": False, "nodes_visited": visited}


async def memory_read_node(state: dict, config: RunnableConfig = None) -> dict:
    """Delegate to real memory_read_node (short_term.py) or stub."""
    if _MEMORY_READ_AVAILABLE:
        return await _real_memory_read(state, config)
    return await _stub_node("memory_read_node", state, config or {})


async def code_agent_node(state: dict, config: RunnableConfig = None) -> dict:
    """Run the CodeAgent sub-graph; falls back to an error message if unavailable."""
    if _AGENTS_AVAILABLE:
        try:
            graph = _get_cached_subgraph("CodeAgent", _build_code_agent)
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
    """Run the DocAgent sub-graph; falls back to an error message if unavailable."""
    if _AGENTS_AVAILABLE:
        try:
            graph = _get_cached_subgraph("DocAgent", _build_doc_agent)
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
    """Run the DebugAgent sub-graph; falls back to an error message if unavailable."""
    if _AGENTS_AVAILABLE:
        try:
            graph = _get_cached_subgraph("DebugAgent", _build_debug_agent)
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
    """Run the ArchAgent sub-graph; falls back to an error message if unavailable."""
    if _AGENTS_AVAILABLE:
        try:
            graph = _get_cached_subgraph("ArchAgent", _build_arch_agent)
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
    """Run the WebAgent sub-graph; falls back to an error message if unavailable."""
    if _AGENTS_AVAILABLE:
        try:
            graph = _get_cached_subgraph("WebAgent", _build_web_agent)
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
    """Merge multi-agent outputs into a single final_response; single-agent is a pass-through."""
    return await _real_synthesizer_node(state, config)


async def hil_check_node(state: dict, config: RunnableConfig = None) -> dict:
    """Delegate to real hil_check_node or stub."""
    if _HIL_AVAILABLE:
        return await _real_hil_check(state, config)
    return await _stub_node("hil_check_node", state, config or {})


async def output_guardrail_node(state: dict, config: RunnableConfig = None) -> dict:
    """Delegate to real output_guardrail_node or stub."""
    if _OUTPUT_GUARDRAIL_AVAILABLE:
        return await _real_output_guardrail(state, config)
    return await _stub_node("output_guardrail_node", state, config or {})


async def response_node(state: dict, config: RunnableConfig = None) -> dict:
    """Assemble final response and fire background RAGAS evaluation."""
    final = state.get("final_response", "[No response]")
    logger.info("[RESPONSE_NODE] final_response length=%d chars", len(final))
    visited = list(state.get("nodes_visited", []))
    visited.append("response_node")

    # Resolve the trace id once for both evaluation pipelines.
    # Prefer the CURRENT request's trace id from config (correct on checkpoint
    # resume, where state still carries the original request's id — M-3);
    # fall back to state, then ambient OTEL context.
    trace_id = None
    try:
        trace_id = ((config or {}).get("configurable") or {}).get("langfuse_trace_id")
    except Exception:  # noqa: BLE001
        pass
    trace_id = trace_id or state.get("langfuse_trace_id") or _current_langfuse_trace_id()

    # F-4: evaluation runs exactly once per logical turn. On checkpoint resume
    # / replay the graph re-enters response_node with state that already has
    # evaluation_queued=True; skip re-scoring so we never publish duplicate
    # RAGAS or online scores onto the trace.
    already_evaluated = bool(state.get("evaluation_queued"))
    if already_evaluated:
        logger.debug("[RESPONSE_NODE] evaluation already performed for this turn — skipping")
        return {
            "final_response": final,
            "evaluation_queued": True,
            "nodes_visited": visited,
        }

    # ── Fire-and-forget RAGAS evaluation ──────────────────────────────────────
    try:
        from app.observability.rag_evaluator import EvaluationSample, RAGEvaluator

        retrieved_context = [
            c.get("content", c.get("page_content", ""))
            for c in state.get("reranked_chunks", [])
            if isinstance(c, dict)
        ]
        sample = EvaluationSample(
            query=state.get("query", ""),
            ground_truth="",
            retrieved_context=retrieved_context,
            answer=final,
            session_id=state.get("session_id", ""),
            source=state.get("intent", "HYBRID"),
            trace_id=trace_id,
        )
        evaluator = RAGEvaluator.get_instance()
        # C-1: evaluate_sample() is synchronous and does blocking RAGAS + judge
        # LLM work. Run it off the event loop so streaming is never blocked, and
        # log (not swallow) any failure via a done-callback.
        loop = asyncio.get_running_loop()
        eval_future = loop.run_in_executor(None, evaluator.evaluate_sample, sample)
        eval_future.add_done_callback(_log_eval_future_result)
        logger.debug("[RESPONSE_NODE] RAGAS evaluation scheduled (executor)")
    except Exception as _eval_err:  # noqa: BLE001
        logger.debug("[RESPONSE_NODE] RAGAS eval skipped: %s", _eval_err)

    # ── Online code-evaluator suite (deterministic, sub-ms, trace-linked) ─────
    # Publishes structure/citation/grounding/routing/guardrail scores onto the
    # same trace via the SDK's background batch queue. Disjoint from the RAGAS
    # score set, so no duplicate evaluations. No-op when unsampled/disabled.
    try:
        from app.observability.evaluation import evaluate_response_online
        # M-3: pass the resolved trace id (config-first) so scores land on the
        # current request's trace even on checkpoint resume.
        evaluate_response_online(state, trace_id_override=trace_id)
    except Exception as _code_eval_err:  # noqa: BLE001
        logger.debug("[RESPONSE_NODE] online code evaluation skipped: %s", _code_eval_err)

    # Pass final_response through so astream_events on_chain_end can emit it
    # as a fallback token event when the LLM streamed via on_chat_model_stream.
    return {
        "final_response": final,
        "evaluation_queued": True,
        "nodes_visited": visited,
    }


async def memory_write_node(state: dict, config: RunnableConfig = None) -> dict:
    """Delegate to real memory_write_node (entity_extractor.py) or stub."""
    if _MEMORY_WRITE_AVAILABLE:
        return await _real_memory_write(state, config)
    return await _stub_node("memory_write_node", state, config or {})


# ─────────────────────────────────────────────────────────────────────────────
# Routing functions (conditional edges)
# ─────────────────────────────────────────────────────────────────────────────

def route_cache(state: dict) -> str:
    """After cache_check_node: hit → response_node, miss → memory_read_node."""
    if state.get("cache_hit", False):
        return "response_node"
    return "memory_read_node"


def dispatch_agents(state: dict) -> list:
    """
    After intent_classifier_node: return a list of Send() objects, one per
    agent in state["routing_agents"].

    LangGraph executes all Send()s in the same superstep, so a two-agent
    response (e.g. CodeAgent + DocAgent) runs in parallel rather than
    sequentially.  Each Send() passes the full current state to the target
    node so every agent sees the same query, retrieved memory, and routing
    fields.

    Fallback: if routing_agents is empty or the cache was already hit,
    this returns a single Send to CodeAgent / response_node so the graph
    never stalls.
    """
    # Cache hit path — nothing to dispatch to an agent
    if state.get("cache_hit", False):
        return [Send("response_node", state)]

    agents: list[str] = state.get("routing_agents", [])

    # Validate — drop any name that isn't a registered node
    _valid = {"CodeAgent", "DocAgent", "DebugAgent", "ArchAgent", "WebAgent"}
    valid_agents = [a for a in agents if a in _valid]

    if not valid_agents:
        logger.warning(
            "[SUPERVISOR] routing_agents=%r contains no valid names — "
            "falling back to CodeAgent",
            agents,
        )
        valid_agents = ["CodeAgent"]

    logger.info("[SUPERVISOR] dispatching agents=%s (parallel=%s)", valid_agents, len(valid_agents) > 1)
    return [Send(agent, state) for agent in valid_agents]


# Kept for backward compatibility with tests that call route_to_agent directly
def route_to_agent(state: dict) -> str:
    """
    Legacy single-dispatch routing function — reads routing_decision and
    returns the agent node name.

    This function is retained so existing tests that call route_to_agent()
    directly continue to pass.  The graph itself now uses dispatch_agents()
    which returns Send() objects for parallel execution.
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

    # Agent sub-graphs
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
    builder.add_node("memory_write_node",     memory_write_node)

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

    # Parallel fan-out: dispatch_agents() returns Send() objects so multiple
    # agents can run in the same superstep for compound queries.
    # The path map lists every valid destination so LangGraph can validate the
    # graph at compile time even though Send() bypasses the string mapping at
    # runtime.
    builder.add_conditional_edges(
        "intent_classifier_node",
        dispatch_agents,
        {
            "CodeAgent":     "CodeAgent",
            "DocAgent":      "DocAgent",
            "DebugAgent":    "DebugAgent",
            "ArchAgent":     "ArchAgent",
            "WebAgent":      "WebAgent",
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
    builder.add_edge("response_node",         "memory_write_node")
    builder.add_edge("memory_write_node",     END)

    # ── Compile ───────────────────────────────────────────────────────────────
    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    graph = builder.compile(**compile_kwargs)

    logger.info(
        "[SUPERVISOR] Graph compiled — nodes=%d, checkpointer=%s, "
        "parallel_dispatch=Send()-based",
        len(builder.nodes),
        type(checkpointer).__name__ if checkpointer else "None",
    )
    return graph
