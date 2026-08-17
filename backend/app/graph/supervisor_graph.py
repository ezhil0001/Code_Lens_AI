"""
Supervisor graph — root StateGraph that orchestrates the full query pipeline.

Entry point for every v2 chat request.  Builds the graph once at startup and
reuses the compiled instance across all requests via get_supervisor_graph().

Node execution order:
  input_guardrail_node    → safety checks (injection, PII, token budget)
  cache_check_node        → semantic cache lookup; hit → skip to response_node
  memory_read_node        → load STM window + LTM facts
  intent_classifier_node  → LLM routing; returns 1–2 agent names
  hil_check_node          → SAFETY GATE: interrupt for human review BEFORE any
                            answer is generated; reject skips agents entirely
  [agent nodes]           → dispatched in parallel via Send() for compound queries
  synthesizer_node        → merge outputs; single-agent is a pass-through
  output_guardrail_node   → code safety scan + PII leak scan + citation warnings
  response_node           → assemble final SSE payload, write to cache + memory

Why the gate sits before the agents
-----------------------------------
It used to run after synthesizer_node. By then the agents and synthesiser had
already produced *and streamed* the answer, so a "DROP TABLE users" request was
delivered to the browser before the reviewer ever saw the banner, and the output
guardrail could only annotate content the user had already read. Gating between
classification and dispatch keeps both controls genuinely preventive: the
classifier supplies routing_confidence for the low-confidence gate, and no
answer-producing node has run yet.

Parallel fan-out
----------------
intent_classifier_node populates state["routing_agents"] with 1–2 agent names.
route_after_hil() reads that list and returns a list of Send() objects, one per
agent.  LangGraph executes all Send()s in the same superstep so two agents run
concurrently.  Each agent writes its answer into agent_responses under its own
key; the _merge_agent_responses reducer in AgentState combines the dicts so no
answer is lost.  The synthesizer then receives the full set.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
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


_EVAL_TIMEOUT_SECONDS = float(os.getenv("RAGAS_EVAL_TIMEOUT_SECONDS", "180"))
# Background evaluation shares the chat path's Groq quota. Each RAGAS sample
# issues several judge calls, so running evaluations in parallel is what
# actually triggers HTTP 429 — and a 429 there can starve real chat requests.
# Evaluation is a background quality job with no latency requirement, so keep
# it at one worker by default.
_EVAL_MAX_WORKERS = int(os.getenv("RAGAS_EVAL_MAX_WORKERS", "1"))
_eval_executor: "Any" = None
_eval_inflight: set[str] = set()
_eval_inflight_lock = threading.Lock()


def _get_eval_executor():
    """Dedicated bounded pool so RAGAS never starves the default executor."""
    global _eval_executor
    if _eval_executor is None:
        from concurrent.futures import ThreadPoolExecutor
        _eval_executor = ThreadPoolExecutor(
            max_workers=_EVAL_MAX_WORKERS, thread_name_prefix="ragas-eval"
        )
    return _eval_executor


def shutdown_eval_executor(wait: bool = False) -> None:
    """Release evaluation workers at process shutdown."""
    global _eval_executor
    if _eval_executor is not None:
        _eval_executor.shutdown(wait=wait, cancel_futures=not wait)
        _eval_executor = None


async def _run_ragas_evaluation(evaluator: "Any", sample: "Any", dedupe_key: str) -> None:
    """Run one RAGAS evaluation with a hard timeout and explicit outcome logs.

    Scheduling failures used to be logged at DEBUG, so a broken evaluator
    produced neither a success nor a failure line and looked like a hang.
    """
    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_get_eval_executor(), evaluator.evaluate_sample, sample),
            timeout=_EVAL_TIMEOUT_SECONDS,
        )
        if result is None:
            logger.warning("[RAGAS] evaluation returned no result (sample skipped)")
        else:
            logger.info(
                "[RAGAS] evaluation completed — faithfulness=%s answer_relevancy=%s "
                "context_recall=%s",
                getattr(result.metrics, "faithfulness", None),
                getattr(result.metrics, "answer_relevancy", None),
                getattr(result.metrics, "context_recall", None),
            )
    except asyncio.TimeoutError:
        logger.error(
            "[RAGAS] evaluation timed out after %.0fs — abandoning sample",
            _EVAL_TIMEOUT_SECONDS,
        )
    except asyncio.CancelledError:
        logger.warning("[RAGAS] evaluation cancelled")
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("[RAGAS] evaluation failed: %s", exc)
    finally:
        with _eval_inflight_lock:
            _eval_inflight.discard(dedupe_key)



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

    # Root observation of this request, so background evaluation nests under
    # "chat.supervisor" instead of becoming a second root in the same trace.
    parent_span_id = None
    try:
        parent_span_id = ((config or {}).get("configurable") or {}).get("langfuse_parent_span_id")
    except Exception:  # noqa: BLE001
        pass
    parent_span_id = parent_span_id or state.get("langfuse_parent_span_id")

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
            parent_span_id=parent_span_id,
        )
        evaluator = RAGEvaluator.get_instance()
        # Dedupe on the trace (falls back to session+query) so a retried or
        # resumed response cannot enqueue the same evaluation twice.
        dedupe_key = trace_id or f"{state.get('session_id','')}::{hash(state.get('query',''))}"
        with _eval_inflight_lock:
            already_running = dedupe_key in _eval_inflight
            if not already_running:
                _eval_inflight.add(dedupe_key)
        if already_running:
            logger.info("[RAGAS] evaluation already in flight for %s — skipping duplicate", dedupe_key[:16])
        else:
            asyncio.create_task(_run_ragas_evaluation(evaluator, sample, dedupe_key))
            logger.info("[RAGAS] evaluation scheduled (timeout=%.0fs)", _EVAL_TIMEOUT_SECONDS)
    except Exception as _eval_err:  # noqa: BLE001
        # Must never be DEBUG: a broken evaluator here is invisible otherwise.
        logger.exception("[RAGAS] could not schedule evaluation: %s", _eval_err)

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
    """Legacy router kept for callers/tests that import it directly.

    The live graph uses :func:`route_after_hil`, which gates *before* any
    answer is generated.
    """
    return "output_guardrail_node"


def route_after_hil(state: dict):
    """Route out of ``hil_check_node``, which now runs BEFORE generation.

    ``hil_check_node`` raises LangGraph's dynamic ``interrupt()`` when a
    request needs review, so control only reaches this router once there is a
    decision (or none was needed).

    * rejected  → straight to the output guardrail; ``hil_check_node`` has
      already put the safe refusal in ``final_response`` and **no agent runs**,
      so no unsafe answer is ever generated.
    * otherwise → the normal ``Send()`` fan-out to the selected agents.
    """
    if state.get("hil_approved") is False:
        logger.info("[SUPERVISOR] HIL rejected — skipping agent generation")
        return "output_guardrail_node"
    return dispatch_agents(state)


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

    # Safety gate BEFORE any answer is produced. hil_check_node needs
    # routing_confidence (set by the classifier) for its low-confidence gate,
    # so it sits directly after classification and directly before dispatch.
    # Running it after the synthesiser — as it used to — meant the agents and
    # synthesiser had already generated AND streamed the answer, so neither the
    # human review nor the output guardrail could prevent disclosure.
    builder.add_edge("intent_classifier_node", "hil_check_node")

    # Parallel fan-out: route_after_hil() returns Send() objects so multiple
    # agents can run in the same superstep for compound queries, or the string
    # "output_guardrail_node" when a reviewer rejected the request.
    # The path map lists every valid destination so LangGraph can validate the
    # graph at compile time even though Send() bypasses the string mapping at
    # runtime.
    builder.add_conditional_edges(
        "hil_check_node",
        route_after_hil,
        {
            "CodeAgent":     "CodeAgent",
            "DocAgent":      "DocAgent",
            "DebugAgent":    "DebugAgent",
            "ArchAgent":     "ArchAgent",
            "WebAgent":      "WebAgent",
            "response_node": "response_node",
            "output_guardrail_node": "output_guardrail_node",
        },
    )

    # All agents → synthesizer
    for _agent in ("CodeAgent", "DocAgent", "DebugAgent", "ArchAgent", "WebAgent"):
        builder.add_edge(_agent, "synthesizer_node")

    builder.add_edge("synthesizer_node", "output_guardrail_node")

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
