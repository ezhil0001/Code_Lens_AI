"""
AgentState — Central LangGraph state schema shared across all graph nodes.

Every node reads from this dict and returns only the fields it modified.

REDUCER FIELDS
--------------
messages:        Annotated[..., operator.add]
    Standard LangGraph list-append reducer — parallel nodes each append
    without clobbering each other.

agent_responses: Annotated[Dict[str, str], _merge_agent_responses]
    Dict-merge reducer introduced to support parallel agent fan-out via the
    Send() API.  When two agents run in the same superstep (e.g. CodeAgent
    and DocAgent both dispatched for a compound query), each writes a single
    {agent_name: answer} entry.  Without this reducer LangGraph would apply
    last-write-wins semantics and silently discard all but one agent's answer.
    _merge_agent_responses(a, b) = {**a, **b} so every agent's contribution
    is preserved and the synthesizer sees the full set.

All other fields: last-write-wins (default LangGraph behaviour).

Invariant: session_id must always be the namespaced form
"{user_id}::{raw_session_id}" by the time it reaches any node.  The v2
chat endpoint builds this before constructing the initial state; nodes
never re-namespace it.  Breaking this would silently mix conversation
history across users.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence

from langchain_core.messages import BaseMessage


def _merge_agent_responses(existing: dict, update: dict) -> dict:
    """
    Merge reducer for agent_responses.

    Called by LangGraph whenever more than one branch in the same superstep
    writes to agent_responses.  Combines the two dicts so every parallel
    agent's answer is preserved:

        {"CodeAgent": "..."} merged with {"DocAgent": "..."}
        → {"CodeAgent": "...", "DocAgent": "..."}

    If two branches write the same agent key (shouldn't happen in practice),
    the later write wins, matching normal last-write-wins expectation.
    """
    return {**existing, **update}


def _merge_nodes_visited(existing: list, update: list) -> list:
    """
    Dedup-merge reducer for nodes_visited.

    Each parallel agent reads the full nodes_visited list at the start of its
    superstep, appends its own entry, and returns the complete list.  Without
    this reducer LangGraph would raise InvalidUpdateError on concurrent writes.

    This reducer unions the two lists by first-occurrence order so the trace
    contains every unique visit without duplicating the shared prefix that
    both agents copied before appending their own entry.
    """
    seen: set[str] = set()
    merged: list[str] = []
    for item in existing + update:
        key = str(item)
        if key not in seen:
            merged.append(item)
            seen.add(key)
    return merged


def _last_non_none(existing: Any, update: Any) -> Any:
    """
    Last-write-wins reducer that prefers a non-None update.

    Used for active_agent: in single-agent queries the one agent sets it;
    in parallel queries whichever agent finished last wins (the synthesizer
    doesn't rely on active_agent for the multi-agent path).
    """
    return update if update is not None else existing


def _merge_dicts(existing: dict, update: dict) -> dict:
    """Dict-merge reducer: later write wins per key (same as agent_responses)."""
    return {**existing, **update}


def _merge_chunk_lists(existing: list, update: list) -> list:
    """Concat-dedup reducer for lists of retrieval chunks / sources / scores.

    When the supervisor dispatches multiple agents in parallel (CodeAgent +
    DebugAgent via ``Send()``), each branch writes ``retrieved_chunks``,
    ``reranked_chunks``, ``rerank_scores`` and ``sources`` in the SAME
    superstep. Without a reducer LangGraph raises
    ``INVALID_CONCURRENT_GRAPH_UPDATE`` ("Can receive only one value per
    step"). This reducer concatenates every branch's contribution and drops
    exact duplicates (chunks are unhashable dicts, so we dedup by a stable
    content key) while preserving first-occurrence order.
    """
    existing = existing or []
    update = update or []
    merged: list = []
    seen: set = set()

    def _key(item: Any) -> str:
        if isinstance(item, dict):
            # Prefer stable identity fields; fall back to full repr.
            for k in ("id", "chunk_id", "source", "content"):
                if k in item and item[k] is not None:
                    return f"{k}:{item[k]}"
            return repr(sorted(item.items(), key=lambda kv: str(kv[0])))
        return repr(item)

    for item in list(existing) + list(update):
        key = _key(item)
        if key not in seen:
            merged.append(item)
            seen.add(key)
    return merged



class AgentState(dict):
    """
    Central state TypedDict for the LangGraph Supervisor graph.

    Implemented as a plain dict subclass with class-level field annotations
    so that LangGraph can introspect the schema while remaining compatible
    with Python 3.11 and standard TypedDict usage patterns.

    REDUCER SEMANTICS
    -----------------
    messages: Annotated[..., operator.add]
        Every node that appends a message works correctly even with parallel
        execution — LangGraph merges lists from concurrent nodes by
        concatenation (not last-write-wins).

    agent_responses: Annotated[Dict[str, str], _merge_agent_responses]
        Dict-merge so parallel Send() branches each contribute their own
        {agent_name: answer} entry without overwriting each other.

    All other fields: last-write-wins (default LangGraph behaviour).
    """

    # ── Core conversation ─────────────────────────────────────────────────────
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # operator.add → each node appends; never overwrites.

    # ── Request context ───────────────────────────────────────────────────────
    user_id: str
    session_id: str          # namespaced: "{user_id}::{raw_session_id}"
    org_id: Optional[str]
    query: str               # original user query — immutable after set by entry node
    query_embedding: Optional[List[float]]  # cached — computed once

    # ── Routing & intent ─────────────────────────────────────────────────────
    intent: Optional[str]            # "CODE_LOOKUP" | "DEBUG" | "ARCHITECTURE" | ...
    routing_decision: Optional[str]  # primary agent name (kept for backward-compat)
    routing_agents: List[str]        # full list from LLM routing — may contain 1..N names
    routing_confidence: float
    metadata_filter: Optional[Dict[str, Any]]  # ChromaDB where= clause

    # ── Retrieval ─────────────────────────────────────────────────────────────
    # Concat-dedup reducers: parallel agents (Code/Debug/Doc/Arch) each write
    # these in the same superstep, so a reducer is REQUIRED to avoid
    # INVALID_CONCURRENT_GRAPH_UPDATE.
    retrieved_chunks: Annotated[List[Dict[str, Any]], _merge_chunk_lists]
    reranked_chunks: Annotated[List[Dict[str, Any]], _merge_chunk_lists]
    rerank_scores: Annotated[List[float], _merge_chunk_lists]  # cross-encoder scores for reranked_chunks
    parent_contexts: Dict[str, str]

    # ── Agent outputs ─────────────────────────────────────────────────────────
    # Dict-merge reducer: parallel Send() branches each write one key → combined here
    agent_responses: Annotated[Dict[str, str], _merge_agent_responses]
    # last-non-none: whichever agent set active_agent last wins (harmless for multi)
    active_agent: Annotated[Optional[str], _last_non_none]
    # list-append reducers: parallel agents each contribute their own entries
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]
    tool_results: Annotated[List[Dict[str, Any]], operator.add]

    # ── Memory ────────────────────────────────────────────────────────────────
    short_term_window: List[Dict[str, Any]]   # last N turns from PostgresChatMessageHistory
    long_term_facts: List[str]                # retrieved LTM facts for this query
    memory_write_queue: List[Dict[str, Any]]  # facts to persist post-turn

    # ── Guardrails ────────────────────────────────────────────────────────────
    guardrail_passed: bool
    guardrail_violations: List[Dict[str, Any]]
    pii_scrubbed_query: Optional[str]

    # ── HIL (Human-in-the-Loop) ───────────────────────────────────────────────
    hil_required: bool
    hil_reason: Optional[str]
    hil_human_input: Optional[str]
    hil_approved: Optional[bool]

    # ── Final response ────────────────────────────────────────────────────────
    final_response: Optional[str]
    # Concat-dedup: parallel agents each append their retrieval sources.
    sources: Annotated[List[Dict[str, Any]], _merge_chunk_lists]
    cache_hit: bool
    evaluation_queued: bool

    # ── Observability ─────────────────────────────────────────────────────────
    span_id: Optional[str]
    langfuse_trace_id: Optional[str]     # deterministic trace id for eval scoring
    graph_checkpoint_id: Optional[str]   # renamed: 'checkpoint_id' is reserved by LangGraph
    # dedup-merge: parallel agents each read + extend the list; combine without duplication
    nodes_visited: Annotated[List[str], _merge_nodes_visited]
    total_latency_ms: float


def make_initial_state(
    query: str,
    user_id: str,
    session_id: str,
    org_id: Optional[str] = None,
) -> dict:
    """
    Build the initial AgentState dict for the start of a new graph run.

    The returned dict contains safe defaults for every field so that nodes
    never encounter a KeyError when reading state fields they didn't set.
    """
    return {
        # Core conversation
        "messages": [],

        # Request context
        "user_id": user_id,
        "session_id": session_id,
        "org_id": org_id,
        "query": query,
        "query_embedding": None,

        # Routing & intent
        "intent": None,
        "routing_decision": None,
        "routing_agents": [],
        "routing_confidence": 0.0,
        "metadata_filter": None,

        # Retrieval
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "rerank_scores": [],
        "parent_contexts": {},

        # Agent outputs
        "agent_responses": {},
        "active_agent": None,
        "tool_calls": [],
        "tool_results": [],

        # Memory
        "short_term_window": [],
        "long_term_facts": [],
        "memory_write_queue": [],

        # Guardrails
        "guardrail_passed": True,
        "guardrail_violations": [],
        "pii_scrubbed_query": None,

        # HIL
        "hil_required": False,
        "hil_reason": None,
        "hil_human_input": None,
        "hil_approved": None,

        # Final response
        "final_response": None,
        "sources": [],
        "cache_hit": False,
        "evaluation_queued": False,

        # Observability
        "span_id": None,
        "langfuse_trace_id": None,
        "graph_checkpoint_id": None,
        "nodes_visited": [],
        "total_latency_ms": 0.0,
    }
