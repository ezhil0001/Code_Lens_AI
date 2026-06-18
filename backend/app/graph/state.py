"""
AgentState — Central LangGraph State Schema
============================================
Phase A: F-01

This TypedDict is the single source of truth that flows through every node
in the LangGraph Supervisor graph.  Think of it as a Redux store:
  - Nodes READ from it freely.
  - Nodes RETURN only the fields they changed (dict, never the full state).
  - operator.add on `messages` means nodes append — they never overwrite.

Design rules:
  - Never mutate the state object directly — return a partial dict.
  - `query` is set by the entry node and is immutable after that.
  - `user_id` / `session_id` must always use the namespaced form
    "{user_id}::{raw_session_id}" (security invariant from Phase 2).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence

from langchain_core.messages import BaseMessage


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
    routing_decision: Optional[str]  # "CodeAgent" | "DocAgent" | "DebugAgent" | ...
    routing_confidence: float
    metadata_filter: Optional[Dict[str, Any]]  # ChromaDB where= clause

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieved_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    parent_contexts: Dict[str, str]

    # ── Agent outputs ─────────────────────────────────────────────────────────
    agent_responses: Dict[str, str]  # agent_name → partial answer
    active_agent: Optional[str]
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]

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
    sources: List[Dict[str, Any]]
    cache_hit: bool
    evaluation_queued: bool

    # ── Observability ─────────────────────────────────────────────────────────
    span_id: Optional[str]
    graph_checkpoint_id: Optional[str]   # renamed: 'checkpoint_id' is reserved by LangGraph
    nodes_visited: List[str]
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
        "routing_confidence": 0.0,
        "metadata_filter": None,

        # Retrieval
        "retrieved_chunks": [],
        "reranked_chunks": [],
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
        "graph_checkpoint_id": None,
        "nodes_visited": [],
        "total_latency_ms": 0.0,
    }
