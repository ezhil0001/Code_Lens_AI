"""
Synthesizer Node — Phase B: F-11
==================================
Merges outputs from one or more agent sub-graphs into a single coherent response.

Single-agent path (most queries):
  - Copies agent_responses[active_agent] → final_response verbatim.
  - Zero LLM cost.

Multi-agent path (HYBRID routing):
  - Deduplicates sources by 'id'.
  - Calls LLM with synthesis prompt to merge N partial answers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return sources with duplicate 'id' fields removed (first-occurrence wins)."""
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for src in sources:
        src_id = str(src.get("id", ""))
        if src_id and src_id in seen:
            continue
        seen.add(src_id)
        deduped.append(src)
    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

async def synthesizer_node(state: dict, config: RunnableConfig = None) -> dict:
    """
    Merge agent responses into a single final_response.

    Single-agent fast path:
        active_agent is set → copy agent_responses[active_agent] verbatim.
    Multi-agent path:
        Multiple keys in agent_responses → LLM synthesis call.
    """
    agent_responses: Dict[str, str] = state.get("agent_responses", {})
    active_agent: Optional[str] = state.get("active_agent")
    sources: List[Dict] = state.get("sources", [])
    visited = list(state.get("nodes_visited", []))
    visited.append("synthesizer_node")

    # ── Deduplicate sources ───────────────────────────────────────────────────
    deduped_sources = deduplicate_sources(sources)

    # ── Single-agent fast path ────────────────────────────────────────────────
    if active_agent and active_agent in agent_responses and len(agent_responses) == 1:
        return {
            "final_response": agent_responses[active_agent],
            "sources": deduped_sources,
            "nodes_visited": visited,
        }

    # ── If no responses at all ────────────────────────────────────────────────
    if not agent_responses:
        return {
            "final_response": "[No agent responses available]",
            "sources": deduped_sources,
            "nodes_visited": visited,
        }

    # ── Multi-agent synthesis path ────────────────────────────────────────────
    # Try LLM synthesis; fall back to concatenation on failure.
    agent_names = list(agent_responses.keys())
    combined_text = "\n\n---\n\n".join(
        f"**{name}:** {resp}" for name, resp in agent_responses.items()
    )

    synthesis_prompt = (
        f"You have received answers from {len(agent_names)} specialized agents "
        f"({', '.join(agent_names)}). "
        "Produce a single, coherent, non-repetitive answer that synthesises their insights. "
        "Cite relevant sources where applicable.\n\n"
        f"Agent Responses:\n{combined_text}"
    )

    final_response: str = combined_text  # default fallback
    try:
        from app.services.pipeline_factory import get_pipeline_factory_cached
        factory = get_pipeline_factory_cached()
        llm = getattr(factory, "get_llm", lambda: None)()
        if llm:
            from langchain_core.messages import HumanMessage  # type: ignore
            ai_msg = await llm.ainvoke([HumanMessage(content=synthesis_prompt)])
            final_response = getattr(ai_msg, "content", combined_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[synthesizer_node] LLM synthesis failed: %s — using concatenation", exc)

    return {
        "final_response": final_response,
        "sources": deduped_sources,
        "nodes_visited": visited,
    }
