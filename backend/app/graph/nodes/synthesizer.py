"""
Synthesizer node — merges agent outputs into the final response.

Single-agent path (the common case):
  Directly promotes agent_responses[active_agent] to final_response.
  No LLM call, zero extra token cost.

Multi-agent path (HYBRID routing where multiple agents ran):
  Deduplicates sources by 'id', then asks the LLM to write a single
  coherent answer that doesn't repeat itself across the partial answers.
  Only reached when HYBRID routing dispatched more than one sub-graph.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def sanitise_source_path(value: Any) -> Any:
    """Reduce a source path to its basename.

    Ingested files land in a server temp dir, so the raw metadata path is an
    absolute host path (``/private/var/folders/.../T/tmpXXXX/SQL.pdf``). That
    was rendered verbatim in the chat UI, disclosing the server filesystem
    layout and the temp-dir naming scheme to every client.
    """
    if not isinstance(value, str) or not value:
        return value
    if "/" not in value and "\\" not in value:
        return value
    return value.replace("\\", "/").rstrip("/").split("/")[-1] or value


_SOURCE_PATH_KEYS = ("file_path", "source", "source_file", "path", "filename")


def deduplicate_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return sources with duplicate 'id' fields removed (first-occurrence wins).

    Also strips absolute host paths — this is the single boundary every agent's
    sources pass through before reaching the client.
    """
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for src in sources:
        src_id = str(src.get("id", ""))
        if src_id and src_id in seen:
            continue
        seen.add(src_id)
        if isinstance(src, dict):
            src = dict(src)
            for key in _SOURCE_PATH_KEYS:
                if key in src:
                    src[key] = sanitise_source_path(src[key])
            meta = src.get("metadata")
            if isinstance(meta, dict):
                meta = dict(meta)
                for key in _SOURCE_PATH_KEYS:
                    if key in meta:
                        meta[key] = sanitise_source_path(meta[key])
                src["metadata"] = meta
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
            chunks: list[str] = []
            async for chunk in llm.astream([HumanMessage(content=synthesis_prompt)]):
                piece = getattr(chunk, "content", "")
                if piece:
                    chunks.append(piece)
            final_response = "".join(chunks) or combined_text
    except Exception as exc:  # noqa: BLE001
        logger.warning("[synthesizer_node] LLM synthesis failed: %s — using concatenation", exc)

    return {
        "final_response": final_response,
        "sources": deduped_sources,
        "nodes_visited": visited,
    }
