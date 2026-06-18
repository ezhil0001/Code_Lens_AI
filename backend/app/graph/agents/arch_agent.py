"""
ArchAgent Sub-Graph — Phase B: F-08
=====================================
Architecture and system design questions — hybrid retrieval across
both code AND documentation simultaneously.

Key characteristic:
  - Uses None metadata_filter (no restriction — queries both collections)
  - LLM prompt instructs synthesis of data-flow narrative
  - All node names prefixed arch_*
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from app.graph.state import AgentState

logger = logging.getLogger(__name__)
MAX_CHARS_PER_SOURCE = 8_000


async def arch_retrieve_node(state: dict, config: RunnableConfig = None) -> dict:
    """Hybrid retrieval: both code and kt_doc (no metadata filter restriction)."""
    import threading

    query: str = state.get("query", "")
    visited = list(state.get("nodes_visited", []))
    visited.append("arch_retrieve_node")

    chunks: List[Dict[str, Any]] = []
    try:
        from app.services.pipeline_factory import get_pipeline_factory_cached
        factory = get_pipeline_factory_cached()
        retriever = factory.get_retriever_engine()
        _lock = getattr(retriever, "_metadata_lock", threading.Lock())
        with _lock:
            result = retriever.retrieve(query=query, top_k=10, metadata_filter=None)
        chunks = result.chunks if result else []
        logger.info("[arch_retrieve_node] retrieved %d hybrid chunks", len(chunks))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[arch_retrieve_node] retrieval failed: %s", exc)

    return {"retrieved_chunks": chunks, "nodes_visited": visited}


async def arch_generate_node(state: dict, config: RunnableConfig = None) -> dict:
    """LLM generation focused on architecture and data-flow synthesis."""
    query: str = state.get("query", "")
    chunks: List[Dict] = state.get("retrieved_chunks", [])[:6]
    visited = list(state.get("nodes_visited", []))
    visited.append("arch_generate_node")

    context_parts: List[str] = []
    sources: List[Dict] = []
    for i, chunk in enumerate(chunks):
        content = chunk.get("content", chunk.get("page_content", ""))[:MAX_CHARS_PER_SOURCE]
        metadata = chunk.get("metadata", {})
        file_path = metadata.get("file_path", metadata.get("source", "unknown"))
        file_type = metadata.get("file_type", "unknown")
        context_parts.append(f"### [{file_type}] {file_path}\n{content}")
        sources.append({
            "id": chunk.get("id", f"arch-{i}"),
            "file_path": file_path,
            "file_type": file_type,
            "score": chunk.get("score", 0.0),
            "content": content[:500],
        })

    context_text = "\n\n".join(context_parts) if context_parts else "No context found."

    system_prompt = (
        "You are CodeLens AI, an expert software architect. "
        "Answer architecture and system design questions by synthesizing "
        "both code structure and documentation. "
        "Describe data flows, component relationships, and design decisions clearly."
    )
    user_prompt = (
        f"Architecture Question: {query}\n\n"
        f"Context (code + docs):\n{context_text}\n\n"
        "Provide a comprehensive architectural explanation."
    )

    response_text: str = "[ArchAgent: LLM unavailable — architecture context assembled]"
    try:
        from app.services.pipeline_factory import get_pipeline_factory_cached
        factory = get_pipeline_factory_cached()
        llm = getattr(factory, "get_llm", lambda: None)()
        if llm:
            from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            ai_msg = await llm.ainvoke(messages)
            response_text = getattr(ai_msg, "content", str(ai_msg))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[arch_generate_node] LLM call failed: %s", exc)

    agent_responses = dict(state.get("agent_responses", {}))
    agent_responses["ArchAgent"] = response_text
    existing_sources = list(state.get("sources", []))
    existing_sources.extend(sources)

    return {
        "active_agent": "ArchAgent",
        "agent_responses": agent_responses,
        "sources": existing_sources,
        "nodes_visited": visited,
    }


def build_arch_agent() -> Any:
    """Build and compile the ArchAgent sub-graph."""
    builder = StateGraph(AgentState)

    builder.add_node("arch_retrieve_node", arch_retrieve_node)
    builder.add_node("arch_generate_node", arch_generate_node)

    builder.set_entry_point("arch_retrieve_node")
    builder.add_edge("arch_retrieve_node", "arch_generate_node")
    builder.add_edge("arch_generate_node", END)

    return builder.compile()
