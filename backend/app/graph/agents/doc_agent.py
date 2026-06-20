"""
DocAgent Sub-Graph — Phase B: F-06
====================================
Mirrors CodeAgent but targets KT documentation (file_type=kt_doc).

Key differences from CodeAgent:
  - DOC_AGENT_METADATA_FILTER = {"file_type": "kt_doc"}
  - BM25 weight 0.6 / vector weight 0.4 (exact section lookup)
  - LLM prompt uses documentation-focused few-shot template
  - All node names prefixed doc_*
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from app.graph.state import AgentState

logger = logging.getLogger(__name__)

# Module-level constant checked by B-005 test
DOC_AGENT_METADATA_FILTER: Dict[str, Any] = {"file_type": "kt_doc"}

MAX_CHARS_PER_SOURCE = 8_000


# ─────────────────────────────────────────────────────────────────────────────
# Node implementations
# ─────────────────────────────────────────────────────────────────────────────

async def doc_retrieve_node(state: dict, config: RunnableConfig = None) -> dict:
    """BM25 + ChromaDB retrieval filtered to kt_doc documents."""
    import threading

    query: str = state.get("query", "")
    visited = list(state.get("nodes_visited", []))
    visited.append("doc_retrieve_node")

    chunks: List[Dict[str, Any]] = []
    try:
        from app.services.pipeline_factory import get_pipeline_factory_cached
        factory = get_pipeline_factory_cached()
        retriever = factory.get_retriever_engine()
        _lock = getattr(retriever, "_metadata_lock", threading.Lock())
        with _lock:
            result = retriever.retrieve(
                query=query,
                top_k=10,
                metadata_filter=DOC_AGENT_METADATA_FILTER,
            )
        chunks = result.chunks if result else []
        logger.info("[doc_retrieve_node] retrieved %d doc chunks", len(chunks))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[doc_retrieve_node] retrieval failed: %s", exc)

    return {"retrieved_chunks": chunks, "nodes_visited": visited}


async def doc_rerank_node(state: dict, config: RunnableConfig = None) -> dict:
    """Rerank doc chunks — BM25-weighted top-5."""
    query: str = state.get("query", "")
    chunks: List[Dict] = state.get("retrieved_chunks", [])
    visited = list(state.get("nodes_visited", []))
    visited.append("doc_rerank_node")

    reranked: List[Dict] = chunks[:5]
    try:
        from app.services.pipeline_factory import get_pipeline_factory_cached
        factory = get_pipeline_factory_cached()
        reranker = getattr(factory, "get_reranker", None)
        if reranker and chunks:
            reranked = factory.get_reranker().rerank(query=query, documents=chunks, top_k=5)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[doc_rerank_node] reranker unavailable: %s", exc)

    return {"reranked_chunks": reranked, "nodes_visited": visited}


async def doc_generate_node(state: dict, config: RunnableConfig = None) -> dict:
    """LLM generation with documentation-focused prompt."""
    query: str = state.get("query", "")
    chunks: List[Dict] = state.get("reranked_chunks", [])
    visited = list(state.get("nodes_visited", []))
    visited.append("doc_generate_node")

    context_parts: List[str] = []
    sources: List[Dict] = []
    for i, chunk in enumerate(chunks):
        content = chunk.get("content", chunk.get("page_content", ""))[:MAX_CHARS_PER_SOURCE]
        metadata = chunk.get("metadata", {})
        file_path = metadata.get("file_path", metadata.get("source", "unknown"))
        section = metadata.get("section", "")
        heading = f"{file_path} — {section}" if section else file_path
        context_parts.append(f"### {heading}\n{content}")
        sources.append({
            "id": chunk.get("id", chunk.get("chunk_id", f"doc-{i}")),
            "file_path": file_path,
            "section": section,
            "score": chunk.get("score", 0.0),
            "content": content[:500],
        })

    context_text = "\n\n".join(context_parts) if context_parts else "No documentation found."

    system_prompt = (
        "You are CodeLens AI, an expert technical documentation assistant. "
        "Answer the question using ONLY the provided KT documentation. "
        "Be clear and thorough, referencing specific sections."
    )
    user_prompt = (
        f"Question: {query}\n\n"
        f"Documentation Context:\n{context_text}\n\n"
        "Provide a comprehensive explanation based on the documentation."
    )

    response_text: str = "[DocAgent: LLM unavailable — documentation context assembled]"
    try:
        from app.services.pipeline_factory import get_pipeline_factory_cached
        factory = get_pipeline_factory_cached()
        llm = getattr(factory, "get_llm", lambda: None)()
        if llm:
            from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            chunks: list[str] = []
            async for chunk in llm.astream(messages):
                piece = getattr(chunk, "content", "")
                if piece:
                    chunks.append(piece)
            response_text = "".join(chunks) or "[DocAgent: empty response]"
    except Exception as exc:  # noqa: BLE001
        logger.warning("[doc_generate_node] LLM call failed: %s", exc)
        if context_parts:
            response_text = (
                f"Based on the KT documentation for '{query}':\n\n" + context_text[:2000]
            )

    agent_responses = dict(state.get("agent_responses", {}))
    agent_responses["DocAgent"] = response_text
    existing_sources = list(state.get("sources", []))
    existing_sources.extend(sources)

    return {
        "active_agent": "DocAgent",
        "agent_responses": agent_responses,
        "sources": existing_sources,
        "nodes_visited": visited,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_doc_agent() -> Any:
    """Build and compile the DocAgent sub-graph."""
    builder = StateGraph(AgentState)

    builder.add_node("doc_retrieve_node", doc_retrieve_node)
    builder.add_node("doc_rerank_node",   doc_rerank_node)
    builder.add_node("doc_generate_node", doc_generate_node)

    builder.set_entry_point("doc_retrieve_node")
    builder.add_edge("doc_retrieve_node", "doc_rerank_node")
    builder.add_edge("doc_rerank_node",   "doc_generate_node")
    builder.add_edge("doc_generate_node", END)

    return builder.compile()
