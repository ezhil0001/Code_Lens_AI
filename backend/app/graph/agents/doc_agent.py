"""
DocAgent — handles queries about KT documentation: onboarding guides,
architecture notes, process documentation, and decision records.

Mirrors the CodeAgent pipeline but applies a file_type=kt_doc filter so
retrieval stays inside the documentation collection.  BM25 is weighted
higher (0.6) than vector (0.4) because documentation queries tend to match
on exact section titles and product-specific terminology that dense embeddings
can miss.  The LLM prompt uses a documentation-focused few-shot template
from few_shot_prompt.py.

Node names are prefixed `doc_` to avoid key collisions in the supervisor graph.
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
        from app.core.database import run_retrieval

        def _do_retrieve():
            # H-1: blocking retrieval on the dedicated pool — never on the event loop.
            # Model inference is serialised at the model boundary
            # (get_embedding_lock / get_reranker_lock), not by a coarse mutex.
            return retriever.retrieve(
                query=query,
                top_k=10,
                metadata_filter=DOC_AGENT_METADATA_FILTER,
            )

        import asyncio
        result = await run_retrieval(_do_retrieve)
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
    scores: List[float] = []
    try:
        from app.services.pipeline_factory import get_pipeline_factory_cached
        factory = get_pipeline_factory_cached()
        reranker = getattr(factory, "get_reranker", None)
        if reranker and chunks:
            # rerank() returns a (docs, scores) tuple
            # Cross-encoder inference is CPU-bound and takes seconds on
            # ~20 candidates. Running it inline blocked the event loop, so
            # /api/health returned 000 under concurrent load. Offload it to the
            # bounded retrieval pool like every other model call.
            from app.core.database import run_retrieval
            _reranker = factory.get_reranker()
            result = await run_retrieval(
                lambda: _reranker.rerank(query=query, documents=chunks, top_k=5)
            )
            if isinstance(result, tuple):
                reranked, scores = result[0], list(result[1] or [])
            else:
                reranked = result
    except Exception as exc:  # noqa: BLE001
        logger.debug("[doc_rerank_node] reranker unavailable: %s", exc)

    # M-5: propagate cross-encoder scores for the retrieval_quality evaluator.
    return {"reranked_chunks": reranked, "rerank_scores": scores, "nodes_visited": visited}


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
        from app.graph.nodes.synthesizer import sanitise_source_path
        # Never expose the server's absolute ingest path to a client.
        file_path = sanitise_source_path(
            metadata.get("file_path", metadata.get("source", "unknown"))
        )
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

    # Build conversation history block from short-term memory window
    history_block = ""
    short_term_window: list = state.get("short_term_window", [])
    if short_term_window:
        history_lines = []
        for turn in short_term_window:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "system":
                history_lines.append(f"[Context] {content}")
            elif role in ("human", "user"):
                history_lines.append(f"User: {content}")
            elif role in ("ai", "assistant"):
                history_lines.append(f"Assistant: {content}")
        if history_lines:
            history_block = "\n\nConversation History:\n" + "\n".join(history_lines)

    user_prompt = (
        f"Question: {query}\n\n"
        f"Documentation Context:\n{context_text}"
        f"{history_block}\n\n"
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
            async for chunk in llm.astream(messages, config):
                piece = getattr(chunk, "content", "")
                if piece:
                    chunks.append(piece)
            response_text = "".join(chunks) or "[DocAgent: empty response]"
    except Exception as exc:  # noqa: BLE001
        logger.error("[doc_generate_node] LLM call failed: %s", exc, exc_info=True)
        # Dumping raw retrieved chunks here read as a confident answer while
        # actually being unrelated corpus text. Say the answer is unavailable.
        response_text = (
            "I couldn't generate an answer just now — the language model was "
            "unavailable. Relevant sources are listed below; please retry."
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
