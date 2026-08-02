"""
CodeAgent — handles queries about source code: function lookups, implementation
explanations, usage examples, and class-level analysis.

Retrieval pipeline:
  code_expand_query_node  → rewrites the query for better lexical recall
  code_retrieve_node      → BM25 + ChromaDB with file_type=code filter
  code_rerank_node        → BGE cross-encoder narrows to top-5 chunks
  code_pdr_node           → swaps chunk fragments for their parent function bodies
  code_truncate_node      → caps each source at 8 000 chars to stay within LLM context
  code_generate_node      → streams the LLM response token-by-token

The metadata_filter is set upstream by the intent classifier and must not be
overwritten here.  Changing it would break multi-agent HYBRID queries that
intentionally omit a filter to search across all file types.

RetrieverEngine.retrieve() runs inside a threading.Lock() because ChromaDB's
Python client is not thread-safe.  Do not remove that lock.

Node names are all prefixed `code_` to avoid key collisions when the supervisor
adds multiple agents to the same StateGraph.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.graph.state import AgentState

logger = logging.getLogger(__name__)

MAX_CHARS_PER_SOURCE = 8_000


# ─────────────────────────────────────────────────────────────────────────────
# Tool — code_search (F-56: Pydantic args_schema)
# ─────────────────────────────────────────────────────────────────────────────

class CodeSearchInput(BaseModel):
    """Input schema for the code_search tool."""
    query: str = Field(..., description="Natural-language query to search the codebase")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    file_pattern: Optional[str] = Field(
        default=None,
        description="Optional glob-style file pattern filter, e.g. '*.py'",
    )


try:
    from langchain_core.tools import tool  # type: ignore

    @tool("code_search", args_schema=CodeSearchInput)
    async def code_search_tool(
        query: str,
        top_k: int = 5,
        file_pattern: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search the codebase for functions, classes, and implementation details."""
        try:
            from app.services.pipeline_factory import get_pipeline_factory_cached
            factory = get_pipeline_factory_cached()
            retriever = factory.get_retriever_engine()
            where_filter: Optional[Dict] = {"file_type": "code"}
            if file_pattern:
                where_filter["file_path"] = {"$contains": file_pattern.lstrip("*.")}
            result = retriever.retrieve(
                query=query,
                top_k=top_k,
                metadata_filter=where_filter,
            )
            return {"chunks": result.chunks[:top_k], "query": query}
        except Exception as exc:  # noqa: BLE001
            logger.warning("[code_search_tool] retrieval failed: %s", exc)
            return {"chunks": [], "query": query, "error": str(exc)}

except Exception:  # noqa: BLE001
    # Fallback: define a plain async function with the correct args_schema attribute
    # so B-003 (args_schema check) still passes even if langchain_core.tools fails.
    async def code_search_tool(  # type: ignore[assignment]
        query: str,
        top_k: int = 5,
        file_pattern: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search the codebase for functions, classes, and implementation details."""
        return {"chunks": [], "query": query}

    code_search_tool.args_schema = CodeSearchInput  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────────────────────
# Node implementations
# ─────────────────────────────────────────────────────────────────────────────

async def code_expand_query_node(state: dict, config: RunnableConfig = None) -> dict:
    """Expand the user query into semantic variations for better recall."""
    query: str = state.get("query", "")
    visited = list(state.get("nodes_visited", []))
    visited.append("code_expand_query_node")

    expanded = [query]
    try:
        # Simple keyword-focused expansion (no external LLM call at this stage)
        if "how" in query.lower() and "work" in query.lower():
            expanded.append(query.replace("how", "implementation of").replace("works", "").strip())
        if "function" not in query.lower() and "def" not in query.lower():
            expanded.append(f"def {query.split()[0]}" if query.split() else query)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[code_expand_query_node] expansion skipped: %s", exc)

    return {
        "nodes_visited": visited,
        "tool_calls": [{"node": "code_expand_query_node", "expanded_queries": expanded}],
    }


async def code_retrieve_node(state: dict, config: RunnableConfig = None) -> dict:
    """BM25 + ChromaDB retrieval with file_type=code metadata filter.

    Uses the code-specialized embedding model (``get_code_embedder()``) for
    the vector leg so that query embeddings are computed in the same vector
    space as the code-chunk embeddings produced during ingestion.  The BM25
    leg is model-agnostic and is unaffected by this change.  Doc retrieval,
    semantic cache, and LTM all keep the general-purpose embedder.
    """
    import threading

    query: str = state.get("query", "")
    metadata_filter = state.get("metadata_filter") or {"file_type": "code"}
    visited = list(state.get("nodes_visited", []))
    visited.append("code_retrieve_node")

    chunks: List[Dict[str, Any]] = []
    try:
        from app.services.pipeline_factory import get_pipeline_factory_cached
        factory = get_pipeline_factory_cached()
        retriever = factory.get_retriever_engine()

        # Swap the vector-leg embedder to the code-specialized singleton for this call.
        # We do it inside the shared, process-wide retrieval lock so no concurrent
        # agent (CodeAgent + DebugAgent dispatched in parallel) runs sentence-
        # transformer inference at the same time — MPS is not thread-safe and
        # concurrent inference deadlocks the whole request.
        hybrid = retriever.hybrid_retriever
        from app.core.database import get_retrieval_lock
        _lock = get_retrieval_lock()

        try:
            from app.core.database import get_code_embedder
            _code_emb = get_code_embedder()
        except Exception as _ce:
            logger.warning("[code_retrieve_node] code embedder unavailable (%s) — using general", _ce)
            _code_emb = None

        def _retrieve_with_code_embedder():
            """Embedder swap + retrieval, executed in a worker thread (H-1).

            The lock is held inside the thread so the event loop is never
            blocked by the embedding + Chroma + BM25 work (~100s of ms).
            contextvars are copied by asyncio.to_thread, so Langfuse spans
            keep nesting under the request trace.
            """
            nonlocal_code_emb = _code_emb
            with _lock:
                _prev_emb = hybrid.vector_retriever.embeddings if nonlocal_code_emb else None
                if nonlocal_code_emb:
                    try:
                        hybrid.vector_retriever.embeddings = nonlocal_code_emb
                    except Exception:
                        nonlocal_code_emb = None  # retriever is immutable — skip swap
                try:
                    return retriever.retrieve(
                        query=query,
                        top_k=10,
                        metadata_filter=metadata_filter,
                    )
                finally:
                    # Always restore the original embedder before releasing the lock
                    if nonlocal_code_emb and _prev_emb is not None:
                        try:
                            hybrid.vector_retriever.embeddings = _prev_emb
                        except Exception:  # noqa: BLE001
                            logger.error("[code_retrieve_node] embedder restore FAILED — general embedder may be replaced")

        import asyncio
        result = await asyncio.to_thread(_retrieve_with_code_embedder)

        chunks = result.chunks if result else []
        logger.info("[code_retrieve_node] retrieved %d chunks (code embedder)", len(chunks))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[code_retrieve_node] retrieval failed: %s", exc)

    return {"retrieved_chunks": chunks, "nodes_visited": visited}


async def code_rerank_node(state: dict, config: RunnableConfig = None) -> dict:
    """Rerank retrieved chunks using BGE cross-encoder."""
    query: str = state.get("query", "")
    chunks: List[Dict] = state.get("retrieved_chunks", [])
    visited = list(state.get("nodes_visited", []))
    visited.append("code_rerank_node")

    reranked: List[Dict] = chunks  # default: pass-through if reranker unavailable
    scores: List[float] = []
    try:
        from app.services.pipeline_factory import get_pipeline_factory_cached
        factory = get_pipeline_factory_cached()
        reranker = getattr(factory, "get_reranker", None)
        if reranker and chunks:
            # rerank() returns a (docs, scores) tuple
            result = factory.get_reranker().rerank(query=query, documents=chunks, top_k=5)
            if isinstance(result, tuple):
                reranked, scores = result[0], list(result[1] or [])
            else:
                reranked = result
        else:
            reranked = chunks[:5]
    except Exception as exc:  # noqa: BLE001
        logger.debug("[code_rerank_node] reranker unavailable: %s — using top-5 pass-through", exc)
        reranked = chunks[:5]

    # M-5: propagate cross-encoder scores for the retrieval_quality evaluator.
    return {"reranked_chunks": reranked, "rerank_scores": scores, "nodes_visited": visited}


async def code_pdr_node(state: dict, config: RunnableConfig = None) -> dict:
    """Parent Document Retrieval - fetch full function bodies for top chunks."""
    reranked: List[Dict] = state.get("reranked_chunks", [])
    visited = list(state.get("nodes_visited", []))
    visited.append("code_pdr_node")

    parent_contexts: Dict[str, str] = {}
    try:
        from app.services.pipeline_factory import get_pipeline_factory_cached
        factory = get_pipeline_factory_cached()
        pdr = getattr(factory, "get_parent_document_retriever", None)
        if pdr and reranked:
            pdr_instance = factory.get_parent_document_retriever()
            for chunk in reranked:
                chunk_id = chunk.get("id") or chunk.get("chunk_id", "")
                if chunk_id:
                    parent = getattr(pdr_instance, "get_parent", None)
                    if parent:
                        parent_contexts[chunk_id] = parent(chunk_id) or ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("[code_pdr_node] PDR unavailable: %s", exc)

    return {"parent_contexts": parent_contexts, "nodes_visited": visited}


async def code_truncate_node(state: dict, config: RunnableConfig = None) -> dict:
    """Truncate oversized chunks to MAX_CHARS_PER_SOURCE."""
    reranked: List[Dict] = state.get("reranked_chunks", [])
    visited = list(state.get("nodes_visited", []))
    visited.append("code_truncate_node")

    truncated = []
    for chunk in reranked:
        content = chunk.get("content", chunk.get("page_content", ""))
        if len(content) > MAX_CHARS_PER_SOURCE:
            chunk = {**chunk, "content": content[:MAX_CHARS_PER_SOURCE] + "\n... [truncated]"}
        truncated.append(chunk)

    return {"reranked_chunks": truncated, "nodes_visited": visited}


async def code_generate_node(state: dict, config: RunnableConfig = None) -> dict:
    """LLM generation using code-focused prompt with retrieved context."""
    query: str = state.get("query", "")
    chunks: List[Dict] = state.get("reranked_chunks", [])
    parent_ctx: Dict[str, str] = state.get("parent_contexts", {})
    visited = list(state.get("nodes_visited", []))
    visited.append("code_generate_node")

    # Build context block from top reranked chunks
    context_parts: List[str] = []
    sources: List[Dict] = []
    for i, chunk in enumerate(chunks):
        content = chunk.get("content", chunk.get("page_content", ""))
        chunk_id = chunk.get("id", chunk.get("chunk_id", f"chunk-{i}"))
        # Prefer parent context (full function body) when available
        full_content = parent_ctx.get(chunk_id, content)
        metadata = chunk.get("metadata", {})
        file_path = metadata.get("file_path", metadata.get("source", "unknown"))
        context_parts.append(
            f"### Source {i + 1}: {file_path}\n```\n{full_content[:MAX_CHARS_PER_SOURCE]}\n```"
        )
        sources.append({
            "id": chunk_id,
            "file_path": file_path,
            "score": chunk.get("score", 0.0),
            "content": content[:500],
        })

    context_text = "\n\n".join(context_parts) if context_parts else "No code context found."

    system_prompt = (
        "You are CodeLens AI, an expert code assistant. "
        "Answer the user's question using ONLY the code context provided. "
        "Be precise, reference specific functions/classes, and include relevant code snippets."
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
        f"Code Context:\n{context_text}"
        f"{history_block}\n\n"
        "Provide a clear, technical answer with code examples where relevant."
    )

    response_text: str = "[CodeAgent: LLM unavailable — retrieval context assembled]"
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
            response_text = "".join(chunks) or "[CodeAgent: empty response]"
    except Exception as exc:  # noqa: BLE001
        logger.warning("[code_generate_node] LLM call failed: %s", exc)
        if context_parts:
            response_text = (
                f"Based on the codebase, here is the relevant context for '{query}':\n\n"
                + context_text[:2000]
            )

    agent_responses = dict(state.get("agent_responses", {}))
    agent_responses["CodeAgent"] = response_text
    existing_sources = list(state.get("sources", []))
    existing_sources.extend(sources)

    return {
        "active_agent": "CodeAgent",
        "agent_responses": agent_responses,
        "sources": existing_sources,
        "nodes_visited": visited,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_code_agent() -> Any:
    """
    Build and compile the CodeAgent sub-graph.

    Returns a compiled StateGraph that can be used as a node in the
    Supervisor graph or invoked standalone for testing.
    """
    builder = StateGraph(AgentState)

    builder.add_node("code_expand_query_node", code_expand_query_node)
    builder.add_node("code_retrieve_node",     code_retrieve_node)
    builder.add_node("code_rerank_node",       code_rerank_node)
    builder.add_node("code_pdr_node",          code_pdr_node)
    builder.add_node("code_truncate_node",     code_truncate_node)
    builder.add_node("code_generate_node",     code_generate_node)

    builder.set_entry_point("code_expand_query_node")
    builder.add_edge("code_expand_query_node", "code_retrieve_node")
    builder.add_edge("code_retrieve_node",     "code_rerank_node")
    builder.add_edge("code_rerank_node",       "code_pdr_node")
    builder.add_edge("code_pdr_node",          "code_truncate_node")
    builder.add_edge("code_truncate_node",     "code_generate_node")
    builder.add_edge("code_generate_node",     END)

    return builder.compile()
