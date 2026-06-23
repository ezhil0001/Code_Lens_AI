"""
DebugAgent — handles error diagnosis, stack trace analysis, and fix suggestions.

The distinguishing step is `debug_parse_error_node`, which extracts structured
error metadata (error type, file path, line number) from the query text before
hitting the retriever.  This makes the subsequent retrieval query much more
precise — instead of searching for the raw error message, we search near the
specific file and function that raised it.

Node sequence:
  debug_parse_error_node  → extract {error_type, file_path, line_number}
  debug_retrieve_node     → code search scoped around the failing code
  debug_pattern_node      → BM25 scan of known error patterns in the corpus
  debug_dependency_node   → find callers of the failing function
  debug_generate_node     → root-cause analysis with optional fix suggestion
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from app.graph.state import AgentState

logger = logging.getLogger(__name__)
MAX_CHARS_PER_SOURCE = 8_000

# Common Python/JS error patterns for structured extraction
_ERROR_PATTERNS = [
    re.compile(r'(?P<error_type>\w+Error|\w+Exception): (?P<message>.+)'),
    re.compile(r'File "(?P<file_path>[^"]+)", line (?P<line_number>\d+)'),
    re.compile(r'at (?P<file_path>\S+):(?P<line_number>\d+)'),   # JS stack frame
]


# ─────────────────────────────────────────────────────────────────────────────
# Node implementations
# ─────────────────────────────────────────────────────────────────────────────

async def debug_parse_error_node(state: dict, config: RunnableConfig = None) -> dict:
    """Extract structured error info from the user query / stack trace."""
    query: str = state.get("query", "")
    visited = list(state.get("nodes_visited", []))
    visited.append("debug_parse_error_node")

    parsed: Dict[str, Any] = {"raw_query": query}
    for pattern in _ERROR_PATTERNS:
        m = pattern.search(query)
        if m:
            parsed.update({k: v for k, v in m.groupdict().items() if v})

    tool_calls = list(state.get("tool_calls", []))
    tool_calls.append({"node": "debug_parse_error_node", "parsed_error": parsed})

    return {"tool_calls": tool_calls, "nodes_visited": visited}


async def debug_retrieve_node(state: dict, config: RunnableConfig = None) -> dict:
    """Retrieve code chunks relevant to the error context."""
    import threading

    query: str = state.get("query", "")
    tool_calls: List[Dict] = state.get("tool_calls", [])
    visited = list(state.get("nodes_visited", []))
    visited.append("debug_retrieve_node")

    # Enrich query with parsed error context if available
    parsed = {}
    for tc in tool_calls:
        if tc.get("node") == "debug_parse_error_node":
            parsed = tc.get("parsed_error", {})

    enriched_query = query
    if parsed.get("error_type"):
        enriched_query = f"{parsed['error_type']} {query}"
    if parsed.get("file_path"):
        enriched_query = f"{enriched_query} file:{parsed['file_path']}"

    chunks: List[Dict[str, Any]] = []
    try:
        from app.services.pipeline_factory import get_pipeline_factory_cached
        factory = get_pipeline_factory_cached()
        retriever = factory.get_retriever_engine()
        _lock = getattr(retriever, "_metadata_lock", threading.Lock())
        with _lock:
            result = retriever.retrieve(
                query=enriched_query,
                top_k=10,
                metadata_filter={"file_type": "code"},
            )
        chunks = result.chunks if result else []
        logger.info("[debug_retrieve_node] retrieved %d chunks", len(chunks))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[debug_retrieve_node] retrieval failed: %s", exc)

    return {"retrieved_chunks": chunks, "nodes_visited": visited}


async def debug_pattern_node(state: dict, config: RunnableConfig = None) -> dict:
    """Look for known error pattern matches in documentation."""
    query: str = state.get("query", "")
    visited = list(state.get("nodes_visited", []))
    visited.append("debug_pattern_node")

    pattern_chunks: List[Dict] = []
    try:
        from app.services.pipeline_factory import get_pipeline_factory_cached
        factory = get_pipeline_factory_cached()
        retriever = factory.get_retriever_engine()
        import threading
        _lock = getattr(retriever, "_metadata_lock", threading.Lock())
        with _lock:
            result = retriever.retrieve(
                query=f"error handling {query}",
                top_k=3,
                metadata_filter=None,
            )
        pattern_chunks = (result.chunks if result else [])[:3]
    except Exception as exc:  # noqa: BLE001
        logger.debug("[debug_pattern_node] pattern search failed: %s", exc)

    existing = list(state.get("retrieved_chunks", []))
    combined = existing + [c for c in pattern_chunks if c not in existing]

    return {"retrieved_chunks": combined[:10], "nodes_visited": visited}


async def debug_dependency_node(state: dict, config: RunnableConfig = None) -> dict:
    """Find callers / dependencies of the failing function."""
    query: str = state.get("query", "")
    visited = list(state.get("nodes_visited", []))
    visited.append("debug_dependency_node")
    # Dependency traversal: Phase J enhancement — pass-through for now
    return {"nodes_visited": visited}


async def debug_generate_node(state: dict, config: RunnableConfig = None) -> dict:
    """Generate root-cause analysis and fix suggestion."""
    query: str = state.get("query", "")
    chunks: List[Dict] = state.get("retrieved_chunks", [])[:5]
    visited = list(state.get("nodes_visited", []))
    visited.append("debug_generate_node")

    context_parts: List[str] = []
    sources: List[Dict] = []
    for i, chunk in enumerate(chunks):
        content = chunk.get("content", chunk.get("page_content", ""))[:MAX_CHARS_PER_SOURCE]
        metadata = chunk.get("metadata", {})
        file_path = metadata.get("file_path", metadata.get("source", "unknown"))
        context_parts.append(f"### {file_path}\n```\n{content}\n```")
        sources.append({
            "id": chunk.get("id", f"debug-{i}"),
            "file_path": file_path,
            "score": chunk.get("score", 0.0),
            "content": content[:500],
        })

    context_text = "\n\n".join(context_parts) if context_parts else "No relevant code found."

    system_prompt = (
        "You are CodeLens AI, an expert debugging assistant. "
        "Analyse the error/issue and provide: "
        "1) Root cause analysis, 2) Explanation, 3) Fix suggestion with code. "
        "Use ONLY the provided code context."
    )
    user_prompt = (
        f"Issue: {query}\n\n"
        f"Relevant Code:\n{context_text}\n\n"
        "Provide root-cause analysis and a specific fix."
    )

    response_text: str = "[DebugAgent: LLM unavailable — error context assembled]"
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
            response_text = "".join(chunks) or "[DebugAgent: empty response]"
    except Exception as exc:  # noqa: BLE001
        logger.warning("[debug_generate_node] LLM call failed: %s", exc)

    agent_responses = dict(state.get("agent_responses", {}))
    agent_responses["DebugAgent"] = response_text
    existing_sources = list(state.get("sources", []))
    existing_sources.extend(sources)

    return {
        "active_agent": "DebugAgent",
        "agent_responses": agent_responses,
        "sources": existing_sources,
        "nodes_visited": visited,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_debug_agent() -> Any:
    """Build and compile the DebugAgent sub-graph."""
    builder = StateGraph(AgentState)

    builder.add_node("debug_parse_error_node",  debug_parse_error_node)
    builder.add_node("debug_retrieve_node",     debug_retrieve_node)
    builder.add_node("debug_pattern_node",      debug_pattern_node)
    builder.add_node("debug_dependency_node",   debug_dependency_node)
    builder.add_node("debug_generate_node",     debug_generate_node)

    builder.set_entry_point("debug_parse_error_node")
    builder.add_edge("debug_parse_error_node", "debug_retrieve_node")
    builder.add_edge("debug_retrieve_node",    "debug_pattern_node")
    builder.add_edge("debug_pattern_node",     "debug_dependency_node")
    builder.add_edge("debug_dependency_node",  "debug_generate_node")
    builder.add_edge("debug_generate_node",    END)

    return builder.compile()
