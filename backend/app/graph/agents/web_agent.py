"""
WebAgent — handles queries that need external information.

Covers CVE lookups, changelog queries, package documentation, and anything
else that requires a live web search rather than the local code corpus.
Uses Tavily when TAVILY_API_KEY is set; falls back to a helpful error message
that tells the user to configure the key rather than silently returning nothing.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from app.graph.state import AgentState

logger = logging.getLogger(__name__)


async def web_search_node(state: dict, config: RunnableConfig = None) -> dict:
    """Call Tavily web search for external references."""
    query: str = state.get("query", "")
    visited = list(state.get("nodes_visited", []))
    visited.append("web_search_node")

    results: List[Dict] = []
    api_key = os.getenv("TAVILY_API_KEY", "")

    if not api_key:
        logger.info("[web_search_node] TAVILY_API_KEY not set — skipping web search")
        agent_responses = dict(state.get("agent_responses", {}))
        agent_responses["WebAgent"] = (
            "[WebAgent: External search unavailable — TAVILY_API_KEY not configured]"
        )
        return {"active_agent": "WebAgent", "agent_responses": agent_responses,
                "nodes_visited": visited}

    try:
        from tavily import TavilyClient  # type: ignore
        from app.observability.tracing import span as _lf_span
        with _lf_span(
            "tavily.search",
            kind="tool",
            input={"query": query[:300], "max_results": 5},
            metadata={"api": "Tavily", "endpoint": "search", "http.method": "POST"},
        ) as _s:
            client = TavilyClient(api_key=api_key)
            # H-1 / timeout hardening: Tavily's client is synchronous — run it
            # in a worker thread with a hard timeout so a hung HTTP call can
            # never block the event loop or stall the stream indefinitely.
            import asyncio
            response = await asyncio.wait_for(
                asyncio.to_thread(client.search, query=query, max_results=5),
                timeout=float(os.getenv("TAVILY_TIMEOUT_SECONDS", "10")),
            )
            results = [
                {
                    "id": f"web-{i}",
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "content": r.get("content", "")[:2000],
                    "score": r.get("score", 0.0),
                }
                for i, r in enumerate(response.get("results", []))
            ]
            _s.update(output={"results": len(results), "urls": [r["url"] for r in results]})
        logger.info("[web_search_node] Tavily returned %d results", len(results))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[web_search_node] Tavily search failed: %s", exc)

    context_text = "\n\n".join(
        f"### {r['title']} ({r['url']})\n{r['content']}" for r in results
    ) if results else "No external results found."

    response_text = f"External search results for '{query}':\n\n{context_text}"

    agent_responses = dict(state.get("agent_responses", {}))
    agent_responses["WebAgent"] = response_text
    existing_sources = list(state.get("sources", []))
    existing_sources.extend(results)
    tool_results = list(state.get("tool_results", []))
    tool_results.extend(results)

    return {
        "active_agent": "WebAgent",
        "agent_responses": agent_responses,
        "sources": existing_sources,
        "tool_results": tool_results,
        "nodes_visited": visited,
    }


def build_web_agent() -> Any:
    """Build and compile the WebAgent sub-graph."""
    builder = StateGraph(AgentState)
    builder.add_node("web_search_node", web_search_node)
    builder.set_entry_point("web_search_node")
    builder.add_edge("web_search_node", END)
    return builder.compile()
