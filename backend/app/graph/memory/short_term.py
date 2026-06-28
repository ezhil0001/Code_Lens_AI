"""
Short-term memory node — loads the recent conversation window into graph state.

Loads the last N turns from PostgresChatMessageHistory and injects them into
AgentState["short_term_window"] so downstream nodes have turn-by-turn context
without re-querying the database.

Two constraints that must not be relaxed:
  - session_id must be the namespaced "{user_id}::{raw_session_id}" form.
    A bare session_id would let one user's history bleed into another's.
  - If the window exceeds STM_MAX_TOKENS words, the oldest turns are trimmed
    rather than passed verbatim.  Large history windows caused context-length
    errors with the Groq API at ~12k tokens before this guard was added.

The node is resilient by design: a history load failure returns an empty
window so the graph can still answer the current query cold.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

# ── Tunables (overridden by env / settings) ──────────────────────────────────
STM_WINDOW_TURNS: int = 10    # max number of conversation turns to load
STM_MAX_TOKENS: int = 4096    # rough word-count budget before trimming


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers (tested directly by C-003, C-004)
# ─────────────────────────────────────────────────────────────────────────────

def build_namespaced_session_id(user_id: str, session_id: str) -> str:
    """Return the security-namespaced session key: '{user_id}::{session_id}'.

    Never pass a bare session_id to PostgresChatMessageHistory — this
    namespace prevents cross-user history leakage.
    """
    # If already namespaced, return as-is
    if "::" in session_id and session_id.startswith(f"{user_id}::"):
        return session_id
    return f"{user_id}::{session_id}"


async def apply_token_budget(
    window: List[Dict[str, Any]],
    max_tokens: int = STM_MAX_TOKENS,
) -> List[Dict[str, Any]]:
    """Trim the oldest turns from *window* until total word-count ≤ max_tokens.

    Uses a simple whitespace word-count as a cheap proxy for token count.
    When the window still exceeds the budget after removing all but the
    last 2 turns, those last 2 turns are returned as-is (never empty).

    Args:
        window:     List of {"role": str, "content": str} dicts, oldest first.
        max_tokens: Maximum word budget (default 4096).

    Returns:
        Trimmed list, always ≤ max_tokens words (best-effort).
    """
    if not window:
        return window

    def _word_count(turns: List[Dict[str, Any]]) -> int:
        return sum(len(t.get("content", "").split()) for t in turns)

    # Fast path: already within budget
    if _word_count(window) <= max_tokens:
        return window

    # Drop oldest turns until within budget, keeping at least 2 turns
    trimmed = list(window)
    while len(trimmed) > 2 and _word_count(trimmed) > max_tokens:
        trimmed.pop(0)

    return trimmed


# ─────────────────────────────────────────────────────────────────────────────
# History loader (graceful — never raises)
# ─────────────────────────────────────────────────────────────────────────────

async def _load_history(namespaced_session_id: str, window_turns: int) -> List[Dict[str, Any]]:
    """Load the last *window_turns* turns from PostgresChatMessageHistory.

    Returns an empty list on any failure so the graph can still proceed.
    """
    try:
        from langchain_community.chat_message_histories import (  # type: ignore
            PostgresChatMessageHistory,
        )
        from app.core.database import build_psycopg_dsn

        connection_string = build_psycopg_dsn()
        history = PostgresChatMessageHistory(
            connection_string=connection_string,
            session_id=namespaced_session_id,
        )
        messages = history.messages  # synchronous; LangChain loads eagerly
        # Convert BaseMessage objects → plain dicts for state storage
        turns: List[Dict[str, Any]] = []
        for msg in messages[-(window_turns * 2):]:  # each turn = user + assistant
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "")
            turns.append({"role": role, "content": content})
        return turns

    except Exception as exc:  # noqa: BLE001
        logger.warning("[STM] history load failed (session=%s): %s",
                       namespaced_session_id, exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

async def memory_read_node(state: dict, config: RunnableConfig = None) -> dict:
    """Load short-term memory window for the current session.

    Reads:
        state["user_id"]    — used for namespace isolation
        state["session_id"] — may be bare or already namespaced
        state["query"]      — used for LTM semantic retrieval

    Writes:
        state["short_term_window"] — list of {role, content} dicts
        state["long_term_facts"]   — list of relevant fact strings from LTM
        state["nodes_visited"]     — appended
    """
    user_id: str = state.get("user_id", "anonymous")
    raw_session_id: str = state.get("session_id", "")
    query: str = state.get("query", "")
    visited = list(state.get("nodes_visited", []))
    visited.append("memory_read_node")

    # Always enforce namespace
    namespaced_id = build_namespaced_session_id(user_id, raw_session_id)

    # Load raw history
    window = await _load_history(namespaced_id, STM_WINDOW_TURNS)

    # Apply token budget guard
    window = await apply_token_budget(window, STM_MAX_TOKENS)

    logger.info("[STM] loaded %d turns for session %s", len(window), namespaced_id)

    # Retrieve relevant facts from long-term memory using pgvector similarity
    ltm_facts: List[str] = []
    if user_id and user_id != "anonymous" and query:
        try:
            from app.graph.memory.long_term_store import get_ltm_store
            ltm_facts = await get_ltm_store().retrieve(user_id=user_id, query=query, top_k=5)
            logger.info("[LTM] retrieved %d facts for user=%s", len(ltm_facts), user_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[LTM] retrieve failed in memory_read_node: %s", exc)

    if ltm_facts:
        # Prepend LTM context as a synthetic system turn so the LLM sees it
        ltm_turn = {
            "role": "system",
            "content": "Remembered facts from previous sessions:\n"
                       + "\n".join(f"- {f}" for f in ltm_facts),
        }
        window = [ltm_turn] + window

    return {
        "short_term_window": window,
        "long_term_facts": ltm_facts,
        "nodes_visited": visited,
    }
