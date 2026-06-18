"""
Short-Term Memory (STM) — Phase C: F-12
=========================================
Loads the last N conversation turns from PostgresChatMessageHistory and
injects them into AgentState as state["short_term_window"].

Key design rules:
  - Session ID MUST be the namespaced value "{user_id}::{raw_session_id}".
    This is a security fix carried over from the monolithic AgentBrain —
    never pass a bare session_id to the history store.
  - Window size is configurable (default 10 turns).
  - Token budget guard: if combined window > max_tokens (default 4096 words),
    oldest turns are trimmed via apply_token_budget() before state injection.
  - The node is resilient: if history load fails, it returns an empty window
    rather than propagating the error (conversation can still proceed).
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

    Writes:
        state["short_term_window"] — list of {role, content} dicts
        state["nodes_visited"]     — appended
    """
    user_id: str = state.get("user_id", "anonymous")
    raw_session_id: str = state.get("session_id", "")
    visited = list(state.get("nodes_visited", []))
    visited.append("memory_read_node")

    # Always enforce namespace
    namespaced_id = build_namespaced_session_id(user_id, raw_session_id)

    # Load raw history
    window = await _load_history(namespaced_id, STM_WINDOW_TURNS)

    # Apply token budget guard
    window = await apply_token_budget(window, STM_MAX_TOKENS)

    logger.info("[STM] loaded %d turns for session %s", len(window), namespaced_id)

    # Merge with any LTM facts already in state (injected by LongTermStore)
    ltm_facts: List[str] = state.get("long_term_facts", [])
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
        "nodes_visited": visited,
    }
