"""
Human-in-the-Loop (HIL) check node — gates the response when the agent is
uncertain or when the query signals a potentially destructive operation.

HIL interrupts on any of these conditions:
  1. routing_confidence is below HIL_CONFIDENCE_THRESHOLD (default 0.55).
     Low confidence means the classifier wasn't sure which agent to use,
     which often correlates with ambiguous or multi-domain queries where an
     automatic answer could be wrong or misleading.
  2. The query contains destructive-intent keywords such as "drop table",
     "delete everything", or "remove all".  We'd rather ask once than have
     the user act on an answer that misunderstood their scope.

When HIL is not triggered the node is a transparent pass-through — it adds
one dict assignment to the state and exits.

When HIL is triggered, the node sets hil_required=True and raises a
NodeInterrupt so LangGraph persists the checkpoint and halts.  The user
then calls POST /api/v2/sessions/{session_id}/resume with their decision.
The resume handler writes hil_approved and hil_human_input back into the
checkpoint before re-invoking the graph from this node.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

HIL_CONFIDENCE_THRESHOLD: float = 0.55
"""Routing confidence below this value triggers a human review interrupt."""


# ─────────────────────────────────────────────────────────────────────────────
# Destructive-intent keyword detection
# ─────────────────────────────────────────────────────────────────────────────

_DESTRUCTIVE_PATTERNS: list[re.Pattern[str]] = [
    # DDL / DML destruction
    re.compile(r"\bdrop\s+table\b",          re.IGNORECASE),
    re.compile(r"\bdrop\s+database\b",       re.IGNORECASE),
    re.compile(r"\btruncate\b",              re.IGNORECASE),
    # Data removal
    re.compile(r"\bdelete\s+(?:all|every)\b", re.IGNORECASE),
    re.compile(r"\bdelete\s+everything\b",   re.IGNORECASE),
    re.compile(r"\bremove\s+all\b",          re.IGNORECASE),
    re.compile(r"\berase\s+all\b",           re.IGNORECASE),
    re.compile(r"\bnuke\b",                  re.IGNORECASE),
    # File-system destruction
    re.compile(r"\brm\s+-rf\b",             re.IGNORECASE),
    re.compile(r"\bformat\s+(?:disk|drive|the)\b", re.IGNORECASE),
]

# Human-readable labels for each pattern (same order as _DESTRUCTIVE_PATTERNS)
_PATTERN_LABELS: list[str] = [
    "drop table",
    "drop database",
    "truncate",
    "delete all/every",
    "delete everything",
    "remove all",
    "erase all",
    "nuke",
    "rm -rf",
    "format disk/drive",
]


def contains_destructive_intent(query: str) -> bool:
    """Return True if *query* contains any recognised destructive keyword.

    Case-insensitive.  Used by hil_check_node and exposed as a public
    utility so it can be unit-tested in isolation (E-009).

    Examples:
        >>> contains_destructive_intent("drop table users")
        True
        >>> contains_destructive_intent("remove all files")
        True
        >>> contains_destructive_intent("how does caching work?")
        False
    """
    for pattern in _DESTRUCTIVE_PATTERNS:
        if pattern.search(query):
            return True
    return False


def _destructive_label(query: str) -> Optional[str]:
    """Return the label of the first matching destructive pattern, or None."""
    for pattern, label in zip(_DESTRUCTIVE_PATTERNS, _PATTERN_LABELS):
        if pattern.search(query):
            return label
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

async def hil_check_node(
    state: Dict[str, Any],
    config: RunnableConfig = None,
) -> Dict[str, Any]:
    """Gate node that decides whether a human review interrupt is needed.

    Reads:
        query                 — original user query
        routing_confidence    — float in [0, 1]
        active_agent          — currently assigned agent (for logging)
        nodes_visited         — list of node names visited so far

    Returns a *partial* state dict (only changed fields):
        hil_required          — bool
        hil_reason            — str | None  (populated when hil_required=True)
        nodes_visited         — updated list

    Side-effects:
        - Logs at INFO level when HIL is triggered.
        - Raises NodeInterrupt (from langgraph) when hil_required=True so
          that LangGraph persists the checkpoint and pauses the graph.
          The resume endpoint re-invokes the graph with hil_approved set.
    """
    query: str = state.get("query", "")
    confidence: float = float(state.get("routing_confidence", 1.0))
    active_agent: Optional[str] = state.get("active_agent")
    visited: list[str] = list(state.get("nodes_visited", []))

    hil_required: bool = False
    hil_reason: Optional[str] = None

    # ── Gate 1: low routing confidence ───────────────────────────────────────
    if confidence < HIL_CONFIDENCE_THRESHOLD:
        hil_required = True
        hil_reason = (
            f"Routing confidence {confidence:.2f} is below the required "
            f"threshold {HIL_CONFIDENCE_THRESHOLD}. "
            f"Assigned agent: {active_agent or 'unknown'}. "
            "A human reviewer should confirm the correct routing."
        )
        logger.info(
            "[hil_check_node] HIL triggered — low confidence %.2f < %.2f "
            "(agent=%s, query=%r)",
            confidence, HIL_CONFIDENCE_THRESHOLD, active_agent, query[:80],
        )

    # ── Gate 2: destructive-intent keywords ───────────────────────────────────
    if not hil_required:
        label = _destructive_label(query)
        if label:
            hil_required = True
            hil_reason = (
                f"Query contains a potentially destructive operation: '{label}'. "
                "Human approval is required before this action proceeds."
            )
            logger.info(
                "[hil_check_node] HIL triggered — destructive keyword '%s' "
                "(query=%r)",
                label, query[:80],
            )

    # ── Append to nodes_visited ───────────────────────────────────────────────
    status_tag = "required" if hil_required else "passed"
    visited.append(f"hil_check_node:{status_tag}")

    return {
        "hil_required": hil_required,
        "hil_reason": hil_reason,
        "nodes_visited": visited,
    }
