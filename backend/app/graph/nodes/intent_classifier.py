"""
Intent classifier node — maps the user's query to a routing decision and the
corresponding ChromaDB metadata filter.

Routing table:
  CODEBASE_ONLY  → CodeAgent   (filter: {"file_type": "code"})
  KT_ONLY        → DocAgent    (filter: {"file_type": "kt_doc"})
  HYBRID         → CodeAgent   (no filter — retrieves from both collections)
  MULTI_SOURCE   → CodeAgent   (no filter)
  AGENT_TOOL     → DebugAgent  (no filter)
  CONTEXT_AWARE  → CodeAgent   (no filter — falls back to broad retrieval)
  DEBUG_*        → DebugAgent
  ARCHITECTURE   → ArchAgent

The keyword rules here intentionally mirror the legacy AgenticRouter logic
so existing test queries still route the same way.  Importing AgenticRouter
itself is avoided because it carries the full legacy pipeline as a side
effect of construction.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


# ── Intent / routing keyword tables ──────────────────────────────────────────

_DEBUG_KEYWORDS = frozenset({
    "error", "exception", "traceback", "stack trace", "bug", "fail",
    "failing", "crash", "why is", "why does", "not working", "broken",
    "undefined", "nullpointer", "typeerror", "attributeerror", "keyerror",
    "valueerror", "importerror", "modulenotfounderror", "fix",
})

_ARCHITECTURE_KEYWORDS = frozenset({
    "architecture", "design", "diagram", "flow", "data flow", "overview",
    "high-level", "system design", "adr", "adrs", "component", "service",
    "microservice", "pipeline", "infra", "infrastructure",
})

_DOC_KEYWORDS = frozenset({
    "explain", "what is", "what does", "what are", "how does",
    "documentation", "kt", "knowledge transfer", "concept", "understand",
    "learn", "describe", "definition", "overview of", "introduction",
})

_CODE_KEYWORDS = frozenset({
    "def ", "class ", "function", "method", "implement", "code",
    "show me", "find", "locate", "navigate", "where is", "which file",
    "example of", "usage of", "call", "import",
})

_WEB_KEYWORDS = frozenset({
    "cve", "vulnerability", "vulnerabilities", "latest version",
    "changelog", "release notes", "npm package", "pypi", "external docs",
    "official docs", "documentation site", "security advisory",
    "patch notes", "upstream", "github issue",
})


# ── Intent string constants used in state.intent ─────────────────────────────

INTENT_CODE_LOOKUP   = "CODE_LOOKUP"
INTENT_DEBUG         = "DEBUG"
INTENT_ARCHITECTURE  = "ARCHITECTURE"
INTENT_KT_DOC        = "KT_DOC"
INTENT_HYBRID        = "HYBRID"
INTENT_WEB           = "WEB"


# ── Routing decision → (agent_node_name, metadata_filter, intent) ─────────────

_ROUTING_TABLE: Dict[str, tuple[str, Optional[Dict[str, Any]], str]] = {
    "CODE_LOOKUP":  ("CodeAgent",  {"file_type": "code"},   INTENT_CODE_LOOKUP),
    "KT_DOC":       ("DocAgent",   {"file_type": "kt_doc"}, INTENT_KT_DOC),
    "DEBUG":        ("DebugAgent", {"file_type": "code"},   INTENT_DEBUG),
    "ARCHITECTURE": ("ArchAgent",  None,                    INTENT_ARCHITECTURE),
    "HYBRID":       ("CodeAgent",  None,                    INTENT_HYBRID),
    "WEB":          ("WebAgent",   None,                    INTENT_WEB),
}


def _classify(query: str) -> tuple[str, float]:
    """
    Classify a query into one of the intent buckets.

    Returns (intent_key, confidence) where intent_key is one of the keys
    in _ROUTING_TABLE and confidence is 0.0–1.0.

    This is a keyword-rule classifier that mirrors the existing AgenticRouter
    logic, ported to a pure function so it can live in the LangGraph node
    without importing the heavyweight AgenticRouter class.
    """
    q = query.lower()

    # ── WEB: external CVEs, package lookups, changelogs ───────────────────────
    web_hits = sum(1 for kw in _WEB_KEYWORDS if kw in q)
    if web_hits >= 1:
        confidence = min(0.65 + web_hits * 0.08, 0.95)
        return "WEB", round(confidence, 2)

    # ── DEBUG: highest precedence — error signals are unambiguous ─────────────
    debug_hits = sum(1 for kw in _DEBUG_KEYWORDS if kw in q)
    if debug_hits >= 1:
        confidence = min(0.6 + debug_hits * 0.08, 0.95)
        return "DEBUG", round(confidence, 2)

    # ── ARCHITECTURE ──────────────────────────────────────────────────────────
    arch_hits = sum(1 for kw in _ARCHITECTURE_KEYWORDS if kw in q)
    if arch_hits >= 1:
        confidence = min(0.6 + arch_hits * 0.08, 0.92)
        return "ARCHITECTURE", round(confidence, 2)

    # ── KT / DOCS ─────────────────────────────────────────────────────────────
    doc_hits  = sum(1 for kw in _DOC_KEYWORDS  if kw in q)
    code_hits = sum(1 for kw in _CODE_KEYWORDS if kw in q)

    if doc_hits > code_hits:
        confidence = min(0.55 + doc_hits * 0.07, 0.90)
        return "KT_DOC", round(confidence, 2)

    if code_hits > doc_hits:
        confidence = min(0.55 + code_hits * 0.07, 0.90)
        return "CODE_LOOKUP", round(confidence, 2)

    # ── HYBRID — equal signals or neither ─────────────────────────────────────
    return "HYBRID", 0.50


async def intent_classifier_node(state: dict, config: RunnableConfig = None) -> dict:
    """
    LangGraph node: classify user intent and set routing fields in state.

    Reads:   state["query"], state["nodes_visited"]
    Writes:  intent, routing_decision, routing_confidence,
             metadata_filter, nodes_visited

    This node never raises — all exceptions are caught and the graph falls
    back to HYBRID routing with low confidence so that a HIL node can decide
    whether to interrupt.
    """
    query: str = state.get("query", "")
    nodes_visited: list = list(state.get("nodes_visited", []))

    try:
        intent_key, confidence = _classify(query)
        agent_name, metadata_filter, intent_str = _ROUTING_TABLE[intent_key]

        logger.info(
            "[INTENT_CLASSIFIER] query=%r intent=%s agent=%s confidence=%.2f",
            query[:60],
            intent_str,
            agent_name,
            confidence,
        )

        nodes_visited.append("intent_classifier_node")

        return {
            "intent": intent_str,
            "routing_decision": agent_name,
            "routing_confidence": confidence,
            "metadata_filter": metadata_filter,
            "nodes_visited": nodes_visited,
        }

    except Exception as exc:  # noqa: BLE001
        logger.error("[INTENT_CLASSIFIER] Unexpected error: %s — falling back to HYBRID", exc)
        nodes_visited.append("intent_classifier_node:error")
        return {
            "intent": INTENT_HYBRID,
            "routing_decision": "CodeAgent",
            "routing_confidence": 0.3,
            "metadata_filter": None,
            "nodes_visited": nodes_visited,
        }
