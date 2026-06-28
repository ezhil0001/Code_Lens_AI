"""
Entity Extractor — extracts salient facts from a completed conversation turn
and queues them for long-term memory storage.
=============================================================================
Runs as a post-turn node (memory_write_node) after every response. Uses a
lightweight LLM call with a structured extraction prompt. Falls back to
heuristic keyword scanning when the LLM is unavailable or too slow.

Entity types:
  user_fact   — what the user is working on, their team, project, goals
  code_fact   — specific modules, functions, bugs, or architectural decisions
  preference  — how the user prefers answers: verbosity, language, format

Tested by:
  C-008 — extract_facts() importable
  C-009 — VALID_ENTITY_TYPES == {"user_fact", "code_fact", "preference"}
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Constants (tested by C-009) ───────────────────────────────────────────────
VALID_ENTITY_TYPES: frozenset[str] = frozenset({"user_fact", "code_fact", "preference"})

# Extraction prompt — instructs the LLM to produce a JSON list
_EXTRACT_PROMPT_TEMPLATE = """\
You are a memory extraction assistant. Review this conversation turn and extract \
factual statements that should be remembered for future sessions.

Focus ONLY on:
  - What the user is working on (user_fact)
  - Specific code modules, functions, bugs, or architectural decisions (code_fact)
  - How the user prefers answers (preference)

Output a valid JSON array. Each item: {{"content": "...", "entity_type": "user_fact|code_fact|preference"}}
Return [] if nothing noteworthy.

Conversation turn:
User:      {user_query}
Assistant: {assistant_response}

JSON output:"""

# Heuristic patterns for fallback extraction (no LLM)
_CODE_FACT_PATTERNS = [
    re.compile(r'\b(function|class|module|service|endpoint|API|database|schema|table)\b', re.I),
    re.compile(r'\b(bug|error|issue|fix|crash|exception|traceback)\b', re.I),
    re.compile(r'\b(authentication|authorization|JWT|OAuth|password|token)\b', re.I),
]
_USER_FACT_PATTERNS = [
    re.compile(r'\bI\s+(am|work|need|want|have|use)\b', re.I),
    re.compile(r'\bwe\s+(are|use|need|have)\b', re.I),
    re.compile(r'\bour\s+(team|project|codebase|system)\b', re.I),
]
_PREFERENCE_PATTERNS = [
    re.compile(r'\b(prefer|rather|always|never|please|don\'t|do not)\b', re.I),
    re.compile(r'\bshort(er)?\s+answer', re.I),
    re.compile(r'\b(example|code snippet|step.by.step)\b', re.I),
]


# ─────────────────────────────────────────────────────────────────────────────
# Core extraction function (C-008)
# ─────────────────────────────────────────────────────────────────────────────

async def extract_facts(
    user_query: str,
    assistant_response: str,
    llm: Optional[Any] = None,
) -> List[Dict[str, str]]:
    """Extract salient facts from a conversation turn.

    Tries LLM extraction first (structured JSON output). Falls back to
    heuristic pattern matching if LLM is unavailable or returns invalid JSON.

    Args:
        user_query:          The user's message for this turn.
        assistant_response:  The assistant's response for this turn.
        llm:                 Optional LangChain LLM instance. If None, heuristics are used.

    Returns:
        List of {"content": str, "entity_type": str} dicts.
        entity_type is always one of VALID_ENTITY_TYPES.
    """
    # ── Try LLM extraction ────────────────────────────────────────────────────
    if llm is not None:
        try:
            facts = await _extract_via_llm(user_query, assistant_response, llm)
            if facts:
                return _validate_facts(facts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[entity_extractor] LLM extraction failed: %s — falling back", exc)

    # ── Heuristic fallback ────────────────────────────────────────────────────
    return _extract_heuristic(user_query, assistant_response)


async def _extract_via_llm(
    user_query: str,
    assistant_response: str,
    llm: Any,
) -> List[Dict[str, str]]:
    """Call the LLM to extract facts and parse JSON response."""
    prompt = _EXTRACT_PROMPT_TEMPLATE.format(
        user_query=user_query[:1000],        # guard against huge queries
        assistant_response=assistant_response[:2000],
    )

    from langchain_core.messages import HumanMessage  # type: ignore
    ai_msg = await llm.ainvoke([HumanMessage(content=prompt)])
    raw = getattr(ai_msg, "content", str(ai_msg)).strip()

    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()

    return json.loads(raw)  # raises json.JSONDecodeError on bad output


def _extract_heuristic(user_query: str, assistant_response: str) -> List[Dict[str, str]]:
    """Cheap pattern-based extraction when LLM is unavailable."""
    combined = f"{user_query} {assistant_response}"
    facts: List[Dict[str, str]] = []

    # Extract code facts from sentences mentioning code artifacts
    sentences = re.split(r'[.!?]\s+', combined)
    for sentence in sentences[:20]:  # cap to avoid runaway
        sentence = sentence.strip()
        if not sentence or len(sentence) < 20:
            continue
        if any(p.search(sentence) for p in _CODE_FACT_PATTERNS):
            facts.append({"content": sentence[:300], "entity_type": "code_fact"})
        elif any(p.search(sentence) for p in _USER_FACT_PATTERNS):
            facts.append({"content": sentence[:300], "entity_type": "user_fact"})
        elif any(p.search(sentence) for p in _PREFERENCE_PATTERNS):
            facts.append({"content": sentence[:300], "entity_type": "preference"})

    # Deduplicate by content prefix
    seen: set[str] = set()
    deduped: List[Dict[str, str]] = []
    for f in facts:
        key = f["content"][:80]
        if key not in seen:
            seen.add(key)
            deduped.append(f)

    return deduped[:10]  # cap at 10 facts per turn


def _validate_facts(raw_facts: List[Any]) -> List[Dict[str, str]]:
    """Validate and normalise extracted facts.

    Drops entries with unknown entity_type and truncates content to 500 chars.
    """
    validated: List[Dict[str, str]] = []
    for item in raw_facts:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        entity_type = str(item.get("entity_type", "user_fact")).strip()
        if not content:
            continue
        if entity_type not in VALID_ENTITY_TYPES:
            entity_type = "user_fact"  # safe default
        validated.append({"content": content[:500], "entity_type": entity_type})
    return validated


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node — memory write (post-turn)
# ─────────────────────────────────────────────────────────────────────────────

async def memory_write_node(state: dict, config=None) -> dict:
    """Post-turn node: extract facts from the completed turn and queue them.

    Reads:
        state["query"]          — the user's question
        state["final_response"] — the assistant's answer
        state["user_id"]        — for namespace isolation
        state["session_id"]     — tagged as source_session

    Writes:
        state["memory_write_queue"] — list of MemoryEntry-like dicts
        state["nodes_visited"]      — appended
    """
    user_id: str = state.get("user_id", "anonymous")
    query: str = state.get("query", "")
    response: str = state.get("final_response", "")
    session_id: str = state.get("session_id", "")
    visited = list(state.get("nodes_visited", []))
    visited.append("memory_write_node")

    if not query or not response:
        return {"nodes_visited": visited, "memory_write_queue": []}

    # Get LLM if available
    llm = None
    try:
        from app.services.pipeline_factory import get_pipeline_factory_cached
        factory = get_pipeline_factory_cached()
        llm = getattr(factory, "get_llm", lambda: None)()
    except Exception:  # noqa: BLE001
        pass

    facts = await extract_facts(query, response, llm)

    # Build write queue entries
    write_queue = [
        {
            "user_id": user_id,
            "content": f["content"],
            "entity_type": f["entity_type"],
            "source_session": session_id,
        }
        for f in facts
    ]

    # Best-effort async persist (non-blocking — failures are logged, not raised)
    if write_queue:
        try:
            from app.graph.memory.long_term_store import get_ltm_store, MemoryEntry
            store = get_ltm_store()
            entries = [MemoryEntry(**e) for e in write_queue]
            stored = await store.store_batch(entries)
            logger.info("[memory_write_node] persisted %d/%d facts for user=%s",
                        stored, len(entries), user_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[memory_write_node] batch store failed: %s", exc)

    return {
        "memory_write_queue": write_queue,
        "nodes_visited": visited,
    }
