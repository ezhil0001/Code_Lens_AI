"""
Intent classifier node — uses an LLM structured-output call to decide which
agent(s) should handle the query, then writes routing fields to state.

Design
------
LLM path (primary):
    A single ChatGroq call with a JSON-schema-constrained output returns
    { "agents": ["CodeAgent", "DocAgent"], "confidence": 0.87 }.
    The node writes *all* returned agents to state["routing_agents"] so
    the supervisor's conditional edge can dispatch them in parallel via
    LangGraph's Send() API.

Keyword-rule fallback (secondary):
    If the LLM call fails (network, rate limit, empty API key) the node
    falls back to the original keyword scoring logic.  This means a
    misconfigured API key degrades to single-agent routing rather than a
    hard failure, which keeps the server running.

Backward compatibility:
    state["routing_decision"] is still written as the *first* agent in the
    list so any code path that reads that single field continues to work.
    state["metadata_filter"] reflects the filter for that primary agent.

LLM model:
    Reuses the same factory / provider that the rest of the app uses
    (GROQ_MODEL env var, default llama-3.1-70b-versatile).  Temperature is
    forced to 0 for deterministic routing decisions.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


# ── Valid agent names ─────────────────────────────────────────────────────────

VALID_AGENTS: frozenset[str] = frozenset({
    "CodeAgent", "DocAgent", "DebugAgent", "ArchAgent", "WebAgent",
})


# ── Intent string constants used in state.intent ─────────────────────────────

INTENT_CODE_LOOKUP   = "CODE_LOOKUP"
INTENT_DEBUG         = "DEBUG"
INTENT_ARCHITECTURE  = "ARCHITECTURE"
INTENT_KT_DOC        = "KT_DOC"
INTENT_HYBRID        = "HYBRID"
INTENT_WEB           = "WEB"


# ── Agent → (metadata_filter, intent_string) ─────────────────────────────────

_AGENT_META: Dict[str, tuple[Optional[Dict[str, Any]], str]] = {
    "CodeAgent":  ({"file_type": "code"},   INTENT_CODE_LOOKUP),
    "DocAgent":   ({"file_type": "kt_doc"}, INTENT_KT_DOC),
    "DebugAgent": ({"file_type": "code"},   INTENT_DEBUG),
    "ArchAgent":  (None,                    INTENT_ARCHITECTURE),
    "WebAgent":   (None,                    INTENT_WEB),
}

# Backward-compat alias used by supervisor_graph.route_to_agent
_ROUTING_TABLE: Dict[str, tuple[str, Optional[Dict[str, Any]], str]] = {
    name: (name, meta[0], meta[1])
    for name, meta in _AGENT_META.items()
}


# ── Keyword fallback tables (used when LLM is unavailable) ───────────────────

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


# ── LLM routing prompt ────────────────────────────────────────────────────────

_ROUTING_SYSTEM_PROMPT = """\
You are an expert routing supervisor for a multi-agent code assistant.

Available agents:
- CodeAgent   : retrieves and explains source code (functions, classes, imports)
- DocAgent    : retrieves and explains documentation, KT docs, technical guides
- DebugAgent  : analyses errors, exceptions, stack traces, and bugs
- ArchAgent   : explains system architecture, data flows, design decisions
- WebAgent    : searches external sources for CVEs, library docs, changelogs

Your job: given a user query, return a JSON object that picks the BEST agent(s).

Rules:
1. Pick ONE agent for focused queries.
2. Pick TWO agents only when the query genuinely needs two distinct capabilities
   (e.g. "explain the architecture AND show me the code for X" needs both
   ArchAgent and CodeAgent).  Do NOT pad with extra agents.
3. Never pick more than two agents.
4. confidence must be between 0.0 and 1.0 and reflect how sure you are.

Respond with ONLY this JSON and nothing else:
{"agents": ["<AgentName>", ...], "confidence": <float>}
"""

_ROUTING_USER_TEMPLATE = "Query: {query}"


# ── LLM-based classifier ──────────────────────────────────────────────────────

async def _llm_classify(query: str) -> tuple[List[str], float]:
    """
    Call the configured LLM to classify the query.

    Returns (agent_list, confidence).  The returned agent names are validated
    against VALID_AGENTS; any unrecognised name is dropped.  If the result is
    empty after validation, raises ValueError so the caller can fall back.
    """
    from app.services.pipeline_factory import get_pipeline_factory_cached

    factory = get_pipeline_factory_cached()
    base_llm = factory.get_llm()
    if base_llm is None:
        raise RuntimeError("LLM not initialised in pipeline factory")

    # Force temperature=0 for deterministic routing; preserve the provider
    # settings by binding only the overrides we need.
    try:
        llm = base_llm.bind(temperature=0, max_tokens=128)
    except Exception:
        llm = base_llm  # bind not supported — use as-is

    from langchain_core.messages import SystemMessage, HumanMessage

    messages = [
        SystemMessage(content=_ROUTING_SYSTEM_PROMPT),
        HumanMessage(content=_ROUTING_USER_TEMPLATE.format(query=query)),
    ]

    try:
        # Hard timeout so a hung provider call can never stall the stream.
        import asyncio, os
        _timeout = float(os.getenv("INTENT_LLM_TIMEOUT_SECONDS", "15"))
        response = await asyncio.wait_for(llm.ainvoke(messages), timeout=_timeout)
        raw = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        raise RuntimeError(f"LLM invoke failed: {exc}") from exc

    # Strip markdown code fences if the model wrapped its JSON
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines()
            if not line.startswith("```")
        ).strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned non-JSON: {cleaned!r}") from exc

    raw_agents: list = payload.get("agents", [])
    confidence: float = float(payload.get("confidence", 0.5))

    # Validate and cap at 2 agents (safety guard against a verbose model)
    valid = [a for a in raw_agents if a in VALID_AGENTS]
    if not valid:
        raise ValueError(f"LLM returned no valid agent names: {raw_agents}")

    return valid[:2], round(min(max(confidence, 0.0), 1.0), 2)


# ── Keyword-rule fallback ─────────────────────────────────────────────────────

def _keyword_classify(query: str) -> tuple[List[str], float]:
    """
    Keyword-rule fallback used when the LLM is unavailable.

    Returns (agent_list, confidence).  Always returns exactly one agent,
    mirroring the previous single-dispatch behaviour so degraded mode is
    safe and predictable.
    """
    q = query.lower()

    web_hits = sum(1 for kw in _WEB_KEYWORDS if kw in q)
    if web_hits >= 1:
        return ["WebAgent"], round(min(0.65 + web_hits * 0.08, 0.95), 2)

    debug_hits = sum(1 for kw in _DEBUG_KEYWORDS if kw in q)
    if debug_hits >= 1:
        return ["DebugAgent"], round(min(0.6 + debug_hits * 0.08, 0.95), 2)

    arch_hits = sum(1 for kw in _ARCHITECTURE_KEYWORDS if kw in q)
    if arch_hits >= 1:
        return ["ArchAgent"], round(min(0.6 + arch_hits * 0.08, 0.92), 2)

    doc_hits  = sum(1 for kw in _DOC_KEYWORDS  if kw in q)
    code_hits = sum(1 for kw in _CODE_KEYWORDS if kw in q)

    if doc_hits > code_hits:
        return ["DocAgent"],  round(min(0.55 + doc_hits  * 0.07, 0.90), 2)
    if code_hits > doc_hits:
        return ["CodeAgent"], round(min(0.55 + code_hits * 0.07, 0.90), 2)

    return ["CodeAgent"], 0.50  # HYBRID fallback — equal or no signals


# ── Node ──────────────────────────────────────────────────────────────────────

async def intent_classifier_node(state: dict, config: RunnableConfig = None) -> dict:
    """
    LangGraph node: classify the query with an LLM call and populate routing state.

    Reads:   state["query"], state["nodes_visited"]
    Writes:  intent, routing_decision, routing_agents, routing_confidence,
             metadata_filter, nodes_visited

    Routing output semantics
    ------------------------
    routing_agents   — full ordered list from the LLM (1–2 names).
                       The supervisor dispatches every agent in this list
                       via Send() so they run in the same superstep.
    routing_decision — first agent in routing_agents; kept for backward
                       compatibility with code that reads the single-value field.
    metadata_filter  — ChromaDB filter for the primary (first) agent.

    Failure policy
    --------------
    LLM errors → keyword fallback → confidence is set to a lower value so
    the HIL node can optionally interrupt if confidence drops below its
    threshold.  The node itself never raises.
    """
    query: str = state.get("query", "")
    nodes_visited: list = list(state.get("nodes_visited", []))

    agents: List[str] = []
    confidence: float = 0.0
    used_llm: bool = False

    # ── Primary: LLM routing ──────────────────────────────────────────────────
    try:
        agents, confidence = await _llm_classify(query)
        used_llm = True
        logger.info(
            "[INTENT_CLASSIFIER] LLM routing → agents=%s confidence=%.2f query=%r",
            agents, confidence, query[:80],
        )
    except Exception as llm_exc:  # noqa: BLE001
        logger.warning(
            "[INTENT_CLASSIFIER] LLM routing failed (%s) — keyword fallback",
            llm_exc,
        )

    # ── Fallback: keyword rules ───────────────────────────────────────────────
    if not agents:
        agents, confidence = _keyword_classify(query)
        logger.info(
            "[INTENT_CLASSIFIER] keyword routing → agents=%s confidence=%.2f query=%r",
            agents, confidence, query[:80],
        )

    # ── Build state delta ─────────────────────────────────────────────────────
    primary_agent = agents[0]
    metadata_filter, intent_str = _AGENT_META.get(primary_agent, (None, INTENT_HYBRID))

    if len(agents) > 1:
        # Multi-agent dispatch — use HYBRID as the umbrella intent label
        intent_str = INTENT_HYBRID

    nodes_visited.append(
        f"intent_classifier_node:{'llm' if used_llm else 'keyword'}"
    )

    return {
        "intent": intent_str,
        "routing_decision": primary_agent,
        "routing_agents": agents,
        "routing_confidence": confidence,
        "metadata_filter": metadata_filter,
        "nodes_visited": nodes_visited,
    }
