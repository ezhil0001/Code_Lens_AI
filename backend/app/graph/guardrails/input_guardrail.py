"""
Input Guardrail Node — Phase F: F-30, F-31, F-32, F-33
=======================================================
A LangGraph node that runs before any retrieval or agent work and
applies a chain of ordered safety checks.

Check chain (in order):
  1. PromptInjectionDetector  — severity: block
  2. TokenBudgetCheck         — severity: block
  3. PIIScrubber              — severity: scrub (continues with clean query)

If a "block" check fails, guardrail_passed is set to False and
final_response is pre-filled with a rejection message — the graph
short-circuits at the cache_check/memory_read stage.

If only "scrub" checks fire, the graph continues with the sanitized
query in state["pii_scrubbed_query"] and state["query"].

Tested by:
  F-001  input_guardrail_node found in this module
  F-002  PromptInjectionDetector blocks known injection strings
  F-003  TokenBudgetCheck blocks queries > 512 tokens
  F-004  Safe query passes guardrail_passed=True
  F-005  PIIScrubber anonymizes email addresses
  F-006  PIIScrubber anonymizes phone numbers
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    """Result returned by every guardrail check."""
    passed: bool
    violation: Optional[str] = None
    sanitized_query: Optional[str] = None   # populated by scrub-severity checks
    severity: str = "block"                 # "block" | "scrub" | "warn"


# ─────────────────────────────────────────────────────────────────────────────
# Check 1 — Prompt Injection Detector  (F-031 / F-002)
# ─────────────────────────────────────────────────────────────────────────────

_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "ignore previous/all instructions"
    (re.compile(r"ignore\s+(previous|all)\s+instructions?", re.IGNORECASE),
     "ignore_instructions"),
    # "you are now [X]" role-play hijack
    (re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
     "role_hijack"),
    # "disregard your/the system/previous prompt"
    (re.compile(r"disregard\s+(your|the)\s+(system|previous)\s+prompt", re.IGNORECASE),
     "disregard_prompt"),
    # "act as a/an different/new/evil/unrestricted AI"
    (re.compile(r"act\s+as\s+(a\s+|an\s+)?(different|new|evil|unrestricted)", re.IGNORECASE),
     "act_as_override"),
    # DAN / jailbreak / developer mode keywords
    (re.compile(r"\b(DAN|jailbreak|developer\s+mode)\b", re.IGNORECASE),
     "jailbreak_keyword"),
    # Token boundary injection  <|...|>
    (re.compile(r"<\|.*?\|>", re.DOTALL),
     "token_boundary_injection"),
    # LLaMA-style instruction tags  [INST] [SYS] [/INST]
    (re.compile(r"\[(INST|SYS|/INST|/SYS)\]", re.IGNORECASE),
     "llama_instruction_tag"),
    # "forget everything"
    (re.compile(r"\bforget\s+every(thing|one)\b", re.IGNORECASE),
     "forget_everything"),
    # "new session" / "new conversation" reset attempt
    (re.compile(r"\b(start\s+a?\s*new\s+(session|conversation)|reset\s+your\s+memory)\b",
                re.IGNORECASE),
     "session_reset"),
    # "bypass" / "override" safety
    (re.compile(r"\b(bypass|override)\s+(safety|filter|restriction|guideline)",
                re.IGNORECASE),
     "bypass_safety"),
    # "pretend you have no restrictions"
    (re.compile(r"\bpretend\s+(you\s+have\s+no|there\s+are\s+no)\s+restriction",
                re.IGNORECASE),
     "pretend_no_restrictions"),
    # Prompt injection continuation marker
    (re.compile(r"---\s*(end|stop)\s+(of\s+)?(system|instruction)", re.IGNORECASE),
     "end_of_system_marker"),
]


class PromptInjectionDetector:
    """Blocks known prompt injection / jailbreak patterns.

    Severity: block — any match stops the guardrail chain immediately.
    """

    name: str = "injection_detector"
    severity: str = "block"

    async def check(self, query: str, state: Dict[str, Any]) -> GuardrailResult:
        for pattern, label in _INJECTION_PATTERNS:
            if pattern.search(query):
                logger.warning(
                    "[injection_detector] blocked pattern=%s query=%r",
                    label, query[:80],
                )
                return GuardrailResult(
                    passed=False,
                    violation=f"Prompt injection detected: {label}",
                    severity="block",
                )
        return GuardrailResult(passed=True, severity="block")


# ─────────────────────────────────────────────────────────────────────────────
# Check 2 — Token Budget Check  (F-033 / F-003)
# ─────────────────────────────────────────────────────────────────────────────

class TokenBudgetCheck:
    """Blocks queries that exceed the maximum allowed token count.

    Uses a simple whitespace-split word count as a proxy for tokens.
    The real tokenizer count is ~30% higher for code-heavy queries, so
    the default budget (512) gives a comfortable margin before the LLM
    context limit is hit.

    Severity: block.
    """

    name: str = "token_budget"
    severity: str = "block"

    def __init__(self, max_tokens: int = 512) -> None:
        self.max_tokens = max_tokens

    async def check(self, query: str, state: Dict[str, Any]) -> GuardrailResult:
        word_count = len(query.split())
        if word_count > self.max_tokens:
            logger.info(
                "[token_budget] blocked — %d words > %d budget",
                word_count, self.max_tokens,
            )
            return GuardrailResult(
                passed=False,
                violation=(
                    f"Query too long: {word_count} words exceed the "
                    f"{self.max_tokens}-token budget."
                ),
                severity="block",
            )
        return GuardrailResult(passed=True, severity="block")


# ─────────────────────────────────────────────────────────────────────────────
# Check 3 — PII Scrubber  (F-030 / F-005, F-006)
# ─────────────────────────────────────────────────────────────────────────────

# Regex-based PII patterns (used when Presidio is not available)
_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Email addresses
    (re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ), "[EMAIL]"),
    # US/International phone numbers  e.g. +1-800-555-0199 / (800) 555-0199
    (re.compile(
        r"(\+?\d{1,3}[\s\-\.]?)?"
        r"(\(?\d{2,4}\)?[\s\-\.]?)"
        r"\d{3,4}[\s\-\.]?\d{4}"
    ), "[PHONE]"),
    # Credit card numbers  (4 groups of 4 digits)
    (re.compile(
        r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"
    ), "[CC_NUMBER]"),
    # US Social Security Numbers  XXX-XX-XXXX
    (re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    ), "[SSN]"),
    # IPv4 addresses
    (re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ), "[IP_ADDRESS]"),
]


class PIIScrubber:
    """Detects and anonymizes PII in the query.

    Attempts to use Presidio if available; falls back to regex patterns.
    Severity: scrub — execution continues with sanitized_query.
    """

    name: str = "pii_scrubber"
    severity: str = "scrub"

    def __init__(self) -> None:
        self._presidio_available = False
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore
            from presidio_anonymizer import AnonymizerEngine  # type: ignore
            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            self._presidio_available = True
            logger.debug("[pii_scrubber] Presidio engine loaded")
        except Exception:
            logger.debug("[pii_scrubber] Presidio unavailable — using regex fallback")

    def _scrub_regex(self, text: str) -> tuple[str, list[str]]:
        """Replace PII using compiled regex patterns."""
        violations: list[str] = []
        result = text
        for pattern, replacement in _PII_PATTERNS:
            new_result = pattern.sub(replacement, result)
            if new_result != result:
                violations.append(
                    f"PII detected and replaced with {replacement}"
                )
                result = new_result
        return result, violations

    def _scrub_presidio(self, text: str) -> tuple[str, list[str]]:
        """Replace PII using Presidio analyzer + anonymizer."""
        from presidio_analyzer import AnalyzerEngine  # type: ignore
        from presidio_anonymizer import AnonymizerEngine  # type: ignore
        from presidio_anonymizer.entities import OperatorConfig  # type: ignore

        results = self._analyzer.analyze(text=text, language="en")
        if not results:
            return text, []

        # Build operator map: replace every entity type with a readable tag
        operators = {
            r.entity_type: OperatorConfig(
                "replace",
                {"new_value": f"[{r.entity_type}]"},
            )
            for r in results
        }
        anonymized = self._anonymizer.anonymize(
            text=text, analyzer_results=results, operators=operators
        )
        violations = [
            f"PII entity [{r.entity_type}] scrubbed"
            for r in results
        ]
        return anonymized.text, violations

    async def check(self, query: str, state: Dict[str, Any]) -> GuardrailResult:
        try:
            if self._presidio_available:
                sanitized, violations = self._scrub_presidio(query)
            else:
                sanitized, violations = self._scrub_regex(query)
        except Exception as exc:
            logger.warning("[pii_scrubber] scrub failed: %s", exc)
            return GuardrailResult(passed=True, sanitized_query=query, severity="scrub")

        if violations:
            logger.info("[pii_scrubber] scrubbed %d PII entities", len(violations))
            return GuardrailResult(
                passed=True,               # scrub — continue, not block
                violation="; ".join(violations),
                sanitized_query=sanitized,
                severity="scrub",
            )
        return GuardrailResult(passed=True, sanitized_query=query, severity="scrub")


# ─────────────────────────────────────────────────────────────────────────────
# Guardrail chain
# ─────────────────────────────────────────────────────────────────────────────

# Default chain instantiated once at import — zero per-request overhead
_TOKEN_BUDGET_CHECK = TokenBudgetCheck(max_tokens=512)
_INJECTION_DETECTOR = PromptInjectionDetector()
_PII_SCRUBBER = PIIScrubber()

INPUT_GUARDRAIL_CHAIN = [
    _INJECTION_DETECTOR,    # must be first: cheapest + most critical
    _TOKEN_BUDGET_CHECK,
    _PII_SCRUBBER,          # last: always runs to clean up PII
]


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

async def input_guardrail_node(
    state: Dict[str, Any],
    config: RunnableConfig = None,
) -> Dict[str, Any]:
    """Gate node that validates and sanitizes the incoming query.

    Reads:
        query          — raw user query
        nodes_visited  — list of node names visited so far

    Returns partial state dict:
        guardrail_passed    — False if any "block" check failed
        guardrail_violations — list of violation dicts
        pii_scrubbed_query  — sanitized query (may equal original)
        query               — updated to sanitized value after PII scrub
        final_response      — set only on block (short-circuits the graph)
        nodes_visited       — updated list
    """
    query: str = state.get("query", "")
    visited: list[str] = list(state.get("nodes_visited", []))
    violations: list[dict] = list(state.get("guardrail_violations", []))

    current_query = query   # may be updated by scrub-severity checks

    for check in INPUT_GUARDRAIL_CHAIN:
        result = await check.check(current_query, state)

        if not result.passed and result.severity == "block":
            visited.append("input_guardrail_node:blocked")
            # Emit Prometheus counter if available
            _emit_guardrail_metric(check.name, "blocked")
            violations.append({"check": check.name, "reason": result.violation})
            return {
                "guardrail_passed": False,
                "guardrail_violations": violations,
                "pii_scrubbed_query": current_query,
                "final_response": f"Request blocked: {result.violation}",
                "nodes_visited": visited,
            }

        if result.violation:
            violations.append({"check": check.name, "reason": result.violation})
            _emit_guardrail_metric(check.name, "scrubbed")

            if result.severity == "scrub" and result.sanitized_query is not None:
                current_query = result.sanitized_query
        else:
            _emit_guardrail_metric(check.name, "passed")

    visited.append("input_guardrail_node:passed")
    return {
        "guardrail_passed": True,
        "guardrail_violations": violations,
        "pii_scrubbed_query": current_query,
        "query": current_query,     # downstream nodes see sanitized version
        "nodes_visited": visited,
    }


def _emit_guardrail_metric(check_name: str, action: str) -> None:
    """Increment GUARDRAIL_EVENTS counter — best-effort, never raises."""
    try:
        from app.observability.quality_metrics import GUARDRAIL_EVENTS  # type: ignore
        if GUARDRAIL_EVENTS is not None:
            GUARDRAIL_EVENTS.labels(check_name=check_name, action=action).inc()
    except Exception:  # noqa: BLE001
        pass
