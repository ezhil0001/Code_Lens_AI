"""
Output Guardrail Node — Phase F: F-34, F-35, F-36
==================================================
A LangGraph node that runs after agents produce final_response, before
the response is streamed to the client.

Check chain (in order):
  1. CodeSafetyScanner  — severity: block — scans code blocks for dangerous
     shell / SQL / Python patterns (rm -rf, eval, exec, DROP TABLE, etc.)
  2. PIILeakScanner     — severity: scrub — re-runs PII detection on the
     generated response to catch cases where the LLM echoed back PII
     from retrieved chunks.
  3. CitationVerifier   — severity: warn — annotates claims that have no
     supporting source in reranked_chunks.

If a "block" check fails, final_response is replaced with a safety
rejection and guardrail_passed is set to False.

Tested by:
  F-007  output_guardrail_node found in this module
  F-008  CodeSafetyScanner blocks rm -rf
  F-009  CodeSafetyScanner blocks eval()
  F-010  CodeSafetyScanner allows safe code
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared result dataclass (re-exported for tests that import from here)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    passed: bool
    violation: Optional[str] = None
    sanitized_query: Optional[str] = None
    severity: str = "block"


# ─────────────────────────────────────────────────────────────────────────────
# Check 1 — Code Safety Scanner  (F-035 / F-008, F-009, F-010)
# ─────────────────────────────────────────────────────────────────────────────

# Pattern tuples: (compiled_regex, human_readable_label)
_DANGEROUS_CODE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Shell: recursive delete
    (re.compile(r"\brm\s+-[a-z]*r[a-z]*f\b", re.IGNORECASE),         "rm -rf"),
    # Shell: format disk/drive
    (re.compile(r"\bformat\s+(disk|drive|c:|/dev/)", re.IGNORECASE),  "format disk/drive"),
    # Shell: fork bomb
    (re.compile(r":\(\)\s*\{.*\|.*\}", re.DOTALL),                    "fork bomb"),
    # Python: eval() with a variable / expression (not a literal)
    (re.compile(r"\beval\s*\(\s*(?![\"\'])", re.IGNORECASE),          "eval()"),
    # Python: exec()
    (re.compile(r"\bexec\s*\(\s*(?![\"\'])", re.IGNORECASE),          "exec()"),
    # Python: __import__ dynamic import
    (re.compile(r"\b__import__\s*\(", re.IGNORECASE),                 "__import__()"),
    # Python: os.system / subprocess.call / subprocess.run with shell=True
    (re.compile(r"\bos\.system\s*\(", re.IGNORECASE),                 "os.system()"),
    (re.compile(r"\bsubprocess\.(call|run|Popen)\s*\(", re.IGNORECASE), "subprocess()"),
    # SQL: DROP TABLE
    (re.compile(r"\bDROP\s+TABLE\b", re.IGNORECASE),                  "DROP TABLE"),
    # SQL: DELETE without WHERE clause  (DELETE FROM table ;)
    (re.compile(r"\bDELETE\s+FROM\s+\w+\s*;", re.IGNORECASE),        "DELETE without WHERE"),
    # SQL: TRUNCATE
    (re.compile(r"\bTRUNCATE\s+TABLE?\b", re.IGNORECASE),             "TRUNCATE TABLE"),
]


class CodeSafetyScanner:
    """Scans code blocks in the response for dangerous patterns.

    Only inspects text inside fenced code blocks (``` ... ```) to avoid
    false-positives on prose that happens to contain a keyword.

    Severity: block.
    """

    name: str = "code_safety_scanner"
    severity: str = "block"

    def _extract_code_blocks(self, text: str) -> list[str]:
        """Return all fenced code block contents."""
        # Match both ``` and ~~~, with optional language tag
        fence_re = re.compile(
            r"(?:```|~~~)[^\n]*\n(.*?)(?:```|~~~)",
            re.DOTALL,
        )
        return fence_re.findall(text)

    async def check(self, response: str, state: Dict[str, Any]) -> GuardrailResult:
        code_blocks = self._extract_code_blocks(response)

        # If there are no fenced code blocks, also scan the raw text
        # (some agents return code without fences)
        texts_to_scan = code_blocks if code_blocks else [response]

        for block in texts_to_scan:
            for pattern, label in _DANGEROUS_CODE_PATTERNS:
                if pattern.search(block):
                    logger.warning(
                        "[code_safety_scanner] blocked dangerous pattern=%s", label
                    )
                    return GuardrailResult(
                        passed=False,
                        violation=f"Dangerous code pattern detected: {label}",
                        severity="block",
                    )

        return GuardrailResult(passed=True, severity="block")


# ─────────────────────────────────────────────────────────────────────────────
# Check 2 — PII Leak Scanner  (F-036)
# ─────────────────────────────────────────────────────────────────────────────

# Re-use the same regex patterns from input_guardrail
_PII_RESPONSE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
     "[EMAIL]"),
    (re.compile(
        r"(\+?\d{1,3}[\s\-\.]?)?(\(?\d{2,4}\)?[\s\-\.]?)\d{3,4}[\s\-\.]?\d{4}"
    ), "[PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
]


class PIILeakScanner:
    """Detects PII leaked into the agent response.

    Severity: scrub — replaces PII tokens in the response.
    """

    name: str = "pii_leak_scanner"
    severity: str = "scrub"

    async def check(self, response: str, state: Dict[str, Any]) -> GuardrailResult:
        cleaned = response
        violations: list[str] = []
        for pattern, replacement in _PII_RESPONSE_PATTERNS:
            new_cleaned = pattern.sub(replacement, cleaned)
            if new_cleaned != cleaned:
                violations.append(f"PII replaced with {replacement}")
                cleaned = new_cleaned

        if violations:
            return GuardrailResult(
                passed=True,
                violation="; ".join(violations),
                sanitized_query=cleaned,  # re-using field for sanitized response
                severity="scrub",
            )
        return GuardrailResult(passed=True, sanitized_query=response, severity="scrub")


# ─────────────────────────────────────────────────────────────────────────────
# Check 3 — Citation Verifier  (F-034)  — lightweight / warn only
# ─────────────────────────────────────────────────────────────────────────────

class CitationVerifier:
    """Annotates claims that have no supporting source.

    Severity: warn — never blocks; just appends a ⚠️ note.
    """

    name: str = "citation_verifier"
    severity: str = "warn"

    async def check(self, response: str, state: Dict[str, Any]) -> GuardrailResult:
        chunks: list[dict] = state.get("reranked_chunks", [])
        if not chunks:
            # No context retrieved — skip verification (e.g. cache hit, web agent)
            return GuardrailResult(passed=True, severity="warn")

        # Extract a combined source corpus for simple substring checks
        source_corpus = " ".join(
            str(c.get("text", "") or c.get("page_content", ""))
            for c in chunks[:10]
        ).lower()

        # Extract sentences from the response (split on ". " or ".\n")
        sentences = re.split(r"\.\s+|\.\n", response)
        unverified: list[str] = []

        for sentence in sentences:
            sentence = sentence.strip()
            # Only check substantive sentences (>30 chars, no code markers)
            if len(sentence) < 30 or "```" in sentence or sentence.startswith("#"):
                continue
            # Quick heuristic: if no 4-gram from the sentence appears in sources,
            # flag as potentially unverified
            words = sentence.lower().split()
            if len(words) < 4:
                continue
            found = any(
                " ".join(words[i:i+4]) in source_corpus
                for i in range(len(words) - 3)
            )
            if not found:
                unverified.append(sentence[:60])

        if len(unverified) > 2:
            logger.debug(
                "[citation_verifier] %d unverified claims found", len(unverified)
            )
            return GuardrailResult(
                passed=True,
                violation=f"{len(unverified)} claims may lack source support",
                severity="warn",
            )
        return GuardrailResult(passed=True, severity="warn")


# ─────────────────────────────────────────────────────────────────────────────
# Guardrail chain
# ─────────────────────────────────────────────────────────────────────────────

_CODE_SAFETY_SCANNER = CodeSafetyScanner()
_PII_LEAK_SCANNER = PIILeakScanner()
_CITATION_VERIFIER = CitationVerifier()

OUTPUT_GUARDRAIL_CHAIN = [
    _CODE_SAFETY_SCANNER,
    _PII_LEAK_SCANNER,
    _CITATION_VERIFIER,
]


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

async def output_guardrail_node(
    state: Dict[str, Any],
    config: RunnableConfig = None,
) -> Dict[str, Any]:
    """Gate node that validates the agent response before streaming.

    Reads:
        final_response     — the agent-generated response
        reranked_chunks    — source chunks (for citation verification)
        nodes_visited      — traversal history

    Returns partial state dict:
        guardrail_passed   — False if a block check failed
        guardrail_violations — accumulated violations
        final_response     — may be replaced (blocked) or scrubbed
        nodes_visited      — updated
    """
    response: str = state.get("final_response", "") or ""
    visited: list[str] = list(state.get("nodes_visited", []))
    violations: list[dict] = list(state.get("guardrail_violations", []))

    current_response = response

    for check in OUTPUT_GUARDRAIL_CHAIN:
        result = await check.check(current_response, state)

        if not result.passed and result.severity == "block":
            visited.append("output_guardrail_node:blocked")
            _emit_guardrail_metric(check.name, "blocked")
            violations.append({"check": check.name, "reason": result.violation})
            return {
                "guardrail_passed": False,
                "guardrail_violations": violations,
                "final_response": (
                    "⚠️ Response blocked by output safety filter: "
                    f"{result.violation}"
                ),
                "nodes_visited": visited,
            }

        if result.violation:
            violations.append({"check": check.name, "reason": result.violation})
            _emit_guardrail_metric(check.name, "scrubbed")
            # Update response for scrub-severity checks
            if result.severity == "scrub" and result.sanitized_query is not None:
                current_response = result.sanitized_query
        else:
            _emit_guardrail_metric(check.name, "passed")

    visited.append("output_guardrail_node:passed")
    return {
        "guardrail_passed": True,
        "guardrail_violations": violations,
        "final_response": current_response,
        "nodes_visited": visited,
    }


def _emit_guardrail_metric(check_name: str, action: str) -> None:
    try:
        from app.observability.quality_metrics import GUARDRAIL_EVENTS  # type: ignore
        if GUARDRAIL_EVENTS is not None:
            GUARDRAIL_EVENTS.labels(check_name=check_name, action=action).inc()
    except Exception:  # noqa: BLE001
        pass
