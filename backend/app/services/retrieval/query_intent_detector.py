"""QueryIntentDetector - Intelligent query classification for weight tuning.

Analyzes query characteristics to determine optimal vector/BM25 weights
and configuration parameters dynamically.

Author: Phase 2 Optimization System
"""

import re
import logging
from enum import Enum
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass


logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Query intent classification."""
    EXACT_MATCH = "exact_match"           # Function/class names
    CONCEPTUAL = "conceptual"             # How-to, patterns, architecture
    MIXED = "mixed"                       # Both specific and conceptual
    TYPO_TOLERANT = "typo_tolerant"       # Likely misspellings
    MULTI_LANGUAGE = "multi_language"     # Language-agnostic concepts
    ERROR_HANDLING = "error_handling"     # Exception/error patterns
    CONFIG_LOOKUP = "config_lookup"       # Configuration values


@dataclass
class IntentAnalysis:
    """Result of query intent detection."""
    intent_type: IntentType
    confidence: float  # 0.0-1.0
    vector_weight: float
    bm25_weight: float
    candidates_k: int
    final_k: int
    skip_expansion: bool
    rerank_needed: bool
    rationale: str
    detected_patterns: List[str]  # Patterns that triggered this classification


class QueryIntentDetector:
    """Detects query intent and recommends optimal retrieval parameters."""
    
    # Patterns for exact code element matching
    EXACT_MATCH_PATTERNS = {
        r"^def\s+\w+": "Function definition",
        r"^class\s+\w+": "Class definition",
        r"^\w+\s*=\s*['\"]": "Variable assignment",
        r"^import\s+": "Import statement",
        r"^from\s+\w+\s+import": "Import statement",
        r"@\w+": "Decorator pattern",
        r"\w+\(\s*\)": "Function call pattern",
    }
    
    # Patterns for conceptual queries
    CONCEPTUAL_PATTERNS = {
        r"(?i)how\s+to": "How-to question",
        r"(?i)what's?\s+(?:the|a)\s+(?:best\s+)?way": "Best practice question",
        r"(?i)explain": "Explanation request",
        r"(?i)tutorial": "Tutorial request",
        r"(?i)pattern": "Pattern inquiry",
        r"(?i)architecture": "Architecture question",
        r"(?i)design": "Design question",
    }
    
    # Patterns for error/exception handling
    ERROR_PATTERNS = {
        r"(?i)error": "Error mention",
        r"(?i)exception": "Exception mention",
        r"(?i)try\s+except": "Try-except pattern",
        r"(?i)catch": "Catch pattern",
        r"(?i)throw": "Throw pattern",
        r"(?i)fail": "Failure mention",
        r"(?i)handle": "Handle mention",
    }
    
    # Patterns for configuration
    CONFIG_PATTERNS = {
        r"^[A-Z_]+\s*=": "Uppercase assignment (constant)",
        r"(?i)config": "Configuration mention",
        r"(?i)setting": "Settings mention",
        r"(?i)environment": "Environment variable",
        r"(?i)timeout": "Timeout config",
        r"(?i)limit|max|min": "Limit/threshold config",
    }
    
    # Patterns suggesting typos/abbreviations
    TYPO_INDICATORS = {
        r"\w{10,}": "Long single word (possible typo)",
        r"[a-z]{2,}(?:[A-Z][a-z]{2,})+": "CamelCase abbreviation",
        r"\b\w{1,3}\b": "Very short word (abbreviation)",
        r"(?<=[a-z])[A-Z]{2,}": "Multiple capital letters mid-word",
    }
    
    def __init__(self):
        """Initialize the detector."""
        self.intent_threshold = 0.6  # Confidence threshold
        self.typo_probability_threshold = 0.4
    
    def detect_intent(
        self,
        query: str,
        verbose: bool = False
    ) -> IntentAnalysis:
        """
        Detect query intent and return optimal parameters.
        
        Args:
            query: User's search query
            verbose: Whether to log detection details
            
        Returns:
            IntentAnalysis with classification and parameters
        """
        query = query.strip()
        
        if not query:
            logger.warning("Empty query provided to intent detector")
            return self._get_default_intent()
        
        # Analyze query characteristics
        detected_patterns = []
        scores = {intent_type: 0.0 for intent_type in IntentType}
        
        # Check exact match patterns
        exact_match_score = self._check_patterns(
            query, self.EXACT_MATCH_PATTERNS, detected_patterns
        )
        scores[IntentType.EXACT_MATCH] = exact_match_score
        
        # Check conceptual patterns
        conceptual_score = self._check_patterns(
            query, self.CONCEPTUAL_PATTERNS, detected_patterns
        )
        scores[IntentType.CONCEPTUAL] = conceptual_score
        
        # Check error patterns
        error_score = self._check_patterns(
            query, self.ERROR_PATTERNS, detected_patterns
        )
        scores[IntentType.ERROR_HANDLING] = error_score
        
        # Check config patterns
        config_score = self._check_patterns(
            query, self.CONFIG_PATTERNS, detected_patterns
        )
        scores[IntentType.CONFIG_LOOKUP] = config_score
        
        # Check for typo indicators
        typo_score = self._detect_typos(query, detected_patterns)
        if typo_score > self.typo_probability_threshold:
            scores[IntentType.TYPO_TOLERANT] = typo_score
        
        # Check for multi-language concepts
        multi_lang_score = self._detect_multi_language(query, detected_patterns)
        scores[IntentType.MULTI_LANGUAGE] = multi_lang_score
        
        # Determine final intent
        final_intent = self._resolve_intent(scores, detected_patterns)
        
        # Get parameters for this intent
        params = self._get_parameters(final_intent, scores)
        
        if verbose:
            logger.info(
                f"Query Intent Detection for '{query}': {final_intent.value} "
                f"(confidence: {params.confidence:.2f}) | Patterns: {detected_patterns}"
            )
        
        return params
    
    def _check_patterns(
        self,
        query: str,
        patterns: Dict[str, str],
        detected_patterns: List[str]
    ) -> float:
        """Check if query matches any patterns."""
        matches = 0
        for pattern, label in patterns.items():
            if re.search(pattern, query):
                matches += 1
                detected_patterns.append(label)
        
        return min(matches / len(patterns), 1.0) if patterns else 0.0
    
    def _detect_typos(self, query: str, detected_patterns: List[str]) -> float:
        """Estimate probability that query contains typos."""
        typo_indicators = 0
        words = query.split()
        
        # Check for long single words without spaces (common typo pattern)
        long_words = [w for w in words if len(w) > 12 and w.isalpha()]
        if long_words:
            typo_indicators += len(long_words)
            detected_patterns.append("Long unbroken word (possible typo)")
        
        # Check for unusual character patterns
        unusual_patterns = sum(1 for w in words if re.search(r"[aeiou]{3,}", w))
        if unusual_patterns:
            typo_indicators += unusual_patterns
            detected_patterns.append("Unusual vowel patterns")
        
        # Check for missing spaces in CamelCase
        camelcase_words = [w for w in words if re.match(r"[a-z]+[A-Z][a-z]+", w)]
        if camelcase_words:
            typo_indicators += len(camelcase_words)
            detected_patterns.append("CamelCase abbreviation")
        
        return min(typo_indicators / max(len(words), 1), 1.0)
    
    def _detect_multi_language(
        self,
        query: str,
        detected_patterns: List[str]
    ) -> float:
        """Detect if query is language-agnostic concept."""
        # Concepts that appear across languages
        multi_lang_concepts = [
            r"(?i)async",
            r"(?i)concur",
            r"(?i)thread",
            r"(?i)promise",
            r"(?i)decorator",
            r"(?i)iterator",
            r"(?i)factory",
            r"(?i)singleton",
            r"(?i)observer",
            r"(?i)middleware",
            r"(?i)context",
            r"(?i)event",
        ]
        
        matches = sum(1 for concept in multi_lang_concepts if re.search(concept, query))
        
        if matches > 0:
            detected_patterns.append(f"Multi-language concept ({matches} detected)")
        
        return min(matches / len(multi_lang_concepts), 1.0)
    
    def _resolve_intent(
        self,
        scores: Dict[IntentType, float],
        detected_patterns: List[str]
    ) -> IntentType:
        """Resolve final intent from scores."""
        # Special case: if both exact and conceptual high, it's mixed
        exact_score = scores[IntentType.EXACT_MATCH]
        conceptual_score = scores[IntentType.CONCEPTUAL]
        
        if (exact_score > 0.3 and conceptual_score > 0.3):
            return IntentType.MIXED
        
        # If typo indicators high, return typo tolerant
        if scores[IntentType.TYPO_TOLERANT] > 0.5:
            return IntentType.TYPO_TOLERANT
        
        # If error patterns high
        if scores[IntentType.ERROR_HANDLING] > 0.4:
            return IntentType.ERROR_HANDLING
        
        # If config patterns high
        if scores[IntentType.CONFIG_LOOKUP] > 0.4:
            return IntentType.CONFIG_LOOKUP
        
        # Get highest scoring intent
        highest_intent = max(scores, key=scores.get)
        if scores[highest_intent] > self.intent_threshold:
            return highest_intent
        
        # Default to mixed if no clear winner
        return IntentType.MIXED
    
    def _get_parameters(
        self,
        intent: IntentType,
        scores: Dict[IntentType, float]
    ) -> IntentAnalysis:
        """Get retrieval parameters for intent type."""
        confidence = scores[intent]
        
        # Base configuration by intent
        configs = {
            IntentType.EXACT_MATCH: {
                "vector_weight": 0.3,
                "bm25_weight": 0.7,
                "candidates_k": 10,
                "final_k": 5,
                "skip_expansion": False,
                "rerank_needed": False,
                "rationale": "Exact matches dominate - BM25 preferred",
            },
            IntentType.CONCEPTUAL: {
                "vector_weight": 0.8,
                "bm25_weight": 0.2,
                "candidates_k": 10,
                "final_k": 5,
                "skip_expansion": False,
                "rerank_needed": True,
                "rationale": "Semantic understanding crucial - Vector preferred",
            },
            IntentType.MIXED: {
                "vector_weight": 0.5,
                "bm25_weight": 0.5,
                "candidates_k": 10,
                "final_k": 5,
                "skip_expansion": False,
                "rerank_needed": True,
                "rationale": "Balanced approach for mixed queries",
            },
            IntentType.TYPO_TOLERANT: {
                "vector_weight": 0.9,
                "bm25_weight": 0.1,
                "candidates_k": 10,
                "final_k": 5,
                "skip_expansion": True,
                "rerank_needed": True,
                "rationale": "Vector resilience to typos - skip expansion to avoid noise",
            },
            IntentType.ERROR_HANDLING: {
                "vector_weight": 0.75,
                "bm25_weight": 0.25,
                "candidates_k": 10,
                "final_k": 5,
                "skip_expansion": False,
                "rerank_needed": True,
                "rationale": "Error patterns are conceptual but specific",
            },
            IntentType.CONFIG_LOOKUP: {
                "vector_weight": 0.4,
                "bm25_weight": 0.6,
                "candidates_k": 10,
                "final_k": 5,
                "skip_expansion": False,
                "rerank_needed": False,
                "rationale": "Config values are exact - BM25 preferred",
            },
            IntentType.MULTI_LANGUAGE: {
                "vector_weight": 0.75,
                "bm25_weight": 0.25,
                "candidates_k": 10,
                "final_k": 5,
                "skip_expansion": False,
                "rerank_needed": True,
                "rationale": "Cross-language concepts need semantic understanding",
            },
        }
        
        config = configs.get(intent, configs[IntentType.MIXED])
        
        return IntentAnalysis(
            intent_type=intent,
            confidence=min(confidence, 1.0),
            vector_weight=config["vector_weight"],
            bm25_weight=config["bm25_weight"],
            candidates_k=config["candidates_k"],
            final_k=config["final_k"],
            skip_expansion=config["skip_expansion"],
            rerank_needed=config["rerank_needed"],
            rationale=config["rationale"],
            detected_patterns=[],
        )
    
    def _get_default_intent(self) -> IntentAnalysis:
        """Return default mixed intent."""
        return IntentAnalysis(
            intent_type=IntentType.MIXED,
            confidence=0.0,
            vector_weight=0.6,
            bm25_weight=0.4,
            candidates_k=10,
            final_k=5,
            skip_expansion=False,
            rerank_needed=True,
            rationale="Default balanced configuration",
            detected_patterns=[],
        )


class AdaptiveWeightStrategy:
    """Manages dynamic weight adjustment based on search results."""
    
    def __init__(
        self,
        enable_adaptive: bool = True,
        min_vector_weight: float = 0.1,
        max_vector_weight: float = 0.95,
    ):
        """Initialize adaptive strategy."""
        self.enable_adaptive = enable_adaptive
        self.min_vector_weight = min_vector_weight
        self.max_vector_weight = max_vector_weight
    
    def adjust_weights_for_empty_bm25(
        self,
        vector_results: int,
        bm25_results: int,
        current_vector_weight: float,
        current_bm25_weight: float,
    ) -> Tuple[float, float]:
        """
        Adjust weights if BM25 returns no results (detected typo/novel term).
        
        Args:
            vector_results: Number of vector search results
            bm25_results: Number of BM25 results
            current_vector_weight: Current vector weight
            current_bm25_weight: Current BM25 weight
            
        Returns:
            Tuple of (adjusted_vector_weight, adjusted_bm25_weight)
        """
        if not self.enable_adaptive:
            return current_vector_weight, current_bm25_weight
        
        # If BM25 returned nothing but vector has results
        if bm25_results == 0 and vector_results > 0:
            logger.warning(
                f"BM25 returned 0 results. Boosting vector weight "
                f"from {current_vector_weight:.2f} to 0.85"
            )
            return 0.85, 0.15
        
        # If both returned results, keep current
        if bm25_results > 0 and vector_results > 0:
            return current_vector_weight, current_bm25_weight
        
        # Fallback
        return current_vector_weight, current_bm25_weight
    
    def adjust_weights_for_low_confidence(
        self,
        top_score: float,
        avg_score: float,
        current_vector_weight: float,
        current_bm25_weight: float,
    ) -> Tuple[float, float]:
        """
        Adjust weights if confidence is low (scores clustered near 0).
        
        Args:
            top_score: Highest scoring result
            avg_score: Average score of top results
            current_vector_weight: Current vector weight
            current_bm25_weight: Current BM25 weight
            
        Returns:
            Tuple of (adjusted_vector_weight, adjusted_bm25_weight)
        """
        if not self.enable_adaptive:
            return current_vector_weight, current_bm25_weight
        
        # If scores are low (< 0.3), boost vector for semantic understanding
        if top_score < 0.3:
            logger.warning(
                f"Low confidence scores (top: {top_score:.2f}, avg: {avg_score:.2f}). "
                f"Boosting vector weight for semantic search."
            )
            new_vector_weight = min(
                current_vector_weight + 0.15,
                self.max_vector_weight
            )
            new_bm25_weight = 1.0 - new_vector_weight
            return new_vector_weight, new_bm25_weight
        
        return current_vector_weight, current_bm25_weight


if __name__ == "__main__":
    # Test the detector
    detector = QueryIntentDetector()
    
    test_queries = [
        "def authenticate",
        "How to handle authentication with JWT?",
        "authenticate usrs",
        "MAX_RETRIES",
        "error handling best practices",
        "async iterator pattern",
    ]
    
    for query in test_queries:
        result = detector.detect_intent(query, verbose=True)
        print(f"\n  → Intent: {result.intent_type.value}")
        print(f"    Weights: vector={result.vector_weight}, bm25={result.bm25_weight}")
        print(f"    Confidence: {result.confidence:.2f}")
