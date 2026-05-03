"""Phase 3: Agentic Router.

Intelligent router that decides:
1. Whether to query codebase, KT documentation, or both
2. Which retrieval strategy to use (based on Phase 2 intent detection)
3. Whether to invoke specialized tools
4. How to combine multiple sources

Routes incoming queries to the most appropriate retrieval and processing path.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RoutingDecision(str, Enum):
    """Possible routing decisions."""
    
    CODEBASE_ONLY = "codebase_only"           # Query code only
    KT_ONLY = "kt_only"                       # Query documentation only
    HYBRID = "hybrid"                         # Query both, then combine
    MULTI_SOURCE = "multi_source"             # Multiple retrieval strategies
    AGENT_TOOL = "agent_tool"                 # Invoke specialized tool
    CONTEXT_AWARE = "context_aware"           # Use conversation history


class RetrievalPriority(str, Enum):
    """Prioritization strategy for results."""
    
    CODE_FIRST = "code_first"                 # Prioritize codebase results
    DOCUMENTATION_FIRST = "documentation_first"  # Prioritize KT docs
    BALANCED = "balanced"                     # Equal weight
    SEMANTIC = "semantic"                     # Order by semantic relevance only


@dataclass
class RoutingConfig:
    """Configuration for routing decisions."""
    
    # Retrieval sources
    enable_codebase_search: bool = True
    enable_documentation_search: bool = True
    enable_multi_source: bool = True
    
    # Strategy selection
    use_query_expansion: bool = True
    use_reranking: bool = True
    enable_dynamic_weights: bool = True
    
    # Result combination
    max_results: int = 5
    include_parent_context: bool = True
    deduplicate_results: bool = True
    
    # Tool invocation
    enable_tool_invocation: bool = True
    available_tools: List[str] = None  # ["code_search", "doc_search", "execute", "analyze"]
    
    def __post_init__(self):
        if self.available_tools is None:
            self.available_tools = ["code_search", "doc_search", "analyze"]


@dataclass
class RoutingResult:
    """Result of routing decision."""
    
    decision: RoutingDecision
    priority: RetrievalPriority
    sources_to_query: List[str]  # ["codebase", "documentation", etc.]
    reasoning: str
    confidence: float  # 0-1, how confident is the router
    recommended_tools: List[str]
    parent_context: Optional[Dict[str, Any]] = None  # Context from Phase 2
    intent_type: Optional[str] = None  # Intent from Phase 2 analysis (e.g., "CODE_LOOKUP", "ARCHITECTURE")
    intent_confidence: float = 0.0  # Confidence of intent detection from Phase 2


def routing_decision_to_metadata_filter(
    result: "RoutingResult",
) -> Optional[Dict[str, Any]]:
    """Translate a RoutingDecision into a Chroma `where=` metadata filter.

    P0 FIX #2: Routing decisions are now ENFORCED at the vector-store layer
    instead of being decorative log lines.

    Mapping:
      - CODEBASE_ONLY  -> {"file_type": "code"}
      - KT_ONLY        -> {"file_type": "kt_doc"}
      - HYBRID / MULTI_SOURCE / CONTEXT_AWARE / AGENT_TOOL -> None (no filter)
    """
    if result is None:
        return None
    decision = getattr(result, "decision", None)
    if decision == RoutingDecision.CODEBASE_ONLY:
        return {"file_type": "code"}
    if decision == RoutingDecision.KT_ONLY:
        return {"file_type": "kt_doc"}
    return None


class AgenticRouter:
    """Intelligently routes queries to appropriate retrieval and processing paths.
    
    ROUTING LOGIC:
    
    1. Analyze User Query
       - Extract intent (from Phase 2)
       - Identify primary goal (what does user want?)
       - Detect required sources (code, docs, both?)
    
    2. Make Routing Decision
       - CODEBASE_ONLY: "Show me the authenticate function"
       - KT_ONLY: "Explain the architecture"
       - HYBRID: "How does authentication work?"
       - AGENT_TOOL: "Execute this code"
    
    3. Select Strategy
       - Prioritization (code vs docs)
       - Retrieval methods (vector vs BM25)
       - Tool invocation if needed
    
    4. Return Routing Result
       - Clear decision
       - Sources to query
       - Reasoning for transparency
       - Tools to invoke
    """
    
    def __init__(self, config: Optional[RoutingConfig] = None):
        """Initialize the router.
        
        Args:
            config: RoutingConfig with routing preferences
        """
        self.config = config or RoutingConfig()
        logger.info("Initialized AgenticRouter")
    
    def route(
        self,
        query: str,
        intent_analysis: Optional[Dict[str, Any]] = None,
        conversation_context: Optional[str] = None,
    ) -> RoutingResult:
        """Route a query to the most appropriate path.
        
        Args:
            query: User's natural language query
            intent_analysis: Phase 2 IntentAnalysis result (optional)
            conversation_context: Recent conversation history for context-awareness
        
        Returns:
            RoutingResult with decision, sources, and reasoning
        """
        try:
            # Step 1: Analyze query characteristics
            query_type = self._classify_query(query)
            query_intent = intent_analysis.get("intent_type") if intent_analysis else None
            intent_confidence = intent_analysis.get("confidence", 0.0) if intent_analysis else 0.0
            
            # Step 2: Determine primary goal
            primary_goal = self._extract_primary_goal(query)
            
            # Step 3: Decide routing
            decision, sources, priority = self._make_routing_decision(
                query_type=query_type,
                query_intent=query_intent,
                primary_goal=primary_goal,
                conversation_context=conversation_context,
            )
            
            # Step 4: Select tools
            tools = self._select_tools(decision, query_type)
            
            # Step 5: Build reasoning
            reasoning = self._build_reasoning(
                query_type, primary_goal, decision, sources
            )
            
            # Step 6: Compute confidence
            confidence = self._compute_confidence(decision, query_type)
            
            result = RoutingResult(
                decision=decision,
                priority=priority,
                sources_to_query=sources,
                reasoning=reasoning,
                confidence=confidence,
                recommended_tools=tools,
                parent_context=intent_analysis,
                intent_type=query_intent,  # FIX #2: Pass intent type from Phase 2
                intent_confidence=intent_confidence,  # FIX #2: Pass intent confidence
            )
            
            logger.info(
                f"Routed query '{query[:50]}...' → {decision.value} "
                f"(confidence: {confidence:.2%}, intent: {query_intent or 'unknown'}, "
                f"intent_confidence: {intent_confidence:.2%})"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error in routing: {e}")
            # Fallback to balanced hybrid search
            return RoutingResult(
                decision=RoutingDecision.HYBRID,
                priority=RetrievalPriority.BALANCED,
                sources_to_query=["codebase", "documentation"],
                reasoning=f"Fallback to hybrid search due to error: {str(e)}",
                confidence=0.5,
                recommended_tools=["code_search", "doc_search"],
            )
    
    def _classify_query(self, query: str) -> str:
        """Classify query into type.
        
        Types:
        - exact_lookup: "def authenticate"
        - conceptual: "How does authentication work?"
        - navigational: "Where is X called?"
        - reference: "Show me examples"
        - explanatory: "Explain Y"
        - operational: "How to do Z?"
        """
        query_lower = query.lower()
        
        # Exact lookup
        if query_lower.startswith(("def ", "class ", "import ")):
            return "exact_lookup"
        
        # Navigational
        if any(kw in query_lower for kw in ["where", "which file", "what calls", "who uses"]):
            return "navigational"
        
        # Reference
        if any(kw in query_lower for kw in ["show me", "example", "demonstrate"]):
            return "reference"
        
        # Operational
        if any(kw in query_lower for kw in ["how to", "how do", "implement", "build"]):
            return "operational"
        
        # Conceptual (default)
        return "conceptual"
    
    def _extract_primary_goal(self, query: str) -> str:
        """Extract what user ultimately wants.
        
        Returns:
        - "understand_code": Wants to understand how code works
        - "learn_concept": Wants to learn a concept
        - "fix_problem": Wants to solve an error
        - "navigate_codebase": Wants to find code location
        - "get_examples": Wants concrete examples
        """
        query_lower = query.lower()
        
        # Fix problem
        if any(kw in query_lower for kw in ["error", "fix", "bug", "fail", "why"]):
            return "fix_problem"
        
        # Navigate codebase
        if any(kw in query_lower for kw in ["where", "find", "locate", "file"]):
            return "navigate_codebase"
        
        # Get examples
        if any(kw in query_lower for kw in ["example", "show", "demonstrate", "pattern"]):
            return "get_examples"
        
        # Learn concept
        if any(kw in query_lower for kw in ["explain", "what is", "learn", "architecture"]):
            return "learn_concept"
        
        # Understand code (default)
        return "understand_code"
    
    def _make_routing_decision(
        self,
        query_type: str,
        query_intent: Optional[str],
        primary_goal: str,
        conversation_context: Optional[str],
    ) -> Tuple[RoutingDecision, List[str], RetrievalPriority]:
        """Make routing decision based on query characteristics.
        
        Decision Matrix:
        
        Query Type × Primary Goal → Decision
        ─────────────────────────────────────
        exact_lookup × navigate_codebase → CODEBASE_ONLY (code-first)
        exact_lookup × understand_code → CODEBASE_ONLY (code-first)
        conceptual × learn_concept → KT_ONLY (docs-first)
        operational × understand_code → HYBRID (balanced)
        reference × get_examples → MULTI_SOURCE (both, then synthesize)
        fix_problem × * → AGENT_TOOL (use debugger)
        * × * (with context) → CONTEXT_AWARE (use conversation)
        """
        
        # Check if we have conversation context for context-aware routing
        if conversation_context:
            return (
                RoutingDecision.CONTEXT_AWARE,
                ["codebase", "documentation"],
                RetrievalPriority.SEMANTIC,
            )
        
        # Fix problem → Use specialized tool
        if primary_goal == "fix_problem":
            return (
                RoutingDecision.AGENT_TOOL,
                ["codebase", "documentation"],
                RetrievalPriority.CODE_FIRST,
            )
        
        # Navigate codebase → Code-first
        if primary_goal == "navigate_codebase":
            return (
                RoutingDecision.CODEBASE_ONLY,
                ["codebase"],
                RetrievalPriority.CODE_FIRST,
            )
        
        # Learn concept → Documentation-first
        if primary_goal == "learn_concept":
            return (
                RoutingDecision.KT_ONLY,
                ["documentation"],
                RetrievalPriority.DOCUMENTATION_FIRST,
            )
        
        # Get examples → Multi-source (diverse perspectives)
        if primary_goal == "get_examples":
            return (
                RoutingDecision.MULTI_SOURCE,
                ["codebase", "documentation"],
                RetrievalPriority.BALANCED,
            )
        
        # Exact lookup queries → Codebase only
        if query_type == "exact_lookup":
            return (
                RoutingDecision.CODEBASE_ONLY,
                ["codebase"],
                RetrievalPriority.CODE_FIRST,
            )
        
        # Default: Hybrid search
        return (
            RoutingDecision.HYBRID,
            ["codebase", "documentation"],
            RetrievalPriority.BALANCED,
        )
    
    def _select_tools(self, decision: RoutingDecision, query_type: str) -> List[str]:
        """Select tools to invoke based on routing decision.
        
        Available tools:
        - code_search: Query codebase
        - doc_search: Query KT documentation
        - analyze: Code analysis
        - execute: Execute code snippets
        """
        tools = []
        
        if not self.config.enable_tool_invocation:
            return tools
        
        available = set(self.config.available_tools)
        
        # Map decision to tools
        if decision == RoutingDecision.CODEBASE_ONLY:
            tools = ["code_search"]
        
        elif decision == RoutingDecision.KT_ONLY:
            tools = ["doc_search"]
        
        elif decision == RoutingDecision.HYBRID:
            tools = ["code_search", "doc_search"]
        
        elif decision == RoutingDecision.MULTI_SOURCE:
            tools = ["code_search", "doc_search", "analyze"]
        
        elif decision == RoutingDecision.AGENT_TOOL:
            tools = ["code_search", "doc_search", "analyze"]
        
        elif decision == RoutingDecision.CONTEXT_AWARE:
            tools = ["code_search", "doc_search"]
        
        # Filter to available tools
        return [t for t in tools if t in available]
    
    def _build_reasoning(
        self,
        query_type: str,
        primary_goal: str,
        decision: RoutingDecision,
        sources: List[str],
    ) -> str:
        """Build human-readable reasoning for the routing decision."""
        reasoning = f"Query type: {query_type.replace('_', ' ')}, "
        reasoning += f"Primary goal: {primary_goal.replace('_', ' ')}, "
        reasoning += f"Decision: {decision.value.replace('_', ' ')}, "
        reasoning += f"Querying: {', '.join(sources)}"
        return reasoning
    
    def _compute_confidence(self, decision: RoutingDecision, query_type: str) -> float:
        """Compute confidence in routing decision (0-1).
        
        Factors:
        - Exact queries → high confidence (0.95)
        - Ambiguous queries → medium confidence (0.70)
        - Complex queries → lower confidence (0.60)
        """
        # Exact lookup queries are always high confidence
        if query_type == "exact_lookup":
            return 0.95
        
        # Operational and reference queries are medium-high confidence
        if query_type in ["operational", "reference"]:
            return 0.85
        
        # Navigational queries are medium confidence
        if query_type == "navigational":
            return 0.75
        
        # Conceptual queries are medium-low confidence (can go multiple ways)
        return 0.70


class RoutingOrchestrator:
    """Orchestrates the routing process with caching and optimization.
    
    Features:
    - Routes queries efficiently
    - Caches routing decisions
    - Learns from user feedback
    - Optimizes routing over time
    """
    
    def __init__(self, router: Optional[AgenticRouter] = None):
        """Initialize orchestrator.
        
        Args:
            router: AgenticRouter instance
        """
        self.router = router or AgenticRouter()
        self.routing_cache: Dict[str, RoutingResult] = {}
        self.decision_history: List[Tuple[str, RoutingResult]] = []
        logger.info("Initialized RoutingOrchestrator")
    
    def route_with_cache(
        self,
        query: str,
        intent_analysis: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> RoutingResult:
        """Route query, using cache if available.
        
        Args:
            query: User query
            intent_analysis: Phase 2 intent analysis
            use_cache: Whether to use cached routing decisions
        
        Returns:
            RoutingResult
        """
        # Check cache
        if use_cache and query in self.routing_cache:
            logger.debug(f"Using cached routing for: {query[:50]}...")
            return self.routing_cache[query]
        
        # Route fresh
        result = self.router.route(query, intent_analysis)
        
        # Cache result
        self.routing_cache[query] = result
        self.decision_history.append((query, result))
        
        # Keep cache bounded
        if len(self.routing_cache) > 1000:
            # Remove oldest entries
            oldest_key = next(iter(self.routing_cache))
            del self.routing_cache[oldest_key]
        
        return result
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get statistics on routing decisions."""
        if not self.decision_history:
            return {}
        
        decisions = [result.decision for _, result in self.decision_history]
        decision_counts = {}
        for decision in decisions:
            decision_counts[decision.value] = decision_counts.get(decision.value, 0) + 1
        
        return {
            "total_routed": len(self.decision_history),
            "cache_size": len(self.routing_cache),
            "decisions": decision_counts,
            "avg_confidence": sum(r.confidence for _, r in self.decision_history) / len(self.decision_history),
        }
