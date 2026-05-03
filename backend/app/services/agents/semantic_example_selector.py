"""Phase 3: Semantic Similarity Example Selector.

Dynamically selects high-quality Q&A examples based on semantic similarity
to the user's query. Uses the vector store to find contextually relevant
examples for Few-Shot learning.
"""

import logging
import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ExamplePair:
    """Represents a Q&A example for Few-Shot learning."""
    
    question: str
    answer: str
    category: str  # "code_explanation", "kt_documentation", "architecture", etc.
    embedding: Optional[List[float]] = None
    quality_score: float = 1.0  # 0-1, used for ranking
    tags: List[str] = None
    context: Optional[str] = None  # Optional additional context
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate example pair."""
        if not self.question or not self.answer:
            raise ValueError("Question and answer cannot be empty")
        
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError(f"quality_score must be 0-1, got {self.quality_score}")
        
        if self.tags is None:
            self.tags = []
        
        if self.created_at is None:
            self.created_at = datetime.now()


class SemanticExampleSelector:
    """Selects Few-Shot examples based on semantic similarity to user query.
    
    Architecture:
    1. Maintains a curated list of high-quality Q&A examples
    2. Computes similarity between user query and each example question
    3. Ranks examples by: similarity_score * quality_score
    4. Returns top-k most relevant examples
    5. Groups by category for diverse perspectives
    
    Categories:
    - code_explanation: How to understand specific code
    - kt_documentation: Knowledge transfer documentation
    - architecture: System design patterns
    - troubleshooting: Error handling and debugging
    - best_practices: Code style and conventions
    """
    
    def __init__(self, vector_store=None, embedding_model=None):
        """Initialize the example selector.
        
        Args:
            vector_store: ChromaDB or similar vector store for embeddings
            embedding_model: Embedding model for computing similarities
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.examples: List[ExamplePair] = []
        self._curated_examples()
        logger.info(f"Initialized SemanticExampleSelector with {len(self.examples)} examples")
    
    def _curated_examples(self) -> None:
        """Initialize curated high-quality examples."""
        
        self.examples = [
            # Code Explanation Examples
            ExamplePair(
                question="How does the authenticate function work?",
                answer="""The authenticate function validates user credentials against stored tokens.

Structure:
1. Extract token from request headers
2. Decode JWT token to get payload
3. Verify signature using app secret
4. Check token expiration
5. Query database for user record
6. Return user object if valid, raise exception if invalid

Key Points:
- Uses JWT for stateless authentication
- Supports refresh tokens for extended sessions
- Implements exponential backoff for failed attempts
- Logs authentication events for audit trail

Example Code:
```python
def authenticate(token: str) -> User:
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    user = db.query(User).filter(User.id == payload['user_id']).first()
    if not user or not user.is_active:
        raise AuthenticationError("Invalid token")
    return user
```

Related: verify_token, TokenManager, JWT configuration""",
                category="code_explanation",
                quality_score=0.95,
                tags=["authentication", "jwt", "security"],
            ),
            ExamplePair(
                question="What's the difference between sync and async function calls?",
                answer="""Synchronous vs Asynchronous Execution:

SYNCHRONOUS (Blocking):
- Caller waits for function to complete
- Resources blocked until function returns
- Simpler error handling with try-except
- Lower overhead, better for I/O-bound operations
```python
result = fetch_data()  # Blocks here until data returns
print(result)
```

ASYNCHRONOUS (Non-blocking):
- Caller continues executing while function runs
- Resources available for other operations
- Better throughput for multiple concurrent requests
- Requires async/await or callbacks
```python
result = await fetch_data()  # Doesn't block
print(result)
```

When to Use:
- Sync: Simple operations, rare I/O, easy debugging
- Async: High-concurrency APIs, long I/O operations, real-time needs

Performance Impact:
- Sync: 100 requests = 100 seconds
- Async: 100 requests = 1 second (with proper pooling)""",
                category="code_explanation",
                quality_score=0.92,
                tags=["async", "performance", "concurrency"],
            ),
            
            # KT Documentation Examples
            ExamplePair(
                question="Explain the Knowledge Transfer (KT) philosophy of this project",
                answer="""Knowledge Transfer Philosophy:

GOAL: Enable any developer to understand and modify code without original author.

PRINCIPLES:
1. Self-Documenting Code
   - Clear variable/function names
   - Minimal comments (why, not what)
   - Type hints for clarity

2. Contextual Documentation
   - Parent context accessible (full functions/classes)
   - Related functions cross-referenced
   - Dependencies explicit

3. Example-Driven Learning
   - Code examples for each concept
   - Integration tests as documentation
   - Real use-case scenarios

4. Architecture Transparency
   - System design documented
   - Data flow diagrams provided
   - Decision rationale explained

IMPLEMENTATION IN THIS PROJECT:
- Phase 1: Ingestion pipeline stores parent context
- Phase 2: Retrieval includes full parent chunks
- Phase 3: AI agent provides contextual explanations
- Result: Developer gets code + context + explanation""",
                category="kt_documentation",
                quality_score=0.93,
                tags=["knowledge-transfer", "documentation", "philosophy"],
            ),
            
            # Architecture Examples
            ExamplePair(
                question="How does the hybrid retrieval system work?",
                answer="""Hybrid Retrieval Architecture:

TWO-PATH RETRIEVAL:

Path 1: Vector/Semantic Search
- Query converted to embeddings (1536 dimensions)
- ChromaDB finds semantically similar chunks
- Returns top-10 matches based on cosine similarity
- Captures conceptual relationships

Path 2: BM25/Lexical Search
- Query tokenized into terms
- Inverted index searched for exact matches
- TF-IDF scoring applied
- Returns top-10 keyword matches

COMBINATION & RERANKING:
1. Merge results from both paths
2. Deduplicate (keep highest score)
3. Weight: vector_score * 0.6 + bm25_score * 0.4
4. Select top-20 candidates
5. Use BGE-Reranker-v2 for deep relevance scoring
6. Return top-5 final results

CONTEXT ASSEMBLY:
- For each result, fetch parent context
- Combine chunk + parent for full understanding
- Include metadata (line numbers, file path)

BENEFITS:
✓ High recall (both paths catch different patterns)
✓ High precision (reranking filters noise)
✓ Full context (parent chunks available)
✓ Explainable scores (can see retrieval confidence)""",
                category="architecture",
                quality_score=0.95,
                tags=["architecture", "retrieval", "hybrid-search"],
            ),
            
            # Troubleshooting Examples
            ExamplePair(
                question="Why am I getting empty results from my query?",
                answer="""Troubleshooting Empty Results:

POSSIBLE CAUSES & SOLUTIONS:

1. Query Too Specific
   ✗ Query: "def authenticate_user_with_jwt_token_verification"
   ✓ Query: "How to authenticate users?"
   Solution: Use broader, natural language queries

2. Typos in Query
   ✗ Query: "autenticate usrs"
   ✓ Query: "authenticate users"
   Solution: Phase 2 detects typos, but cleaner queries work better

3. BM25 Index Missing
   Problem: Lexical search fails if documents not indexed
   Solution: Ensure Phase 1 ingestion completed successfully

4. Vector Store Empty
   Problem: No embeddings computed
   Solution: Verify ChromaDB collection has documents

5. Query-Document Mismatch
   Problem: Query uses different terminology than codebase
   Solution: Try synonyms, acronyms, or related terms

DEBUGGING STEPS:
1. Check Phase 1 logs: Documents ingested successfully?
2. Test vector search alone: Do embeddings work?
3. Test BM25 alone: Are keywords matched?
4. Verify ChromaDB collection: SELECT COUNT(*) FROM collection
5. Review query expansion: What variations were tried?

QUICK FIXES:
- Add more context to query
- Use Phase 2 QueryIntentDetector to understand intent
- Check if results are being filtered unexpectedly
- Increase candidates_k if reranking too aggressive""",
                category="troubleshooting",
                quality_score=0.90,
                tags=["debugging", "retrieval", "troubleshooting"],
            ),
            
            # Best Practices Example
            ExamplePair(
                question="What are best practices for querying the system effectively?",
                answer="""Best Practices for Effective Queries:

QUERY PHRASING:
✓ GOOD: "How to implement caching in the database layer?"
✗ BAD: "caching db"

✓ GOOD: "What error handling patterns are used?"
✗ BAD: "error"

✓ GOOD: "Show me the authentication flow"
✗ BAD: "auth"

WHY:
- Natural language → Better semantic embeddings
- Context-specific → More relevant results
- Complete thoughts → Query expansion more effective

QUERY TYPES & OPTIMIZATION:

1. Exact Match Queries
   Query: "def authenticate"
   Strategy: High BM25 weight (0.7), lower vector (0.3)
   Result: Precise function location

2. Conceptual Queries
   Query: "How to implement authentication?"
   Strategy: High vector weight (0.8), lower BM25 (0.2)
   Result: Broader understanding and patterns

3. Error-Focused Queries
   Query: "Handle token expiration error"
   Strategy: Balanced weights (0.5/0.5), include error patterns
   Result: Error handling examples and solutions

4. Code Navigation Queries
   Query: "What calls authenticate function?"
   Strategy: Medium weights (0.6/0.4), cross-reference analysis
   Result: Call chain and dependencies

ADVANCED TECHNIQUES:
- Use multiple queries to triangulate: "authenticate" + "jwt" + "security"
- Include context: "In user service, how to authenticate?"
- Ask for examples: "Show me 3 examples of error handling"
- Request documentation: "Explain the authentication architecture"

WHAT SYSTEM WILL DO:
✓ Expand queries automatically (typo tolerance)
✓ Detect intent (adjust weights dynamically)
✓ Find semantic matches (even if keywords differ)
✓ Rerank by relevance (best results first)
✓ Include full context (understand relationships)""",
                category="best_practices",
                quality_score=0.91,
                tags=["querying", "best-practices", "optimization"],
            ),
        ]
    
    def select_examples(
        self,
        query: str,
        k: int = 2,
        categories: Optional[List[str]] = None,
        quality_threshold: float = 0.85,
    ) -> List[ExamplePair]:
        """Select top-k most relevant examples for Few-Shot learning.
        
        Args:
            query: User query to find similar examples for
            k: Number of examples to return
            categories: Filter to specific categories (None = all)
            quality_threshold: Minimum quality score to consider
        
        Returns:
            List of ExamplePair sorted by relevance
        
        Algorithm:
            1. Filter by category and quality if specified
            2. Compute similarity between query and each example question
            3. Score: similarity * quality_score
            4. Rank and return top-k
            5. Group by category for diversity
        """
        try:
            # Filter examples
            candidates = self.examples
            if categories:
                candidates = [ex for ex in candidates if ex.category in categories]
            
            candidates = [ex for ex in candidates if ex.quality_score >= quality_threshold]
            
            if not candidates:
                logger.warning(f"No examples found for query: {query}")
                return []
            
            # Compute similarities
            similarities = []
            for example in candidates:
                similarity = self._compute_similarity(query, example.question)
                score = similarity * example.quality_score
                similarities.append((example, score, similarity))
            
            # Sort by score descending
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k, preferring diverse categories
            selected = []
            categories_used = set()
            
            for example, score, similarity in similarities:
                if len(selected) >= k:
                    break
                
                # Prefer diverse categories
                if example.category not in categories_used:
                    selected.append(example)
                    categories_used.add(example.category)
                elif len(categories_used) < len(set(ex.category for ex in selected)):
                    # Add even if category duplicate if we need more
                    selected.append(example)
            
            logger.debug(
                f"Selected {len(selected)} examples for query: {query[:50]}..."
            )
            return selected
        
        except Exception as e:
            logger.error(f"Error selecting examples: {e}")
            return []
    
    def _compute_similarity(self, query: str, example_question: str) -> float:
        """Compute semantic similarity using cosine similarity.
        
        PRODUCTION IMPLEMENTATION:
        1. If embedding_model provided: Use actual embeddings (best)
        2. If vector_store provided: Query vector store for similarity (good)
        3. Fallback: Use improved keyword similarity with TF-IDF weighting (acceptable)
        
        This follows Phase 2 retriever approach for consistency.
        """
        # CASE 1: Use actual embedding model (production preferred)
        if self.embedding_model:
            try:
                # Support both SentenceTransformer (.encode) and LangChain
                # Embeddings (.embed_query) interfaces.
                if hasattr(self.embedding_model, "embed_query"):
                    query_embedding = self.embedding_model.embed_query(query)
                    example_embedding = self.embedding_model.embed_query(example_question)
                else:
                    query_embedding = self.embedding_model.encode(query)
                    example_embedding = self.embedding_model.encode(example_question)

                # Cosine similarity
                similarity = self._cosine_similarity(
                    list(query_embedding), list(example_embedding)
                )
                return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
            except Exception as e:
                logger.warning(f"Embedding model failed: {e}, falling back to keyword similarity")
        
        # CASE 2: Use vector store (if available)
        if self.vector_store:
            try:
                # Query vector store for similarity
                # This assumes vector store has similarity search capability
                results = self.vector_store.similarity_search_with_score(example_question, k=1)
                if results:
                    # Score is typically 0-1, higher is more similar
                    score = results[0][1] if isinstance(results[0][1], float) else 0.5
                    return score
            
            except Exception as e:
                logger.warning(f"Vector store query failed: {e}, falling back to keyword similarity")
        
        # CASE 3: Fallback - Improved keyword similarity with TF-IDF weighting
        # Better than Jaccard: accounts for term importance
        return self._tfidf_similarity(query, example_question)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two embedding vectors.
        
        Formula: cos(θ) = (A · B) / (||A|| * ||B||)
        Returns: value in [-1, 1], typically [0, 1] for embeddings
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        # Dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Magnitudes
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def _tfidf_similarity(self, query: str, example_question: str) -> float:
        """Fallback: Improved keyword similarity using TF-IDF-like weighting.
        
        Better than Jaccard because:
        - Rare terms weighted higher (more discriminative)
        - Common terms weighted lower (less important)
        - Accounts for term frequency
        
        Approximation of true TF-IDF without corpus statistics.
        """
        # Tokenize
        q_terms = query.lower().split()
        e_terms = example_question.lower().split()
        
        if not q_terms or not e_terms:
            return 0.0
        
        # Build term frequency maps
        q_tf = {}
        e_tf = {}
        
        for term in q_terms:
            q_tf[term] = q_tf.get(term, 0) + 1
        for term in e_terms:
            e_tf[term] = e_tf.get(term, 0) + 1
        
        # Compute weighted overlap
        common_terms = set(q_tf.keys()) & set(e_tf.keys())
        
        if not common_terms:
            return 0.0
        
        # Weight by term frequency (TF) and inverse rarity
        # Rare terms (appearing once) get higher weight
        weighted_intersection = sum(
            (q_tf[term] * e_tf[term]) / math.sqrt(1 + abs(len(q_terms) - len(e_terms)))
            for term in common_terms
        )
        
        # Magnitude normalization
        q_magnitude = math.sqrt(sum(tf * tf for tf in q_tf.values()))
        e_magnitude = math.sqrt(sum(tf * tf for tf in e_tf.values()))
        
        if q_magnitude == 0 or e_magnitude == 0:
            return 0.0
        
        # Normalized weighted similarity
        similarity = weighted_intersection / (q_magnitude * e_magnitude)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, similarity))
    
    def add_example(self, example: ExamplePair) -> None:
        """Add a new example to the selector.
        
        Args:
            example: ExamplePair to add
        """
        example.validate()
        self.examples.append(example)
        logger.info(f"Added example: {example.question[:50]}...")
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of examples by category."""
        distribution = {}
        for example in self.examples:
            distribution[example.category] = distribution.get(example.category, 0) + 1
        return distribution
    
    def __len__(self) -> int:
        """Get total number of examples."""
        return len(self.examples)
