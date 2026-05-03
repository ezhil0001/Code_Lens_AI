"""Phase 2: High-Performance Hybrid Retrieval Engine with Reranking.

This module implements a dual-path retrieval system combining:
1. Vector-based retrieval (ChromaDB + semantic embeddings)
2. Lexical/keyword retrieval (BM25)
3. Query expansion for improved recall
4. Reranking with BGE-Reranker for top-K selection
5. Parent context assembly from PDR
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

# FIX #3: OpenTelemetry instrumentation
try:
    from opentelemetry import trace
    HAS_OTEL = True
    tracer = trace.get_tracer(__name__)
except ImportError:
    HAS_OTEL = False
    tracer = None

try:
    # langchain >= 1.0 moved EnsembleRetriever to langchain_classic;
    # fall back to the legacy path for older installs.
    try:
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:
        from langchain.retrievers import EnsembleRetriever  # type: ignore
    from langchain_community.retrievers import BM25Retriever
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    try:
        from langchain_core.documents import Document
    except ImportError:
        from langchain.schema import Document  # type: ignore
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
except ImportError as e:
    raise ImportError(
        f"LangChain dependencies required: {e}. "
        "Install: pip install langchain langchain-classic langchain-community "
        "langchain-chroma langchain-huggingface rank_bm25"
    )

try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False

# Import query intent detector for dynamic weight tuning
try:
    from app.services.retrieval.query_intent_detector import QueryIntentDetector, AdaptiveWeightStrategy, IntentAnalysis
    HAS_INTENT_DETECTOR = True
except ImportError:
    HAS_INTENT_DETECTOR = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("QueryIntentDetector not available. Dynamic weight tuning disabled.")

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval engine with all necessary context."""
    
    query: str
    chunks: List[Dict[str, Any]]  # Retrieved chunks
    parent_contexts: Dict[str, str]  # chunk_id -> parent content mapping
    expanded_queries: List[str]  # Queries used for expansion
    retrieval_time_ms: float
    reranking_time_ms: float
    total_time_ms: float
    candidates_before_rerank: int
    final_count: int
    top_k_scores: List[float]  # Relevance scores from reranker
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "chunks": self.chunks,
            "parent_contexts": self.parent_contexts,
            "expanded_queries": self.expanded_queries,
            "retrieval_time_ms": self.retrieval_time_ms,
            "reranking_time_ms": self.reranking_time_ms,
            "total_time_ms": self.total_time_ms,
            "candidates_before_rerank": self.candidates_before_rerank,
            "final_count": self.final_count,
            "top_k_scores": self.top_k_scores,
            "metadata": self.metadata or {},
        }


class QueryExpander:
    """Generate multiple query variations for improved recall."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """Initialize query expander.
        
        Args:
            model_name: SentenceTransformer model for semantic similarity
        """
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            logger.warning("SentenceTransformer not available for query expansion")
            self.model = None
    
    def expand_query(
        self,
        query: str,
        max_variations: int = 3,
    ) -> List[str]:
        """Expand query into semantic variations.
        
        Strategy:
        1. Original query
        2. Keyword-focused variation (most important terms)
        3. Concept-focused variation (high-level concepts)
        4. Technical variation (domain-specific terms)
        
        Args:
            query: User's original query
            max_variations: Number of variations to generate
            
        Returns:
            List of query variations including original
        """
        variations = [query]  # Always include original
        
        try:
            # Strategy 1: Remove stop words, keep main concepts
            keywords = self._extract_keywords(query)
            if keywords and len(variations) < max_variations:
                keyword_query = " ".join(keywords)
                if keyword_query != query:
                    variations.append(keyword_query)
                    logger.debug(f"Keyword variation: {keyword_query}")
            
            # Strategy 2: Synonym/concept expansion (simple version)
            if len(variations) < max_variations:
                concept_query = self._expand_concepts(query)
                if concept_query != query and concept_query not in variations:
                    variations.append(concept_query)
                    logger.debug(f"Concept variation: {concept_query}")
            
            # Strategy 3: Technical term expansion
            if len(variations) < max_variations:
                technical_query = self._add_technical_context(query)
                if technical_query != query and technical_query not in variations:
                    variations.append(technical_query)
                    logger.debug(f"Technical variation: {technical_query}")
        
        except Exception as e:
            logger.warning(f"Error in query expansion: {str(e)}")
            # Return at least original query
            return variations[:max_variations]
        
        return variations[:max_variations]
    
    @staticmethod
    def _extract_keywords(query: str) -> List[str]:
        """Extract main keywords from query by removing common stop words."""
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can", "what",
            "how", "where", "when", "why", "which", "who", "whom"
        }
        
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords
    
    @staticmethod
    def _expand_concepts(query: str) -> str:
        """Expand with related concepts."""
        # Simple concept mapping
        concept_map = {
            "auth": "authentication authorization security",
            "error": "exception error handling debugging",
            "config": "configuration settings parameters",
            "database": "db sql query data storage",
            "api": "endpoint request response http",
            "cache": "caching memory storage performance",
            "queue": "messaging async job queue",
        }
        
        query_lower = query.lower()
        for concept, expansion in concept_map.items():
            if concept in query_lower:
                return f"{query} {expansion}"
        
        return query
    
    @staticmethod
    def _add_technical_context(query: str) -> str:
        """Add technical context to query."""
        # Add domain hints if not present
        technical_context = {
            "function": "method implementation code",
            "class": "class definition structure interface",
            "test": "unit test testing specification",
            "schema": "data structure table fields columns",
            "migration": "database migration schema change alter",
        }
        
        query_lower = query.lower()
        for term, context in technical_context.items():
            if term in query_lower:
                return f"{query} context: {context}"
        
        return query


class RerankingEngine:
    """Rerank retrieval results using BGE-Reranker-v2 via CrossEncoder.

    Production-grade: loads `BAAI/bge-reranker-v2-m3` and reranks
    top-N candidates down to top-K with true cross-encoder relevance.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
    ):
        """Initialize reranking engine.

        Args:
            model_name: HuggingFace cross-encoder model name.
            device: 'cuda', 'mps', or 'cpu'. Auto-detected if None.
        """
        if not HAS_CROSS_ENCODER:
            raise ImportError(
                "sentence-transformers is required for BGE reranking. "
                "Install: pip install sentence-transformers"
            )

        self.model_name = model_name
        # MPS (Apple Silicon) has known issues with CrossEncoder batch inference.
        # Force CPU for reranker — it's actually faster than MPS for this workload.
        if device is None:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"  # MPS intentionally skipped for CrossEncoder
        self.device = device
        try:
            self.cross_encoder = CrossEncoder(model_name, device=device, max_length=512)
            logger.info(f"✅ Loaded BGE Reranker: {model_name} on device={device}")
        except Exception as e:
            logger.error(f"Failed to load BGE reranker '{model_name}': {e}")
            raise RuntimeError(
                f"Could not initialize BGE reranker. Verify the model name "
                f"and that sentence-transformers can fetch it."
            ) from e

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Rerank documents by true cross-encoder relevance.

        Args:
            query: Search query.
            documents: Candidate documents (typically top-20 from hybrid retrieval).
            top_k: Number of top results to return (typically 5).

        Returns:
            Tuple of (top_k_documents, relevance_scores).
        """
        if not documents:
            return [], []

        try:
            pairs = [[query, doc.get("content", "")] for doc in documents]
            scores = self.cross_encoder.predict(
    pairs,
    convert_to_numpy=True,
    show_progress_bar=False,
    batch_size=32,   # CPU-la 32 optimal — MPS overhead illa
)

            scored = list(zip(documents, scores.tolist()))
            scored.sort(key=lambda x: x[1], reverse=True)

            top = scored[:top_k]
            reranked_docs = [d for d, _ in top]
            reranked_scores = [float(s) for _, s in top]

            logger.info(
                f"BGE reranked {len(documents)} → top-{len(reranked_docs)} "
                f"(top score: {reranked_scores[0]:.3f})"
            )
            return reranked_docs, reranked_scores

        except Exception as e:
            logger.error(f"BGE reranking failed: {e}", exc_info=True)
            # G6 FIX — Fail-soft: return original retrieval order AND preserve
            # each candidate's original retrieval score (vector-cosine / RRF /
            # ensemble score) instead of zeroing them out. Downstream
            # `confidence_score` and observability dashboards stay accurate
            # during a reranker outage; LLM input is unchanged either way.
            fallback_docs = documents[:top_k]
            fallback_scores = [
                float(d.get("score", 0.0)) for d in fallback_docs
            ]
            logger.warning(
                f"Reranker fail-soft: returning {len(fallback_docs)} candidates "
                f"with original scores (top: {fallback_scores[0] if fallback_scores else 0.0:.3f})"
            )
            return fallback_docs, fallback_scores


class _ChromaCollectionRetriever(BaseRetriever):
    """Thin LangChain BaseRetriever wrapper around a raw chromadb Collection.

    Embeds the query with the same HuggingFace model used at ingestion and
    queries the collection directly — gives EnsembleRetriever a uniform
    interface alongside BM25Retriever.
    """

    collection: Any
    embeddings: Any
    k: int = 20
    metadata_filter: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        try:
            query_embedding = self.embeddings.embed_query(query)
            query_kwargs: Dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": self.k,
                "include": ["documents", "metadatas", "distances"],
            }
            # Apply Chroma metadata filter (P0 fix #2: enforce routing decision)
            if self.metadata_filter:
                query_kwargs["where"] = self.metadata_filter
            results = self.collection.query(**query_kwargs)
            documents_text = (results.get("documents") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]

            docs: List[Document] = []
            for text, meta, dist in zip(documents_text, metadatas, distances):
                meta = dict(meta or {})
                meta["score"] = float(1.0 - dist)
                meta["retrieval_method"] = "vector"
                docs.append(Document(page_content=text or "", metadata=meta))
            return docs
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return []


class HybridRetriever:
    """Hybrid retriever combining ChromaDB (dense) + BM25 (lexical)
    via LangChain's EnsembleRetriever (Reciprocal Rank Fusion).

    Features:
      - Idiomatic LangChain `EnsembleRetriever` with weighted RRF
      - Optional dynamic weight adjustment via QueryIntentDetector
      - Parent context lookup for PDR
    """

    def __init__(
        self,
        chroma_collection,
        documents_for_bm25: Optional[List[Document]] = None,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        parent_store: Optional[Dict[str, str]] = None,
        enable_dynamic_weights: bool = True,
        candidate_k: int = 20,
    ):
        """Initialize hybrid retriever.

        Args:
            chroma_collection: ChromaDB collection.
            documents_for_bm25: LangChain Documents for BM25 indexing.
            embedding_model: HuggingFace embedding model used at ingestion.
            vector_weight: Initial weight for vector retriever (0-1).
            bm25_weight: Initial weight for BM25 retriever (0-1).
            parent_store: Map chunk/parent_id -> parent content.
            enable_dynamic_weights: Adjust weights via QueryIntentDetector.
            candidate_k: Candidates to fetch from each retriever.
        """
        self.chroma_collection = chroma_collection
        self.parent_store = parent_store or {}
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.candidate_k = candidate_k

        # L1 FIX: serialize per-call mutation of the shared `_ChromaCollectionRetriever.metadata_filter`.
        # `HybridRetriever` is a process-wide singleton (built by pipeline_factory)
        # so concurrent FastAPI handlers must NOT race on the filter attribute.
        self._filter_lock = threading.Lock()

        # L2 FIX: reuse the process-wide singleton embedder instead of loading a
        # second 500 MB model copy. Falls back to a per-instance load if the
        # singleton helper is unavailable (e.g. during isolated unit tests).
        try:
            from app.core.database import get_embedder
            self.embeddings = get_embedder()
        except Exception as _e:  # pragma: no cover
            logger.warning(
                f"Singleton embedder unavailable ({_e}); falling back to local instance."
            )
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_retriever = _ChromaCollectionRetriever(
            collection=chroma_collection,
            embeddings=self.embeddings,
            k=candidate_k,
        )

        if documents_for_bm25:
            self.bm25_retriever = BM25Retriever.from_documents(documents_for_bm25)
            self.bm25_retriever.k = candidate_k
            logger.info(
                f"BM25Retriever initialized with {len(documents_for_bm25)} documents"
            )
        else:
            self.bm25_retriever = None
            logger.warning("No BM25 documents provided — using vector-only retrieval")

        # Build EnsembleRetriever (RRF fusion)
        if self.bm25_retriever is not None:
            self.ensemble = EnsembleRetriever(
                retrievers=[self.vector_retriever, self.bm25_retriever],
                weights=[vector_weight, bm25_weight],
            )
            logger.info(
                f"✅ EnsembleRetriever ready (vector={vector_weight}, bm25={bm25_weight})"
            )
        else:
            self.ensemble = self.vector_retriever
            logger.info("✅ Vector-only retrieval (no BM25 corpus)")

        # Dynamic weight tuning
        self.enable_dynamic_weights = enable_dynamic_weights and HAS_INTENT_DETECTOR
        if self.enable_dynamic_weights:
            self.intent_detector = QueryIntentDetector()
            self.adaptive_strategy = AdaptiveWeightStrategy(enable_adaptive=True)
            logger.info("Dynamic weight tuning enabled")
        else:
            self.intent_detector = None
            self.adaptive_strategy = None

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        results, _ = self.retrieve_with_intent(query, top_k, metadata_filter=metadata_filter)
        return results

    def retrieve_with_intent(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Run hybrid retrieval; optionally retune weights per-query."""
        if HAS_OTEL and tracer:
            with tracer.start_as_current_span("retriever.retrieve_with_intent") as span:
                span.set_attribute("query", query[:100])
                span.set_attribute("top_k", top_k)
                if metadata_filter:
                    span.set_attribute("metadata_filter", str(metadata_filter))
                return self._retrieve_impl(query, top_k, span, metadata_filter)
        return self._retrieve_impl(query, top_k, None, metadata_filter)

    def _retrieve_impl(
        self,
        query: str,
        top_k: int,
        span: Optional[Any],
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        intent_dict: Optional[Dict[str, Any]] = None
        ensemble = self.ensemble

        # L1 FIX: serialize the mutate-use-restore region on `vector_retriever.metadata_filter`.
        # Without this lock, concurrent requests on the singleton retriever leak filters
        # across each other (HYBRID query receiving KT-only results, etc.).
        with self._filter_lock:
            previous_filter = getattr(self.vector_retriever, "metadata_filter", None)
            if metadata_filter:
                try:
                    self.vector_retriever.metadata_filter = metadata_filter
                except Exception:
                    pass

            # Optional dynamic weight tuning — rebuild EnsembleRetriever if needed
            if self.enable_dynamic_weights and self.bm25_retriever is not None:
                try:
                    intent_analysis: IntentAnalysis = self.intent_detector.detect_intent(
                        query, verbose=False
                    )
                    v_w = intent_analysis.vector_weight
                    b_w = intent_analysis.bm25_weight
                    if abs(v_w - self.vector_weight) > 0.05 or abs(
                        b_w - self.bm25_weight
                    ) > 0.05:
                        ensemble = EnsembleRetriever(
                            retrievers=[self.vector_retriever, self.bm25_retriever],
                            weights=[v_w, b_w],
                        )
                        logger.info(
                            f"Adapted weights for intent={intent_analysis.intent_type.value}: "
                            f"vector={v_w:.2f}, bm25={b_w:.2f}"
                        )
                    intent_dict = {
                        "intent_type": intent_analysis.intent_type.value,
                        "confidence": getattr(intent_analysis, "confidence", 0.8),
                        "vector_weight": v_w,
                        "bm25_weight": b_w,
                    }
                    if span:
                        span.set_attribute("intent_type", intent_analysis.intent_type.value)
                except Exception as e:
                    logger.warning(f"Intent detection failed: {e}")

            # Run ensemble (RRF inside LangChain) — INSIDE the lock so the filter
            # cannot be mutated by another request while Chroma is querying.
            try:
                docs: List[Document] = ensemble.invoke(query)
            except Exception as e:
                logger.error(f"EnsembleRetriever failed: {e}", exc_info=True)
                # restore prior filter before releasing the lock
                try:
                    self.vector_retriever.metadata_filter = previous_filter
                except Exception:
                    pass
                return [], intent_dict

            # Restore prior filter inside the lock so the next waiter sees a clean slate.
            try:
                self.vector_retriever.metadata_filter = previous_filter
            except Exception:
                pass

        # ---- end critical section ----

        # Post-filter BM25 hits (BM25Retriever has no native metadata filter).
        # Safe to do outside the lock: `docs` is a local list, no shared state.
        if metadata_filter:
            def _matches(meta: Dict[str, Any]) -> bool:
                meta = meta or {}
                for k, v in metadata_filter.items():
                    if isinstance(v, dict) and "$eq" in v:
                        if meta.get(k) != v["$eq"]:
                            return False
                    else:
                        if meta.get(k) != v:
                            return False
                return True
            docs = [d for d in docs if _matches(d.metadata or {})]

        results: List[Dict[str, Any]] = []
        for d in docs[:top_k]:
            results.append(
                {
                    "content": d.page_content,
                    "metadata": d.metadata or {},
                    "score": float((d.metadata or {}).get("score", 0.0)),
                    "retrieval_method": (d.metadata or {}).get(
                        "retrieval_method", "ensemble"
                    ),
                }
            )

        if span:
            span.set_attribute("final_results_count", len(results))
        logger.info(
            f"Hybrid retrieval returned {len(results)} candidates for: {query[:60]}"
        )
        return results, intent_dict

    @staticmethod
    def _deduplicate_results(
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen: Dict[str, Dict[str, Any]] = {}
        for r in results:
            key = r.get("content", "")
            if not key:
                continue
            if key not in seen or r.get("score", 0) > seen[key].get("score", 0):
                seen[key] = r
        return list(seen.values())


class RetrieverEngine:
    """End-to-end retrieval engine with query expansion, hybrid search, and reranking."""
    
    def __init__(
        self,
        chroma_collection,
        documents_for_bm25: Optional[List[Document]] = None,
        parent_store: Optional[Dict[str, str]] = None,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        enable_query_expansion: bool = True,
        enable_reranking: bool = True,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ):
        """Initialize end-to-end retrieval engine.
        
        Args:
            chroma_collection: ChromaDB collection for vector search
            documents_for_bm25: LangChain Document objects for BM25
            parent_store: Dict mapping chunk_id -> parent context
            embedding_model: Model for query expansion
            reranker_model: Model for reranking
            enable_query_expansion: Enable query expansion
            enable_reranking: Enable reranking
            vector_weight: Weight for vector results
            bm25_weight: Weight for BM25 results
        """
        self.chroma_collection = chroma_collection
        self.parent_store = parent_store or {}
        self.enable_query_expansion = enable_query_expansion
        self.enable_reranking = enable_reranking
        
        # Initialize components
        self.query_expander = QueryExpander(embedding_model) if enable_query_expansion else None
        self.reranking_engine = RerankingEngine(reranker_model) if enable_reranking else None
        self.hybrid_retriever = HybridRetriever(
            chroma_collection=chroma_collection,
            documents_for_bm25=documents_for_bm25,
            embedding_model=embedding_model,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            parent_store=parent_store,
        )
        
        logger.info("RetrieverEngine initialized")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidates_k: int = 20,
        expand_queries: bool = True,
        use_dynamic_weights: bool = True,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """End-to-end retrieval with query expansion, hybrid search, and reranking.

        Args:
            query: User's search query
            top_k: Final number of results to return
            candidates_k: Number of candidates to retrieve before reranking
            expand_queries: Enable query expansion for this request
            use_dynamic_weights: Toggle adaptive vector/BM25 weighting per-query
                (P0 FIX #1 — was previously rejected as unknown kwarg).
            metadata_filter: Optional Chroma `where=` filter (P0 FIX #2 —
                routing decisions become real metadata constraints).

        Returns:
            RetrievalResult with chunks, parent contexts, and metadata
        """
        start_time = time.time()

        # P0 FIX #1: honor caller-controlled dynamic weights for this invocation
        previous_dynamic = getattr(self.hybrid_retriever, "enable_dynamic_weights", False)
        self.hybrid_retriever.enable_dynamic_weights = bool(
            use_dynamic_weights and HAS_INTENT_DETECTOR and self.hybrid_retriever.bm25_retriever is not None
        )

        try:
            # Step 1: Query Expansion
            expanded_queries = [query]
            if expand_queries and self.query_expander:
                expanded_queries = self.query_expander.expand_query(query, max_variations=3)
                logger.info(f"Query expanded to {len(expanded_queries)} variations")

            # Step 2: Hybrid Retrieval (parallel for expanded queries)
            retrieval_start = time.time()
            candidates = []

            for expanded_q in expanded_queries:
                results = self.hybrid_retriever.retrieve(
                    expanded_q,
                    top_k=candidates_k,
                    metadata_filter=metadata_filter,
                )
                candidates.extend(results)
            
            # Deduplicate and sort by score
            candidates = self.hybrid_retriever._deduplicate_results(candidates)
            candidates.sort(key=lambda x: x["score"], reverse=True)
            candidates = candidates[:candidates_k]
            
            retrieval_time = (time.time() - retrieval_start) * 1000
            logger.info(f"Retrieved {len(candidates)} candidates in {retrieval_time:.2f}ms")
            
            # Step 3: Reranking
            reranking_start = time.time()
            top_chunks = candidates
            rerank_scores = [c.get("score", 0.0) for c in candidates[:top_k]]
            
            if self.enable_reranking and self.reranking_engine and len(candidates) > 0:
                top_chunks, rerank_scores = self.reranking_engine.rerank(
                    query=query,
                    documents=candidates,
                    top_k=top_k,
                )
                logger.info(f"Reranked to top-{len(top_chunks)}")
            else:
                top_chunks = candidates[:top_k]
            
            reranking_time = (time.time() - reranking_start) * 1000
            
            # Step 4: Assemble final results with parent context
            final_chunks = []
            parent_contexts = {}
            
            for chunk in top_chunks:
                chunk_id = chunk.get("metadata", {}).get("chunk_id", "")
                parent_id = chunk.get("metadata", {}).get("parent_id", "")
                
                # Get parent context if available
                parent_context = None
                if parent_id and parent_id in self.parent_store:
                    parent_context = self.parent_store[parent_id]
                    parent_contexts[chunk_id] = parent_context
                
                final_chunks.append({
                    "chunk_id": chunk_id,
                    "content": chunk.get("content", ""),
                    "metadata": chunk.get("metadata", {}),
                    "parent_id": parent_id,
                    "parent_context": parent_context,
                    "score": chunk.get("score", 0.0),
                })
            
            total_time = (time.time() - start_time) * 1000
            
            # Build result
            result = RetrievalResult(
                query=query,
                chunks=final_chunks,
                parent_contexts=parent_contexts,
                expanded_queries=expanded_queries,
                retrieval_time_ms=retrieval_time,
                reranking_time_ms=reranking_time,
                total_time_ms=total_time,
                candidates_before_rerank=len(candidates),
                final_count=len(final_chunks),
                top_k_scores=rerank_scores,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "query_expansion_enabled": self.enable_query_expansion,
                    "reranking_enabled": self.enable_reranking,
                },
            )
            
            logger.info(
                f"Retrieval complete: {result.candidates_before_rerank} candidates → "
                f"{result.final_count} final results in {total_time:.2f}ms"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error in retrieval: {e}", exc_info=True)
            # Return empty result
            return RetrievalResult(
                query=query,
                chunks=[],
                parent_contexts={},
                expanded_queries=[query],
                retrieval_time_ms=0.0,
                reranking_time_ms=0.0,
                total_time_ms=(time.time() - start_time) * 1000,
                candidates_before_rerank=0,
                final_count=0,
                top_k_scores=[],
                metadata={"error": str(e)},
            )
        finally:
            # Restore prior dynamic-weight state so per-call toggles do not leak.
            try:
                self.hybrid_retriever.enable_dynamic_weights = previous_dynamic
            except Exception:
                pass
