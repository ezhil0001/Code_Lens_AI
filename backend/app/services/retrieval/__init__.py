"""Phase 2: Retrieval Service Module.

High-performance hybrid retrieval engine with:
- Dual-path retrieval (vector + lexical)
- Query expansion for improved recall
- Reranking with BGE-Reranker or FlashRank
- Parent context assembly from PDR
"""

from .retriever_engine import (
    RetrieverEngine,
    HybridRetriever,
    RerankingEngine,
    QueryExpander,
    RetrievalResult,
)
from .retrieval_config import (
    RetrieverConfig,
    DEFAULT_CONFIG,
    DEV_CONFIG,
    PRODUCTION_CONFIG,
    FAST_CONFIG,
    COMPREHENSIVE_CONFIG,
    get_config,
    RERANKER_MODELS,
    EMBEDDING_MODELS,
)

__all__ = [
    # Engine components
    "RetrieverEngine",
    "HybridRetriever",
    "RerankingEngine",
    "QueryExpander",
    "RetrievalResult",
    # Configuration
    "RetrieverConfig",
    "DEFAULT_CONFIG",
    "DEV_CONFIG",
    "PRODUCTION_CONFIG",
    "FAST_CONFIG",
    "COMPREHENSIVE_CONFIG",
    "get_config",
    "RERANKER_MODELS",
    "EMBEDDING_MODELS",
]
