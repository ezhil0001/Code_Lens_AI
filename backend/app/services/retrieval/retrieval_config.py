"""Phase 2: Retrieval Configuration and Constants.

Defines configuration for the hybrid retrieval engine including models,
parameters, and optimization settings.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import os


@dataclass
class RetrieverConfig:
    """Configuration for the retrieval engine."""
    
    # Vector search parameters
    vector_model: str = "sentence-transformers/all-mpnet-base-v2"
    vector_weight: float = 0.6  # Weight in hybrid search
    
    # BM25 lexical search parameters
    bm25_weight: float = 0.4  # Weight in hybrid search
    
    # Reranking parameters
    reranker_model: str = "BAAI/bge-reranker-v2-m3"  # BGE-Reranker v2 (multilingual, 128M params)
    use_flashrank: bool = True  # Use FlashRank if available (faster inference)
    
    # Query expansion
    enable_query_expansion: bool = True
    max_query_variations: int = 3  # Original + 2 variations
    
    # Reranking
    enable_reranking: bool = True
    candidates_k: int = 10  # Fetch 20 candidates before reranking
    final_k: int = 5  # Return top-5 after reranking
    
    # Performance parameters
    chunk_size: int = 500  # Size of chunks for retrieval
    timeout_seconds: float = 30.0  # Timeout for retrieval operations
    
    # ChromaDB parameters. The collection is NOT configurable here: retrieval
    # and ingestion must always address the same corpus, so both go through
    # IngestionService.get_chroma_collection() (CANONICAL_COLLECTION). A
    # separate name here previously pointed at a stale per-upload collection.
    chroma_persist_dir: str = "./chroma_db"
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        errors = []
        
        if not 0.0 <= self.vector_weight <= 1.0:
            errors.append(f"vector_weight must be 0.0-1.0, got {self.vector_weight}")
        
        if not 0.0 <= self.bm25_weight <= 1.0:
            errors.append(f"bm25_weight must be 0.0-1.0, got {self.bm25_weight}")
        
        if self.candidates_k < self.final_k:
            errors.append(f"candidates_k ({self.candidates_k}) must be >= final_k ({self.final_k})")
        
        if self.final_k < 1:
            errors.append(f"final_k must be >= 1, got {self.final_k}")
        
        if self.max_query_variations < 1:
            errors.append(f"max_query_variations must be >= 1, got {self.max_query_variations}")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "vector_model": self.vector_model,
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
            "reranker_model": self.reranker_model,
            "use_flashrank": self.use_flashrank,
            "enable_query_expansion": self.enable_query_expansion,
            "max_query_variations": self.max_query_variations,
            "enable_reranking": self.enable_reranking,
            "candidates_k": self.candidates_k,
            "final_k": self.final_k,
            "chunk_size": self.chunk_size,
            "timeout_seconds": self.timeout_seconds,
        }


# Preset configurations for different use cases

DEFAULT_CONFIG = RetrieverConfig()

# Development configuration: Balanced speed and quality
DEV_CONFIG = RetrieverConfig(
    vector_weight=0.5,
    bm25_weight=0.5,
    enable_query_expansion=True,
    enable_reranking=True,
    candidates_k=15,  # Fewer candidates for faster iteration
    final_k=3,
    use_flashrank=False,  # Use slower but more available CrossEncoder
)

# Production configuration: High quality, optimized for accuracy
PRODUCTION_CONFIG = RetrieverConfig(
    vector_weight=0.6,
    bm25_weight=0.4,
    enable_query_expansion=True,
    enable_reranking=True,
    use_flashrank=True,
    candidates_k=20,
    final_k=5,
    reranker_model="BAAI/bge-reranker-v2-m3",
)

# Fast configuration: Speed optimized (for real-time scenarios)
FAST_CONFIG = RetrieverConfig(
    vector_weight=0.7,
    bm25_weight=0.3,
    enable_query_expansion=False,  # Skip expansion
    enable_reranking=False,  # Skip reranking
    candidates_k=10,
    final_k=3,
)

# Comprehensive configuration: High recall, slower but thorough
COMPREHENSIVE_CONFIG = RetrieverConfig(
    vector_weight=0.5,
    bm25_weight=0.5,
    enable_query_expansion=True,
    enable_reranking=True,
    use_flashrank=True,
    candidates_k=30,  # More candidates for better coverage
    final_k=10,  # Return more results
    max_query_variations=5,
)


# Model recommendations based on use case

RERANKER_MODELS = {
    "bge-reranker-v2-m3": {
        "name": "BAAI/bge-reranker-v2-m3",
        "description": "Multilingual (128M params), best for production",
        "languages": "100+",
        "latency_ms": 50,
    },
    "bge-reranker-v2-m3-large": {
        "name": "BAAI/bge-reranker-v2-large",
        "description": "Large multilingual model (350M params), highest quality",
        "languages": "100+",
        "latency_ms": 150,
    },
    "bge-reranker-base": {
        "name": "BAAI/bge-reranker-base",
        "description": "Fast English-focused reranker",
        "languages": "English",
        "latency_ms": 30,
    },
}

EMBEDDING_MODELS = {
    "all-mpnet-base-v2": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "description": "Fast, high-quality, multilingual",
        "size_mb": 438,
        "latency_ms": 20,
    },
    "all-MiniLM-L6-v2": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Ultra-fast for query expansion",
        "size_mb": 90,
        "latency_ms": 5,
    },
    "bge-large-en-v1.5": {
        "name": "BAAI/bge-large-en-v1.5",
        "description": "High-quality English embeddings",
        "size_mb": 1344,
        "latency_ms": 50,
    },
}


def get_config(name: str) -> RetrieverConfig:
    """Get preset configuration by name.
    
    Args:
        name: Configuration name (default, dev, prod, fast, comprehensive)
        
    Returns:
        RetrieverConfig instance
        
    Raises:
        ValueError: If configuration name not found
    """
    configs = {
        "default": DEFAULT_CONFIG,
        "dev": DEV_CONFIG,
        "prod": PRODUCTION_CONFIG,
        "production": PRODUCTION_CONFIG,
        "fast": FAST_CONFIG,
        "comprehensive": COMPREHENSIVE_CONFIG,
    }
    
    if name not in configs:
        raise ValueError(
            f"Unknown configuration: {name}. "
            f"Available: {', '.join(configs.keys())}"
        )
    
    return configs[name]
