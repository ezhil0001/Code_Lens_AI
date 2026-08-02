"""RAG Evaluation Framework — measures answer quality using Ragas metrics.

Evaluates the RAG pipeline on four dimensions every time a response is
generated. Results are stored in SQLite and streamed to Langfuse so
quality regressions show up in the Langfuse dashboard before users notice them.

1. **Faithfulness**: Does the answer use only the provided context?
2. **Context Recall**: What percentage of ground truth is recalled?
3. **Answer Relevancy**: How relevant is the answer to the query?

Uses local Ollama LLM as the judge (no external API calls).
Stores results in SQLite for performance tracking over time.
"""

import logging
import time
import sqlite3
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import csv
import os

logger = logging.getLogger(__name__)

# ==================== Ragas Integration ====================

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, context_recall, answer_relevancy
    from datasets import Dataset
    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False
    logger.warning("Ragas not installed. Install with: pip install ragas datasets")

_EVAL_PROVIDER = os.getenv("EVAL_LLM_PROVIDER", "groq").lower()
HAS_OLLAMA = False
_USE_GROQ_EVAL = False

if _EVAL_PROVIDER == "ollama":
    try:
        from langchain_ollama import ChatOllama, OllamaLLM as Ollama
        HAS_OLLAMA = True
    except ImportError:
        logger.warning("langchain_ollama not installed — pip install langchain-ollama")

elif _EVAL_PROVIDER == "groq":
    try:
        from langchain_groq import ChatGroq
        _USE_GROQ_EVAL = True
        HAS_OLLAMA = True
    except ImportError:
        logger.warning("langchain_groq not installed")


# ==================== Data Models ====================

@dataclass
class EvaluationSample:
    """Single RAG evaluation sample."""
    
    query: str
    ground_truth: str  # Expected answer
    retrieved_context: List[str]  # Retrieved documents
    answer: str  # Generated answer
    
    # Optional metadata
    session_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    source: Optional[str] = None  # Which query type (code, docs, etc.)
    trace_id: Optional[str] = None  # Langfuse trace to attach scores to
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "ground_truth": self.ground_truth,
            "retrieved_context": self.retrieved_context,
            "answer": self.answer,
            "session_id": self.session_id or "unknown",
            "timestamp": (self.timestamp or datetime.now()).isoformat(),
            "source": self.source or "general",
        }


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a single sample."""
    
    faithfulness: float = 0.0  # 0-1, higher is better
    context_recall: float = 0.0  # 0-1, higher is better
    answer_relevancy: float = 0.0  # 0-1, higher is better
    
    # Aggregated score
    aggregate_score: float = field(init=False)
    
    # Metadata
    evaluation_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    model_used: str = "unknown"
    
    def __post_init__(self):
        """Calculate aggregate score."""
        scores = [self.faithfulness, self.context_recall, self.answer_relevancy]
        self.aggregate_score = sum(scores) / len(scores) if scores else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "faithfulness": self.faithfulness,
            "context_recall": self.context_recall,
            "answer_relevancy": self.answer_relevancy,
            "aggregate_score": self.aggregate_score,
            "evaluation_time_ms": self.evaluation_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "model_used": self.model_used,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    
    sample: EvaluationSample
    metrics: EvaluationMetrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **self.sample.to_dict(),
            **self.metrics.to_dict(),
        }


# ==================== SQLite Storage ====================

class EvaluationDatabase:
    """SQLite database for storing evaluation results."""
    
    def __init__(self, db_path: str = "evaluation_results.db"):
        """Initialize database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    
                    -- Sample data
                    query TEXT NOT NULL,
                    ground_truth TEXT,
                    retrieved_context TEXT,  -- JSON array
                    answer TEXT,
                    
                    -- Metrics
                    faithfulness REAL,
                    context_recall REAL,
                    answer_relevancy REAL,
                    aggregate_score REAL,
                    
                    -- Metadata
                    session_id TEXT,
                    source TEXT,
                    evaluation_time_ms REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_used TEXT,
                    
                    -- Indexing
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices for performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id 
                ON evaluations(session_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source 
                ON evaluations(source)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON evaluations(timestamp)
            """)
            
            conn.commit()
            logger.info(f"✓ Evaluation database initialized: {self.db_path}")
    
    def store_result(self, result: EvaluationResult) -> int:
        """Store evaluation result in database.
        
        Args:
            result: EvaluationResult instance
            
        Returns:
            Database row ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO evaluations (
                        query, ground_truth, retrieved_context, answer,
                        faithfulness, context_recall, answer_relevancy, aggregate_score,
                        session_id, source, evaluation_time_ms, model_used, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.sample.query,
                    result.sample.ground_truth,
                    json.dumps(result.sample.retrieved_context),
                    result.sample.answer,
                    result.metrics.faithfulness,
                    result.metrics.context_recall,
                    result.metrics.answer_relevancy,
                    result.metrics.aggregate_score,
                    result.sample.session_id,
                    result.sample.source,
                    result.metrics.evaluation_time_ms,
                    result.metrics.model_used,
                    result.metrics.timestamp.isoformat(),
                ))
                
                conn.commit()
                row_id = cursor.lastrowid
                logger.info(f"✓ Stored evaluation result (ID: {row_id})")
                return row_id
        
        except Exception as e:
            logger.error(f"Failed to store evaluation result: {e}")
            raise
    
    def get_results_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all evaluation results for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of result dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM evaluations 
                WHERE session_id = ? 
                ORDER BY timestamp DESC
            """, (session_id,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_results_by_source(self, source: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get evaluation results by source type.
        
        Args:
            source: Source type (code, docs, etc.)
            limit: Maximum results to return
            
        Returns:
            List of result dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM evaluations 
                WHERE source = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (source, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_aggregate_metrics(self, source: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregate metrics across all evaluations.
        
        Args:
            source: Optional source filter
            
        Returns:
            Dictionary with aggregate statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            where_clause = "WHERE source = ?" if source else ""
            params = (source,) if source else ()
            
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_evaluations,
                    AVG(faithfulness) as avg_faithfulness,
                    AVG(context_recall) as avg_context_recall,
                    AVG(answer_relevancy) as avg_answer_relevancy,
                    AVG(aggregate_score) as avg_aggregate_score,
                    MIN(aggregate_score) as min_aggregate_score,
                    MAX(aggregate_score) as max_aggregate_score
                FROM evaluations
                {where_clause}
            """, params)
            
            row = cursor.fetchone()
            return dict(zip([desc[0] for desc in cursor.description], row)) if row else {}
    
    def export_to_csv(self, output_path: str, source: Optional[str] = None):
        """Export results to CSV file.
        
        Args:
            output_path: Path to output CSV file
            source: Optional source filter
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                where_clause = "WHERE source = ?" if source else ""
                params = (source,) if source else ()
                
                cursor.execute(f"""
                    SELECT * FROM evaluations {where_clause} 
                    ORDER BY timestamp DESC
                """, params)
                
                rows = cursor.fetchall()
                
                if not rows:
                    logger.warning(f"No results to export for source: {source}")
                    return
                
                # Write CSV
                with open(output_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=dict(rows[0]).keys())
                    writer.writeheader()
                    writer.writerows([dict(row) for row in rows])
                
                logger.info(f"✓ Exported {len(rows)} results to {output_path}")
        
        except Exception as e:
            logger.error(f"Failed to export results to CSV: {e}")


# ==================== RAG Evaluator ====================

class RAGEvaluator:
    """Main RAG evaluation engine using Ragas framework."""

    # Process-wide singleton (P2 #7): instantiating per-request reloaded the
    # Ollama LLM handle and the SQLite path; we now reuse one instance.
    _instance: "Optional[RAGEvaluator]" = None
    _instance_lock = None

    @classmethod
    def get_instance(
        cls,
        db_path: str = "evaluation_results.db",
    ) -> "RAGEvaluator":
        """Return the process-wide RAGEvaluator singleton (thread-safe)."""
        if cls._instance is not None:
            return cls._instance
        if cls._instance_lock is None:
            import threading
            cls._instance_lock = threading.Lock()
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(db_path=db_path)
        return cls._instance
    
    def __init__(
        self,
        db_path: str = "evaluation_results.db",
    ):
        """Initialize RAG evaluator.
        
        Args:
            db_path: SQLite database path
        """
        self.db = EvaluationDatabase(db_path)
        self.eval_model = os.getenv("GROQ_MODEL", os.getenv("OLLAMA_EVAL_MODEL", "unknown"))
        self.has_ragas = HAS_RAGAS
        self.has_ollama = HAS_OLLAMA
        
        # Setup LLM for evaluation
        self.evaluator_llm = self._setup_evaluator_llm()
        # RAGAS metrics like answer_relevancy need an embeddings model. Without
        # one, RAGAS silently defaults to OpenAI embeddings and calls
        # /openai/v1/... — which fails whenever OPENAI_API_KEY is a placeholder,
        # forcing every evaluation onto the lexical fallback. Reuse the app's
        # local HuggingFace embedder so evaluation is fully self-contained.
        self.evaluator_embeddings = self._setup_evaluator_embeddings()

    def _setup_evaluator_embeddings(self):
        """Local embeddings for RAGAS so it never reaches out to OpenAI."""
        try:
            from app.core.database import get_embedder
            emb = get_embedder()
            if emb is not None:
                logger.info("✓ RAGAS embeddings: local HuggingFace singleton")
                return emb
        except Exception as e:  # noqa: BLE001
            logger.warning(f"RAGAS local embeddings unavailable: {e}")
        return None

        provider = os.getenv("EVAL_LLM_PROVIDER", "groq").lower()

        if provider == "ollama":
            try:
                from langchain_ollama import OllamaLLM
                llm = OllamaLLM(
                    model=os.getenv("OLLAMA_EVAL_MODEL", "mistral"),
                    base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
                )
                logger.info(f"✓ Ollama evaluator: {os.getenv('OLLAMA_EVAL_MODEL', 'mistral')}")
                return llm
            except Exception as e:
                logger.warning(f"Ollama evaluator failed: {e}")
                return None

        if provider == "groq":
            try:
                from langchain_groq import ChatGroq
                llm = ChatGroq(
                    api_key=os.getenv("GROQ_API_KEY", ""),
                    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                    temperature=0.0,
                )
                logger.info(f"✓ Groq evaluator: {os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')}")
                return llm
            except Exception as e:
                logger.warning(f"Groq evaluator failed: {e}")
                return None

        logger.warning(f"Unknown EVAL_LLM_PROVIDER: {provider}")
        return None
    
    def evaluate_sample(self, sample: EvaluationSample) -> Optional[EvaluationResult]:
        """Evaluate a single RAG sample.
        
        Uses Ragas framework to compute:
        - Faithfulness (hallucination detection)
        - Context Recall (ground truth coverage)
        - Answer Relevancy (query alignment)
        
        Args:
            sample: EvaluationSample to evaluate
            
        Returns:
            EvaluationResult with computed metrics, or None if evaluation fails
        """
        if not self.has_ragas or not self.evaluator_llm:
            logger.warning("Ragas or Ollama not available - cannot evaluate")
            return self._fallback_evaluate(sample)

        # Background-job span: joins the originating request trace via
        # sample.trace_id (M-2 — executor threads lose OTEL context, so we pin
        # the trace explicitly) instead of rooting a separate trace.
        from app.observability.tracing import span as _lf_span
        with _lf_span(
            "ragas.evaluate_sample",
            kind="evaluator",
            input={"query": sample.query[:300], "contexts": len(sample.retrieved_context)},
            metadata={
                "job.type": "ragas_evaluation",
                "request.source": "background-job",
            },
            trace_id=getattr(sample, "trace_id", None),
        ):
            return self._evaluate_sample_inner(sample)

    def _evaluate_sample_inner(self, sample: EvaluationSample) -> Optional[EvaluationResult]:
        try:
            start_time = time.time()
            
            # Prepare dataset for Ragas
            dataset_dict = {
                "question": [sample.query],
                "answer": [sample.answer],
                "contexts": [sample.retrieved_context],
                "ground_truth": [sample.ground_truth],
            }
            
            dataset = Dataset.from_dict(dataset_dict)
            
            # Run evaluation
            _eval_kwargs = {
                "metrics": [faithfulness, context_recall, answer_relevancy],
                "llm": self.evaluator_llm,
            }
            # Pin local embeddings so answer_relevancy never calls OpenAI.
            if getattr(self, "evaluator_embeddings", None) is not None:
                _eval_kwargs["embeddings"] = self.evaluator_embeddings
            result = evaluate(dataset, **_eval_kwargs)
            
            eval_time_ms = (time.time() - start_time) * 1000
            
            # Extract scores
            metrics = EvaluationMetrics(
                faithfulness=float(result["faithfulness"][0]) if "faithfulness" in result else 0.0,
                context_recall=float(result["context_recall"][0]) if "context_recall" in result else 0.0,
                answer_relevancy=float(result["answer_relevancy"][0]) if "answer_relevancy" in result else 0.0,
                evaluation_time_ms=eval_time_ms,
                model_used=self.eval_model,
            )
            
            eval_result = EvaluationResult(sample=sample, metrics=metrics)
            self._publish_to_langfuse(sample, metrics)
            return eval_result
        
        except Exception as e:
            logger.error(f"Ragas evaluation failed: {e}. Using fallback.")
            return self._fallback_evaluate(sample)
    
    def _fallback_evaluate(self, sample: EvaluationSample) -> EvaluationResult:
        """Fallback evaluation using simple heuristics.
        
        Used when Ragas/Ollama not available. Provides basic scoring:
        - Faithfulness: Text overlap between answer and context
        - Context Recall: How much ground truth appears in context
        - Answer Relevancy: Query similarity to answer
        
        Args:
            sample: EvaluationSample to evaluate
            
        Returns:
            EvaluationResult with heuristic scores
        """
        start_time = time.time()
        
        # Faithfulness: Check answer uses context
        answer_lower = sample.answer.lower()
        context_text = " ".join(sample.retrieved_context).lower()
        faithfulness_score = self._text_overlap(answer_lower, context_text)
        
        # Context Recall: Check ground truth in context
        ground_truth_lower = sample.ground_truth.lower()
        context_recall_score = self._text_overlap(ground_truth_lower, context_text)
        
        # Answer Relevancy: Check query-answer similarity
        query_lower = sample.query.lower()
        answer_relevancy_score = self._text_overlap(query_lower, answer_lower)
        
        eval_time_ms = (time.time() - start_time) * 1000
        
        metrics = EvaluationMetrics(
            faithfulness=min(max(faithfulness_score, 0.0), 1.0),
            context_recall=min(max(context_recall_score, 0.0), 1.0),
            answer_relevancy=min(max(answer_relevancy_score, 0.0), 1.0),
            evaluation_time_ms=eval_time_ms,
            model_used="heuristic-fallback",
        )
        
        self._publish_to_langfuse(sample, metrics)
        return EvaluationResult(sample=sample, metrics=metrics)

    def _publish_to_langfuse(
        self,
        sample: "EvaluationSample",
        metrics: "EvaluationMetrics",
    ) -> None:
        """Push evaluation scores onto the originating Langfuse trace.

        Records faithfulness, context recall, answer relevancy, and the
        aggregate as Langfuse scores so quality is visible per-trace and can be
        aggregated in the Langfuse dashboard. Never raises — a scoring failure
        must not affect evaluation or the request.
        """
        trace_id = getattr(sample, "trace_id", None)
        if not trace_id:
            return
        try:
            from app.observability.langfuse_client import score_current_trace, is_enabled
            if not is_enabled():
                return
            score_map = {
                "faithfulness": metrics.faithfulness,
                "context_recall": metrics.context_recall,
                "answer_relevancy": metrics.answer_relevancy,
                "ragas_aggregate": metrics.aggregate_score,
            }
            for name, value in score_map.items():
                score_current_trace(
                    trace_id=trace_id,
                    name=name,
                    value=float(value),
                    comment=f"model={metrics.model_used}",
                    data_type="NUMERIC",
                )
            logger.debug("[rag_evaluator] published %d scores to Langfuse trace %s",
                         len(score_map), trace_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[rag_evaluator] Langfuse score publish skipped: %s", exc)
    
    @staticmethod
    def _text_overlap(text1: str, text2: str) -> float:
        """Calculate text overlap as a simple similarity score.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Overlap score 0-1
        """
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        return overlap / total if total > 0 else 0.0
    
    def evaluate_batch(self, samples: List[EvaluationSample]) -> List[EvaluationResult]:
        """Evaluate a batch of samples.
        
        Args:
            samples: List of EvaluationSample instances
            
        Returns:
            List of EvaluationResult instances
        """
        results = []
        
        for i, sample in enumerate(samples):
            logger.info(f"Evaluating sample {i+1}/{len(samples)}: {sample.query[:50]}...")
            
            result = self.evaluate_sample(sample)
            if result:
                results.append(result)
                
                # Store in database
                self.db.store_result(result)
        
        logger.info(f"✓ Completed batch evaluation: {len(results)}/{len(samples)} samples")
        return results
    
    def get_performance_report(self, source: Optional[str] = None) -> Dict[str, Any]:
        """Get performance report for evaluated samples.
        
        Args:
            source: Optional source filter (code, docs, etc.)
            
        Returns:
            Dictionary with performance metrics and analysis
        """
        metrics = self.db.get_aggregate_metrics(source)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "source": source or "all",
            "metrics": metrics,
            "analysis": {
                "faithfulness_status": self._get_status(
                    metrics.get("avg_faithfulness", 0.0)
                ),
                "recall_status": self._get_status(
                    metrics.get("avg_context_recall", 0.0)
                ),
                "relevancy_status": self._get_status(
                    metrics.get("avg_answer_relevancy", 0.0)
                ),
                "overall_status": self._get_status(
                    metrics.get("avg_aggregate_score", 0.0)
                ),
            },
            "recommendations": self._generate_recommendations(metrics),
        }
        
        return report
    
    @staticmethod
    def _get_status(score: float) -> str:
        """Get status label for a score.
        
        Args:
            score: Score 0-1
            
        Returns:
            Status label
        """
        if score >= 0.85:
            return "✓ Excellent"
        elif score >= 0.70:
            return "⚠ Good"
        elif score >= 0.50:
            return "⚠ Fair"
        else:
            return "✗ Poor"
    
    @staticmethod
    def _generate_recommendations(metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on metrics.
        
        Args:
            metrics: Aggregate metrics
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if metrics.get("avg_faithfulness", 1.0) < 0.70:
            recommendations.append(
                "Low faithfulness detected. Review retrieval quality and "
                "consider improving reranking or reducing context size."
            )
        
        if metrics.get("avg_context_recall", 1.0) < 0.70:
            recommendations.append(
                "Low context recall. Improve query expansion or increase k "
                "in retrieval to capture more ground truth."
            )
        
        if metrics.get("avg_answer_relevancy", 1.0) < 0.70:
            recommendations.append(
                "Low answer relevancy. Consider improving prompt engineering "
                "or LLM model selection."
            )
        
        if not recommendations:
            recommendations.append("Performance is good. Continue monitoring.")
        
        return recommendations


# ==================== Example Usage ====================

def example_evaluation():
    """Example evaluation workflow."""
    
    if not HAS_RAGAS:
        logger.warning("Ragas not installed. Skipping example.")
        return
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Create sample evaluations
    samples = [
        EvaluationSample(
            query="How do I configure PostgreSQL for production?",
            ground_truth="Use pgbouncer for connection pooling, enable SSL, "
                        "configure max_connections, and setup backups.",
            retrieved_context=[
                "PostgreSQL configuration best practices...",
                "Connection pooling with pgbouncer...",
                "SSL setup for PostgreSQL...",
            ],
            answer="To configure PostgreSQL for production, you should enable SSL, "
                   "use pgbouncer for connection pooling, and increase max_connections.",
            session_id="session-001",
            source="docs",
        ),
        EvaluationSample(
            query="What is the retriever_engine.py file used for?",
            ground_truth="It implements hybrid retrieval combining vector search, "
                        "BM25, query expansion, and BGE reranking.",
            retrieved_context=[
                "High-Performance Hybrid Retrieval Engine...",
                "Combines vector-based and lexical retrieval...",
                "Implements BGE-Reranker for top-K selection...",
            ],
            answer="The retriever_engine.py implements a hybrid retrieval system that "
                   "combines vector search and BM25 with BGE cross-encoder reranking.",
            session_id="session-001",
            source="code",
        ),
    ]
    
    # Evaluate batch
    results = evaluator.evaluate_batch(samples)
    
    # Generate report
    for source in ["docs", "code"]:
        report = evaluator.get_performance_report(source)
        logger.info(f"Performance Report ({source}):\n{json.dumps(report, indent=2)}")
    
    # Export results
    evaluator.db.export_to_csv("evaluation_results.csv")
