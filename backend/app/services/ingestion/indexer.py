"""Incremental indexing with hash-based deduplication."""

import hashlib
import logging
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class HashManager:
    """Manage document hashes for deduplication."""

    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute SHA256 hash of content.

        Args:
            content: Content to hash.

        Returns:
            Hex digest of SHA256 hash.
            
        Raises:
            ValueError: If content is invalid.
            RuntimeError: If hashing fails.
        """
        try:
            if not isinstance(content, str):
                raise ValueError(f"Content must be string, got {type(content)}")
            
            return hashlib.sha256(content.encode()).hexdigest()
            
        except ValueError as e:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to compute hash: {str(e)}")

    @staticmethod
    def compute_chunk_hash(chunk: dict) -> str:
        """Compute hash of a chunk.

        Args:
            chunk: Chunk dictionary with content and metadata.

        Returns:
            Hex digest of SHA256 hash.
        """
        content = chunk.get("content", "")
        source = chunk.get("metadata", {}).get("source", "")
        chunk_index = chunk.get("metadata", {}).get("chunk_index", 0)

        # Combine for stable hash
        combined = f"{source}:{chunk_index}:{content}"
        return hashlib.sha256(combined.encode()).hexdigest()


class IncrementalIndexer:
    """Track and manage incremental indexing state."""

    def __init__(self, db_connection=None):
        """Initialize the indexer.

        Args:
            db_connection: Database connection object.
        """
        self.db_connection = db_connection
        self.local_cache = {}  # In-memory cache for hashes

    def get_record_manager(self, namespace: str = "documents"):
        """Get SQLRecordManager instance for tracking.

        Args:
            namespace: Namespace for record tracking.

        Returns:
            Record manager instance.
        """
        try:
            from langchain.indexes import SQLRecordManager

            if not self.db_connection:
                raise ValueError("Database connection not configured")

            db_url = self._build_connection_string()
            
            record_manager = SQLRecordManager(
                namespace=namespace,
                db_url=db_url,
            )
            record_manager.create_schema()

            return record_manager

        except ImportError:
            logger.warning(
                "SQLRecordManager not available. Using in-memory cache."
            )
            return InMemoryRecordManager(namespace)

    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string.

        Returns:
            Connection string.
        """
        if hasattr(self.db_connection, "url"):
            return str(self.db_connection.url)

        # Manual construction
        host = getattr(self.db_connection, "host", "localhost")
        port = getattr(self.db_connection, "port", 5432)
        database = getattr(self.db_connection, "database", "codelens")
        user = getattr(self.db_connection, "user", "postgres")
        password = getattr(self.db_connection, "password", "")

        if password:
            return f"postgresql://{user}:{password}@{host}:{port}/{database}"
        else:
            return f"postgresql://{user}@{host}:{port}/{database}"

    def track_document(
        self,
        document_id: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> Dict:
        """Track a document for deduplication.

        Args:
            document_id: Unique document identifier.
            content: Document content.
            metadata: Document metadata.

        Returns:
            Record dictionary with hash and timestamp.
            
        Raises:
            ValueError: If parameters are invalid.
            RuntimeError: If tracking fails.
        """
        try:
            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError("document_id must be a non-empty string")
            
            if not isinstance(content, str):
                raise ValueError(f"content must be string, got {type(content)}")
            
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError(f"metadata must be dict, got {type(metadata)}")
            
            try:
                content_hash = HashManager.compute_hash(content)
            except Exception as e:
                raise RuntimeError(f"Failed to compute hash: {str(e)}")

            record = {
                "id": document_id,
                "hash": content_hash,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {},
                "changed": False,
            }

            # Check if hash exists
            if document_id in self.local_cache:
                old_hash = self.local_cache[document_id].get("hash")
                record["changed"] = old_hash != content_hash

            self.local_cache[document_id] = record

            return record
            
        except (ValueError, RuntimeError) as e:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to track document: {str(e)}")

    def has_changed(self, document_id: str, content: str) -> bool:
        """Check if a document has changed since last index.

        Args:
            document_id: Document identifier.
            content: Current document content.

        Returns:
            True if document has changed, False otherwise.
            
        Raises:
            ValueError: If parameters are invalid.
            RuntimeError: If comparison fails.
        """
        try:
            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError("document_id must be a non-empty string")
            
            if not isinstance(content, str):
                raise ValueError(f"content must be string, got {type(content)}")
            
            if document_id not in self.local_cache:
                return True

            try:
                current_hash = HashManager.compute_hash(content)
            except Exception as e:
                raise RuntimeError(f"Failed to compute hash: {str(e)}")
            
            stored_hash = self.local_cache[document_id].get("hash")

            return current_hash != stored_hash
            
        except (ValueError, RuntimeError) as e:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to check if document changed: {str(e)}")

    def filter_changed_chunks(self, chunks: List[dict]) -> Tuple[List[dict], List[dict]]:
        """Separate changed and unchanged chunks.

        Args:
            chunks: List of chunks to filter.

        Returns:
            Tuple of (changed_chunks, unchanged_chunks).
            
        Raises:
            ValueError: If chunks list is invalid.
            RuntimeError: If filtering fails.
        """
        try:
            if not isinstance(chunks, list):
                raise ValueError(f"Chunks must be a list, got {type(chunks)}")
            
            if not chunks:
                logger.warning("No chunks to filter")
                return [], []
            
            changed = []
            unchanged = []
            error_count = 0

            for chunk_idx, chunk in enumerate(chunks):
                try:
                    if not isinstance(chunk, dict):
                        logger.warning(f"Chunk {chunk_idx} is not a dict, skipping")
                        error_count += 1
                        continue
                    
                    chunk_hash = HashManager.compute_chunk_hash(chunk)
                    metadata = chunk.get("metadata", {})
                    source = metadata.get("source", "")
                    chunk_id = f"{source}:{chunk_hash}"

                    if self.has_changed(chunk_id, chunk.get("content", "")):
                        changed.append(chunk)
                    else:
                        unchanged.append(chunk)
                        
                except Exception as e:
                    error_count += 1
                    logger.warning(f"Error filtering chunk {chunk_idx}: {str(e)}")
                    # Treat errors as changed to be safe
                    changed.append(chunk)
                    continue

            if error_count > 0:
                logger.warning(
                    f"Filtered chunks with {error_count} errors: "
                    f"{len(changed)} changed, {len(unchanged)} unchanged"
                )
            else:
                logger.info(
                    f"Filtered chunks: {len(changed)} changed, {len(unchanged)} unchanged"
                )

            return changed, unchanged
            
        except ValueError as e:
            logger.error(f"Validation error in filter_changed_chunks: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in filter_changed_chunks: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to filter chunks: {str(e)}")

    def get_indexing_stats(self) -> Dict:
        """Get indexing statistics.

        Returns:
            Statistics dictionary.
        """
        total_tracked = len(self.local_cache)
        changed_count = sum(
            1 for r in self.local_cache.values() if r.get("changed", False)
        )

        return {
            "total_tracked": total_tracked,
            "changed": changed_count,
            "unchanged": total_tracked - changed_count,
            "cache_size_mb": sum(
                len(json.dumps(r)) for r in self.local_cache.values()
            ) / (1024 * 1024),
        }


class InMemoryRecordManager:
    """Fallback in-memory record manager for development."""

    def __init__(self, namespace: str = "documents"):
        """Initialize in-memory record manager.

        Args:
            namespace: Namespace for records.
        """
        self.namespace = namespace
        self.records = {}

    def create_schema(self):
        """Create schema (no-op for in-memory)."""
        pass

    def add_records(self, records: List[Dict]):
        """Add records.

        Args:
            records: List of record dictionaries.
        """
        for record in records:
            key = f"{self.namespace}:{record['id']}"
            self.records[key] = record

    def get_records(self, ids: List[str]) -> List[Dict]:
        """Get records by IDs.

        Args:
            ids: List of record IDs.

        Returns:
            List of record dictionaries.
        """
        return [
            self.records.get(f"{self.namespace}:{id_}") for id_ in ids
        ]

    def exists(self, record_id: str) -> bool:
        """Check if record exists.

        Args:
            record_id: Record ID to check.

        Returns:
            True if exists, False otherwise.
        """
        return f"{self.namespace}:{record_id}" in self.records


class IndexingMetrics:
    """Track indexing metrics."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.documents_processed = 0
        self.chunks_created = 0
        self.chunks_skipped = 0
        self.embeddings_generated = 0
        self.start_time = None
        self.end_time = None

    def record_processing(
        self,
        documents_count: int,
        chunks_count: int,
        skipped_count: int = 0,
    ):
        """Record processing metrics.

        Args:
            documents_count: Number of documents processed.
            chunks_count: Number of chunks created.
            skipped_count: Number of chunks skipped.
        """
        self.documents_processed += documents_count
        self.chunks_created += chunks_count
        self.chunks_skipped += skipped_count

    def record_embedding(self, count: int):
        """Record embedding generation.

        Args:
            count: Number of embeddings generated.
        """
        self.embeddings_generated += count

    def get_summary(self) -> Dict:
        """Get metrics summary.

        Returns:
            Metrics dictionary.
        """
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()

        return {
            "documents_processed": self.documents_processed,
            "chunks_created": self.chunks_created,
            "chunks_skipped": self.chunks_skipped,
            "embeddings_generated": self.embeddings_generated,
            "duration_seconds": duration,
            "chunks_per_second": (
                self.chunks_created / duration if duration and duration > 0 else 0
            ),
        }
