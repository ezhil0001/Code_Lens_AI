"""Expert-level context-aware ingestion pipeline with incremental indexing and enrichment."""

import logging
import time
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Centralized debug logger — emits [EMBEDDING] / [STORAGE] tags during ingest.
from app.core.logger import logger as flow_logger, timed, log_step, log_success

logger = logging.getLogger(__name__)


class ContextAwareIngestionPipeline:
    """End-to-end ingestion pipeline: Load → Split → PDR → Embed → Store."""

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        child_chunk_size: int = 400,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        persist_directory: str = "./chroma_db",
        manifest_path: str = "./manifest.json",
        enable_incremental_indexing: bool = True,
        enable_enrichment: bool = True,
    ):
        """Initialize expert-level pipeline with incremental indexing and enrichment.
        
        Args:
            chunk_size: Size of chunks from language-aware splitter
            chunk_overlap: Overlap between chunks
            child_chunk_size: Target size for child chunks (for embedding)
            embedding_model: HuggingFace embedding model
            persist_directory: Directory for ChromaDB persistence
            manifest_path: Path to hash manifest for incremental indexing
            enable_incremental_indexing: Enable hash-based deduplication
            enable_enrichment: Enable contextual enrichment
        """
        try:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.child_chunk_size = child_chunk_size
            self.embedding_model = embedding_model
            self.persist_directory = persist_directory
            self.manifest_path = Path(manifest_path)
            self.enable_incremental_indexing = enable_incremental_indexing
            self.enable_enrichment = enable_enrichment
            
            # Load or create manifest for incremental indexing
            self.manifest = self._load_manifest() if enable_incremental_indexing else {}
            
            logger.info("Initialized Expert-Level ContextAwareIngestionPipeline")
            logger.info(f"  Chunk size: {chunk_size}")
            logger.info(f"  Child chunk size: {child_chunk_size}")
            logger.info(f"  Embedding model: {embedding_model}")
            logger.info(f"  Incremental indexing: {enable_incremental_indexing}")
            logger.info(f"  Contextual enrichment: {enable_enrichment}")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {str(e)}")
            raise RuntimeError(f"Failed to initialize pipeline: {str(e)}")

    def _load_manifest(self) -> Dict:
        """Load hash manifest from disk.
        
        Returns:
            Dictionary mapping file paths to hashes.
        """
        try:
            if self.manifest_path.exists():
                with open(self.manifest_path, "r") as f:
                    manifest = json.load(f)
                logger.info(f"Loaded manifest with {len(manifest)} tracked files")
                return manifest
            else:
                logger.info("Creating new manifest for incremental indexing")
                return {}
        except Exception as e:
            logger.error(f"Error loading manifest: {str(e)}")
            return {}

    def _save_manifest(self) -> None:
        """Save hash manifest to disk."""
        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.manifest_path, "w") as f:
                json.dump(self.manifest, f, indent=2)
            logger.info(f"Saved manifest with {len(self.manifest)} tracked files")
        except Exception as e:
            logger.error(f"Error saving manifest: {str(e)}")

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of a file.
        
        Args:
            file_path: Path to file.
            
        Returns:
            Hex digest of file hash.
        """
        try:
            from app.services.ingestion.indexer import HashManager
            
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            return HashManager.compute_hash(content)
        except Exception as e:
            logger.warning(f"Error computing hash for {file_path}: {str(e)}")
            return None

    def _should_reindex_file(self, file_path: str, current_hash: str) -> bool:
        """Check if file should be reindexed based on hash.
        
        Args:
            file_path: Path to file.
            current_hash: Current hash of file.
            
        Returns:
            True if file has changed and should be reindexed.
        """
        if not self.enable_incremental_indexing:
            return True
        
        stored_hash = self.manifest.get(file_path)
        changed = stored_hash != current_hash
        
        if not changed:
            logger.debug(f"Skipping unchanged file: {file_path}")
        else:
            logger.debug(f"Reindexing changed file: {file_path}")
        
        return changed

    def _enrich_chunk(self, chunk: Dict) -> Dict:
        """Enrich chunk with contextual information.
        
        Args:
            chunk: Chunk dictionary with content and metadata.
            
        Returns:
            Enhanced chunk with contextual header prepended.
        """
        if not self.enable_enrichment:
            return chunk
        
        try:
            from app.services.ingestion.enricher import ContextualEnricher
            
            enricher = ContextualEnricher()
            enriched = enricher.extract_context_info(chunk)
            
            # Build contextual header
            context = enriched.get("context", {})
            header_parts = []
            
            # Add imports
            if context.get("imports"):
                header_parts.append(f"[Imports]: {', '.join(context['imports'][:3])}")
            
            # Add classes
            if context.get("classes"):
                header_parts.append(f"[Classes]: {', '.join(context['classes'][:2])}")
            
            # Add functions
            if context.get("functions"):
                header_parts.append(f"[Functions]: {', '.join(context['functions'][:3])}")
            
            if header_parts:
                header = " | ".join(header_parts) + "\n"
                enriched["content"] = header + enriched["content"]
            
            return enriched
            
        except Exception as e:
            logger.warning(f"Error enriching chunk: {str(e)}")
            return chunk

    def ingest(
        self,
        source_paths: List[str] = None,
        source_type: str = "file_system",
        enrichment_enabled: bool = True,
    ) -> Dict:
        """
        Generic ingestion method that delegates to specialized methods.
        
        Args:
            source_paths: Paths to files or directories to ingest
            source_type: Type of source ("file_system", "codebase", "documents")
            enrichment_enabled: Whether to enable contextual enrichment
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            logger.info(f"[INGEST] Generic ingest() called with source_type={source_type}")
            
            if source_type == "file_system":
                # Ingest individual files uploaded through HTTP
                if not source_paths:
                    return {
                        "status": "error",
                        "error": "No source paths provided",
                        "documents_indexed": 0,
                        "chunks_created": 0,
                    }
                logger.info(f"[INGEST] Ingesting {len(source_paths)} files from file system")
                return self._ingest_files(source_paths, enrichment_enabled)
            
            elif source_type == "codebase":
                # Ingest entire codebase directory
                code_dir = source_paths[0] if source_paths else "./"
                logger.info(f"[INGEST] Ingesting codebase from {code_dir}")
                return self.ingest_codebase(code_dir)
            
            elif source_type == "documents":
                # Ingest knowledge base documents
                logger.info(f"[INGEST] Ingesting {len(source_paths) if source_paths else 0} documents")
                return self.ingest_kt_documents(source_paths, enrichment_enabled)
            
            else:
                return {
                    "status": "error",
                    "error": f"Unknown source_type: {source_type}",
                    "documents_indexed": 0,
                    "chunks_created": 0,
                }
        
        except Exception as e:
            logger.error(f"[INGEST] Error in generic ingest(): {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "documents_indexed": 0,
                "chunks_created": 0,
            }

    def _ingest_files(
        self,
        source_paths: List[str],
        enrichment_enabled: bool = True,
    ) -> Dict:
        """
        Ingest individual files (uploaded through HTTP).
        
        Args:
            source_paths: List of file paths to ingest
            enrichment_enabled: Whether to enable contextual enrichment
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            start_time = time.time()
            logger.info(f"[INGEST-FILES] Starting ingestion of {len(source_paths)} files")
            log_step("[FILE UPLOAD]", f"_ingest_files called with {len(source_paths)} path(s)")
            for sp in source_paths[:5]:
                flow_logger.bind(tag="[FILE UPLOAD]").debug(f"  • {Path(sp).name}")
            if len(source_paths) > 5:
                flow_logger.bind(tag="[FILE UPLOAD]").debug(f"  … +{len(source_paths)-5} more")
            
            # Import modules
            from app.services.ingestion.multi_modal_loader import MultiModalLoader
            from app.services.ingestion.language_aware_splitter import LanguageAwareSplitter
            from app.services.ingestion.parent_document_retriever import (
                ParentDocumentStore, PDRStrategy
            )
            from app.services.ingestion.chroma_vector_store import (
                EmbeddingEngine, ChromaVectorStore
            )
            
            # Stage 1: Load files
            logger.info("[PHASE-1] Loading files...")
            # Shared PDR parent store for this ingestion session.
            # In-memory for now; parents link chunks to their page/function context.
            parent_store = ParentDocumentStore(backend="memory")
            loader = MultiModalLoader()
            all_documents = []
            
            # Load all files as KT documents (PDFs, markdown, etc from HTTP uploads)
            try:
                logger.info(f"  Loading {len(source_paths)} files...")
                # Create a temporary directory containing all files
                import tempfile
                import shutil
                
                # Create temp dir for batch loading
                temp_dir = tempfile.mkdtemp()
                for src_path in source_paths:
                    src = Path(src_path)
                    dst = Path(temp_dir) / src.name
                    shutil.copy(src_path, dst)
                
                # Load all files at once using load_kt_documents
                documents = loader.load_kt_documents(
                    directory_path=temp_dir,
                    file_patterns=None,  # Load all supported KT formats
                    recursive=False
                )
                all_documents.extend(documents)
                logger.info(f"    ✓ Loaded {len(documents)} document(s) from {len(source_paths)} file(s)")
                
                # Cleanup temp dir
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"  ✗ Error loading files: {str(e)}")
            
            if not all_documents:
                logger.warning("[PHASE-1] No documents loaded")
                return {
                    "status": "error",
                    "error": "No valid documents loaded",
                    "documents_indexed": 0,
                    "chunks_created": 0,
                }
            
            logger.info(f"[PHASE-1] ✓ Loaded {len(all_documents)} total documents")
            
            # Stage 2: Split documents
            logger.info("[PHASE-2] Splitting documents with language-aware splitter...")
            splitter = LanguageAwareSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            
            chunks = []  # Will contain dicts from splitter
            phase2_start = time.time()
            chunk_counter = 0
            
            for doc in all_documents:
                try:
                    # Ensure document has page_content
                    if not hasattr(doc, 'page_content') or not doc.page_content:
                        logger.warning(f"  ✗ Document missing page_content: {doc.metadata.get('source', 'unknown')}")
                        continue
                    
                    language = doc.metadata.get("language", "python")
                    # split_code returns a list of dicts: {"content": ..., "metadata": ...}
                    doc_chunks = splitter.split_code(doc, language=language)
                    
                    # Ensure each chunk has a unique id for PDR strategy
                    for chunk in doc_chunks:
                        chunk["id"] = f"chunk_{chunk_counter}"
                        chunk_counter += 1
                    
                    chunks.extend(doc_chunks)
                    logger.info(f"  ✓ Split {doc.metadata.get('source')} into {len(doc_chunks)} chunks")
                    
                    # Log sample chunks with details (chunks are dicts)
                    for chunk_idx, chunk in enumerate(doc_chunks[:3], 1):
                        chunk_content = chunk.get("content", "")[:100]
                        chunk_len = len(chunk.get("content", ""))
                        logger.info(f"     ├─ Chunk {chunk_idx}: {chunk_len} chars")
                        logger.info(f"     │  Preview: {chunk_content.strip()}...")
                        
                except Exception as e:
                    logger.warning(f"  ✗ Error splitting {doc.metadata.get('source')}: {str(e)}")
            
            phase2_time = time.time() - phase2_start
            logger.info(f"[PHASE-2] ✓ Created {len(chunks)} chunks (size: {self.chunk_size}, overlap: {self.chunk_overlap})")
            logger.info(f"[PHASE-2] ⏱  Time taken: {phase2_time:.2f}s")
            
            # Stage 3: Create parent document structure (PDR)
            logger.info("[PHASE-3] Creating parent document retrieval structure...")

            # For prose documents (PDFs/markdown) where AST extraction yields
            # no function-level parents, register each *original page/document*
            # (from Phase 1 all_documents) as its own parent so retrieval
            # returns a page of context rather than the entire file.
            prose_languages = {"markdown", "md", "txt", "text", "pdf", "rst"}
            for doc in all_documents:
                doc_lang = (doc.metadata.get("language") or "").lower()
                if doc_lang in prose_languages:
                    src = doc.metadata.get("source", "unknown")
                    page = doc.metadata.get("page", 0)
                    pid = f"parent::{src}::page::{page}"
                    if not parent_store.get_parent(pid):
                        parent_store.add_parent(
                            parent_id=pid,
                            content=doc.page_content,
                            metadata={**doc.metadata, "scope": "page", "parent_name": f"page_{page}"},
                        )

            pdr_strategy = PDRStrategy(
                child_chunk_size=self.child_chunk_size,
                parent_store=parent_store,  # shared store — not a throwaway
            )

            # Use the correct method name and pass metadata-enriched chunks
            enhanced_chunks = pdr_strategy.create_child_parent_pairs(chunks)

            # Extract parent and child statistics
            # Count distinct parents actually referenced by chunks (excludes
            # pre-registered page parents that happen to have no children).
            num_parents = len(set(
                chunk.get("parent_id") for chunk in enhanced_chunks
                if chunk.get("parent_id") and "::__module__" not in chunk.get("parent_id", "")
            ))
            num_children = len(enhanced_chunks)
            scope_counts = {}
            for c in enhanced_chunks:
                s = (c.get("metadata") or {}).get("scope", "unknown")
                scope_counts[s] = scope_counts.get(s, 0) + 1

            log_success("[PDR]",
                f"parents={num_parents}  chunks={num_children}  "
                f"scopes={scope_counts}"
            )
            logger.info(f"[PHASE-3] ✓ Created PDR structure: {num_parents} parents, {num_children} children")
            
            # Stage 4: Generate embeddings
            # Code chunks use the code-specialized embedder; prose chunks use
            # the shared general-purpose embedder.  Both are 768-dim singletons
            # so they land in the same ChromaDB collection without schema changes.
            logger.info("[PHASE-4] Generating embeddings...")
            log_step("[EMBEDDING]", f"model={self.embedding_model} chunks={len(enhanced_chunks)}")

            _CODE_LANGUAGES = frozenset({
                "python", "typescript", "javascript", "java", "cpp", "c",
                "go", "rust", "ruby", "kotlin", "swift", "scala", "shell",
                "bash", "sh", "tsx", "jsx",
            })

            # Only load the 329 MB code embedder if at least one chunk is code.
            _has_code_chunks = any(
                (c.get("metadata") or {}).get("file_type", "").lower() == "code"
                or (c.get("metadata") or {}).get("language", "").lower() in _CODE_LANGUAGES
                for c in enhanced_chunks
            )

            _code_embed_engine = None
            if _has_code_chunks:
                try:
                    from app.core.database import get_code_embedder, get_code_embed_model_name
                    _code_embed_engine = EmbeddingEngine(
                        model_name=get_code_embed_model_name(),
                        embedder=get_code_embedder(),
                    )
                except Exception as _ce:
                    logger.warning(f"[EMBEDDING] Code embedder unavailable ({_ce}), falling back to general")

            _general_embed_engine = EmbeddingEngine(model_name=self.embedding_model)

            def _pick_engine(chunk: dict) -> "EmbeddingEngine":
                """Return the code embedder for code chunks, general for prose."""
                if _code_embed_engine is None:
                    return _general_embed_engine
                meta = chunk.get("metadata") or {}
                lang = (meta.get("language") or "").lower()
                ftype = (meta.get("file_type") or "").lower()
                if ftype == "code" or lang in _CODE_LANGUAGES:
                    return _code_embed_engine
                return _general_embed_engine

            embeddings = []
            chunk_embedding_map = {}  # Map chunk id to embedding
            phase4_start = time.time()
            with timed("[EMBEDDING]") as embed_ctx:
                for i, chunk in enumerate(enhanced_chunks):
                    try:
                        embed_start = time.time()
                        engine = _pick_engine(chunk)
                        embedding = engine.embed_text(chunk["content"])
                        embed_time = time.time() - embed_start
                        embeddings.append(embedding)
                        chunk_embedding_map[chunk["id"]] = embedding
                        
                        # Log sample embeddings with full details
                        if i < 3:  # Show first 3 embeddings
                            chunk_preview = chunk["content"][:80].replace('\n', ' ').strip()
                            vector_dim = len(embedding) if embedding else 0
                            sample_values = embedding[:3] if embedding else []
                            flow_logger.bind(tag="[EMBEDDING]").debug(
                                f"chunk#{i} dim={vector_dim} t={embed_time*1000:.1f}ms "
                                f"model={engine.model_name} "
                                f"sample={sample_values} preview='{chunk_preview}...'"
                            )
                        
                        if (i + 1) % 10 == 0:
                            flow_logger.bind(tag="[EMBEDDING]").info(
                                f"progress {i + 1}/{len(enhanced_chunks)}"
                            )
                    except Exception as e:
                        flow_logger.bind(tag="[EMBEDDING]").warning(
                            f"chunk#{i} failed: {e}"
                        )
                embed_ctx["count"] = len(embeddings)
                embed_ctx["dim"] = len(embeddings[0]) if embeddings else 0
            
            phase4_time = time.time() - phase4_start
            vector_dim = len(embeddings[0]) if embeddings else 0
            logger.info(f"[PHASE-4] ✓ Generated {len(embeddings)} embeddings ({vector_dim}-dimensional)")
            logger.info(f"[PHASE-4] ⏱  Total time: {phase4_time:.2f}s")
            
            # Stage 5: Store in ChromaDB
            # Append to the active collection so the singleton RetrieverEngine
            # and BM25 index continue to see ALL documents (old + new).
            # A fresh timestamp collection is only created on the very first
            # ingest run; subsequent uploads land in the same collection.
            logger.info("[PHASE-5] Storing vectors in ChromaDB...")
            try:
                from app.services.ingestion.ingestion_service import IngestionService
                _live = IngestionService.get_chroma_collection()
                collection_id = _live.name
            except Exception:
                collection_id = f"documents_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            log_step("[STORAGE]", f"backend=ChromaDB collection={collection_id} persist_dir={self.persist_directory}")
            
            vector_store = ChromaVectorStore(
                collection_name=collection_id,
                persist_directory=self.persist_directory,
            )
            
            # Enrich chunk metadata before storing
            enriched_chunks = []
            for i, (chunk, embedding) in enumerate(zip(enhanced_chunks, embeddings)):
                enriched_chunk = {
                    **chunk,
                    "metadata": {
                        **chunk.get("metadata", {}),
                        "chunk_index": i,
                        "embedding_model": self.embedding_model,
                        "vector_dimension": len(embedding) if embedding else 0,
                    }
                }
                enriched_chunks.append(enriched_chunk)
            
            # Batch store all chunks — ChromaDB (primary)
            with timed("[STORAGE]") as store_ctx:
                vector_store.add_documents(enriched_chunks, embeddings)
                store_ctx["vectors"] = len(enriched_chunks)
                store_ctx["collection"] = collection_id
            stored_count = len(enriched_chunks)
            logger.info(f"[PHASE-5] ✓ Stored {stored_count} vectors in ChromaDB (collection: {collection_id})")
            log_success("[STORAGE]", f"Persisted {stored_count} vectors → {collection_id}")

            # Parallel pgvector write — active only when VECTOR_STORE_BACKEND includes "pgvector"
            try:
                from app.services.retrieval.pgvector_store import (
                    PgVectorDocumentStore, pgvector_enabled,
                )
                if pgvector_enabled():
                    pg_store = PgVectorDocumentStore()
                    pg_upserted = pg_store.insert_chunks(enriched_chunks, embeddings)
                    logger.info(f"[PHASE-5] ✓ Also mirrored {pg_upserted} chunks → pgvector document_chunks")
            except Exception as _pg_err:  # noqa: BLE001
                logger.warning(f"[PHASE-5] pgvector mirror skipped: {_pg_err}")

            # Stage 6: Update manifest for incremental indexing
            logger.info("[PHASE-6] Updating manifest for incremental indexing...")
            for file_path in source_paths:
                try:
                    file_hash = self._compute_file_hash(file_path)
                    if file_hash:
                        self.manifest[file_path] = file_hash
                except Exception as e:
                    logger.warning(f"  Error hashing {file_path}: {str(e)}")
            
            self._save_manifest()
            logger.info(f"[PHASE-6] ✓ Updated manifest with {len(source_paths)} files")
            
            total_time = time.time() - start_time
            
            logger.info("[INGEST-COMPLETE] Ingestion successful!")
            logger.info(f"  Total time: {total_time:.2f}s")
            logger.info(f"  Documents: {len(all_documents)}")
            logger.info(f"  Chunks: {len(chunks)}")
            logger.info(f"  Vectors: {stored_count} ({vector_dim}D)")
            
            return {
                "status": "success",
                "documents_indexed": len(all_documents),
                "chunks_created": len(chunks),
                "parent_docs_stored": num_parents,
                "child_chunks_embedded": len(embeddings),
                "vectors_stored": stored_count,
                "vector_dimension": vector_dim,
                "collection_id": collection_id,
                "processing_time_seconds": total_time,
                "metrics": {
                    "total_time": total_time,
                    "files": len(source_paths),
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "embedding_model": self.embedding_model,
                }
            }
        
        except Exception as e:
            logger.error(f"[INGEST-FILES] Error: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "documents_indexed": 0,
                "chunks_created": 0,
            }

    def ingest_codebase(
        self,
        code_directory: str,
        code_patterns: Optional[List[str]] = None,
    ) -> Dict:
        """Expert-level codebase ingestion with incremental indexing and enrichment.
        
        Pipeline:
        1. Load source code files
        2. Check hashes for incremental indexing (skip unchanged files)
        3. Split using language-aware splitters
        4. Enrich with contextual information
        5. Apply PDR strategy (mark parents/children)
        6. Generate embeddings for children
        7. Store in ChromaDB
        
        Args:
            code_directory: Path to codebase directory
            code_patterns: File patterns (e.g., ["*.py", "*.ts"])
            
        Returns:
            Dictionary with ingestion results and metrics
        """
        try:
            start_time = time.time()
            logger.info(f"Starting expert-level codebase ingestion: {code_directory}")
            
            # Import modules
            from app.services.ingestion.multi_modal_loader import MultiModalLoader
            from app.services.ingestion.language_aware_splitter import LanguageAwareSplitter
            from app.services.ingestion.parent_document_retriever import (
                ParentDocumentStore, PDRStrategy
            )
            from app.services.ingestion.chroma_vector_store import (
                EmbeddingEngine, ChromaVectorStore
            )
            
            # Stage 1: Load code files
            logger.info("Stage 1: Loading source code with hash tracking...")
            loader = MultiModalLoader()
            all_documents = loader.load_source_code(
                code_directory,
                code_patterns or ["*.py", "*.ts", "*.tsx", "*.js"],
            )
            load_time = time.time() - start_time
            logger.info(f"  Loaded {len(all_documents)} files in {load_time:.2f}s")
            
            # Stage 1.5: Incremental indexing - filter unchanged files
            logger.info("Stage 1.5: Filtering unchanged files (incremental indexing)...")
            filtered_documents = []
            skipped_count = 0
            failed_files = []
            
            for doc in all_documents:
                try:
                    source_path = doc.metadata.get("source", "")
                    
                    # Compute hash
                    file_hash = self._compute_file_hash(source_path)
                    
                    # Check if file should be reindexed
                    if file_hash and self._should_reindex_file(source_path, file_hash):
                        filtered_documents.append(doc)
                        self.manifest[source_path] = file_hash  # Update manifest
                    elif file_hash:
                        skipped_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {doc.metadata.get('source', 'unknown')}: {str(e)}")
                    failed_files.append({
                        "file": doc.metadata.get("source", "unknown"),
                        "error": str(e)
                    })
            
            logger.info(f"  Filtered documents: {len(filtered_documents)} to process, {skipped_count} skipped")
            
            # Save updated manifest
            self._save_manifest()
            
            # Stage 2: Split documents using language-aware splitter
            logger.info("Stage 2: Splitting documents with language awareness...")
            split_start = time.time()
            splitter = LanguageAwareSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            
            chunks = []
            split_failed = 0
            
            for doc in filtered_documents:
                try:
                    doc_chunks = splitter.split_code(
                        [doc],
                        language=doc.metadata.get("language", "python")
                    )
                    chunks.extend(doc_chunks)
                except Exception as e:
                    logger.warning(f"Error splitting {doc.metadata.get('source')}: {str(e)}")
                    split_failed += 1
                    failed_files.append({
                        "file": doc.metadata.get("source", "unknown"),
                        "stage": "split",
                        "error": str(e)
                    })
            
            split_time = time.time() - split_start
            logger.info(f"  Created {len(chunks)} chunks in {split_time:.2f}s ({split_failed} failed)")
            
            # Stage 2.5: Enrich chunks with contextual information
            logger.info("Stage 2.5: Enriching chunks with contextual information...")
            enrich_start = time.time()
            enriched_chunks = []
            enrich_failed = 0
            
            for chunk in chunks:
                try:
                    enriched = self._enrich_chunk(chunk)
                    # Ensure metadata propagation
                    enriched["metadata"] = {
                        **enriched.get("metadata", {}),
                        "file_path": enriched["metadata"].get("source", ""),
                        "source_type": enriched["metadata"].get("file_type", "code"),
                        "start_line": enriched["metadata"].get("start_line", 0),
                    }
                    enriched_chunks.append(enriched)
                except Exception as e:
                    logger.warning(f"Error enriching chunk: {str(e)}")
                    enrich_failed += 1
                    enriched_chunks.append(chunk)
            
            enrich_time = time.time() - enrich_start
            logger.info(f"  Enriched {len(enriched_chunks)} chunks in {enrich_time:.2f}s ({enrich_failed} failed)")
            
            # Stage 3: Apply PDR strategy
            logger.info("Stage 3: Applying Parent Document Retrieval...")
            pdr_start = time.time()
            parent_store = ParentDocumentStore(backend="memory")
            pdr = PDRStrategy(
                parent_store=parent_store,
                child_chunk_size=self.child_chunk_size,
            )
            enhanced_chunks = pdr.create_child_parent_pairs(enriched_chunks)
            embedding_ready = pdr.prepare_for_embedding(enhanced_chunks)
            pdr_time = time.time() - pdr_start
            logger.info(f"  Created PDR pairs in {pdr_time:.2f}s")
            logger.info(f"  Parent store: {parent_store.get_statistics()}")
            logger.info(f"  Ready for embedding: {len(embedding_ready)} child chunks")
            
            # Stage 4: Generate embeddings
            # ingest_codebase handles only source code — always use the
            # code-specialized embedder for best retrieval quality.
            logger.info("Stage 4: Generating embeddings (code-specialized model)...")
            embed_start = time.time()
            try:
                from app.core.database import get_code_embedder, get_code_embed_model_name
                embedding_engine = EmbeddingEngine(
                    model_name=get_code_embed_model_name(),
                    embedder=get_code_embedder(),
                )
                logger.info(f"  Using code-specialized embedder: {get_code_embed_model_name()}")
            except Exception as _ce:
                logger.warning(f"  Code embedder unavailable ({_ce}), falling back to general")
                embedding_engine = EmbeddingEngine(model_name=self.embedding_model)

            # Extract texts for embedding
            texts_to_embed = [chunk["content"] for chunk in embedding_ready]
            embeddings = embedding_engine.embed_batch(texts_to_embed)
            embed_time = time.time() - embed_start
            logger.info(f"  Generated {len(embeddings)} embeddings in {embed_time:.2f}s")
            
            # Stage 5: Store in ChromaDB
            logger.info("Stage 5: Storing in ChromaDB...")
            store_start = time.time()
            vector_store = ChromaVectorStore(
                collection_name="code_ingestion",
                persist_directory=self.persist_directory,
            )
            vector_store.add_documents(embedding_ready, embeddings)
            vector_store.persist()
            store_time = time.time() - store_start
            logger.info(f"  Stored in ChromaDB in {store_time:.2f}s")
            logger.info(f"  ChromaDB stats: {vector_store.get_statistics()}")

            # Mirror write to pgvector when VECTOR_STORE_BACKEND includes "pgvector"
            try:
                from app.services.retrieval.pgvector_store import (
                    PgVectorDocumentStore, pgvector_enabled,
                )
                if pgvector_enabled():
                    pg_store = PgVectorDocumentStore()
                    pg_upserted = pg_store.insert_chunks(embedding_ready, embeddings)
                    logger.info(f"  ✓ Also mirrored {pg_upserted} chunks → pgvector document_chunks")
            except Exception as _pg_err:  # noqa: BLE001
                logger.warning(f"  pgvector mirror skipped: {_pg_err}")

            # Calculate total metrics
            total_time = time.time() - start_time
            
            result = {
                "status": "success",
                "pipeline": "codebase_ingestion",
                "metrics": {
                    "documents_loaded": len(all_documents),
                    "documents_processed": len(filtered_documents),
                    "documents_skipped": skipped_count,
                    "chunks_created": len(chunks),
                    "chunks_enriched": len(enriched_chunks),
                    "parent_documents": parent_store.get_statistics()["total_parents"],
                    "child_chunks_embedded": len(embedding_ready),
                    "embeddings_generated": len(embeddings),
                    "total_time_seconds": total_time,
                    "cost_savings_percent": (skipped_count / len(all_documents) * 100) if all_documents else 0,
                    "stages": {
                        "load": load_time,
                        "filter": time.time() - (start_time + load_time),
                        "split": split_time,
                        "enrich": enrich_time,
                        "pdr": pdr_time,
                        "embed": embed_time,
                        "store": store_time,
                    },
                    "failures": {
                        "split_failures": split_failed,
                        "enrich_failures": enrich_failed,
                        "total_failed_files": len(failed_files),
                    }
                },
                "vector_store": vector_store.get_statistics(),
                "failed_files": failed_files if failed_files else None,
            }
            
            logger.info(f"Codebase ingestion completed in {total_time:.2f}s")
            if failed_files:
                logger.warning(f"  {len(failed_files)} files failed processing")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ingest_codebase: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "metrics": {},
            }

    def ingest_kt_documents(
        self,
        kt_directory: str,
        kt_patterns: Optional[List[str]] = None,
    ) -> Dict:
        """Expert-level KT documents ingestion with error resilience.
        
        Args:
            kt_directory: Path to KT documents directory
            kt_patterns: File patterns (e.g., ["*.pdf", "*.md"])
            
        Returns:
            Dictionary with ingestion results and metrics
        """
        try:
            start_time = time.time()
            logger.info(f"Starting expert-level KT documents ingestion: {kt_directory}")
            
            # Import modules
            from app.services.ingestion.multi_modal_loader import MultiModalLoader
            from app.services.ingestion.language_aware_splitter import LanguageAwareSplitter
            from app.services.ingestion.parent_document_retriever import (
                ParentDocumentStore, PDRStrategy
            )
            from app.services.ingestion.chroma_vector_store import (
                EmbeddingEngine, ChromaVectorStore
            )
            
            # Stage 1: Load KT documents with error resilience
            logger.info("Stage 1: Loading KT documents with error resilience...")
            loader = MultiModalLoader()
            all_documents = []
            failed_files = []
            
            # Try to load PDFs
            try:
                pdf_docs = loader.load_kt_documents(
                    kt_directory,
                    ["*.pdf"] if not kt_patterns else [p for p in kt_patterns if p.endswith(".pdf")],
                )
                all_documents.extend(pdf_docs)
                logger.info(f"  Loaded {len(pdf_docs)} PDF files")
            except Exception as e:
                logger.error(f"Error loading PDFs: {str(e)}")
                failed_files.append({"type": "pdf", "error": str(e)})
            
            # Try to load Markdown
            try:
                md_docs = loader.load_kt_documents(
                    kt_directory,
                    ["*.md"] if not kt_patterns else [p for p in kt_patterns if p.endswith(".md")],
                )
                all_documents.extend(md_docs)
                logger.info(f"  Loaded {len(md_docs)} Markdown files")
            except Exception as e:
                logger.error(f"Error loading Markdown: {str(e)}")
                failed_files.append({"type": "markdown", "error": str(e)})
            
            load_time = time.time() - start_time
            logger.info(f"  Total: {len(all_documents)} documents loaded in {load_time:.2f}s")
            
            if not all_documents:
                logger.warning("No documents loaded")
                return {
                    "status": "warning",
                    "message": "No documents found",
                    "failed_files": failed_files,
                    "metrics": {},
                }
            
            # Stage 1.5: Incremental indexing
            logger.info("Stage 1.5: Filtering unchanged documents...")
            filtered_documents = []
            skipped_count = 0
            
            for doc in all_documents:
                try:
                    source_path = doc.metadata.get("source", "")
                    file_hash = self._compute_file_hash(source_path)
                    
                    if file_hash and self._should_reindex_file(source_path, file_hash):
                        filtered_documents.append(doc)
                        self.manifest[source_path] = file_hash
                    elif file_hash:
                        skipped_count += 1
                except Exception as e:
                    logger.warning(f"Error processing {doc.metadata.get('source')}: {str(e)}")
                    failed_files.append({
                        "file": doc.metadata.get("source", "unknown"),
                        "error": str(e)
                    })
            
            self._save_manifest()
            logger.info(f"  Filtered: {len(filtered_documents)} to process, {skipped_count} skipped")
            
            # Stage 2: Split documents with error resilience
            logger.info("Stage 2: Splitting documents...")
            split_start = time.time()
            splitter = LanguageAwareSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            
            chunks = []
            split_failed = 0
            
            for doc in filtered_documents:
                try:
                    doc_chunks = splitter.split_documents_batch([doc])
                    chunks.extend(doc_chunks)
                except Exception as e:
                    logger.warning(f"Error splitting {doc.metadata.get('source')}: {str(e)}")
                    split_failed += 1
                    failed_files.append({
                        "file": doc.metadata.get("source", "unknown"),
                        "stage": "split",
                        "error": str(e)
                    })
            
            split_time = time.time() - split_start
            logger.info(f"  Created {len(chunks)} chunks in {split_time:.2f}s ({split_failed} failed)")
            
            # Stage 2.5: Enrich chunks
            logger.info("Stage 2.5: Enriching chunks...")
            enrich_start = time.time()
            enriched_chunks = []
            enrich_failed = 0
            
            for chunk in chunks:
                try:
                    enriched = self._enrich_chunk(chunk)
                    enriched["metadata"] = {
                        **enriched.get("metadata", {}),
                        "file_path": enriched["metadata"].get("source", ""),
                        "source_type": "kt_document",
                    }
                    enriched_chunks.append(enriched)
                except Exception as e:
                    logger.warning(f"Error enriching chunk: {str(e)}")
                    enrich_failed += 1
                    enriched_chunks.append(chunk)
            
            enrich_time = time.time() - enrich_start
            logger.info(f"  Enriched {len(enriched_chunks)} chunks in {enrich_time:.2f}s")
            
            # Stage 3: Apply PDR
            logger.info("Stage 3: Applying PDR...")
            pdr_start = time.time()
            parent_store = ParentDocumentStore(backend="memory")
            pdr = PDRStrategy(
                parent_store=parent_store,
                child_chunk_size=self.child_chunk_size,
            )
            enhanced_chunks = pdr.create_child_parent_pairs(enriched_chunks)
            embedding_ready = pdr.prepare_for_embedding(enhanced_chunks)
            pdr_time = time.time() - pdr_start
            logger.info(f"  Created PDR pairs in {pdr_time:.2f}s")
            
            # Stage 4: Generate embeddings
            # ingest_kt_documents handles prose (PDF/markdown/docs) exclusively —
            # intentionally keeps the general-purpose embedder so doc retrieval,
            # semantic cache, and LTM recall are unaffected by the code embedder.
            logger.info("Stage 4: Generating embeddings...")
            embed_start = time.time()
            embedding_engine = EmbeddingEngine(model_name=self.embedding_model)
            texts_to_embed = [chunk["content"] for chunk in embedding_ready]
            embeddings = embedding_engine.embed_batch(texts_to_embed)
            embed_time = time.time() - embed_start
            logger.info(f"  Generated {len(embeddings)} embeddings in {embed_time:.2f}s")
            
            # Stage 5: Store in ChromaDB
            logger.info("Stage 5: Storing in ChromaDB...")
            store_start = time.time()
            vector_store = ChromaVectorStore(
                collection_name="kt_documents",
                persist_directory=self.persist_directory,
            )
            vector_store.add_documents(embedding_ready, embeddings)
            vector_store.persist()
            store_time = time.time() - store_start
            logger.info(f"  Stored in ChromaDB in {store_time:.2f}s")

            # Mirror write to pgvector when VECTOR_STORE_BACKEND includes "pgvector"
            try:
                from app.services.retrieval.pgvector_store import (
                    PgVectorDocumentStore, pgvector_enabled,
                )
                if pgvector_enabled():
                    pg_store = PgVectorDocumentStore()
                    pg_upserted = pg_store.insert_chunks(embedding_ready, embeddings)
                    logger.info(f"  ✓ Also mirrored {pg_upserted} chunks → pgvector document_chunks")
            except Exception as _pg_err:  # noqa: BLE001
                logger.warning(f"  pgvector mirror skipped: {_pg_err}")

            # Calculate metrics
            total_time = time.time() - start_time
            
            result = {
                "status": "success",
                "pipeline": "kt_documents_ingestion",
                "metrics": {
                    "documents_loaded": len(all_documents),
                    "documents_processed": len(filtered_documents),
                    "documents_skipped": skipped_count,
                    "chunks_created": len(chunks),
                    "chunks_enriched": len(enriched_chunks),
                    "parent_documents": parent_store.get_statistics()["total_parents"],
                    "child_chunks_embedded": len(embedding_ready),
                    "embeddings_generated": len(embeddings),
                    "total_time_seconds": total_time,
                    "cost_savings_percent": (skipped_count / len(all_documents) * 100) if all_documents else 0,
                    "stages": {
                        "load": load_time,
                        "filter": time.time() - (start_time + load_time),
                        "split": split_time,
                        "enrich": enrich_time,
                        "pdr": pdr_time,
                        "embed": embed_time,
                        "store": store_time,
                    },
                    "failures": {
                        "split_failures": split_failed,
                        "enrich_failures": enrich_failed,
                    }
                },
                "vector_store": vector_store.get_statistics(),
                "failed_files": failed_files if failed_files else None,
            }
            
            logger.info(f"KT documents ingestion completed in {total_time:.2f}s")
            if failed_files:
                logger.warning(f"  {len(failed_files)} items failed processing")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ingest_kt_documents: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "metrics": {},
            }

    def get_status(self) -> Dict:
        """Get current ingestion status and statistics.
        
        Returns:
            Dictionary with ingestion status
        """
        try:
            from app.services.ingestion.ingestion_service import IngestionService

            collection = IngestionService.get_chroma_collection()
            metadatas = (collection.get(include=["metadatas"]).get("metadatas")) or []
            sources = {
                (m or {}).get("source")
                for m in metadatas
                if (m or {}).get("source")
            }
            chunk_count = collection.count()

            return {
                "status": "ready",
                "documents_indexed": len(sources),
                "total_chunks": chunk_count,
                "num_collections": 1,
                "vector_store": {
                    "collection_name": collection.name,
                    "document_count": chunk_count,
                },
            }
        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def clear_all(self) -> Dict:
        """Clear all ingested documents and reset the system.
        
        Returns:
            Dictionary with clear operation status
        """
        try:
            from app.services.ingestion.chroma_vector_store import ChromaVectorStore
            import shutil
            
            # Clear ChromaDB
            if Path(self.persist_directory).exists():
                shutil.rmtree(self.persist_directory)
                logger.info(f"Cleared ChromaDB: {self.persist_directory}")
            
            # Reset manifest
            self.manifest = {}
            self._save_manifest()
            logger.info("Cleared ingestion manifest")
            
            return {
                "status": "success",
                "message": "All documents cleared",
                "collections_deleted": 0,
                "chunks_deleted": 0,
            }
        except Exception as e:
            logger.error(f"Error clearing documents: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
