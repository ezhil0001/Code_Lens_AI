"""
Document Ingestion Route Handler (FastAPI Endpoint Layer)

This module provides HTTP endpoints that delegate to the service layer.
All ingestion logic is orchestrated by ContextAwareIngestionPipeline in services.

Flow: HTTP Request → Route Handler → Service Layer → Detailed Logging
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status, BackgroundTasks, Depends
from typing import List, Dict, Optional
import ipaddress
import logging
import socket
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
import tempfile
import shutil

# Import service layer
from app.services.ingestion import (
    ContextAwareIngestionPipeline,
    MultiModalLoader,
)
from app.core.config import get_settings

router = APIRouter(
    prefix="/api/v1/ingest",
    tags=["ingestion"]
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auth dependencies (C-2)
# ---------------------------------------------------------------------------

def _resolve_user_dep():
    """Return the real JWT dependency (app.routes.auth.get_current_user).

    Falls back to an anonymous stub ONLY when the auth stack cannot be
    imported (stripped test environments without python-jose / DB).
    """
    try:
        from app.routes.auth import get_current_user  # type: ignore
        return get_current_user
    except Exception:  # noqa: BLE001
        logger.warning("[ingest] auth stack unavailable — anonymous dependency (non-prod only)")
        async def _anon():
            class _User:
                id = "anonymous"
                email = "anonymous@local"
                role = None
            return _User()
        return _anon


_current_user_dep = _resolve_user_dep()


def _is_admin(user) -> bool:
    """True when the user's primary role is admin/superadmin."""
    role = getattr(user, "role", None)
    role_name = getattr(role, "name", None) or (role if isinstance(role, str) else None)
    return str(role_name).lower() in ("admin", "superadmin")


async def _require_admin(current_user=Depends(_current_user_dep)):
    """Dependency: authenticated AND admin/superadmin. 403 otherwise."""
    if not _is_admin(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required for this operation",
        )
    return current_user


# ---------------------------------------------------------------------------
# SSRF protection (C-3)
# ---------------------------------------------------------------------------

_ALLOWED_URL_SCHEMES = ("http", "https")


def _is_forbidden_ip(ip: str) -> bool:
    """True for loopback, private (RFC1918), link-local, reserved, and
    IPv6-local ranges — i.e. anything that could reach internal services
    or cloud metadata endpoints (169.254.169.254)."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True  # unparseable → reject
    return (
        addr.is_loopback
        or addr.is_private
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def validate_ingest_url(url: str) -> str:
    """Validate a user-supplied URL against SSRF. Raises HTTPException(400).

    Checks: scheme allowlist, hostname presence, and DNS resolution of the
    host — EVERY resolved address must be public. This blocks localhost,
    127.0.0.1, ::1, RFC1918, link-local/metadata (169.254.x.x), and DNS
    names that resolve to internal addresses.
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() not in _ALLOWED_URL_SCHEMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"URL scheme {parsed.scheme!r} not allowed (http/https only)",
        )
    host = parsed.hostname
    if not host:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL has no hostname",
        )
    # Literal IP fast path
    try:
        if _is_forbidden_ip(host):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="URL resolves to a forbidden (internal/private) address",
            )
        # host was a valid, public literal IP
        ipaddress.ip_address(host)
        return url
    except ValueError:
        pass  # hostname, not an IP literal → resolve via DNS
    try:
        infos = socket.getaddrinfo(host, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"URL hostname could not be resolved: {exc}",
        )
    for info in infos:
        ip = info[4][0]
        if _is_forbidden_ip(ip):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="URL resolves to a forbidden (internal/private) address",
            )
    return url



# Global pipeline instance (initialized on first use)
_pipeline_instance: Optional[ContextAwareIngestionPipeline] = None


def _get_pipeline() -> ContextAwareIngestionPipeline:
    """
    Get or initialize the ingestion pipeline (singleton pattern).
    
    Returns:
        Initialized ContextAwareIngestionPipeline instance
    """
    global _pipeline_instance
    
    if _pipeline_instance is None:
        logger.info("🏗️  Initializing ContextAwareIngestionPipeline...")
        _pipeline_instance = ContextAwareIngestionPipeline(
            chunk_size=1500,
            chunk_overlap=200,
            child_chunk_size=400,
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            persist_directory="./chroma_db",
            manifest_path="./manifest.json",
            enable_incremental_indexing=True,
            enable_enrichment=True,
        )
        logger.info("✅ ContextAwareIngestionPipeline ready")
    
    return _pipeline_instance


@router.post("/documents")
async def ingest_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    current_user=Depends(_current_user_dep),
):
    """
    Upload and ingest documents through the service layer.
    
    FLOW:
    1. Validate file types
    2. Save files to temp directory
    3. Delegate to ContextAwareIngestionPipeline
    4. Pipeline orchestrates: Load → Split (language-aware) → PDR → Embed → Store
    
    Supported formats: MD, TXT, PDF, PY, TS, JS, JAVA, CPP, C, H, GO, RS,
    HTML, CSS, JSON, YAML, TOML, SH
    
    Args:
        files: List of files to ingest
        
    Returns:
        Ingestion status with session details and logging information
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )

    ingestion_session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logger.info("=" * 100)
    logger.info(f"🚀 DOCUMENT INGESTION SESSION STARTED: {ingestion_session_id}")
    logger.info(f"   Files to ingest: {len(files)}")
    for idx, file in enumerate(files, 1):
        logger.info(f"      {idx}. {file.filename}")
    logger.info("=" * 100)

    try:
        # Get ingestion pipeline
        pipeline = _get_pipeline()
        
        # Create temp directory for uploaded files
        temp_dir = tempfile.mkdtemp(prefix=f"ingest_{ingestion_session_id}_")
        logger.info(f"� Created temporary directory: {temp_dir}")
        
        allowed_extensions = {
            '.md', '.txt', '.pdf',
            '.py', '.ts', '.js', '.java', '.cpp', '.c', '.h', '.go', '.rs',
            '.html', '.css', '.json', '.yaml', '.yml', '.toml', '.sh',
        }
        
        saved_files = []
        errors = []
        
        # Save uploaded files to temp directory
        logger.info("💾 Saving uploaded files to temporary directory...")
        for file_idx, file in enumerate(files, 1):
            try:
                file_ext = Path(file.filename).suffix.lower()
                logger.info(f"   [{file_idx}/{len(files)}] Processing '{file.filename}'...")
                
                if file_ext not in allowed_extensions:
                    error_msg = f"Unsupported file type: {file_ext}"
                    logger.warning(f"      ✗ {error_msg}")
                    errors.append(f"{file.filename}: {error_msg}")
                    continue
                
                # Read and save file
                content = await file.read()
                file_path = Path(temp_dir) / file.filename
                
                # Write file synchronously (acceptable for temp file operations)
                with open(file_path, "wb") as f:
                    f.write(content)
                
                logger.info(f"      ✓ Saved {len(content)} bytes to {file_path}")
                saved_files.append(str(file_path))
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"      ✗ Error saving file: {error_msg}")
                errors.append(f"{file.filename}: {error_msg}")
        
        if not saved_files:
            logger.error("❌ No valid files were saved")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid files provided for ingestion"
            )
        
        logger.info(f"✅ Successfully saved {len(saved_files)} files")
        
        # Delegate to service layer pipeline
        logger.info("\n" + "=" * 100)
        logger.info("📊 INVOKING SERVICE LAYER: ContextAwareIngestionPipeline")
        logger.info("=" * 100)
        logger.info("INGESTION FLOW:")
        logger.info("  1️⃣  MultiModalLoader   → Detect & load files")
        logger.info("  2️⃣  LanguageAwareSplitter → Function-boundary split + code awareness")
        logger.info("  3️⃣  PDRStrategy (Parent Document Retrieval) → Store parent context")
        logger.info("  4️⃣  EmbeddingEngine (all-mpnet-base-v2) → Generate embeddings")
        logger.info("  5️⃣  ChromaVectorStore → Store in vector database")
        logger.info("  6️⃣  HashManager → Incremental indexing tracking")
        logger.info("=" * 100)
        
        try:
            # Run the pipeline - this handles ALL logging internally
            result = pipeline.ingest(
                source_paths=saved_files,
                source_type="file_system",
                enrichment_enabled=True,
            )
            
            logger.info("\n" + "=" * 100)
            logger.info(f"✅ SERVICE LAYER INGESTION COMPLETED")
            logger.info(f"   Ingestion Session: {ingestion_session_id}")
            logger.info(f"   Result Status: {result.get('status', 'unknown')}")
            logger.info(f"   Documents Indexed: {result.get('documents_indexed', 0)}")
            logger.info(f"   Chunks Created: {result.get('chunks_created', 0)}")
            logger.info(f"   Parent Docs Stored: {result.get('parent_docs_stored', 0)}")
            if 'collection_id' in result:
                logger.info(f"   ChromaDB Collection: {result['collection_id']}")
            logger.info("=" * 100)
            
        except Exception as e:
            logger.error(f"❌ SERVICE LAYER ERROR: {str(e)}", exc_info=True)
            raise
        
        # Schedule BM25 index rebuild as a background task so the HTTP response
        # is not blocked.  The retriever will serve lexical queries with the
        # previous (stale) index and log a warning while the rebuild runs.
        def _rebuild_bm25() -> None:
            try:
                from app.services.pipeline_factory import get_pipeline_factory
                get_pipeline_factory().refresh_bm25_index()
            except Exception as _bg_err:
                logger.error(f"[BM25_REBUILD] Background task error: {_bg_err}", exc_info=True)

        background_tasks.add_task(_rebuild_bm25)
        logger.info("[BM25_REBUILD] BM25 index rebuild scheduled as background task")

        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"🧹 Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"⚠️  Could not clean temp directory: {str(e)}")
        
        return {
            "status": result.get("status", "success"),
            "ingestion_session_id": ingestion_session_id,
            "files_ingested": len(saved_files),
            "total_files": len(files),
            "errors": errors if errors else None,
            "ingestion_details": {
                "documents_indexed": result.get("documents_indexed", 0),
                "chunks_created": result.get("chunks_created", 0),
                "parent_docs_stored": result.get("parent_docs_stored", 0),
                "collection_id": result.get("collection_id"),
                "processing_time_seconds": result.get("processing_time_seconds", 0),
            },
            "message": f"Successfully ingested {len(saved_files)}/{len(files)} files"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 CRITICAL: Document ingestion failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )



# ==================== Helper Functions (Delegate to Service Layer) ====================

# Note: PDF extraction, embedding generation, and all processing now handled by
# ContextAwareIngestionPipeline in the service layer. These helpers are legacy
# and kept only for reference - they are NOT used in the refactored flow.


@router.post("/url")
async def ingest_url(
    background_tasks: BackgroundTasks,
    url_data: dict,
    current_user=Depends(_current_user_dep),
):
    """
    Ingest content from a URL through the service layer.
    
    FLOW:
    1. Validate and fetch URL
    2. Save content to temp file
    3. Delegate to ContextAwareIngestionPipeline
    4. Pipeline orchestrates all processing
    
    Args:
        url_data: Dictionary with 'url' key containing the URL to ingest
        
    Returns:
        Ingestion status
    """
    ingestion_session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logger.info("=" * 100)
    logger.info(f"🚀 URL INGESTION SESSION STARTED: {ingestion_session_id}")
    logger.info("=" * 100)
    
    try:
        ingest_url = url_data.get("url", "").strip()
        logger.info(f"📍 Target URL: {ingest_url}")
        
        if not ingest_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="URL is required"
            )

        # C-3: SSRF protection — scheme allowlist + DNS resolution check
        # rejecting loopback/private/link-local/metadata addresses.
        validate_ingest_url(ingest_url)
        
        # Get ingestion pipeline
        pipeline = _get_pipeline()
        
        logger.info("\n" + "=" * 100)
        logger.info("📊 INVOKING SERVICE LAYER: ContextAwareIngestionPipeline")
        logger.info("=" * 100)
        logger.info("URL INGESTION FLOW:")
        logger.info("  1️⃣  MultiModalLoader   → Fetch & detect URL type")
        logger.info("  2️⃣  LanguageAwareSplitter → Semantic chunk splitting")
        logger.info("  3️⃣  PDRStrategy (Parent Document Retrieval) → Store parent context")
        logger.info("  4️⃣  EmbeddingEngine (all-mpnet-base-v2) → Generate embeddings")
        logger.info("  5️⃣  ChromaVectorStore → Store in vector database")
        logger.info("=" * 100)
        
        try:
            # Delegate to service layer
            result = pipeline.ingest(
                source_paths=[ingest_url],
                source_type="web_url",
                enrichment_enabled=True,
            )
            
            logger.info("\n" + "=" * 100)
            logger.info(f"✅ SERVICE LAYER URL INGESTION COMPLETED")
            logger.info(f"   Ingestion Session: {ingestion_session_id}")
            logger.info(f"   URL: {ingest_url}")
            logger.info(f"   Result Status: {result.get('status', 'unknown')}")
            logger.info(f"   Chunks Created: {result.get('chunks_created', 0)}")
            logger.info(f"   Collection ID: {result.get('collection_id')}")
            logger.info("=" * 100)
            
        except Exception as e:
            logger.error(f"❌ SERVICE LAYER ERROR: {str(e)}", exc_info=True)
            raise
        
        # Schedule BM25 rebuild so newly ingested URL content is immediately
        # visible to lexical search without a process restart.
        def _rebuild_bm25_url() -> None:
            try:
                from app.services.pipeline_factory import get_pipeline_factory
                get_pipeline_factory().refresh_bm25_index()
            except Exception as _bg_err:
                logger.error(f"[BM25_REBUILD] URL background task error: {_bg_err}", exc_info=True)

        background_tasks.add_task(_rebuild_bm25_url)
        logger.info("[BM25_REBUILD] BM25 index rebuild scheduled (URL ingest)")

        return {
            "status": result.get("status", "success"),
            "ingestion_session_id": ingestion_session_id,
            "url": ingest_url,
            "ingestion_details": {
                "chunks_created": result.get("chunks_created", 0),
                "parent_docs_stored": result.get("parent_docs_stored", 0),
                "collection_id": result.get("collection_id"),
                "processing_time_seconds": result.get("processing_time_seconds", 0),
            },
            "message": "Successfully ingested URL"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 CRITICAL: URL ingestion failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"URL ingestion failed: {str(e)}"
        )


@router.get("/status")
async def get_ingestion_status(current_user=Depends(_current_user_dep)):
    """
    Get ingestion status from the service layer.
    
    Returns current statistics about indexed documents and chunks from the
    ContextAwareIngestionPipeline.
    
    Returns:
        Current ingestion statistics with collection details
    """
    try:
        logger.info("📊 Fetching ingestion status from service layer...")
        
        pipeline = _get_pipeline()
        
        # Get status from service layer
        status_dict = pipeline.get_status()
        
        logger.info(f"✓ Status retrieved:")
        logger.info(f"   Documents indexed: {status_dict.get('documents_indexed', 0)}")
        logger.info(f"   Total chunks: {status_dict.get('total_chunks', 0)}")
        logger.info(f"   Collections: {status_dict.get('num_collections', 0)}")
        
        return status_dict
        
    except Exception as e:
        logger.error(f"Error getting ingestion status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve ingestion status"
        )


@router.delete("/clear")
async def clear_documents(current_user=Depends(_require_admin)):
    """
    Clear all ingested documents through the service layer.

    C-2: destructive — restricted to admin/superadmin users.
    
    Delegates to ContextAwareIngestionPipeline to clear ChromaDB collections
    and reset ingestion tracking.
    
    Returns:
        Clear operation status with details
    """
    logger.info("=" * 100)
    logger.info("🗑️  CLEARING ALL INGESTED DOCUMENTS")
    logger.info("=" * 100)
    
    try:
        pipeline = _get_pipeline()
        
        logger.info("📂 Clearing all ChromaDB collections via service layer...")
        result = pipeline.clear_all()
        
        logger.info("=" * 100)
        logger.info(f"✅ CLEANUP COMPLETED")
        logger.info(f"   Collections deleted: {result.get('collections_deleted', 0)}")
        logger.info(f"   Chunks deleted: {result.get('chunks_deleted', 0)}")
        logger.info("=" * 100)
        
        return {
            "status": "success",
            "message": "All documents cleared through service layer",
            "details": result
        }
        
    except Exception as e:
        logger.error(f"💥 Error clearing documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not clear documents"
        )
