"""Ingestion Service — Bridge between ingestion pipeline (writes) and
retrieval pipeline (reads).

Exposes:
    - get_chroma_collection() : Live ChromaDB collection for vector search
    - load_documents_for_bm25(): LangChain Documents for lexical retrieval
    - load_parent_store()      : Parent ID -> full content map for PDR
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover
    from langchain.schema import Document  # type: ignore

logger = logging.getLogger(__name__)


# Defaults must match ContextAwareIngestionPipeline + the ingest route
PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DEFAULT_COLLECTION = os.getenv(
    "CHROMA_DEFAULT_COLLECTION", "codelens_ingestion"
)


class IngestionService:
    """Read-side accessor for assets produced by the ingestion pipeline."""

    _client: Optional[chromadb.api.ClientAPI] = None

    # ------------------------------------------------------------------ #
    # ChromaDB                                                            #
    # ------------------------------------------------------------------ #
    @classmethod
    def _get_client(cls) -> chromadb.api.ClientAPI:
        if cls._client is None:
            Path(PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)
            cls._client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
            logger.info(f"ChromaDB PersistentClient ready at {PERSIST_DIRECTORY}")
        return cls._client

    @classmethod
    def get_chroma_collection(
        cls, collection_name: Optional[str] = None
    ):
        """Return the live ChromaDB collection.

        If `collection_name` is None, picks the most recently created
        `documents_*` collection or falls back to DEFAULT_COLLECTION.
        """
        client = cls._get_client()
        target = collection_name

        if target is None:
            try:
                cols = client.list_collections()
                doc_cols = sorted(
                    [c.name for c in cols if c.name.startswith("documents_")],
                    reverse=True,
                )
                target = doc_cols[0] if doc_cols else DEFAULT_COLLECTION
            except Exception as e:
                logger.warning(f"Could not list collections: {e}")
                target = DEFAULT_COLLECTION

        collection = client.get_or_create_collection(
            name=target, metadata={"hnsw:space": "cosine"}
        )
        logger.info(
            f"Loaded ChromaDB collection '{target}' "
            f"({collection.count()} vectors)"
        )
        return collection

    # ------------------------------------------------------------------ #
    # BM25 documents                                                      #
    # ------------------------------------------------------------------ #
    @classmethod
    def load_documents_for_bm25(
        cls, collection_name: Optional[str] = None
    ) -> List[Document]:
        """Reconstruct LangChain Document objects from ChromaDB so BM25 can
        index the same corpus that powers vector search.
        """
        try:
            collection = cls.get_chroma_collection(collection_name)
            data = collection.get(include=["documents", "metadatas"])
            documents_text: List[str] = data.get("documents") or []
            metadatas: List[Dict] = data.get("metadatas") or [{}] * len(
                documents_text
            )
            ids: List[str] = data.get("ids") or [
                f"doc_{i}" for i in range(len(documents_text))
            ]

            docs: List[Document] = []
            for doc_id, text, meta in zip(ids, documents_text, metadatas):
                if not text:
                    continue
                meta = dict(meta or {})
                meta.setdefault("chunk_id", doc_id)
                docs.append(Document(page_content=text, metadata=meta))

            logger.info(f"Built {len(docs)} LangChain Documents for BM25")
            return docs
        except Exception as e:
            logger.error(f"Failed to load BM25 documents: {e}", exc_info=True)
            return []

    # ------------------------------------------------------------------ #
    # Parent store (PDR)                                                  #
    # ------------------------------------------------------------------ #
    @classmethod
    def load_parent_store(
        cls, collection_name: Optional[str] = None
    ) -> Dict[str, str]:
        """Build a parent_id -> parent_content mapping.

        Strategy: each chunk's metadata stores `parent_id` (and optionally
        `parent_content`). We aggregate chunks belonging to the same parent
        and concatenate them as a faithful parent reconstruction.
        """
        try:
            collection = cls.get_chroma_collection(collection_name)
            data = collection.get(include=["documents", "metadatas"])
            documents_text: List[str] = data.get("documents") or []
            metadatas: List[Dict] = data.get("metadatas") or [
                {}
            ] * len(documents_text)

            parent_store: Dict[str, List[str]] = {}
            for text, meta in zip(documents_text, metadatas):
                meta = meta or {}
                parent_id = meta.get("parent_id") or meta.get("source")
                if not parent_id or not text:
                    continue
                parent_store.setdefault(parent_id, []).append(text)

            flattened = {pid: "\n\n".join(parts) for pid, parts in parent_store.items()}
            logger.info(f"Built parent store with {len(flattened)} parents")
            return flattened
        except Exception as e:
            logger.error(f"Failed to load parent store: {e}", exc_info=True)
            return {}


# Convenience module-level functions (factory-friendly)
def get_chroma_collection(collection_name: Optional[str] = None):
    return IngestionService.get_chroma_collection(collection_name)


def load_documents_for_bm25(collection_name: Optional[str] = None) -> List[Document]:
    return IngestionService.load_documents_for_bm25(collection_name)


def load_parent_store(collection_name: Optional[str] = None) -> Dict[str, str]:
    return IngestionService.load_parent_store(collection_name)
