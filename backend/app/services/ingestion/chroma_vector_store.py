"""ChromaDB vector store with HuggingFace embeddings."""

import logging
from typing import List, Dict, Optional
import chromadb

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Manage embeddings using HuggingFace models."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
    ):
        """Initialize embedding engine.

        Args:
            model_name: HuggingFace model name for embeddings (used only as a
                fallback identifier — the actual model comes from the shared
                singleton in `app.core.database`).

        G2 FIX — Use the process-wide singleton `get_embedder()` instead of
        loading a second 500 MB model copy. During co-resident ingestion +
        serving (dev / single-container deploys), this saves ~500 MB RSS and
        ~3-5 s of warmup time. The retriever and the ingestion pipeline now
        share one embedder, guaranteeing vector consistency by construction.
        """
        try:
            self.model_name = model_name
            try:
                from app.core.database import get_embedder
                self.embeddings = get_embedder()
                logger.info(
                    f"✅ Ingestion using SHARED singleton embedder "
                    f"(model={model_name})"
                )
            except Exception as singleton_err:
                # Fallback for isolated tests / scripts that import this
                # module without the full app context.
                logger.warning(
                    f"Singleton embedder unavailable ({singleton_err}); "
                    f"falling back to local instance."
                )
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    encode_kwargs={"normalize_embeddings": True},
                )
                logger.info(f"Initialized local embeddings: {model_name}")

        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise RuntimeError(f"Failed to initialize embeddings: {str(e)}")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
            
        Raises:
            RuntimeError: If embedding fails
        """
        try:
            if not text or not isinstance(text, str):
                raise ValueError("Text must be non-empty string")
            
            embedding = self.embeddings.embed_query(text)
            logger.debug(f"Embedded text of length {len(text)}")
            
            return embedding
            
        except ValueError as e:
            logger.error(f"Validation error in embed_text: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise RuntimeError(f"Failed to embed text: {str(e)}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If texts is invalid
        """
        try:
            if not isinstance(texts, list):
                raise ValueError(f"Texts must be list, got {type(texts)}")
            
            if not texts:
                raise ValueError("Texts list cannot be empty")
            
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Embedded batch of {len(texts)} texts")
            
            return embeddings
            
        except ValueError as e:
            logger.error(f"Validation error in embed_batch: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error embedding batch: {str(e)}")
            raise RuntimeError(f"Failed to embed batch: {str(e)}")


class ChromaVectorStore:
    """ChromaDB vector store with persistence."""

    def __init__(
        self,
        collection_name: str = "codelens_ingestion",
        persist_directory: str = "./chroma_db",
    ):
        """Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistent storage
        """
        try:
            # Initialize Chroma client with new PersistentClient API
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection_name = collection_name
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Cosine similarity for embeddings
            )
            
            logger.info(f"Initialized ChromaDB: {collection_name}")
            logger.info(f"Persist directory: {persist_directory}")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise RuntimeError(f"Failed to initialize ChromaDB: {str(e)}")

    def add_documents(
        self,
        documents: List[Dict],
        embeddings: List[List[float]],
    ) -> None:
        """Add documents with embeddings to vector store.
        
        Args:
            documents: List of document chunks
            embeddings: List of corresponding embeddings
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If adding fails
        """
        try:
            # Validate inputs
            if not isinstance(documents, list):
                raise ValueError(f"Documents must be list, got {type(documents)}")
            
            if not isinstance(embeddings, list):
                raise ValueError(f"Embeddings must be list, got {type(embeddings)}")
            
            if len(documents) != len(embeddings):
                raise ValueError(f"Documents ({len(documents)}) and embeddings ({len(embeddings)}) mismatch")
            
            if not documents:
                logger.warning("No documents to add")
                return
            
            # Prepare data for Chroma
            ids = []
            metadatas = []
            documents_list = []
            embeddings_list = []
            
            for doc, embedding in zip(documents, embeddings):
                try:
                    doc_id = doc.get("id", f"doc_{len(ids)}")
                    ids.append(doc_id)
                    
                    metadatas.append(doc.get("metadata", {}))
                    documents_list.append(doc.get("content", ""))
                    embeddings_list.append(embedding)
                    
                except Exception as e:
                    logger.warning(f"Error preparing document: {str(e)}")
                    continue
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents_list,
                metadatas=metadatas,
            )
            
            logger.info(f"Added {len(ids)} documents to ChromaDB")
            
        except ValueError as e:
            logger.error(f"Validation error in add_documents: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to add documents to ChromaDB: {str(e)}")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of search results
            
        Raises:
            ValueError: If inputs are invalid
        """
        try:
            if not isinstance(query_embedding, list):
                raise ValueError("Query embedding must be list")
            
            if top_k <= 0:
                raise ValueError("top_k must be positive")
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )
            
            # Format results
            formatted_results = []
            if results and results.get("ids"):
                for i in range(len(results["ids"][0])):
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results.get("distances", [[]])[0][i] if results.get("distances") else 0,
                    })
            
            logger.debug(f"Search returned {len(formatted_results)} results")
            
            return formatted_results
            
        except ValueError as e:
            logger.error(f"Validation error in search: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            raise RuntimeError(f"Failed to search: {str(e)}")

    def delete_collection(self) -> None:
        """Delete the collection and all data."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise RuntimeError(f"Failed to delete collection: {str(e)}")

    def persist(self) -> None:
        """Persist data to disk."""
        try:
            self.client.persist()
            logger.info("Persisted ChromaDB to disk")
            
        except Exception as e:
            logger.warning(f"Error persisting: {str(e)}")

    def get_statistics(self) -> Dict:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {}
