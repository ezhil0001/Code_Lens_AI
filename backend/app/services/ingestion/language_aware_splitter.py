"""Language-specific text splitting for code and documents."""

import logging
from typing import List, Optional, Dict
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class LanguageAwareSplitter:
    """Split code and documents while preserving logical boundaries."""

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
    ):
        """Initialize splitter.
        
        Args:
            chunk_size: Size of each chunk in characters (1500 recommended)
            chunk_overlap: Overlap between chunks in characters (200 recommended)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter_cache: Dict = {}
        
        logger.info(f"Initialized splitter: chunk_size={chunk_size}, overlap={chunk_overlap}")

    def _get_language_splitter(self, language: str) -> RecursiveCharacterTextSplitter:
        """Get language-specific splitter with caching.
        
        Args:
            language: Programming language or "document"
            
        Returns:
            Language-specific RecursiveCharacterTextSplitter
        """
        try:
            # Check cache first
            if language in self._splitter_cache:
                logger.debug(f"Using cached splitter for {language}")
                return self._splitter_cache[language]
            
            language = language.lower()
            
            # Try to get language-specific splitter from LangChain
            try:
                # Map string language to Language enum
                language_map = {
                    "python": Language.PYTHON,
                    "typescript": Language.TS,
                    "ts": Language.TS,
                    "tsx": Language.TS,
                    "javascript": Language.JS,
                    "js": Language.JS,
                    "jsx": Language.JS,
                    "java": Language.JAVA,
                    "cpp": Language.CPP,
                    "c": Language.C,
                    "go": Language.GO,
                    "rust": Language.RUST,
                    "rs": Language.RUST,
                    "csharp": Language.CSHARP,
                    "cs": Language.CSHARP,
                    "php": Language.PHP,
                }
                
                lang_enum = language_map.get(language)
                
                if lang_enum:
                    # Create language-specific splitter
                    splitter = RecursiveCharacterTextSplitter.from_language(
                        language=lang_enum,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    )
                    logger.info(f"Created language-specific splitter for {language}")
                else:
                    # Prose / document types (markdown, txt, pdf extracted text)
                    # intentionally use the generic splitter — not a warning.
                    _PROSE_TYPES = {"markdown", "md", "txt", "text", "pdf", "rst", "html"}
                    if language in _PROSE_TYPES:
                        logger.info(f"Using generic splitter for prose type: {language}")
                    else:
                        logger.warning(f"Language {language} not recognized, using generic splitter")
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    )
                
            except Exception as e:
                # Fallback to generic splitter
                logger.warning(f"Error creating language splitter for {language}: {str(e)}")
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
            
            # Cache the splitter
            self._splitter_cache[language] = splitter
            
            return splitter
            
        except Exception as e:
            logger.error(f"Error in _get_language_splitter: {str(e)}")
            # Ultimate fallback
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

    def split_code(self, document, language: str) -> List[Dict]:
        """Split source code preserving functions and classes.
        
        Args:
            document: Document with page_content and metadata
            language: Programming language
            
        Returns:
            List of chunk dictionaries with metadata
            
        Raises:
            ValueError: If document is invalid
            RuntimeError: If splitting fails
        """
        try:
            # Validate document
            if document is None:
                raise ValueError("Document cannot be None")
            
            if not hasattr(document, "page_content"):
                raise ValueError("Document must have page_content attribute")
            
            content = document.page_content
            
            if not content or len(content) == 0:
                logger.debug("Document has no content")
                return []
            
            # Get language-specific splitter
            splitter = self._get_language_splitter(language)
            
            # Split the code
            chunks = splitter.split_text(content)
            logger.info(f"Split {language} code into {len(chunks)} chunks")
            
            # Add metadata to each chunk
            chunks_with_metadata = []
            for i, chunk in enumerate(chunks):
                chunk_dict = {
                    "content": chunk,
                    "is_parent": False,  # Will be marked as parent later
                    "metadata": {
                        **document.metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "language": language,
                        "chunk_type": "code",
                        "content_length": len(chunk),
                        "line_count": len(chunk.split("\n")),
                    }
                }
                chunks_with_metadata.append(chunk_dict)
            
            return chunks_with_metadata
            
        except ValueError as e:
            logger.error(f"Validation error in split_code: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error splitting code: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to split code: {str(e)}")

    def split_document(self, document) -> List[Dict]:
        """Split KT document (PDF, Markdown).
        
        Args:
            document: Document with page_content and metadata
            
        Returns:
            List of chunk dictionaries with metadata
            
        Raises:
            ValueError: If document is invalid
        """
        try:
            # Validate document
            if document is None:
                raise ValueError("Document cannot be None")
            
            if not hasattr(document, "page_content"):
                raise ValueError("Document must have page_content attribute")
            
            content = document.page_content
            
            if not content or len(content) == 0:
                logger.debug("Document has no content")
                return []
            
            # Use generic splitter for documents
            splitter = self._get_language_splitter("document")
            
            # Split the document
            chunks = splitter.split_text(content)
            logger.info(f"Split document into {len(chunks)} chunks")
            
            # Add metadata to each chunk
            chunks_with_metadata = []
            for i, chunk in enumerate(chunks):
                chunk_dict = {
                    "content": chunk,
                    "is_parent": False,
                    "metadata": {
                        **document.metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_type": "document",
                        "content_length": len(chunk),
                        "word_count": len(chunk.split()),
                    }
                }
                chunks_with_metadata.append(chunk_dict)
            
            return chunks_with_metadata
            
        except ValueError as e:
            logger.error(f"Validation error in split_document: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error splitting document: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to split document: {str(e)}")

    def split_documents_batch(self, documents: List) -> List[Dict]:
        """Split multiple documents (both code and KT docs).
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of all chunks from all documents
            
        Raises:
            ValueError: If documents is not a list
        """
        try:
            if not isinstance(documents, list):
                raise ValueError(f"Documents must be list, got {type(documents)}")
            
            all_chunks = []
            error_count = 0
            
            for doc_idx, doc in enumerate(documents):
                try:
                    file_type = doc.metadata.get("file_type", "document")
                    language = doc.metadata.get("language", "unknown")
                    
                    if file_type == "code":
                        chunks = self.split_code(doc, language)
                    else:
                        chunks = self.split_document(doc)
                    
                    all_chunks.extend(chunks)
                    logger.debug(f"Processed document {doc_idx}: {len(chunks)} chunks")
                    
                except Exception as e:
                    logger.warning(f"Error splitting document {doc_idx}: {str(e)}")
                    error_count += 1
                    continue
            
            logger.info(f"Split batch: {len(all_chunks)} total chunks, {error_count} errors")
            
            return all_chunks
            
        except ValueError as e:
            logger.error(f"Validation error in split_documents_batch: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error in split_documents_batch: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to split documents batch: {str(e)}")
