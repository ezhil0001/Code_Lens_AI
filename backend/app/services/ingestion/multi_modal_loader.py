"""Multi-modal document loader for KT docs and source code."""

import logging
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# Centralized debug logger — gives [FILE UPLOAD] / [SPLITTING] play-by-play
# in the terminal during ingestion. The plain stdlib `logger` below is kept
# (and bridged into loguru via app.core.logger) so all existing call-sites
# remain functional and consistently formatted.
from app.core.logger import logger as flow_logger, timed, log_step, log_success

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata for loaded documents."""
    file_path: str
    file_type: str  # "code" or "kt_doc"
    language: Optional[str]  # "python", "typescript", etc.
    file_size: int
    loaded_at: str
    line_count: Optional[int] = None


class MultiModalLoader:
    """Load both KT documents (PDF/Markdown) and source code files."""

    # Supported code file extensions
    CODE_EXTENSIONS = {
        ".js": Language.JS,
        ".ts": Language.TS,
        ".tsx": Language.TS,
        ".py": Language.PYTHON,
        ".java": Language.JAVA,
        ".cpp": Language.CPP,
        ".c": Language.C,
        ".go": Language.GO,
        ".rs": Language.RUST,
        ".cs": Language.CSHARP,
        ".php": Language.PHP,
    }

    # Supported KT document extensions
    KT_DOC_EXTENSIONS = {".pdf", ".md", ".txt"}

    def __init__(self, config: Optional[dict] = None):
        """Initialize multi-modal loader.
        
        Args:
            config: Configuration dictionary with paths and patterns.
        """
        self.config = config or {}
        self.loaded_documents: List = []
        self.document_metadata: List[DocumentMetadata] = []

    def load_source_code(
        self,
        directory_path: str,
        file_patterns: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List:
        """Load source code files with language-specific parsing.
        
        Args:
            directory_path: Path to directory containing source files
            file_patterns: File patterns to match (e.g., ["*.py", "*.ts"])
            recursive: Whether to search recursively
            
        Returns:
            List of loaded code documents with metadata
            
        Raises:
            ValueError: If directory doesn't exist or is invalid
            RuntimeError: If loading fails completely
        """
        try:
            directory_path = Path(directory_path).resolve()
            
            if not directory_path.exists():
                raise ValueError(f"Directory does not exist: {directory_path}")
            
            if not directory_path.is_dir():
                raise ValueError(f"Path is not a directory: {directory_path}")
            
            if not file_patterns:
                file_patterns = ["*.py", "*.ts", "*.tsx", "*.js", "*.java", "*.cpp"]
            
            log_step("[FILE UPLOAD]", f"Scanning source tree: {directory_path}")
            log_step("[FILE UPLOAD]", f"Glob patterns: {file_patterns}  recursive={recursive}")

            all_documents = []
            errors = []

            with timed("[FILE UPLOAD]") as ctx:
                # Load each file pattern
                for pattern in file_patterns:
                    try:
                        loader = DirectoryLoader(
                            path=str(directory_path),
                            glob=f"**/{pattern}" if recursive else pattern,
                            silent_errors=True,
                            show_progress=True,
                        )

                        docs = loader.load()
                        flow_logger.bind(tag="[FILE UPLOAD]").debug(
                            f"pattern={pattern} → {len(docs)} file(s)"
                        )

                        # Add file_type and language metadata + emit per-file SPLITTING tag
                        for doc in docs:
                            file_path = doc.metadata.get("source", "")
                            ext = Path(file_path).suffix
                            lang_enum = self.CODE_EXTENSIONS.get(ext)
                            # Convert Language enum to string value for metadata storage
                            lang_str = lang_enum.value if lang_enum else ext.lstrip(".") or "unknown"
                            doc.metadata["file_type"] = "code"
                            doc.metadata["language"] = lang_str
                            doc.metadata["line_count"] = len(doc.page_content.split("\n"))
                            flow_logger.bind(tag="[SPLITTING]").debug(
                                f"strategy=language-aware lang={lang_str} "
                                f"file={Path(file_path).name} lines={doc.metadata['line_count']}"
                            )

                        all_documents.extend(docs)

                    except Exception as e:
                        error_msg = f"Error loading {pattern}: {str(e)}"
                        flow_logger.bind(tag="[FILE UPLOAD]").error(error_msg)
                        errors.append(error_msg)
                        continue

                ctx["files"] = len(all_documents)
                ctx["patterns"] = len(file_patterns)

            if not all_documents and errors:
                error_summary = " | ".join(errors)
                raise RuntimeError(f"Failed to load any source files. Errors: {error_summary}")

            self.loaded_documents.extend(all_documents)
            log_success("[FILE UPLOAD]", f"Loaded {len(all_documents)} source file(s) from {directory_path.name}")

            return all_documents
            
        except (ValueError, RuntimeError) as e:
            flow_logger.bind(tag="[FILE UPLOAD]").error(f"Validation error in load_source_code: {e}")
            raise
        except Exception as e:
            flow_logger.bind(tag="[FILE UPLOAD]").opt(exception=True).error(
                f"Unexpected error in load_source_code: {e}"
            )
            raise RuntimeError(f"Failed to load source code: {str(e)}")

    def load_kt_documents(
        self,
        directory_path: str,
        file_patterns: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List:
        """Load Knowledge Transfer documents (PDF, Markdown, etc.).
        
        Args:
            directory_path: Path to directory containing KT documents
            file_patterns: File patterns to match (e.g., ["*.pdf", "*.md"])
            recursive: Whether to search recursively
            
        Returns:
            List of loaded KT documents with metadata
            
        Raises:
            ValueError: If directory doesn't exist or is invalid
        """
        try:
            directory_path = Path(directory_path).resolve()
            
            if not directory_path.exists():
                raise ValueError(f"Directory does not exist: {directory_path}")
            
            if not directory_path.is_dir():
                raise ValueError(f"Path is not a directory: {directory_path}")
            
            if not file_patterns:
                file_patterns = ["*.pdf", "*.md", "*.txt"]
            
            log_step("[FILE UPLOAD]", f"Scanning KT directory: {directory_path}")
            log_step("[FILE UPLOAD]", f"KT patterns: {file_patterns}  recursive={recursive}")
            
            all_documents = []
            errors = []
            
            # Load PDF files
            if any(p.endswith(".pdf") for p in file_patterns):
                try:
                    with timed("[FILE UPLOAD] pdf") as ctx:
                        pdf_loader = DirectoryLoader(
                            path=str(directory_path),
                            glob="**/*.pdf" if recursive else "*.pdf",
                            loader_cls=PyPDFLoader,
                            silent_errors=True,
                        )
                        pdf_docs = pdf_loader.load()
                        ctx["count"] = len(pdf_docs)

                    for doc in pdf_docs:
                        doc.metadata["file_type"] = "kt_doc"
                        doc.metadata["language"] = "markdown"  # Use markdown for PDF text content
                        flow_logger.bind(tag="[SPLITTING]").debug(
                            f"strategy=markdown-prose source=pdf "
                            f"file={Path(doc.metadata.get('source','?')).name} "
                            f"chars={len(doc.page_content)}"
                        )

                    all_documents.extend(pdf_docs)
                    
                except Exception as e:
                    error_msg = f"Error loading PDF files: {str(e)}"
                    flow_logger.bind(tag="[FILE UPLOAD]").error(error_msg)
                    errors.append(error_msg)
            
            # Load Markdown/Text files
            for pattern in file_patterns:
                if pattern.endswith((".md", ".txt")):
                    try:
                        with timed(f"[FILE UPLOAD] {pattern}") as ctx:
                            loader = DirectoryLoader(
                                path=str(directory_path),
                                glob=f"**/{pattern}" if recursive else pattern,
                                loader_cls=TextLoader,
                                loader_kwargs={"encoding": "utf-8"},
                                silent_errors=True,
                                show_progress=True,
                            )
                            docs = loader.load()
                            ctx["count"] = len(docs)

                        for doc in docs:
                            doc.metadata["file_type"] = "kt_doc"
                            doc.metadata["language"] = pattern.lstrip("*.")
                            flow_logger.bind(tag="[SPLITTING]").debug(
                                f"strategy=markdown-prose ext={pattern.lstrip('*.')} "
                                f"file={Path(doc.metadata.get('source','?')).name} "
                                f"chars={len(doc.page_content)}"
                            )

                        all_documents.extend(docs)
                        
                    except Exception as e:
                        error_msg = f"Error loading {pattern}: {str(e)}"
                        flow_logger.bind(tag="[FILE UPLOAD]").error(error_msg)
                        errors.append(error_msg)
                        continue
            
            if not all_documents and errors:
                error_summary = " | ".join(errors)
                flow_logger.bind(tag="[FILE UPLOAD]").warning(
                    f"No KT documents loaded. Errors: {error_summary}"
                )
                return []
            
            self.loaded_documents.extend(all_documents)
            log_success("[FILE UPLOAD]", f"Loaded {len(all_documents)} KT document(s)")
            
            return all_documents
            
        except ValueError as e:
            flow_logger.bind(tag="[FILE UPLOAD]").error(f"Validation error in load_kt_documents: {e}")
            raise
        except Exception as e:
            flow_logger.bind(tag="[FILE UPLOAD]").opt(exception=True).error(
                f"Unexpected error in load_kt_documents: {e}"
            )
            raise RuntimeError(f"Failed to load KT documents: {str(e)}")

    def load_all(
        self,
        code_directory: Optional[str] = None,
        kt_directory: Optional[str] = None,
        code_patterns: Optional[List[str]] = None,
        kt_patterns: Optional[List[str]] = None,
    ) -> Dict[str, List]:
        """Load both code and KT documents.
        
        Args:
            code_directory: Directory containing source code
            kt_directory: Directory containing KT documents
            code_patterns: Patterns for code files
            kt_patterns: Patterns for KT documents
            
        Returns:
            Dictionary with 'code' and 'kt_docs' keys
        """
        try:
            result = {"code": [], "kt_docs": []}
            
            # Load source code
            if code_directory:
                try:
                    code_docs = self.load_source_code(
                        code_directory,
                        code_patterns,
                        recursive=True
                    )
                    result["code"] = code_docs
                except Exception as e:
                    logger.error(f"Failed to load source code: {str(e)}")
            
            # Load KT documents
            if kt_directory:
                try:
                    kt_docs = self.load_kt_documents(
                        kt_directory,
                        kt_patterns,
                        recursive=True
                    )
                    result["kt_docs"] = kt_docs
                except Exception as e:
                    logger.error(f"Failed to load KT documents: {str(e)}")
            
            logger.info(f"Total loaded: {len(result['code'])} code, {len(result['kt_docs'])} KT docs")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in load_all: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load documents: {str(e)}")

    def get_statistics(self) -> dict:
        """Get loading statistics."""
        code_docs = [d for d in self.loaded_documents if d.metadata.get("file_type") == "code"]
        kt_docs = [d for d in self.loaded_documents if d.metadata.get("file_type") == "kt_doc"]
        
        total_chars = sum(len(doc.page_content) for doc in self.loaded_documents)
        
        return {
            "total_documents": len(self.loaded_documents),
            "code_documents": len(code_docs),
            "kt_documents": len(kt_docs),
            "total_characters": total_chars,
            "average_doc_size": total_chars / len(self.loaded_documents) if self.loaded_documents else 0,
        }
