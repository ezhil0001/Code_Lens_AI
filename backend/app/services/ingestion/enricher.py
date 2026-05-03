"""Contextual enrichment for code chunks."""

import logging
from typing import Optional, List
import re

logger = logging.getLogger(__name__)


class ContextualEnricher:
    """Add contextual information to code chunks."""

    # Regular expressions for function/class detection
    FUNCTION_PATTERNS = {
        "python": r"^\s*(async\s+)?def\s+(\w+)\s*\(",
        "typescript": r"^\s*(async\s+)?(public|private|protected)?\s*(\w+)\s*\(",
        "javascript": r"^\s*(async\s+)?function\s+(\w+)\s*\(|^\s*(\w+)\s*=\s*(async\s*)?\(",
        "java": r"^\s*(public|private|protected)?\s*(static)?\s*(\w+)\s+(\w+)\s*\(",
    }

    CLASS_PATTERNS = {
        "python": r"^class\s+(\w+)",
        "typescript": r"^(export\s+)?(abstract\s+)?class\s+(\w+)",
        "javascript": r"^class\s+(\w+)",
        "java": r"^(public|private|protected)?\s*(abstract)?\s*class\s+(\w+)",
    }

    IMPORT_PATTERNS = {
        "python": r"^(import|from)\s+[\w.]+",
        "typescript": r"^import\s+.*from\s+['\"]",
        "javascript": r"^(import|require)\s*",
        "java": r"^import\s+[\w.]+",
    }

    def __init__(self):
        """Initialize the enricher."""
        self.compiled_patterns = {}

    def _compile_patterns(self):
        """Compile regex patterns for performance."""
        if not self.compiled_patterns:
            for language, patterns_dict in {
                "functions": self.FUNCTION_PATTERNS,
                "classes": self.CLASS_PATTERNS,
                "imports": self.IMPORT_PATTERNS,
            }.items():
                self.compiled_patterns[language] = {}
                for lang, pattern in patterns_dict.items():
                    self.compiled_patterns[language][lang] = re.compile(
                        pattern, re.MULTILINE
                    )

    def extract_context_info(self, chunk: dict) -> dict:
        """Extract contextual information from a code chunk.

        Args:
            chunk: Dictionary with 'content' and 'metadata' keys.

        Returns:
            Enhanced chunk with context information.
            
        Raises:
            ValueError: If chunk is invalid.
            RuntimeError: If extraction fails.
        """
        try:
            if not isinstance(chunk, dict):
                raise ValueError(f"Chunk must be a dict, got {type(chunk)}")
            
            if "content" not in chunk or "metadata" not in chunk:
                raise ValueError("Chunk must have 'content' and 'metadata' keys")
            
            self._compile_patterns()

            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})
            
            if not isinstance(content, str):
                raise ValueError(f"Content must be string, got {type(content)}")
            
            if not content.strip():
                logger.warning("Empty content in chunk")
                return {
                    **chunk,
                    "context": {
                        "functions": [],
                        "classes": [],
                        "imports": [],
                        "summary": "Empty chunk",
                    },
                }
            
            language = metadata.get("language", "python").lower()

            try:
                context_info = {
                    "functions": self._extract_functions(content, language),
                    "classes": self._extract_classes(content, language),
                    "imports": self._extract_imports(content, language),
                    "summary": self._generate_summary(content, language),
                }
            except Exception as e:
                logger.error(f"Error extracting context: {str(e)}")
                context_info = {
                    "functions": [],
                    "classes": [],
                    "imports": [],
                    "summary": "Error extracting context",
                }

            return {
                **chunk,
                "context": context_info,
            }
            
        except ValueError as e:
            logger.error(f"Validation error in extract_context_info: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in extract_context_info: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to extract context: {str(e)}")

    def _extract_functions(self, content: str, language: str) -> List[str]:
        """Extract function names from content.

        Args:
            content: Code content.
            language: Programming language.

        Returns:
            List of function names.
        """
        if language not in self.FUNCTION_PATTERNS:
            return []

        pattern = self.compiled_patterns.get("functions", {}).get(language)
        if not pattern:
            return []

        matches = pattern.findall(content)
        functions = []

        for match in matches:
            # Handle different group indices depending on language
            if isinstance(match, tuple):
                # Find first non-empty group
                func_name = next((g for g in match if g and not g.isspace()), None)
            else:
                func_name = match

            if func_name and func_name.strip():
                functions.append(func_name.strip())

        return functions

    def _extract_classes(self, content: str, language: str) -> List[str]:
        """Extract class names from content.

        Args:
            content: Code content.
            language: Programming language.

        Returns:
            List of class names.
        """
        if language not in self.CLASS_PATTERNS:
            return []

        pattern = self.compiled_patterns.get("classes", {}).get(language)
        if not pattern:
            return []

        matches = pattern.findall(content)
        classes = []

        for match in matches:
            # Handle different group indices depending on language
            if isinstance(match, tuple):
                # Find first non-empty group (skip modifiers)
                class_name = next((g for g in match if g and not g.isspace()), None)
            else:
                class_name = match

            if class_name and class_name.strip():
                classes.append(class_name.strip())

        return classes

    def _extract_imports(self, content: str, language: str) -> List[str]:
        """Extract import statements from content.

        Args:
            content: Code content.
            language: Programming language.

        Returns:
            List of import statements.
        """
        if language not in self.IMPORT_PATTERNS:
            return []

        pattern = self.compiled_patterns.get("imports", {}).get(language)
        if not pattern:
            return []

        matches = pattern.findall(content)
        return matches[:10]  # Limit to first 10 imports

    def _generate_summary(self, content: str, language: str) -> str:
        """Generate a brief summary of the chunk.

        Args:
            content: Code content.
            language: Programming language.

        Returns:
            Summary string.
        """
        lines = content.strip().split("\n")
        
        # Find first non-empty, non-comment line
        summary_lines = []
        for line in lines[:5]:  # Look at first 5 lines
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped:
                continue
            if language == "python" and stripped.startswith("#"):
                continue
            if language in ["javascript", "typescript"] and stripped.startswith("//"):
                continue
            
            summary_lines.append(stripped[:100])  # Limit to 100 chars
            
            if len(summary_lines) >= 2:
                break

        return " | ".join(summary_lines) if summary_lines else "Code snippet"

    def enrich_chunks(self, chunks: List[dict]) -> List[dict]:
        """Enrich multiple chunks with contextual information.

        Args:
            chunks: List of chunk dictionaries.

        Returns:
            List of enriched chunks.
            
        Raises:
            ValueError: If chunks list is invalid.
            RuntimeError: If enrichment fails catastrophically.
        """
        try:
            if not isinstance(chunks, list):
                raise ValueError(f"Chunks must be a list, got {type(chunks)}")
            
            if not chunks:
                logger.warning("No chunks to enrich")
                return []
            
            enriched = []
            error_count = 0

            for chunk_idx, chunk in enumerate(chunks):
                try:
                    if not isinstance(chunk, dict):
                        logger.warning(f"Chunk {chunk_idx} is not a dict, skipping")
                        error_count += 1
                        continue
                    
                    enriched_chunk = self.extract_context_info(chunk)
                    enriched.append(enriched_chunk)
                    
                except Exception as e:
                    error_count += 1
                    logger.warning(f"Error enriching chunk {chunk_idx}: {str(e)}")
                    continue

            if error_count > 0:
                logger.warning(f"Enriched {len(enriched)} chunks with {error_count} errors")
            else:
                logger.info(f"Enriched {len(enriched)} chunks with context information")

            return enriched
            
        except ValueError as e:
            logger.error(f"Validation error in enrich_chunks: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in enrich_chunks: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to enrich chunks: {str(e)}")

    def format_for_embedding(self, chunk: dict) -> str:
        """Format a chunk for embedding with context prepended.

        Args:
            chunk: Enriched chunk dictionary.

        Returns:
            Formatted string for embedding.
            
        Raises:
            ValueError: If chunk is invalid.
        """
        try:
            if not isinstance(chunk, dict):
                raise ValueError(f"Chunk must be a dict, got {type(chunk)}")
            
            metadata = chunk.get("metadata", {})
            context = chunk.get("context", {})
            content = chunk.get("content", "")

            if not isinstance(content, str):
                raise ValueError(f"Content must be string, got {type(content)}")

            # Build context prefix
            prefix_parts = []

            try:
                # File information
                source = metadata.get("source", "unknown")
                if source:
                    prefix_parts.append(f"File: {source}")

                # Functions/Classes
                functions = context.get("functions", [])
                classes = context.get("classes", [])
                
                if classes and isinstance(classes, list):
                    prefix_parts.append(f"Classes: {', '.join(classes[:3])}")
                
                if functions and isinstance(functions, list):
                    prefix_parts.append(f"Functions: {', '.join(functions[:3])}")

                # Summary
                summary = context.get("summary", "")
                if summary and isinstance(summary, str):
                    prefix_parts.append(f"Summary: {summary}")
                    
            except Exception as e:
                logger.warning(f"Error building context prefix: {str(e)}")

            # Combine
            prefix = "\n".join(prefix_parts)

            if prefix:
                return f"{prefix}\n\nContent:\n{content}"
            else:
                return content
                
        except ValueError as e:
            logger.error(f"Validation error in format_for_embedding: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in format_for_embedding: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to format chunk: {str(e)}")

    def get_enrichment_stats(self) -> dict:
        """Get statistics about enrichment.

        Returns:
            Statistics dictionary.
        """
        return {
            "supported_languages": list(self.FUNCTION_PATTERNS.keys()),
            "patterns_compiled": bool(self.compiled_patterns),
        }
