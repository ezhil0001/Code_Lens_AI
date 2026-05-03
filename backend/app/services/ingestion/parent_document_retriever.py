"""Parent Document Retrieval (PDR) strategy for context-aware retrieval."""

import ast
import logging
import re
import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


# ---- Function/class boundary extractors -------------------------------------

# JS/TS: covers `function foo(...)`, `const foo = (...) =>`, `class Foo {`,
# and `foo(...) {` method-style declarations. Best-effort regex (real parsing
# would require @babel/parser); good enough for chunk grouping.
_JS_BOUNDARY_RE = re.compile(
    r"^(?:export\s+)?(?:async\s+)?(?:function\s+(?P<fn>\w+)|class\s+(?P<cls>\w+)|"
    r"const\s+(?P<arrow>\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)",
    re.MULTILINE,
)


def _extract_python_parents(source: str) -> List[Tuple[str, str, int, int]]:
    """Return [(name, body_text, start_line, end_line)] for each top-level
    function/class in `source`. Falls back to empty list on parse error."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    parents: List[Tuple[str, str, int, int]] = []
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = (getattr(node, "lineno", 1) or 1) - 1
            end = (getattr(node, "end_lineno", start + 1) or (start + 1))
            body = "\n".join(lines[start:end])
            parents.append((node.name, body, start + 1, end))
    return parents


def _extract_js_parents(source: str) -> List[Tuple[str, str, int, int]]:
    """Best-effort JS/TS function/class extraction by scanning for boundary
    declarations and balancing braces from the matched line."""
    parents: List[Tuple[str, str, int, int]] = []
    lines = source.splitlines()
    line_starts = [0]
    for ln in lines:
        line_starts.append(line_starts[-1] + len(ln) + 1)

    for m in _JS_BOUNDARY_RE.finditer(source):
        name = m.group("fn") or m.group("cls") or m.group("arrow") or "anon"
        decl_start = m.start()
        # Find first '{' after declaration; if none (arrow expression), take
        # 30 lines as a heuristic block.
        brace_idx = source.find("{", decl_start)
        if brace_idx == -1:
            start_line = source.count("\n", 0, decl_start)
            end_line = min(start_line + 30, len(lines))
            body = "\n".join(lines[start_line:end_line])
            parents.append((name, body, start_line + 1, end_line))
            continue
        depth = 0
        end_idx = brace_idx
        for i in range(brace_idx, len(source)):
            ch = source[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        start_line = source.count("\n", 0, decl_start)
        end_line = source.count("\n", 0, end_idx) + 1
        body = "\n".join(lines[start_line:end_line])
        parents.append((name, body, start_line + 1, end_line))
    return parents


def extract_function_level_parents(
    source: str, language: str
) -> List[Tuple[str, str, int, int]]:
    """Public entry point: dispatch to language-specific parent extractor.

    Returns list of (parent_name, parent_body, start_line, end_line). When
    no parents can be detected (unknown language, parse error, plain doc),
    returns an empty list so the caller can fall back to file-level grouping.
    """
    if not source:
        return []
    lang = (language or "").lower()
    if lang in {"python", "py"}:
        return _extract_python_parents(source)
    if lang in {"javascript", "js", "typescript", "ts", "tsx", "jsx"}:
        return _extract_js_parents(source)
    return []


class ParentDocumentStore:
    """Store parent documents (full functions/pages) with their child chunks."""

    def __init__(self, backend: str = "memory"):
        """Initialize parent document store.
        
        Args:
            backend: Storage backend - "memory" for InMemoryStore, "redis" for Redis
        """
        self.backend = backend
        self.parent_store: Dict[str, Dict] = {}  # In-memory storage
        logger.info(f"Initialized ParentDocumentStore with backend: {backend}")

    def add_parent(self, parent_id: str, content: str, metadata: Dict) -> None:
        """Store parent document.
        
        Args:
            parent_id: Unique identifier for parent document
            content: Full content of parent (e.g., entire function)
            metadata: Metadata dictionary
        """
        try:
            self.parent_store[parent_id] = {
                "content": content,
                "metadata": metadata,
                "stored_at": datetime.now().isoformat(),
                "child_count": 0,
            }
            logger.debug(f"Added parent: {parent_id}")
            
        except Exception as e:
            logger.error(f"Error adding parent {parent_id}: {str(e)}")
            raise

    def add_child(self, parent_id: str, child_id: str) -> None:
        """Link child chunk to parent document.
        
        Args:
            parent_id: Parent document ID
            child_id: Child chunk ID
        """
        try:
            if parent_id in self.parent_store:
                self.parent_store[parent_id]["child_count"] += 1
                logger.debug(f"Linked child {child_id} to parent {parent_id}")
            else:
                logger.warning(f"Parent {parent_id} not found")
                
        except Exception as e:
            logger.error(f"Error linking child to parent: {str(e)}")
            raise

    def get_parent(self, parent_id: str) -> Optional[Dict]:
        """Retrieve parent document.
        
        Args:
            parent_id: Parent document ID
            
        Returns:
            Parent document dictionary or None
        """
        try:
            return self.parent_store.get(parent_id)
            
        except Exception as e:
            logger.error(f"Error retrieving parent {parent_id}: {str(e)}")
            return None

    def get_statistics(self) -> Dict:
        """Get store statistics."""
        return {
            "total_parents": len(self.parent_store),
            "total_children": sum(p["child_count"] for p in self.parent_store.values()),
            "backend": self.backend,
        }


class PDRStrategy:
    """Parent Document Retrieval strategy.
    
    - Child chunks: 400-500 tokens for embeddings (high accuracy search)
    - Parent chunks: Full function/page for context (full understanding)
    """

    def __init__(
        self,
        parent_store: ParentDocumentStore,
        child_chunk_size: int = 400,
        max_child_chunks_per_parent: int = 5,
    ):
        """Initialize PDR strategy.
        
        Args:
            parent_store: ParentDocumentStore instance
            child_chunk_size: Target size for child chunks (tokens)
            max_child_chunks_per_parent: Maximum children per parent
        """
        self.parent_store = parent_store
        self.child_chunk_size = child_chunk_size
        self.max_child_chunks_per_parent = max_child_chunks_per_parent
        logger.info(f"Initialized PDR: child_size={child_chunk_size} tokens")

    def create_child_parent_pairs(
        self,
        chunks: List[Dict],
    ) -> List[Dict]:
        """Create child-parent pairs for PDR using FUNCTION-LEVEL boundaries.

        P1 FIX: Previous implementation grouped every chunk of a file under
        a single synthetic `parent_<n>` (file-scoped). The LLM was therefore
        receiving an entire file as "parent context", which dilutes relevance.

        New strategy:
          1. Group raw chunks by source file.
          2. Reconstruct the file body from the chunks.
          3. Run language-aware parent extraction (Python AST / JS regex).
          4. Register each function/class as its own parent in the store.
          5. For each child chunk, locate the enclosing parent by line range
             (or substring match) and link `parent_id` accordingly.
          6. Chunks not enclosed by any function (top-level imports, module
             docstring) keep a synthetic file-level parent — preserving
             backward compatibility.

        Args:
            chunks: List of chunks from language-aware splitter

        Returns:
            List of enhanced chunks with proper `parent_id` references
        """
        try:
            if not isinstance(chunks, list):
                raise ValueError(f"Chunks must be list, got {type(chunks)}")

            # ---- 1. Group by source file --------------------------------- #
            by_source: Dict[str, List[Dict]] = {}
            for chunk in chunks:
                src = (chunk.get("metadata") or {}).get("source", "unknown")
                by_source.setdefault(src, []).append(chunk)

            enhanced_chunks: List[Dict] = []

            for source, file_chunks in by_source.items():
                language = (file_chunks[0].get("metadata") or {}).get(
                    "language", "unknown"
                )

                # ---- 2. Reconstruct file body from chunks ---------------- #
                # Chunks may overlap; we just concatenate (good enough for
                # parent extraction since extractors operate on text spans).
                full_text = "\n".join(c.get("content", "") for c in file_chunks)

                # ---- 3. Extract function/class parents ------------------- #
                parents = extract_function_level_parents(full_text, language)

                # Register parents in store with stable IDs derived from name+source
                parent_intervals: List[Tuple[str, str, int, int]] = []
                for name, body, start, end in parents:
                    parent_id = f"parent::{source}::{name}::{start}-{end}"
                    self.parent_store.add_parent(
                        parent_id=parent_id,
                        content=body,
                        metadata={
                            "source": source,
                            "language": language,
                            "parent_name": name,
                            "start_line": start,
                            "end_line": end,
                            "scope": "function_or_class",
                        },
                    )
                    parent_intervals.append((parent_id, body, start, end))

                # File-level fallback parent for orphan chunks
                file_parent_id = f"parent::{source}::__module__"
                self.parent_store.add_parent(
                    parent_id=file_parent_id,
                    content=full_text,
                    metadata={
                        "source": source,
                        "language": language,
                        "parent_name": "__module__",
                        "scope": "file",
                    },
                )

                # ---- 4. Link each child chunk to its enclosing parent ----- #
                for chunk in file_chunks:
                    content = chunk.get("content", "")
                    chunk_md = dict(chunk.get("metadata") or {})
                    enclosing_parent_id = file_parent_id

                    if parent_intervals:
                        # Code path: match by substring containment against
                        # function/class bodies.
                        probe = content.strip()[:80]
                        if probe:
                            for pid, body, _s, _e in parent_intervals:
                                if probe in body:
                                    enclosing_parent_id = pid
                                    break
                    else:
                        # Prose path (PDF / markdown): link chunk to its
                        # page-level parent if one was pre-registered.
                        # PyPDFLoader sets metadata["page"] = 0-based page #.
                        page = chunk_md.get("page")
                        if page is not None:
                            page_pid = f"parent::{source}::page::{page}"
                            if self.parent_store.get_parent(page_pid):
                                enclosing_parent_id = page_pid

                    self.parent_store.add_child(
                        enclosing_parent_id, chunk_md.get("chunk_id", "")
                    )

                    chunk_md["parent_id"] = enclosing_parent_id
                    chunk_md["scope"] = (
                        "page" if "::page::" in enclosing_parent_id
                        else "function_or_class" if enclosing_parent_id != file_parent_id
                        else "file"
                    )

                    enhanced_chunks.append({
                        **chunk,
                        "is_parent": False,
                        "parent_id": enclosing_parent_id,
                        "metadata": chunk_md,
                        "chunk_type": "child",
                    })

            logger.info(
                f"Created {len(enhanced_chunks)} child-parent pairs "
                f"({len(self.parent_store.parent_store)} parents registered)"
            )
            logger.info(f"Store stats: {self.parent_store.get_statistics()}")
            return enhanced_chunks

        except ValueError as e:
            logger.error(f"Validation error in create_child_parent_pairs: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error in create_child_parent_pairs: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to create child-parent pairs: {str(e)}")

    def _is_parent_candidate(self, content: str, chunk: Dict) -> bool:
        """Determine if chunk should be stored as parent.
        
        Args:
            content: Chunk content
            chunk: Chunk dictionary with metadata
            
        Returns:
            True if chunk is large enough to be parent
        """
        try:
            # Estimate tokens (rough: 1 token ≈ 4 chars)
            estimated_tokens = len(content) / 4
            
            # If larger than 2x child size, make it parent
            is_large = estimated_tokens > (self.child_chunk_size * 2)
            
            # If contains multiple functions/classes, make it parent
            is_multi_function = content.count("def ") > 1 or content.count("class ") > 1
            
            return is_large or is_multi_function
            
        except Exception as e:
            logger.debug(f"Error in _is_parent_candidate: {str(e)}")
            return False

    def prepare_for_embedding(
        self,
        child_parent_pairs: List[Dict],
    ) -> List[Dict]:
        """Prepare chunks for embedding (only children, not parents).
        
        Args:
            child_parent_pairs: Enhanced chunks from create_child_parent_pairs
            
        Returns:
            List of child chunks ready for embedding
        """
        try:
            embedding_ready = []
            
            for chunk in child_parent_pairs:
                try:
                    # Only embed child chunks (not parents, which are for context)
                    if not chunk.get("is_parent", False):
                        embedding_chunk = {
                            "id": chunk.get("id", ""),
                            "content": chunk.get("content", ""),
                            "parent_id": chunk.get("parent_id", ""),
                            "metadata": chunk.get("metadata", {}),
                        }
                        embedding_ready.append(embedding_chunk)
                    
                except Exception as e:
                    logger.warning(f"Error preparing chunk for embedding: {str(e)}")
                    continue
            
            logger.info(f"Prepared {len(embedding_ready)} child chunks for embedding")
            
            return embedding_ready
            
        except Exception as e:
            logger.error(f"Error in prepare_for_embedding: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to prepare chunks for embedding: {str(e)}")

    def retrieve_context(
        self,
        child_id: str,
    ) -> Optional[str]:
        """Retrieve full parent context for a child chunk.
        
        Args:
            child_id: Child chunk ID
            
        Returns:
            Full parent content or None
        """
        try:
            # In real implementation, look up child_id to find parent_id
            # For now, this is a placeholder
            logger.debug(f"Retrieving context for child {child_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return None
