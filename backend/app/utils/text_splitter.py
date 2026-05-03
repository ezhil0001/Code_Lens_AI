"""Text Splitting Utilities for Code-aware Processing"""

from typing import List


class CodeTextSplitter:
    """
    Code-aware text splitter for chunking source code
    while preserving semantic meaning
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separators: List[str] = None,
    ):
        """
        Initialize code text splitter
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            separators: List of separators to use (in order of preference)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\nclass ",
            "\n\ndef ",
            "\n\n",
            "\n",
            " ",
            "",
        ]
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks while preserving code structure
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        chunks = []
        separator = self.separators[-1]
        
        for _s in self.separators:
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                break
        
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        
        return self._merge_splits(splits, separator)
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits while respecting chunk size and overlap"""
        separator_len = len(separator)
        
        good_splits = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged_text = self._merge_good_splits(good_splits, separator)
                    for m in merged_text:
                        if m:
                            return good_splits + self._merge_splits([m], separator)
                    good_splits = []
                other_info = self._merge_good_splits([s], separator)
                return other_info
        
        return self._merge_good_splits(good_splits, separator)
    
    def _merge_good_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge good splits into chunks of appropriate size"""
        separator_len = len(separator)
        chunks = []
        current_chunk = []
        total_len = 0
        
        for s in splits:
            s_len = len(s)
            if total_len + s_len + (separator_len if current_chunk else 0) > self.chunk_size:
                if current_chunk:
                    chunk = separator.join(current_chunk)
                    if chunk:
                        chunks.append(chunk)
                    current_chunk = []
                    total_len = 0
            
            current_chunk.append(s)
            total_len += s_len + (separator_len if len(current_chunk) > 1 else 0)
        
        if current_chunk:
            chunk = separator.join(current_chunk)
            if chunk:
                chunks.append(chunk)
        
        return chunks
