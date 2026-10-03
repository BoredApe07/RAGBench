import re
from typing import List
from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    """
    Pydantic schema representing an in-memory chunk produced by the chunker
    before it is saved to the database.
    """
    chunk_index: int = Field(..., description="0-indexed position within parent document")
    content: str = Field(..., description="The chunk text")
    char_start: int = Field(..., description="Start character offset in original text")
    char_end: int = Field(..., description="End character offset in original text")
    token_count: int = Field(..., description="Estimated token count (approx. words * 1.3)")


class RecursiveChunker:
    """
    Splits text recursively using natural separators (paragraphs, lines, sentences, spaces)
    with a sliding-window overlap to preserve context across chunk boundaries.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " "]

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[TextChunk]:
        """
        Splits raw text into a list of TextChunk objects with offset tracking.
        """
        if not text or not text.strip():
            return []

        raw_chunks = self._recursive_split(text, self.DEFAULT_SEPARATORS)
        
        # Merge small splits and enforce sliding-window overlap
        merged_chunks = self._merge_splits_with_overlap(raw_chunks, text)
        return merged_chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively splits text using the first separator that successfully divides it.
        """
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        # Find the highest-priority separator present in text
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split text by chosen separator
        splits = text.split(separator) if separator else list(text)

        for split in splits:
            if not split.strip():
                continue
            if len(split) <= self.chunk_size:
                final_chunks.append(split)
            else:
                if new_separators:
                    # Recurse with lower-priority separators
                    sub_splits = self._recursive_split(split, new_separators)
                    final_chunks.extend(sub_splits)
                else:
                    # Fallback hard cut if no separators remain
                    for idx in range(0, len(split), self.chunk_size - self.chunk_overlap):
                        final_chunks.append(split[idx : idx + self.chunk_size])

        return final_chunks

    def _merge_splits_with_overlap(self, splits: List[str], full_text: str) -> List[TextChunk]:
        """
        Combines small splits together up to chunk_size, applying chunk_overlap
        and locating exact character offsets inside the original text.
        """
        chunks: List[TextChunk] = []
        current_chunk_parts: List[str] = []
        current_len = 0
        search_start_cursor = 0

        for split in splits:
            split_len = len(split)
            if current_len + split_len > self.chunk_size and current_chunk_parts:
                # Emit current accumulated chunk
                chunk_str = " ".join(current_chunk_parts).strip()
                start_idx, end_idx = self._find_offsets(full_text, chunk_str, search_start_cursor)
                token_est = max(1, int(len(chunk_str.split()) * 1.3))

                chunks.append(TextChunk(
                    chunk_index=len(chunks),
                    content=chunk_str,
                    char_start=start_idx,
                    char_end=end_idx,
                    token_count=token_est
                ))

                # Slide cursor forward for next offset searches
                search_start_cursor = max(0, start_idx + len(chunk_str) - self.chunk_overlap)

                # Keep overlap pieces from the tail for the next chunk
                overlap_accum = []
                overlap_accum_len = 0
                for part in reversed(current_chunk_parts):
                    if overlap_accum_len + len(part) <= self.chunk_overlap:
                        overlap_accum.insert(0, part)
                        overlap_accum_len += len(part)
                    else:
                        break
                current_chunk_parts = overlap_accum
                current_len = overlap_accum_len

            current_chunk_parts.append(split)
            current_len += split_len

        # Emit any trailing chunk
        if current_chunk_parts:
            chunk_str = " ".join(current_chunk_parts).strip()
            start_idx, end_idx = self._find_offsets(full_text, chunk_str, search_start_cursor)
            token_est = max(1, int(len(chunk_str.split()) * 1.3))
            chunks.append(TextChunk(
                chunk_index=len(chunks),
                content=chunk_str,
                char_start=start_idx,
                char_end=end_idx,
                token_count=token_est
            ))

        return chunks

    def _find_offsets(self, full_text: str, snippet: str, search_start: int) -> tuple[int, int]:
        """
        Finds the exact [start, end] char positions of a chunk in the full text.
        """
        # Search from search_start first
        idx = full_text.find(snippet[:min(40, len(snippet))], search_start)
        if idx == -1:
            idx = full_text.find(snippet[:min(40, len(snippet))])
        if idx == -1:
            return (search_start, search_start + len(snippet))
        return (idx, idx + len(snippet))
