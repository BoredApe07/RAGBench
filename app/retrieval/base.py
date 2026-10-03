from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, Field
from app.database.models import Chunk


class SearchResult(BaseModel):
    """
    Standardized result returned by any Vector Store implementation.
    Decouples retrieval consumers (evaluation, generation) from the underlying database.
    """
    chunk_id: int = Field(..., description="Primary key of the chunk in the relational database")
    document_id: int = Field(..., description="Parent document ID")
    chunk_index: int = Field(..., description="Position index of the chunk in the parent document")
    content: str = Field(..., description="The chunk text snippet")
    score: float = Field(..., description="Relevance score (e.g., cosine similarity, higher is better)")


class BaseVectorStore(ABC):
    """
    Abstract Base Class (Interface) for Vector Database implementations.
    Guarantees that ChromaDB, PostgreSQL pgvector, or any future provider
    can be swapped without altering upstream business logic.
    """

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> int:
        """
        Embeds and stores a list of Chunk objects in the vector index.
        Returns the number of chunks successfully indexed.
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Embeds a text query and returns the top_k most similar chunks.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Returns the total number of vectors currently stored in the index.
        """
        pass

    @abstractmethod
    def delete_document_chunks(self, document_id: int) -> int:
        """
        Deletes all vector embeddings associated with a given document_id.
        """
        pass
