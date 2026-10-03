from typing import List, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.database.models import Chunk
from app.retrieval.base import BaseVectorStore, SearchResult
from app.retrieval.embedder import LocalEmbedder


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB implementation of BaseVectorStore.
    Persists dense embeddings to local disk in DuckDB/SQLite format.
    """

    COLLECTION_NAME = "ragbench_chunks"

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedder: Optional[LocalEmbedder] = None
    ):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        # Ensure the storage directory exists on disk
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self.embedder = embedder or LocalEmbedder()

        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Get or create the collection with Cosine similarity space
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """
        Embeds and stores a list of Chunk objects in ChromaDB.
        """
        if not chunks:
            return 0

        # Prepare data lists for ChromaDB batch insertion
        ids = [str(c.id) for c in chunks]
        texts = [c.content for c in chunks]
        metadatas = [
            {
                "chunk_id": c.id,
                "document_id": c.document_id,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count
            }
            for c in chunks
        ]

        # Generate 384-dimensional embeddings in batches
        embeddings = self.embedder.embed_documents(texts) # type: ignore

        # Upsert into Chroma (UPdate and inSERT)
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings, # type: ignore
            documents=texts, # type: ignore
            metadatas=metadatas # type: ignore
        )

        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Embeds the query and retrieves the top_k most similar chunks.
        """
        if self.count() == 0:
            return []

        # Generate query vector
        query_vector = self.embedder.embed_query(query)

        # Query ChromaDB collection
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"]
        )

        search_results: List[SearchResult] = []

        # Chroma returns lists of lists (one per query)
        if not results or not results["ids"] or not results["ids"][0]:
            return search_results

        ids = results["ids"][0]
        documents = results["documents"][0] # type: ignore
        metadatas = results["metadatas"][0] # type: ignore
        distances = results["distances"][0] # type: ignore

        for chunk_id_str, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
            # Chroma returns Cosine Distance (0 = identical, 1 = unrelated)
            # Convert to Cosine Similarity score (1 = identical, 0 = unrelated)
            similarity_score = max(0.0, 1.0 - float(dist))

            search_results.append(SearchResult(
                chunk_id=int(chunk_id_str),
                document_id=int(meta["document_id"]), # type: ignore
                chunk_index=int(meta["chunk_index"]), # type: ignore
                content=doc_text,
                score=round(similarity_score, 4)
            ))

        return search_results

    def count(self) -> int:
        """
        Returns the total number of vectors in the collection.
        """
        return self.collection.count()

    def delete_document_chunks(self, document_id: int) -> int:
        """
        Deletes all chunks belonging to a specific document_id.
        """
        initial_count = self.count()
        self.collection.delete(where={"document_id": document_id})
        return initial_count - self.count()

