from typing import List, Optional
from sentence_transformers import SentenceTransformer
from app.config import settings


class LocalEmbedder:
    """
    Local embedding generator using Sentence-Transformers (all-MiniLM-L6-v2).
    Generates 384-dimensional dense vectors on CPU with zero cloud API costs.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        # Load the pre-trained model into memory
        self.model = SentenceTransformer(self.model_name)

    @property
    def dimension(self) -> int:
        """
        Returns the embedding vector size (384 for all-MiniLM-L6-v2).
        """
        return self.model.get_embedding_dimension() or 384

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query string into a vector.
        """
        if not query.strip():
            raise ValueError("Query string cannot be empty.")
        
        # normalize_embeddings=True ensures vectors have a length (norm) of 1,
        # which makes cosine similarity equivalent to a simple dot product!
        embedding = self.model.encode(query, normalize_embeddings=True)
        return embedding.tolist()

    def embed_documents(self, documents: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Embeds a list of document chunks in batches for high throughput.
        """
        if not documents:
            return []

        embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.tolist()

