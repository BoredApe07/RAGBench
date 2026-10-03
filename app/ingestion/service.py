import hashlib
from typing import List, Optional, Tuple
from pathlib import Path
from sqlalchemy.orm import Session

from app.config import settings
from app.database.models import Document, Chunk
from app.ingestion.chunker import RecursiveChunker, TextChunk


class IngestionService:
    """
    Manages document ingestion into the relational database.
    Handles duplicate prevention (SHA-256 hashing) and chunk persistence.
    """

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        size = chunk_size or settings.CHUNK_SIZE
        overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.chunker = RecursiveChunker(chunk_size=size, chunk_overlap=overlap)

    @staticmethod
    def compute_sha256(text: str) -> str:
        """
        Generates a deterministic cryptographic fingerprint of document content.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def ingest_text(
        self,
        db: Session,
        filename: str,
        text: str,
        skip_if_exists: bool = True
    ) -> Tuple[Document, List[Chunk], bool]:
        """
        Ingests a document from raw text into SQLite/PostgreSQL.
        Returns:
            (Document, List[Chunk], is_new: bool)
        """
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError(f"Document '{filename}' contains no readable text.")

        file_hash = self.compute_sha256(cleaned_text)
        file_size = len(cleaned_text.encode("utf-8"))

        # Check for existing document by hash (idempotency)
        existing_doc = db.query(Document).filter(Document.file_hash == file_hash).first()
        if existing_doc and skip_if_exists:
            chunks = db.query(Chunk).filter(Chunk.document_id == existing_doc.id).order_by(Chunk.chunk_index).all()
            return existing_doc, chunks, False

        # Create new Document record
        doc = Document(
            filename=filename,
            file_hash=file_hash,
            file_size_bytes=file_size
        )
        db.add(doc)
        db.flush()  # Flush to generate doc.id before creating chunks

        # Split text into chunks
        text_chunks: List[TextChunk] = self.chunker.split_text(cleaned_text)

        # Create Chunk database records
        db_chunks = []
        for tc in text_chunks:
            chunk = Chunk(
                document_id=doc.id,
                chunk_index=tc.chunk_index,
                content=tc.content,
                char_start=tc.char_start,
                char_end=tc.char_end,
                token_count=tc.token_count
            )
            db_chunks.append(chunk)

        db.add_all(db_chunks)
        db.commit()
        db.refresh(doc)

        return doc, db_chunks, True

    def ingest_file(self, db: Session, file_path: Path, skip_if_exists: bool = True) -> Tuple[Document, List[Chunk], bool]:
        """
        Reads a local text/markdown file and ingests it.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        text = path.read_text(encoding="utf-8", errors="replace")
        return self.ingest_text(db, filename=path.name, text=text, skip_if_exists=skip_if_exists)
