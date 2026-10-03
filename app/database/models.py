from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Document(Base):
    """
    Represents an ingested document file.
    Tracks file metadata, hash (to prevent duplicate uploads), and status.
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), unique=True, index=True, nullable=False)  # SHA-256 hash
    file_size_bytes = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # 1 Document has Many Chunks (one-to-many relationship)
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename='{self.filename}', chunks={len(self.chunks)})>"


class Chunk(Base):
    """
    Represents an extracted text chunk from a Document.
    Linked to its parent Document via foreign key `document_id`.
    """
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)  # 0, 1, 2... relative to parent document
    content = Column(Text, nullable=False)
    char_start = Column(Integer, nullable=False, default=0)
    char_end = Column(Integer, nullable=False, default=0)
    token_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Parent relationship
    document = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<Chunk(id={self.id}, doc_id={self.document_id}, index={self.chunk_index})>"


class EvaluationBenchmark(Base):
    """
    Represents a ground-truth benchmark query used to evaluate retrieval and answer quality.
    """
    __tablename__ = "evaluation_benchmarks"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    expected_answer = Column(Text, nullable=True)
    # List of chunk IDs or document IDs that are ground-truth relevant for this query
    ground_truth_chunk_ids = Column(JSON, nullable=False, default=list)
    category = Column(String(100), nullable=True)  # e.g., 'factual', 'summary', 'multi-hop'
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class EvaluationRun(Base):
    """
    Tracks historical evaluation experiments.
    Stores quantitative metrics (Recall@K, Precision@K, Faithfulness, p95 latency).
    Enables systematic A/B comparison between RAG configurations.
    """
    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_name = Column(String(100), nullable=False)  # e.g. "baseline-vector", "hybrid-bm25-vector", "hybrid-rerank"
    pipeline_config = Column(JSON, nullable=False, default=dict)
    
    # Retrieval Metrics
    total_queries = Column(Integer, nullable=False, default=0)
    recall_at_5 = Column(Float, nullable=True)
    precision_at_5 = Column(Float, nullable=True)
    mrr = Column(Float, nullable=True)  # Mean Reciprocal Rank

    # Answer Quality Metrics (LLM-as-a-judge)
    faithfulness_score = Column(Float, nullable=True)   # Groundedness / hallucination check (0.0 to 1.0)
    relevance_score = Column(Float, nullable=True)      # Answer relevance to query (0.0 to 1.0)

    # Performance / Latency Metrics
    avg_latency_ms = Column(Float, nullable=True)
    p50_latency_ms = Column(Float, nullable=True)
    p95_latency_ms = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

