from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """
    Application Settings.
    Reads automatically from .env file and environment variables.
    Provides type validation and defaults.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Gemini LLM Settings
    GEMINI_API_KEY: str = Field(..., description="Google Gemini API key")
    GEMINI_MODEL: str = Field("gemini-flash-lite-latest", description="Gemini model name")

    # Relational Database Settings (SQLite default, can be PostgreSQL)
    DATABASE_URL: str = Field("sqlite:///./data/ragbench.db", description="SQLAlchemy database URL")

    # Vector Database Settings
    CHROMA_PERSIST_DIR: str = Field("./data/chroma_db", description="ChromaDB persistent directory")
    EMBEDDING_MODEL_NAME: str = Field("all-MiniLM-L6-v2", description="Sentence Transformers model name")

    # Retrieval & Chunking Hyperparameters
    TOP_K: int = Field(5, description="Number of chunks to retrieve for generation")
    CHUNK_SIZE: int = Field(500, description="Target character size per chunk")
    CHUNK_OVERLAP: int = Field(50, description="Overlapping characters between adjacent chunks")


# Singleton instance to import across the app
settings = Settings() # type: ignore
