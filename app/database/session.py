import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.config import settings
from app.database.models import Base

# Ensure the parent directory for SQLite exists (e.g., ./data/)
if settings.DATABASE_URL.startswith("sqlite"):
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    parent_dir = Path(db_path).parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(settings.DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """
    Initializes the database schema by creating all tables if they do not already exist.
    """
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    Dependency generator for FastAPI endpoints to yield a database session per request,
    ensuring proper cleanup/closing.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

