"""
Database initialization and connection management.
"""

from sqlalchemy import create_engine, text, inspect # type: ignore
from sqlalchemy.orm import sessionmaker, Session # type: ignore
from sqlalchemy.pool import StaticPool # type: ignore
from pgvector.sqlalchemy import Vector # type: ignore
from app.config import settings
from app.models import Base
import structlog # type: ignore

logger = structlog.get_logger()

# Create engine
engine = create_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,  # Verify connections before using
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    Initialize database:
    1. Create pgvector extension
    2. Create all tables
    3. Create indexes
    """
    logger.info("Initializing database...")
    
    with engine.begin() as conn:
        # Enable pgvector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create indexes
    with engine.begin() as conn:
        # Index on document_id for fast chunk lookups
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id 
            ON document_chunks(document_id)
        """))
        
        # Index on embedding for vector search (ANN)
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding 
            ON document_chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """))
        
        # Index on created_at for time-based queries
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_query_logs_created_at 
            ON query_logs(created_at DESC)
        """))
        
        conn.commit()
    
    logger.info("Database initialized successfully")

def health_check() -> bool:
    """Check if database is reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        return False

def clear_db():
    Base.metadata.drop_all(bind=engine)
    logger.warning("Database cleared - all tables dropped")

if __name__ == "__main__":
    init_db()
    print("Database initialized")
