"""
Application configuration.

Loads from environment variables with fallbacks.
"""

from pydantic_settings import BaseSettings # type: ignore
from functools import lru_cache
import os

class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://sdre_user:dev_password@localhost:5432/sdre_db"
    )
    
    # Redis
    REDIS_URL: str = os.getenv(
        "REDIS_URL",
        "redis://localhost:6379/0"
    )
    
    # LLM Configuration
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Model Selection
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    GENERATION_MODEL: str = "mixtral-8x7b-32768"  # Groq
    
    # Retrieval Settings
    DENSE_TOP_K: int = 20  # Over-retrieve before reranking
    FINAL_TOP_K: int = 10  # Final context chunks
    BM25_TOP_K: int = 20
    
    # Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # Caching
    CACHE_TTL_SECONDS: int = 86400  # 24 hours
    CACHE_MAX_SIZE: int = 1000
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Performance
    MAX_WORKERS: int = 4
    TIMEOUT_SECONDS: int = 30
    
    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

# Convenience
settings = get_settings()
