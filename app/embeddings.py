"""
Embedding management using sentence-transformers.
Handles encoding text and storing in pgvector.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np
import structlog

logger = structlog.get_logger()

class EmbeddingManager:
    """
    Manages embeddings using sentence-transformers.
    
    Uses all-MiniLM-L6-v2 model.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model name
        """
        logger.info("Loading embedding model", model=model_name)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded", dim=self.embedding_dim)
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def embed_single(self, text: str) -> List[float]:
        """
        Embed a single text.
        
        Args:
            text: Single text string
            
        Returns:
            List of floats (embedding vector)
        """
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        embeddings = self.model.encode([text1, text2], convert_to_numpy=True)
        # Cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)

# Singleton instance
_embedding_manager: Optional[EmbeddingManager] = None

def get_embedding_manager() -> EmbeddingManager:
    """Get or create embedding manager (singleton)."""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager
