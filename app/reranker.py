"""
Cross-encoder reranking using sentence-transformers.

Improves retrieval ranking using a trained reranker model.
"""

from sentence_transformers import CrossEncoder
from typing import List, Dict, Optional
import structlog
import time

logger = structlog.get_logger()

class CrossEncoderReranker:
    """
    Reranks retrieved documents using cross-encoder.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """
        Initialize reranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        logger.info("Loading reranker model", model=model_name)
        self.model = CrossEncoder(model_name)
        logger.info("Reranker loaded")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: Search query
            documents: List of document dicts with 'text' key
            top_k: Keep top K documents after reranking
            
        Returns:
            Sorted list of documents with 'rerank_score' added
        """
        start_time = time.time()
        
        if not documents:
            return []
        
        # Prepare pairs for cross-encoder
        query_doc_pairs = [[query, doc["text"]] for doc in documents]
        
        # Get scores
        scores = self.model.predict(query_doc_pairs)
        
        # Add scores to documents
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
        
        # Sort by rerank score
        documents.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        latency_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "Reranking completed",
            input_docs=len(documents),
            latency_ms=latency_ms,
        )
        
        if top_k:
            documents = documents[:top_k]
        
        return documents

# Singleton
_reranker: Optional[CrossEncoderReranker] = None

def get_reranker() -> CrossEncoderReranker:
    """Get or create reranker (singleton)."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker
