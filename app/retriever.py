"""
Hybrid retriever combining dense (pgvector) and sparse (BM25) search.
Main retrieval pipeline.
"""

from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
import structlog
import time

from app.models import DocumentChunk
from app.embeddings import get_embedding_manager
from app.bm25_retriever import get_bm25_index

logger = structlog.get_logger()

class HybridRetriever:
    """
    Combines dense vector search and sparse keyword search.
    """
    
    def __init__(self, db: Session):
        """
        Initialize retriever.
        
        Args:
            db: Database session
        """
        self.db = db
        self.embedding_manager = get_embedding_manager()
        self.bm25_index = get_bm25_index()
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_dense: bool = True,
        use_sparse: bool = True,
    ) -> List[Dict]:
        """
        Retrieve relevant documents using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_dense: Use dense vector search
            use_sparse: Use sparse (BM25) search
            
        Returns:
            List of dicts with keys: chunk_id, doc_id, text, score, metadata
        """
        start_time = time.time()
        results = {}  # chunk_id -> {text, scores, metadata}
        
        # Dense retrieval
        if use_dense:
            dense_results = self._dense_retrieve(query, top_k * 2)
            for chunk_id, score, text, doc_id in dense_results:
                if chunk_id not in results:
                    results[chunk_id] = {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "text": text,
                        "dense_score": score,
                        "sparse_score": 0,
                    }
                else:
                    results[chunk_id]["dense_score"] = score
        
        # Sparse retrieval
        if use_sparse:
            sparse_results = self._sparse_retrieve(query, top_k * 2)
            for chunk_id, score, text, doc_id in sparse_results:
                if chunk_id not in results:
                    results[chunk_id] = {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "text": text,
                        "dense_score": 0,
                        "sparse_score": score,
                    }
                else:
                    results[chunk_id]["sparse_score"] = score
        
        # Combine scores (60% dense, 40% sparse)
        final_results = []
        for chunk_id, data in results.items():
            combined_score = (
                0.6 * data.get("dense_score", 0) +
                0.4 * data.get("sparse_score", 0)
            )
            data["combined_score"] = combined_score
            final_results.append(data)
        
        # Sort by combined score
        final_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        latency_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "Retrieval completed",
            query_len=len(query),
            results=len(final_results[:top_k]),
            latency_ms=latency_ms,
        )
        
        return final_results[:top_k]
    
    def _dense_retrieve(self, query: str, top_k: int) -> List[tuple]:
        """
        Dense retrieval using pgvector.
        
        Returns:
            List of (chunk_id, score, text, doc_id)
        """
        try:
            # Embed query
            query_embedding = self.embedding_manager.embed_single(query)
            
            # Search in pgvector
            results = self.db.query(
                DocumentChunk.id,
                DocumentChunk.text,
                DocumentChunk.document_id,
                # Use cosine distance (operator <->)
                text(f"1 - (embedding <-> '{query_embedding}'::vector) as score")
            ).order_by(
                text(f"embedding <-> '{query_embedding}'::vector")
            ).limit(top_k).all()
            
            # Format results
            formatted = [
                (result[0], float(result[3]), result[1], result[2])
                for result in results
            ]
            
            return formatted
        except Exception as e:
            logger.error("Dense retrieval failed", error=str(e))
            return []
    
    def _sparse_retrieve(self, query: str, top_k: int) -> List[tuple]:
        """
        Sparse retrieval using BM25.
        
        Returns:
            List of (chunk_id, score, text, doc_id)
        """
        try:
            # Get BM25 results
            bm25_results = self.bm25_index.search(query, top_k)
            
            # Convert to (chunk_id, score, text, doc_id)
            formatted = []
            for chunk_id, score in bm25_results:
                chunk = self.db.query(DocumentChunk).filter(
                    DocumentChunk.id == chunk_id
                ).first()
                if chunk:
                    formatted.append((chunk_id, score, chunk.text, chunk.document_id))
            
            return formatted
        except Exception as e:
            logger.error("Sparse retrieval failed", error=str(e))
            return []

def get_retriever(db: Session) -> HybridRetriever:
    """Create retriever for given database session."""
    return HybridRetriever(db)
