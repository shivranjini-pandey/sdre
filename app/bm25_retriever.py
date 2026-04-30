"""
BM25 (sparse) retriever using whoosh.
Provides keyword-based retrieval as fallback to dense search.
"""

from typing import List, Dict, Tuple
import json
from pathlib import Path
import structlog

logger = structlog.get_logger()

class SimpleBM25:
    """
    Simple BM25 implementation.
    
    This is a simplified version for MVP.
    """
    
    def __init__(self):
        """Initialize BM25 scorer."""
        self.documents = {}  # id -> text
        self.term_index = {}  # term -> list of (doc_id, tf)
        self.doc_lengths = {}  # id -> length
        self.avg_doc_length = 0
        self.doc_count = 0
    
    def index_document(self, doc_id: str, text: str):
        """Index a document."""
        self.documents[doc_id] = text
        
        # Tokenize (simple)
        tokens = text.lower().split()
        self.doc_lengths[doc_id] = len(tokens)
        self.doc_count += 1
        
        # Build term index
        term_freq = {}
        for token in tokens:
            if len(token) > 2:  # Skip short tokens
                term_freq[token] = term_freq.get(token, 0) + 1
        
        for term, freq in term_freq.items():
            if term not in self.term_index:
                self.term_index[term] = []
            self.term_index[term].append((doc_id, freq))
        
        # Update average doc length
        self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search using BM25.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (doc_id, score) tuples
        """
        query_tokens = query.lower().split()
        scores = {}
        
        k1 = 1.5  # BM25 parameter
        b = 0.75  # BM25 parameter
        
        for token in query_tokens:
            if token not in self.term_index:
                continue
            
            idf = len(self.documents) / len([d for d, _ in self.term_index[token]])
            
            for doc_id, freq in self.term_index[token]:
                doc_len = self.doc_lengths.get(doc_id, 0)
                
                # BM25 formula
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * (doc_len / self.avg_doc_length))
                score = idf * (numerator / denominator)
                
                scores[doc_id] = scores.get(doc_id, 0) + score
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

# Singleton
_bm25_index: SimpleBM25 = SimpleBM25()

def get_bm25_index() -> SimpleBM25:
    """Get BM25 index (singleton)."""
    return _bm25_index
