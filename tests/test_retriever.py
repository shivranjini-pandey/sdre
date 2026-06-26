"""
Tests for retrieval pipeline.
"""

import pytest
from app.embeddings import get_embedding_manager
from app.bm25_retriever import get_bm25_index

def test_embedding_manager():
    """Test that embedding manager works."""
    em = get_embedding_manager()
    
    texts = ["Hello world", "How are you"]
    embeddings = em.embed(texts)
    
    assert embeddings.shape == (2, 384)  # all-MiniLM-L6-v2 has 384 dims

def test_bm25_index():
    """Test BM25 indexing and search."""
    bm25 = get_bm25_index()
    
    # Clear and add documents
    bm25.term_index.clear()
    bm25.documents.clear()
    bm25.doc_lengths.clear()
    
    bm25.index_document("doc1", "The quick brown fox jumps over the lazy dog")
    bm25.index_document("doc2", "A fast red fox runs away")
    bm25.index_document("doc3", "The dog is very lazy")
    
    # Search
    results = bm25.search("quick fox", top_k=2)
    
    assert len(results) > 0
    assert results[0][0] == "doc1"  # "doc1" should rank highest

def test_embedding_similarity():
    """Test similarity computation."""
    em = get_embedding_manager()
    
    sim1 = em.similarity("The cat is on the mat", "A cat sits on a mat")
    sim2 = em.similarity("The cat is on the mat", "A dog is running")
    
    # Similar texts should have higher similarity
    assert sim1 > sim2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
