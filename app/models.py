"""
Database models for SDRE.

SQLAlchemy ORM models for documents, chunks, queries, and eval results.
"""

from sqlalchemy import Column, String, DateTime, Float, JSON, Integer, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector
from datetime import datetime
import uuid

Base = declarative_base()

class Document(Base):
    """Represents a full document in the system."""
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255), nullable=False, index=True)
    source = Column(String(512))  # URL or file path
    content = Column(Text)  # Full document text (for reference)
    file_type = Column(String(50))  # pdf, docx, txt, etc.
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Metadata storage (domain, author, date, etc.)
    metadata_json = Column(JSON, default={})

class DocumentChunk(Base):
    """Represents a chunk of a document (for retrieval)."""
    __tablename__ = "document_chunks"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    
    # Dense embedding (384-dim for all-MiniLM-L6-v2)
    embedding = Column(Vector(384), nullable=False)
    
    # For BM25 search (sparse)
    bm25_terms = Column(JSON)  # List of (term, score) tuples
    
    # Metadata
    page_number = Column(Integer)  # If from PDF
    section = Column(String(255))  # Chapter, section, etc.
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

class QueryLog(Base):
    """Logs every query for analytics and eval."""
    __tablename__ = "query_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query_text = Column(Text, nullable=False)
    answer = Column(Text)
    
    # Retrieved document IDs
    retrieved_doc_ids = Column(JSON)  # List of doc UUIDs
    retrieved_chunks = Column(JSON)  # List of chunk UUIDs
    
    # Generated response
    citations = Column(JSON)  # List of citation strings
    confidence = Column(Float)
    
    # Eval metrics
    eval_scores = Column(JSON)  # {faithfulness, relevance, precision, recall}
    
    # Performance
    latency_ms = Column(Integer)
    retrieval_ms = Column(Integer)
    generation_ms = Column(Integer)
    
    # Cost
    cost_usd = Column(Float)
    tokens_prompt = Column(Integer)
    tokens_completion = Column(Integer)
    
    # Metadata
    user_id = Column(String(255))
    session_id = Column(String(255))
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

class EvalResult(Base):
    """Ground truth eval dataset."""
    __tablename__ = "eval_results"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query_text = Column(Text, nullable=False)
    ground_truth = Column(Text)  # Expected answer
    
    # List of relevant document IDs
    relevant_docs = Column(JSON)
    
    # System response (link to query_log)
    query_log_id = Column(String)
    
    # Manual annotations
    manually_relevant = Column(JSON)  # List of doc IDs manually marked as relevant
    manual_notes = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class MetricsSnapshot(Base):
    """System-wide metrics snapshot."""
    __tablename__ = "metrics_snapshots"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Counts
    total_documents = Column(Integer)
    total_chunks = Column(Integer)
    total_queries = Column(Integer)
    
    # Averages
    avg_latency_ms = Column(Float)
    avg_cost_usd = Column(Float)
    avg_faithfulness = Column(Float)
    avg_relevance = Column(Float)
    
    # Cache
    cache_hit_rate = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
