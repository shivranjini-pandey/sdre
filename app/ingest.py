"""
Document ingestion pipeline.

Loads documents, chunks them, embeds chunks, and stores in database.
"""

from pathlib import Path
from sqlalchemy.orm import Session
from typing import Optional
import uuid
import structlog

from app.loaders import DocumentLoader, DocumentChunker
from app.embeddings import get_embedding_manager
from app.bm25_retriever import get_bm25_index
from app.models import Document, DocumentChunk

logger = structlog.get_logger()

class DocumentIngester:
    """
    Ingests documents: loads → chunks → embeds → stores.
    """
    
    def __init__(self, db: Session):
        """
        Initialize ingester.
        
        Args:
            db: Database session
        """
        self.db = db
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker()
        self.embedding_manager = get_embedding_manager()
        self.bm25_index = get_bm25_index()
    
    def ingest_file(
        self,
        file_path: str,
        title: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Document:
        """
        Ingest a single file.
        
        Args:
            file_path: Path to file
            title: Document title (defaults to filename)
            metadata: Additional metadata
            
        Returns:
            Document instance
        """
        logger.info("Starting document ingestion", file_path=file_path)
        
        # Load document
        content, file_metadata = self.loader.load(file_path)
        
        if not title:
            title = Path(file_path).stem
        
        # Combine metadata
        if metadata:
            file_metadata.update(metadata)
        
        # Chunk document
        chunks = self.chunker.chunk(content)
        logger.info("Document chunked", num_chunks=len(chunks))
        
        # Create document record
        doc = Document(
            title=title,
            source=file_path,
            content=content[:1000],  # Store first 1000 chars only
            file_type=file_metadata.get("file_type", "unknown"),
            chunk_count=len(chunks),
            metadata_json=file_metadata,
        )
        self.db.add(doc)
        self.db.commit()
        
        logger.info("Document created in DB", doc_id=doc.id)
        
        # Embed and store chunks
        self._ingest_chunks(doc.id, chunks)
        
        logger.info("Document ingestion complete", doc_id=doc.id, chunks=len(chunks))
        return doc
    
    def _ingest_chunks(self, doc_id: str, chunks: list):
        """
        Embed chunks and store in database.
        
        Args:
            doc_id: Document ID
            chunks: List of text chunks
        """
        # Embed all chunks at once (faster)
        logger.info("Embedding chunks", num_chunks=len(chunks))
        embeddings = self.embedding_manager.embed(chunks)
        
        # Store chunks
        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk = DocumentChunk(
                document_id=doc_id,
                chunk_index=idx,
                text=chunk_text,
                embedding=embedding.tolist(),  # Convert numpy to list
            )
            self.db.add(chunk)
            self.db.flush()
            # Also index in BM25
            self.bm25_index.index_document(chunk.id, chunk_text)
        
        self.db.commit()
        logger.info("Chunks stored in database", count=len(chunks))

def get_ingester(db: Session) -> DocumentIngester:
    """Create ingester for given database session."""
    return DocumentIngester(db)
