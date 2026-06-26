"""
FastAPI application for SDRE.

REST API endpoints for querying and ingesting documents.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from pathlib import Path
import uuid
import time
import structlog
from contextlib import asynccontextmanager

from app.config import settings
from app.database import init_db, get_db, SessionLocal, health_check
from app.retriever import get_retriever
from app.reranker import get_reranker
from app.llm import get_llm_manager
from app.ingest import get_ingester
from app.models import QueryLog

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranking: bool = True

class QueryResponse(BaseModel):
    request_id: str
    answer: str
    confidence: float
    sources: list
    latency_ms: int
    cost_usd: float

# Startup & shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting SDRE API...")
    init_db()
    if not health_check():
        logger.error("Database health check failed!")
    logger.info("Startup complete")
    yield
    # Shutdown
    logger.info("Shutting down SDRE API...")

app = FastAPI(
    title="Semantic Document Retrieval Engine",
    version="0.1.0",
    lifespan=lifespan,
)

# Routes
@app.get("/health")
async def health():
    """Health check endpoint."""
    db_ok = health_check()
    return {
        "status": "healthy" if db_ok else "degraded",
        "database": "connected" if db_ok else "disconnected",
    }

@app.post("/query", response_model=QueryResponse)
async def query_documents(
    req: QueryRequest,
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Query documents and get an answer.
    
    Args:
        req: Query request with query text and parameters
        db: Database session
        background_tasks: Background task queue
        
    Returns:
        Answer with sources and metrics
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        logger.info("Query started", request_id=request_id, query=req.query)
        
        # 1. Retrieve documents
        retriever = get_retriever(db)
        retrieved = await retriever.retrieve(req.query, top_k=req.top_k * 2)
        
        if not retrieved:
            return QueryResponse(
                request_id=request_id,
                answer="No relevant documents found.",
                confidence=0.0,
                sources=[],
                latency_ms=int((time.time() - start_time) * 1000),
                cost_usd=0.0,
            )
        
        # 2. Rerank 
        if req.use_reranking:
            reranker = get_reranker()
            retrieved = reranker.rerank(req.query, retrieved, top_k=req.top_k)
        else:
            retrieved = retrieved[:req.top_k]
        
        # 3. Generate answer
        context_texts = [doc["text"] for doc in retrieved]
        llm = get_llm_manager(settings.GROQ_API_KEY)
        answer, cost_usd, prompt_tokens, completion_tokens = llm.generate(
            req.query,
            context_texts,
        )
        
        # 4. Extract sources
        sources = [
            {
                "doc_id": doc["doc_id"],
                "text": doc["text"][:200],
                "score": doc.get("rerank_score", doc.get("combined_score", 0)),
            }
            for doc in retrieved[:3]
        ]
        
        # 5. Log to database (async)
        latency_ms = int((time.time() - start_time) * 1000)
        background_tasks.add_task(
            _log_query,
            request_id=request_id,
            query=req.query,
            answer=answer,
            retrieved_doc_ids=[doc["doc_id"] for doc in retrieved],
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
        
        return QueryResponse(
            request_id=request_id,
            answer=answer,
            confidence=0.85,  # Placeholder
            sources=sources,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )
    
    except Exception as e:
        logger.error("Query failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_documents(
    files: list[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
):
    """
    Ingest documents for retrieval.
    
    Args:
        files: List of files to ingest
        background_tasks: Background task queue
        db: Database session
        
    Returns:
        Status and document IDs
    """
    request_id = str(uuid.uuid4())
    
    try:
        # Save files and ingest asynchronously
        file_paths = []
        for file in files:
            file_path = f"/tmp/{file.filename}"
            with open(file_path, "wb") as f:
                f.write(await file.read())
            file_paths.append(file_path)
        
        # Ingest in background
        background_tasks.add_task(_ingest_files, request_id, file_paths, db)
        
        return {
            "request_id": request_id,
            "status": "ingesting",
            "file_count": len(files),
        }
    
    except Exception as e:
        logger.error("Ingestion failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics")
async def get_metrics(db: Session = Depends(get_db)):
    """Get system metrics."""
    from sqlalchemy import func
    
    total_docs = db.query(func.count(DocumentChunk.document_id)).scalar()
    total_queries = db.query(func.count(QueryLog.id)).scalar()
    
    return {
        "documents": total_docs or 0,
        "queries": total_queries or 0,
        "database": "connected" if health_check() else "disconnected",
    }

# Background tasks
async def _log_query(
    request_id: str,
    query: str,
    answer: str,
    retrieved_doc_ids: list,
    cost_usd: float,
    latency_ms: int,
):
    """Log query to database."""
    db = SessionLocal()
    try:
        query_log = QueryLog(
            query_text=query,
            answer=answer[:500],
            retrieved_doc_ids=retrieved_doc_ids,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
        db.add(query_log)
        db.commit()
    finally:
        db.close()

async def _ingest_files(request_id: str, file_paths: list, db: Session):
    """Ingest files in background."""
    try:
        ingester = get_ingester(db)
        for file_path in file_paths:
            ingester.ingest_file(file_path)
            Path(file_path).unlink()  # Clean up
        logger.info("Ingestion complete", request_id=request_id, files=len(file_paths))
    except Exception as e:
        logger.error("Background ingestion failed", request_id=request_id, error=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
