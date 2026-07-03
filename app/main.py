"""
FastAPI application for SDRE.

REST API endpoints for querying and ingesting documents.
"""

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import structlog
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.cache import get_cache
from app.config import settings
from app.database import SessionLocal, get_db, health_check, init_db
from app.models import DocumentChunk, QueryLog
from app.router import get_router
from monitoring.metrics import (
    get_metrics_output,
    record_cache_event,
    record_error,
    record_ingestion,
    record_query,
    update_document_counts,
)

structlog.configure(
    processors=[structlog.processors.JSONRenderer()],
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()


# --- Pydantic models ---

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_cache: bool = True


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    intent: str
    confidence: float
    sources: list
    latency_ms: int
    cost_usd: float
    cache_hit: bool = False


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting SDRE Phase 2 API...")
    init_db()
    cache = get_cache()
    cache_ok = await cache.health_check()
    db_ok = health_check()

    if not db_ok:
        logger.error("Database health check failed on startup")
    if not cache_ok:
        logger.warning("Redis unavailable — caching disabled")

    logger.info("Startup complete", db=db_ok, cache=cache_ok)
    yield

    logger.info("Shutting down...")
    await cache.close()


app = FastAPI(
    title="Semantic Document Retrieval Engine",
    version="0.2.0",
    lifespan=lifespan,
)


# --- Routes ---

@app.get("/health")
async def health():
    """Health check including cache status."""
    cache = get_cache()
    db_ok = health_check()
    cache_ok = await cache.health_check()

    return {
        "status": "healthy" if (db_ok and cache_ok) else "degraded",
        "database": "connected" if db_ok else "disconnected",
        "cache": "connected" if cache_ok else "disconnected",
        "version": "0.2.0",
    }


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    req: QueryRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Query documents using multi-agent routing.

    Automatically classifies query intent and routes to the
    appropriate retrieval strategy.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    cache = get_cache()

    # 1. Check cache
    if req.use_cache:
        cached = await cache.get_query_response(req.query)
        if cached:
            record_cache_event("query", hit=True)
            cached["request_id"] = request_id
            cached["cache_hit"] = True
            return QueryResponse(**cached)
        record_cache_event("query", hit=False)

    try:
        # 2. Route through multi-agent graph
        router = get_router()
        state = await router.run(req.query, db)

        if state.get("error") and not state.get("answer"):
            raise ValueError(state["error"])

        intent = state.get("intent", "unknown")
        answer = state.get("answer", "No answer generated.")
        cost_usd = state.get("cost_usd", 0.0)

        reranked = state.get("reranked_chunks") or []
        sources = [
            {
                "doc_id": c["doc_id"],
                "text": c["text"][:200],
                "score": c.get("rerank_score", c.get("combined_score", 0)),
            }
            for c in reranked[:3]
        ]

        latency_ms = int((time.time() - start_time) * 1000)

        response_data = {
            "request_id": request_id,
            "answer": answer,
            "intent": intent.value if hasattr(intent, "value") else str(intent),
            "confidence": 0.87,  # TODO: derive from rerank scores
            "sources": sources,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "cache_hit": False,
        }

        # 3. Cache result and log in background
        if req.use_cache:
            background_tasks.add_task(cache.set_query_response, req.query, response_data)

        background_tasks.add_task(
            _log_query,
            request_id=request_id,
            query=req.query,
            answer=answer,
            retrieved_doc_ids=[c.get("doc_id") for c in reranked],
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )

        record_query(
            intent=response_data["intent"],
            latency_seconds=latency_ms / 1000,
            cost_usd=cost_usd,
        )

        return QueryResponse(**response_data)

    except Exception as e:
        record_error("query_error")
        logger.error("Query failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest_documents(
    files: list[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
):
    """
    Ingest documents. Invalidates query cache after ingestion.
    """
    request_id = str(uuid.uuid4())

    try:
        file_paths = []
        for file in files:
            file_path = f"/tmp/{file.filename}"
            with open(file_path, "wb") as f:
                f.write(await file.read())
            file_paths.append((file_path, file.filename or ""))

        background_tasks.add_task(_ingest_and_invalidate, request_id, file_paths, db)

        return {
            "request_id": request_id,
            "status": "ingesting",
            "file_count": len(files),
        }

    except Exception as e:
        record_error("ingestion_error")
        logger.error("Ingestion failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics/prometheus")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    output, content_type = get_metrics_output()
    return Response(content=output, media_type=content_type)


@app.get("/metrics")
async def get_metrics(db: Session = Depends(get_db)):
    """Human-readable system metrics."""
    cache = get_cache()
    cache_stats = await cache.get_stats()

    doc_count = db.query(func.count(func.distinct(DocumentChunk.document_id))).scalar() or 0
    chunk_count = db.query(func.count(DocumentChunk.id)).scalar() or 0
    query_count = db.query(func.count(QueryLog.id)).scalar() or 0

    update_document_counts(doc_count, chunk_count)

    return {
        "documents": doc_count,
        "chunks": chunk_count,
        "queries": query_count,
        "cache": cache_stats,
        "database": "connected" if health_check() else "disconnected",
    }


@app.delete("/cache")
async def flush_cache():
    """Flush all cached query responses (admin use)."""
    cache = get_cache()
    try:
        client = await cache._get_client()
        keys = [k async for k in client.scan_iter("sdre:query:*")]
        for k in keys:
            await client.delete(k)
        return {"status": "ok", "keys_deleted": len(keys)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Background tasks ---

async def _log_query(
    request_id: str,
    query: str,
    answer: str,
    retrieved_doc_ids: list,
    cost_usd: float,
    latency_ms: int,
):
    db = SessionLocal()
    try:
        log = QueryLog(
            query_text=query,
            answer=answer[:500],
            retrieved_doc_ids=retrieved_doc_ids,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
        db.add(log)
        db.commit()
    finally:
        db.close()


async def _ingest_and_invalidate(request_id: str, file_paths: list[tuple], db: Session):
    """Ingest files and invalidate cache."""
    from app.ingest import get_ingester

    cache = get_cache()
    db = SessionLocal()

    try:
        ingester = get_ingester(db)
        for file_path, filename in file_paths:
            ext = Path(filename).suffix.lstrip(".") or "unknown"
            try:
                doc = ingester.ingest_file(file_path, title=Path(filename).stem)
                record_ingestion(file_type=ext, status="success")
                # Invalidate query cache — new doc may affect results
                await cache.invalidate_document(doc.id)
                Path(file_path).unlink(missing_ok=True)
            except Exception as e:
                record_ingestion(file_type=ext, status="error")
                logger.error("File ingestion failed", file=filename, error=str(e))

        logger.info("Ingestion complete", request_id=request_id, count=len(file_paths))
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")