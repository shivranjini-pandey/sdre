"""
Prometheus metrics for SDRE monitoring.

Tracks:
- Query latency (p50, p95, p99)
- Cache hit/miss rates
- Retrieval quality scores
- LLM cost and token usage
- Error rates by type
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import time
from functools import wraps
from typing import Callable
import structlog

logger = structlog.get_logger()

# --- Registry ---
registry = CollectorRegistry()

# --- Counters ---
QUERY_TOTAL = Counter(
    "sdre_queries_total",
    "Total number of queries processed",
    ["intent", "status"],  # labels
    registry=registry,
)

CACHE_HITS = Counter(
    "sdre_cache_hits_total",
    "Total cache hits",
    ["cache_type"],
    registry=registry,
)

CACHE_MISSES = Counter(
    "sdre_cache_misses_total",
    "Total cache misses",
    ["cache_type"],
    registry=registry,
)

INGESTION_TOTAL = Counter(
    "sdre_ingestion_total",
    "Total documents ingested",
    ["file_type", "status"],
    registry=registry,
)

LLM_TOKENS_TOTAL = Counter(
    "sdre_llm_tokens_total",
    "Total LLM tokens used",
    ["token_type"],  # prompt / completion
    registry=registry,
)

ERROR_TOTAL = Counter(
    "sdre_errors_total",
    "Total errors by type",
    ["error_type"],
    registry=registry,
)

# --- Histograms (latency distributions) ---
QUERY_LATENCY = Histogram(
    "sdre_query_latency_seconds",
    "Query end-to-end latency in seconds",
    ["intent"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
    registry=registry,
)

RETRIEVAL_LATENCY = Histogram(
    "sdre_retrieval_latency_seconds",
    "Retrieval latency in seconds",
    ["retrieval_type"],  # dense / sparse / hybrid
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
    registry=registry,
)

RERANKING_LATENCY = Histogram(
    "sdre_reranking_latency_seconds",
    "Reranking latency in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0],
    registry=registry,
)

GENERATION_LATENCY = Histogram(
    "sdre_generation_latency_seconds",
    "LLM generation latency in seconds",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    registry=registry,
)

LLM_COST = Histogram(
    "sdre_llm_cost_usd",
    "LLM cost per query in USD",
    buckets=[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
    registry=registry,
)

# --- Gauges (current values) ---
DOCUMENTS_TOTAL = Gauge(
    "sdre_documents_total",
    "Total documents in the system",
    registry=registry,
)

CHUNKS_TOTAL = Gauge(
    "sdre_chunks_total",
    "Total document chunks in the system",
    registry=registry,
)

CACHE_HIT_RATE = Gauge(
    "sdre_cache_hit_rate",
    "Current cache hit rate (0-1)",
    registry=registry,
)

RETRIEVAL_SCORE_AVG = Gauge(
    "sdre_retrieval_score_avg",
    "Average retrieval score for recent queries",
    registry=registry,
)


# --- Helpers ---

def record_query(intent: str, latency_seconds: float, cost_usd: float,
                 status: str = "success", prompt_tokens: int = 0,
                 completion_tokens: int = 0):
    """Record metrics for a completed query."""
    QUERY_TOTAL.labels(intent=intent, status=status).inc()
    QUERY_LATENCY.labels(intent=intent).observe(latency_seconds)
    LLM_COST.observe(cost_usd)
    LLM_TOKENS_TOTAL.labels(token_type="prompt").inc(prompt_tokens)
    LLM_TOKENS_TOTAL.labels(token_type="completion").inc(completion_tokens)


def record_retrieval(retrieval_type: str, latency_seconds: float):
    """Record retrieval latency."""
    RETRIEVAL_LATENCY.labels(retrieval_type=retrieval_type).observe(latency_seconds)


def record_cache_event(cache_type: str, hit: bool):
    """Record a cache hit or miss."""
    if hit:
        CACHE_HITS.labels(cache_type=cache_type).inc()
    else:
        CACHE_MISSES.labels(cache_type=cache_type).inc()


def record_error(error_type: str):
    """Record an error."""
    ERROR_TOTAL.labels(error_type=error_type).inc()


def record_ingestion(file_type: str, status: str = "success"):
    """Record a document ingestion."""
    INGESTION_TOTAL.labels(file_type=file_type, status=status).inc()


def update_document_counts(doc_count: int, chunk_count: int):
    """Update document and chunk gauges."""
    DOCUMENTS_TOTAL.set(doc_count)
    CHUNKS_TOTAL.set(chunk_count)


def get_metrics_output() -> tuple[bytes, str]:
    """Get Prometheus metrics output."""
    return generate_latest(registry), CONTENT_TYPE_LATEST


# --- Decorator ---

def track_latency(histogram: Histogram, label_value: str = "default"):
    """Decorator to track function latency in a histogram."""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                histogram.labels(label_value).observe(time.time() - start)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                histogram.labels(label_value).observe(time.time() - start)

        import asyncio
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
