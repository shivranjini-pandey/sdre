"""
Redis caching layer for query results and embeddings.

Caches:
- Query responses (full answer + sources)
- Embedding vectors (avoid re-encoding same text)
- BM25 search results (for repeated queries)
"""

import json
import hashlib
from typing import Optional, Any
import structlog
import redis.asyncio as aioredis

from app.config import settings

logger = structlog.get_logger()


def _hash_key(prefix: str, value: str) -> str:
    """Generate a cache key from a prefix and value."""
    h = hashlib.sha256(value.encode()).hexdigest()[:16]
    return f"sdre:{prefix}:{h}"


class CacheManager:
    """
    Async Redis cache manager.

    Handles serialization, TTL, and cache key namespacing.
    """

    def __init__(self, redis_url: str, ttl_seconds: int = 86400):
        self.redis_url = redis_url
        self.ttl = ttl_seconds
        self._client: Optional[aioredis.Redis] = None

    async def _get_client(self) -> aioredis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            self._client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._client

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Returns:
            Deserialized value or None if not found / on error
        """
        try:
            client = await self._get_client()
            raw = await client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.warning("Cache get failed", key=key, error=str(e))
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.

        Returns:
            True if set successfully
        """
        try:
            client = await self._get_client()
            serialized = json.dumps(value)
            await client.setex(key, ttl or self.ttl, serialized)
            return True
        except Exception as e:
            logger.warning("Cache set failed", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            client = await self._get_client()
            await client.delete(key)
            return True
        except Exception as e:
            logger.warning("Cache delete failed", key=key, error=str(e))
            return False

    async def health_check(self) -> bool:
        """Check if Redis is reachable."""
        try:
            client = await self._get_client()
            await client.ping()
            return True
        except Exception:
            return False

    async def close(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None

    # --- Domain-specific helpers ---

    async def get_query_response(self, query: str) -> Optional[dict]:
        """Get cached response for a query."""
        key = _hash_key("query", query.lower().strip())
        result = await self.get(key)
        if result:
            logger.info("Cache hit", type="query_response", query=query[:50])
        return result

    async def set_query_response(self, query: str, response: dict) -> bool:
        """Cache a query response."""
        key = _hash_key("query", query.lower().strip())
        return await self.set(key, response)

    async def get_embedding(self, text: str) -> Optional[list]:
        """Get cached embedding for text."""
        key = _hash_key("embedding", text)
        return await self.get(key)

    async def set_embedding(self, text: str, embedding: list) -> bool:
        """Cache an embedding vector. Use longer TTL — embeddings don't change."""
        key = _hash_key("embedding", text)
        return await self.set(key, embedding, ttl=self.ttl * 7)  # 7 days

    async def invalidate_document(self, doc_id: str):
        """
        Invalidate all cached queries when a document is updated."""
        
        try:
            client = await self._get_client()
            # Scan and delete all query cache keys
            async for key in client.scan_iter("sdre:query:*"):
                await client.delete(key)
            logger.info("Query cache invalidated", doc_id=doc_id)
        except Exception as e:
            logger.warning("Cache invalidation failed", error=str(e))

    async def get_stats(self) -> dict:
        """Get cache statistics."""
        try:
            client = await self._get_client()
            info = await client.info("stats")
            keyspace = await client.info("keyspace")

            query_keys = len([k async for k in client.scan_iter("sdre:query:*")])
            embedding_keys = len([k async for k in client.scan_iter("sdre:embedding:*")])

            return {
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0) /
                    max(1, info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0))
                ),
                "query_cache_entries": query_keys,
                "embedding_cache_entries": embedding_keys,
            }
        except Exception as e:
            logger.warning("Cache stats failed", error=str(e))
            return {}


# Singleton
_cache: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """Get or create cache manager (singleton)."""
    global _cache
    if _cache is None:
        _cache = CacheManager(
            redis_url=settings.REDIS_URL,
            ttl_seconds=settings.CACHE_TTL_SECONDS,
        )
    return _cache