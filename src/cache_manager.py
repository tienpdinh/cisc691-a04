"""
Redis-based caching manager for RAG system components.

Provides caching capabilities for:
- LLM responses
- Document retrieval results
- Query embeddings
- Vector search results
"""

import json
import hashlib
import logging
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
import redis
from dataclasses import dataclass, asdict

@dataclass
class CacheStats:
    """Cache statistics for monitoring."""
    hits: int = 0
    misses: int = 0
    total_keys: int = 0
    memory_usage_mb: float = 0.0
    hit_rate: float = 0.0

class CacheManager:
    """
    Centralized cache manager using Redis for RAG system components.
    
    Features:
    - Automatic key generation with hashing
    - TTL management for different data types
    - Cache statistics and monitoring
    - Fallback handling when Redis is unavailable
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cache manager with configuration.
        
        Args:
            config: Configuration dictionary with cache settings
        """
        self.config = config
        self.cache_config = config.get('cache', {})
        self.enabled = self.cache_config.get('enabled', False)
        self.logger = logging.getLogger(__name__)
        
        # Cache statistics
        self._stats = CacheStats()
        
        # Initialize Redis client
        self.redis_client = None
        if self.enabled:
            self._init_redis()
    
    def _init_redis(self) -> None:
        """Initialize Redis connection with error handling."""
        try:
            self.redis_client = redis.Redis(
                host=self.cache_config.get('redis_host', 'localhost'),
                port=self.cache_config.get('redis_port', 6379),
                db=self.cache_config.get('redis_db', 0),
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.redis_client.ping()
            self.logger.info("Redis cache connected successfully")
            
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            self.redis_client = None
            self.enabled = False
    
    def _generate_key(self, prefix: str, data: Union[str, Dict, List]) -> str:
        """
        Generate cache key with consistent hashing.
        
        Args:
            prefix: Key prefix (e.g., 'llm', 'retrieval', 'embedding')
            data: Data to hash for key generation
            
        Returns:
            Generated cache key
        """
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        hash_obj = hashlib.sha256(data_str.encode())
        hash_hex = hash_obj.hexdigest()[:16]  # Use first 16 chars
        
        return f"rag:{prefix}:{hash_hex}"
    
    def _get_ttl(self, cache_type: str) -> int:
        """Get TTL for specific cache type."""
        ttl_config = self.cache_config.get('ttl_seconds', {})
        return ttl_config.get(cache_type, 3600)  # Default 1 hour
    
    async def get(self, key: str, cache_type: str = 'default') -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key or data to generate key from
            cache_type: Type of cache for TTL determination
            
        Returns:
            Cached value or None if not found
        """
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            # Generate key if needed
            if not key.startswith('rag:'):
                key = self._generate_key(cache_type, key)
            
            value = self.redis_client.get(key)
            
            if value:
                self._stats.hits += 1
                self.logger.debug(f"Cache hit for key: {key}")
                return json.loads(value)
            else:
                self._stats.misses += 1
                self.logger.debug(f"Cache miss for key: {key}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Cache get error: {e}")
            self._stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, cache_type: str = 'default') -> bool:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key or data to generate key from
            value: Value to cache
            cache_type: Type of cache for TTL determination
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            # Generate key if needed
            if not key.startswith('rag:'):
                key = self._generate_key(cache_type, key)
            
            ttl = self._get_ttl(cache_type)
            serialized_value = json.dumps(value)
            
            result = self.redis_client.setex(key, ttl, serialized_value)
            
            if result:
                self.logger.debug(f"Cache set for key: {key} (TTL: {ttl}s)")
            
            return bool(result)
            
        except Exception as e:
            self.logger.warning(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str, cache_type: str = 'default') -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key or data to generate key from
            cache_type: Type of cache for key generation
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            # Generate key if needed
            if not key.startswith('rag:'):
                key = self._generate_key(cache_type, key)
            
            result = self.redis_client.delete(key)
            self.logger.debug(f"Cache delete for key: {key}")
            return bool(result)
            
        except Exception as e:
            self.logger.warning(f"Cache delete error: {e}")
            return False
    
    async def clear_prefix(self, prefix: str) -> int:
        """
        Clear all keys with given prefix.
        
        Args:
            prefix: Key prefix to clear (e.g., 'llm', 'retrieval')
            
        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self.redis_client:
            return 0
        
        try:
            pattern = f"rag:{prefix}:*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                deleted = self.redis_client.delete(*keys)
                self.logger.info(f"Cleared {deleted} keys with prefix: {prefix}")
                return deleted
            
            return 0
            
        except Exception as e:
            self.logger.warning(f"Cache clear error: {e}")
            return 0
    
    async def get_stats(self) -> CacheStats:
        """
        Get cache statistics.
        
        Returns:
            CacheStats object with current statistics
        """
        if not self.enabled or not self.redis_client:
            return self._stats
        
        try:
            # Update stats from Redis
            info = self.redis_client.info('memory')
            self._stats.memory_usage_mb = info.get('used_memory', 0) / (1024 * 1024)
            
            # Count total keys
            keys = self.redis_client.keys('rag:*')
            self._stats.total_keys = len(keys)
            
            # Calculate hit rate
            total_requests = self._stats.hits + self._stats.misses
            if total_requests > 0:
                self._stats.hit_rate = self._stats.hits / total_requests
            
            return self._stats
            
        except Exception as e:
            self.logger.warning(f"Cache stats error: {e}")
            return self._stats
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform cache health check.
        
        Returns:
            Health status and statistics
        """
        if not self.enabled:
            return {
                'status': 'disabled',
                'enabled': False,
                'message': 'Caching is disabled in configuration'
            }
        
        if not self.redis_client:
            return {
                'status': 'error',
                'enabled': False,
                'message': 'Redis client not initialized'
            }
        
        try:
            # Test Redis connection
            self.redis_client.ping()
            stats = await self.get_stats()
            
            return {
                'status': 'healthy',
                'enabled': True,
                'stats': asdict(stats),
                'redis_info': {
                    'host': self.cache_config.get('redis_host'),
                    'port': self.cache_config.get('redis_port'),
                    'db': self.cache_config.get('redis_db')
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'enabled': False,
                'message': f'Redis health check failed: {e}'
            }
    
    def invalidate_query_cache(self, query: str) -> None:
        """
        Invalidate all cache entries related to a specific query.
        
        Args:
            query: Query string to invalidate
        """
        if not self.enabled or not self.redis_client:
            return
        
        try:
            # Generate possible cache keys for this query
            cache_types = ['query_responses', 'llm_responses', 'document_retrieval']
            
            for cache_type in cache_types:
                key = self._generate_key(cache_type, query)
                self.redis_client.delete(key)
            
            # Also clear embeddings cache for this query
            embedding_key = self._generate_key('embeddings', query)
            self.redis_client.delete(embedding_key)
            
            self.logger.debug(f"Invalidated cache for query: {query[:50]}...")
            
        except Exception as e:
            self.logger.warning(f"Cache invalidation error: {e}")

# Global cache manager instance (singleton)
_cache_manager = None

def get_cache_manager(config: Dict[str, Any] = None) -> CacheManager:
    """
    Get global cache manager instance.
    
    Args:
        config: Configuration dictionary (only used on first call)
        
    Returns:
        CacheManager instance
    """
    global _cache_manager
    
    if _cache_manager is None and config is not None:
        _cache_manager = CacheManager(config)
    
    return _cache_manager