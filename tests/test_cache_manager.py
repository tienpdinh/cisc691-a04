"""
Tests for the Redis cache manager implementation.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from src.cache_manager import CacheManager, CacheStats, get_cache_manager


@pytest.fixture
def cache_config():
    """Sample cache configuration for testing."""
    return {
        'cache': {
            'enabled': True,
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 0,
            'ttl_seconds': {
                'llm_responses': 3600,
                'document_retrieval': 1800,
                'query_responses': 900,
                'embeddings': 7200
            },
            'max_cache_size_mb': 100
        }
    }


@pytest.fixture
def disabled_cache_config():
    """Cache configuration with caching disabled."""
    return {
        'cache': {
            'enabled': False,
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 0
        }
    }


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock_redis = Mock()
    mock_redis.ping.return_value = True
    mock_redis.get.return_value = None
    mock_redis.setex.return_value = True
    mock_redis.delete.return_value = 1
    mock_redis.keys.return_value = []
    mock_redis.info.return_value = {'used_memory': 1024 * 1024}  # 1MB
    return mock_redis


class TestCacheManager:
    """Test cases for CacheManager class."""

    def test_cache_manager_init_enabled(self, cache_config):
        """Test CacheManager initialization with caching enabled."""
        with patch('src.cache_manager.redis.Redis') as mock_redis_class:
            mock_redis_instance = Mock()
            mock_redis_instance.ping.return_value = True
            mock_redis_class.return_value = mock_redis_instance
            
            cache_manager = CacheManager(cache_config)
            
            assert cache_manager.enabled is True
            assert cache_manager.redis_client is not None
            mock_redis_class.assert_called_once()

    def test_cache_manager_init_disabled(self, disabled_cache_config):
        """Test CacheManager initialization with caching disabled."""
        cache_manager = CacheManager(disabled_cache_config)
        
        assert cache_manager.enabled is False
        assert cache_manager.redis_client is None

    def test_cache_manager_init_redis_failure(self, cache_config):
        """Test CacheManager initialization when Redis connection fails."""
        with patch('src.cache_manager.redis.Redis') as mock_redis_class:
            mock_redis_class.side_effect = Exception("Connection failed")
            
            cache_manager = CacheManager(cache_config)
            
            assert cache_manager.enabled is False
            assert cache_manager.redis_client is None

    def test_generate_key_string(self, cache_config):
        """Test cache key generation with string data."""
        cache_manager = CacheManager(cache_config)
        
        key = cache_manager._generate_key('test', 'hello world')
        
        assert key.startswith('rag:test:')
        assert len(key.split(':')[2]) == 16  # Hash length

    def test_generate_key_dict(self, cache_config):
        """Test cache key generation with dictionary data."""
        cache_manager = CacheManager(cache_config)
        
        data = {'query': 'test', 'model': 'llama3.1'}
        key = cache_manager._generate_key('query', data)
        
        assert key.startswith('rag:query:')
        assert len(key.split(':')[2]) == 16

    def test_generate_key_consistency(self, cache_config):
        """Test that same data generates same key."""
        cache_manager = CacheManager(cache_config)
        
        data = {'query': 'test', 'model': 'llama3.1'}
        key1 = cache_manager._generate_key('query', data)
        key2 = cache_manager._generate_key('query', data)
        
        assert key1 == key2

    def test_get_ttl(self, cache_config):
        """Test TTL retrieval for different cache types."""
        cache_manager = CacheManager(cache_config)
        
        assert cache_manager._get_ttl('llm_responses') == 3600
        assert cache_manager._get_ttl('document_retrieval') == 1800
        assert cache_manager._get_ttl('unknown_type') == 3600  # Default

    @pytest.mark.asyncio
    async def test_get_cache_disabled(self, disabled_cache_config):
        """Test get operation when caching is disabled."""
        cache_manager = CacheManager(disabled_cache_config)
        
        result = await cache_manager.get('test_key', 'test')
        
        assert result is None

    @pytest.mark.asyncio
    async def test_get_cache_miss(self, cache_config, mock_redis):
        """Test get operation with cache miss."""
        with patch('src.cache_manager.redis.Redis', return_value=mock_redis):
            cache_manager = CacheManager(cache_config)
            mock_redis.get.return_value = None
            
            result = await cache_manager.get('test_key', 'test')
            
            assert result is None
            assert cache_manager._stats.misses == 1

    @pytest.mark.asyncio
    async def test_get_cache_hit(self, cache_config, mock_redis):
        """Test get operation with cache hit."""
        with patch('src.cache_manager.redis.Redis', return_value=mock_redis):
            cache_manager = CacheManager(cache_config)
            test_data = {'response': 'Hello World'}
            mock_redis.get.return_value = json.dumps(test_data)
            
            result = await cache_manager.get('test_key', 'test')
            
            assert result == test_data
            assert cache_manager._stats.hits == 1

    @pytest.mark.asyncio
    async def test_set_cache_disabled(self, disabled_cache_config):
        """Test set operation when caching is disabled."""
        cache_manager = CacheManager(disabled_cache_config)
        
        result = await cache_manager.set('test_key', {'data': 'test'}, 'test')
        
        assert result is False

    @pytest.mark.asyncio
    async def test_set_cache_success(self, cache_config, mock_redis):
        """Test successful set operation."""
        with patch('src.cache_manager.redis.Redis', return_value=mock_redis):
            cache_manager = CacheManager(cache_config)
            mock_redis.setex.return_value = True
            
            test_data = {'response': 'Hello World'}
            result = await cache_manager.set('test_key', test_data, 'llm_responses')
            
            assert result is True
            mock_redis.setex.assert_called_once()
            # Check TTL was used correctly
            call_args = mock_redis.setex.call_args
            assert call_args[0][1] == 3600  # TTL for llm_responses

    @pytest.mark.asyncio
    async def test_delete_cache_success(self, cache_config, mock_redis):
        """Test successful delete operation."""
        with patch('src.cache_manager.redis.Redis', return_value=mock_redis):
            cache_manager = CacheManager(cache_config)
            mock_redis.delete.return_value = 1
            
            result = await cache_manager.delete('test_key', 'test')
            
            assert result is True
            mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_prefix(self, cache_config, mock_redis):
        """Test clearing keys with prefix."""
        with patch('src.cache_manager.redis.Redis', return_value=mock_redis):
            cache_manager = CacheManager(cache_config)
            mock_redis.keys.return_value = ['rag:llm:key1', 'rag:llm:key2']
            mock_redis.delete.return_value = 2
            
            result = await cache_manager.clear_prefix('llm')
            
            assert result == 2
            mock_redis.keys.assert_called_with('rag:llm:*')
            mock_redis.delete.assert_called_with('rag:llm:key1', 'rag:llm:key2')

    @pytest.mark.asyncio
    async def test_get_stats(self, cache_config, mock_redis):
        """Test getting cache statistics."""
        with patch('src.cache_manager.redis.Redis', return_value=mock_redis):
            cache_manager = CacheManager(cache_config)
            cache_manager._stats.hits = 10
            cache_manager._stats.misses = 5
            mock_redis.info.return_value = {'used_memory': 2048 * 1024}  # 2MB
            mock_redis.keys.return_value = ['key1', 'key2', 'key3']
            
            stats = await cache_manager.get_stats()
            
            assert stats.hits == 10
            assert stats.misses == 5
            assert stats.total_keys == 3
            assert stats.memory_usage_mb == 2.0
            assert stats.hit_rate == 10 / 15  # 10 hits out of 15 total

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, cache_config, mock_redis):
        """Test health check when cache is healthy."""
        with patch('src.cache_manager.redis.Redis', return_value=mock_redis):
            cache_manager = CacheManager(cache_config)
            mock_redis.ping.return_value = True
            mock_redis.info.return_value = {'used_memory': 1024 * 1024}
            mock_redis.keys.return_value = ['key1']
            
            health = await cache_manager.health_check()
            
            assert health['status'] == 'healthy'
            assert health['enabled'] is True
            assert 'stats' in health
            assert 'redis_info' in health

    @pytest.mark.asyncio
    async def test_health_check_disabled(self, disabled_cache_config):
        """Test health check when caching is disabled."""
        cache_manager = CacheManager(disabled_cache_config)
        
        health = await cache_manager.health_check()
        
        assert health['status'] == 'disabled'
        assert health['enabled'] is False

    @pytest.mark.asyncio
    async def test_health_check_error(self, cache_config, mock_redis):
        """Test health check when Redis is unhealthy."""
        with patch('src.cache_manager.redis.Redis', return_value=mock_redis):
            cache_manager = CacheManager(cache_config)
            mock_redis.ping.side_effect = Exception("Connection failed")
            
            health = await cache_manager.health_check()
            
            assert health['status'] == 'error'
            assert health['enabled'] is False
            assert 'Redis health check failed' in health['message']

    def test_invalidate_query_cache(self, cache_config, mock_redis):
        """Test query cache invalidation."""
        with patch('src.cache_manager.redis.Redis', return_value=mock_redis):
            cache_manager = CacheManager(cache_config)
            
            cache_manager.invalidate_query_cache("What is AI?")
            
            # Should delete multiple keys related to the query
            assert mock_redis.delete.call_count >= 3  # Different cache types


class TestGlobalCacheManager:
    """Test cases for global cache manager singleton."""

    def test_get_cache_manager_singleton(self, cache_config):
        """Test that get_cache_manager returns singleton instance."""
        with patch('src.cache_manager.redis.Redis'):
            # Reset global instance
            import src.cache_manager
            src.cache_manager._cache_manager = None
            
            manager1 = get_cache_manager(cache_config)
            manager2 = get_cache_manager()  # No config on second call
            
            assert manager1 is manager2

    def test_get_cache_manager_no_config(self):
        """Test get_cache_manager with no config returns None."""
        import src.cache_manager
        src.cache_manager._cache_manager = None
        
        manager = get_cache_manager()
        
        assert manager is None


class TestCacheStats:
    """Test cases for CacheStats dataclass."""

    def test_cache_stats_defaults(self):
        """Test CacheStats default values."""
        stats = CacheStats()
        
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.total_keys == 0
        assert stats.memory_usage_mb == 0.0
        assert stats.hit_rate == 0.0

    def test_cache_stats_custom_values(self):
        """Test CacheStats with custom values."""
        stats = CacheStats(
            hits=100,
            misses=25,
            total_keys=50,
            memory_usage_mb=5.5,
            hit_rate=0.8
        )
        
        assert stats.hits == 100
        assert stats.misses == 25
        assert stats.total_keys == 50
        assert stats.memory_usage_mb == 5.5
        assert stats.hit_rate == 0.8