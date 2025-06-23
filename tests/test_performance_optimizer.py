"""
Tests for Performance Optimizer Module
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from src.performance_optimizer import (
    PerformanceOptimizer,
    MemoryOptimizer,
    ConnectionPoolOptimizer,
    ResponseOptimizer,
    BatchProcessor,
    OptimizationConfig,
    PerformanceMetrics,
    optimize_performance
)


@pytest.fixture
def config():
    """Test configuration."""
    return OptimizationConfig(
        max_connections=10,
        connection_timeout=5,
        memory_threshold_mb=100,
        cpu_threshold_percent=50.0,
        cache_ttl_seconds=30,
        batch_size=3,
        gc_threshold=100,
        monitor_interval=1
    )


@pytest.fixture
def memory_optimizer(config):
    """Memory optimizer fixture."""
    return MemoryOptimizer(config)


@pytest.fixture
def connection_optimizer(config):
    """Connection optimizer fixture."""
    return ConnectionPoolOptimizer(config)


@pytest.fixture
def response_optimizer(config):
    """Response optimizer fixture."""
    return ResponseOptimizer(config)


@pytest.fixture
def batch_processor(config):
    """Batch processor fixture."""
    return BatchProcessor(config)


@pytest.fixture
def performance_optimizer(config):
    """Performance optimizer fixture."""
    return PerformanceOptimizer(config)


class TestMemoryOptimizer:
    """Test memory optimization functionality."""
    
    @patch('psutil.Process')
    def test_get_memory_usage(self, mock_process, memory_optimizer):
        """Test memory usage measurement."""
        # Mock memory info
        mock_process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
        
        memory_usage = memory_optimizer.get_memory_usage()
        assert memory_usage == 100.0
        assert len(memory_optimizer._memory_history) == 1
    
    @patch('gc.collect')
    @patch('weakref.finalize._registry')
    def test_optimize_memory(self, mock_registry, mock_gc, memory_optimizer):
        """Test memory optimization."""
        mock_gc.return_value = [10, 5, 2]
        mock_registry.clear = Mock()
        
        with patch.object(memory_optimizer, 'get_memory_usage', side_effect=[150.0, 120.0]):
            result = memory_optimizer.optimize_memory()
        
        assert result["initial_memory_mb"] == 150.0
        assert result["final_memory_mb"] == 120.0
        assert result["freed_mb"] == 30.0
        assert result["gc_collections"] == [10, 5, 2]
        assert mock_gc.called
        assert mock_registry.clear.called
    
    def test_get_memory_trend(self, memory_optimizer):
        """Test memory trend analysis."""
        # Add some memory history
        memory_optimizer._memory_history.extend([
            (time.time() - 60, 100.0),
            (time.time(), 110.0)
        ])
        
        trend = memory_optimizer.get_memory_trend()
        assert "trend" in trend
        assert "avg_usage" in trend
        assert trend["avg_usage"] == 105.0


class TestConnectionPoolOptimizer:
    """Test connection pool optimization."""
    
    def test_track_connection_create(self, connection_optimizer):
        """Test connection creation tracking."""
        connection_optimizer.track_connection("create")
        
        metrics = connection_optimizer.get_pool_metrics()
        assert metrics["active_connections"] == 1
        assert metrics["pool_stats"]["created"] == 1
    
    def test_track_connection_close(self, connection_optimizer):
        """Test connection close tracking."""
        connection_optimizer.track_connection("create")
        connection_optimizer.track_connection("close")
        
        metrics = connection_optimizer.get_pool_metrics()
        assert metrics["active_connections"] == 0
        assert metrics["pool_stats"]["closed"] == 1
    
    def test_pool_utilization_calculation(self, connection_optimizer):
        """Test pool utilization calculation."""
        for _ in range(5):
            connection_optimizer.track_connection("create")
        
        metrics = connection_optimizer.get_pool_metrics()
        assert metrics["utilization_percent"] == 50.0  # 5/10 * 100
    
    def test_high_utilization_recommendations(self, connection_optimizer):
        """Test recommendations for high utilization."""
        # Create connections to reach high utilization
        for _ in range(9):
            connection_optimizer.track_connection("create")
        
        metrics = connection_optimizer.get_pool_metrics()
        recommendations = metrics["recommendations"]
        assert any("increasing max_connections" in rec for rec in recommendations)


class TestResponseOptimizer:
    """Test response optimization functionality."""
    
    def test_cache_and_retrieve_response(self, response_optimizer):
        """Test caching and retrieving responses."""
        key = "test_key"
        response = {"data": "test_response"}
        
        # Cache response
        assert response_optimizer.cache_response(key, response)
        
        # Retrieve response
        cached = response_optimizer.get_cached_response(key)
        assert cached == response
        
        # Check cache stats
        metrics = response_optimizer.get_cache_metrics()
        assert metrics["cache_stats"]["hits"] == 1
        assert metrics["cache_stats"]["misses"] == 0
    
    def test_cache_miss(self, response_optimizer):
        """Test cache miss scenario."""
        cached = response_optimizer.get_cached_response("non_existent_key")
        assert cached is None
        
        metrics = response_optimizer.get_cache_metrics()
        assert metrics["cache_stats"]["misses"] == 1
    
    def test_cache_expiry(self, response_optimizer):
        """Test cache expiry functionality."""
        key = "expire_test"
        response = {"data": "will_expire"}
        
        # Cache with very short TTL
        response_optimizer.cache_response(key, response, ttl=1)
        
        # Should be available immediately
        assert response_optimizer.get_cached_response(key) == response
        
        # Wait for expiry
        time.sleep(1.1)
        
        # Should be expired
        assert response_optimizer.get_cached_response(key) is None
    
    def test_response_time_tracking(self, response_optimizer):
        """Test response time tracking."""
        response_times = [0.1, 0.2, 0.3]
        for rt in response_times:
            response_optimizer.track_response_time(rt)
        
        metrics = response_optimizer.get_cache_metrics()
        assert abs(metrics["avg_response_time"] - 0.2) < 0.001
    
    def test_cache_hit_rate_calculation(self, response_optimizer):
        """Test cache hit rate calculation."""
        # Generate hits and misses
        response_optimizer.cache_response("key1", {"data": "test"})
        response_optimizer.get_cached_response("key1")  # hit
        response_optimizer.get_cached_response("key2")  # miss
        
        metrics = response_optimizer.get_cache_metrics()
        assert metrics["hit_rate_percent"] == 50.0  # 1 hit, 1 miss


class TestBatchProcessor:
    """Test batch processing functionality."""
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, batch_processor):
        """Test batch processing."""
        # Override batch processing for testing
        async def mock_batch_process(requests):
            return [f"processed_{req}" for req in requests]
        
        batch_processor._batch_process_requests = mock_batch_process
        
        # Add requests to trigger batch processing
        futures = []
        for i in range(3):  # batch_size = 3
            future = await batch_processor.add_to_batch(f"request_{i}")
            futures.append(future)
        
        # Wait for batch processing
        results = await asyncio.gather(*futures)
        
        assert results == ["processed_request_0", "processed_request_1", "processed_request_2"]
        
        # Check batch stats
        metrics = batch_processor.get_batch_metrics()
        assert metrics["batch_stats"]["batches_processed"] == 1
        assert metrics["batch_stats"]["total_requests"] == 3
    
    def test_batch_metrics(self, batch_processor):
        """Test batch metrics calculation."""
        batch_processor._batch_stats = {
            "batches_processed": 2,
            "total_requests": 10,
            "avg_batch_size": 0,
            "processing_time": 2.0
        }
        
        metrics = batch_processor.get_batch_metrics()
        # avg_batch_size is calculated in get_batch_metrics()
        expected_avg_batch_size = 10 / 2  # total_requests / batches_processed
        assert metrics["batch_stats"]["avg_batch_size"] == expected_avg_batch_size
        assert metrics["avg_processing_time"] == 1.0


class TestPerformanceOptimizer:
    """Test main performance optimizer."""
    
    @patch('psutil.cpu_percent')
    def test_get_comprehensive_metrics(self, mock_cpu, performance_optimizer):
        """Test comprehensive metrics collection."""
        mock_cpu.return_value = 25.0
        
        with patch.object(performance_optimizer.memory_optimizer, 'get_memory_usage', return_value=50.0):
            asyncio.run(self._test_comprehensive_metrics(performance_optimizer))
    
    async def _test_comprehensive_metrics(self, performance_optimizer):
        """Helper for async metrics test."""
        metrics = await performance_optimizer.get_comprehensive_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.memory_usage_mb == 50.0
        assert metrics.cpu_usage_percent == 25.0
        assert metrics.performance_score > 0
    
    def test_performance_score_calculation(self, performance_optimizer):
        """Test performance score calculation."""
        # Test perfect score scenario
        score = performance_optimizer._calculate_performance_score(
            memory_usage=25.0,  # 25% of threshold
            cpu_usage=10.0,     # Low CPU
            cache_hit_rate=90.0  # High cache hit rate
        )
        
        assert score > 80  # Should be high score
        
        # Test poor performance scenario
        score = performance_optimizer._calculate_performance_score(
            memory_usage=150.0,  # Above threshold
            cpu_usage=90.0,      # High CPU
            cache_hit_rate=10.0  # Low cache hit rate
        )
        
        assert score < 50  # Should be low score
    
    @pytest.mark.asyncio
    async def test_optimize_all(self, performance_optimizer):
        """Test comprehensive optimization."""
        with patch.object(performance_optimizer.memory_optimizer, 'optimize_memory') as mock_optimize:
            mock_optimize.return_value = {"freed_mb": 10.0}
            
            with patch.object(performance_optimizer, 'get_comprehensive_metrics') as mock_metrics:
                mock_metrics.return_value = PerformanceMetrics(
                    timestamp="2023-01-01T00:00:00",
                    memory_usage_mb=50.0,
                    cpu_usage_percent=25.0,
                    active_connections=5,
                    cache_hit_rate=75.0,
                    avg_response_time=0.1,
                    optimization_suggestions=[],
                    performance_score=85.0
                )
                
                results = await performance_optimizer.optimize_all()
                
                assert "memory" in results
                assert "metrics" in results
                assert mock_optimize.called
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, performance_optimizer):
        """Test monitoring start/stop lifecycle."""
        assert not performance_optimizer._monitoring_active
        
        # Start monitoring
        performance_optimizer.start_monitoring()
        assert performance_optimizer._monitoring_active
        assert performance_optimizer._monitor_task is not None
        
        # Stop monitoring
        await performance_optimizer.stop_monitoring()
        assert not performance_optimizer._monitoring_active


class TestOptimizePerformanceDecorator:
    """Test performance optimization decorator."""
    
    @pytest.mark.asyncio
    async def test_decorator_without_optimizer(self):
        """Test decorator when no optimizer is available."""
        @optimize_performance()
        async def test_function():
            return "test_result"
        
        result = await test_function()
        assert result == "test_result"
    
    @pytest.mark.asyncio
    async def test_decorator_with_caching(self, performance_optimizer):
        """Test decorator with caching enabled."""
        @optimize_performance(cache_key="test_cache")
        async def test_function(value):
            return f"processed_{value}"
        
        # Attach optimizer to function
        test_function._optimizer = performance_optimizer
        
        # First call - should cache
        result1 = await test_function("input")
        assert result1 == "processed_input"
        
        # Second call - should use cache
        result2 = await test_function("input")
        assert result2 == "processed_input"
        
        # Verify cache was used
        cache_metrics = performance_optimizer.response_optimizer.get_cache_metrics()
        assert cache_metrics["cache_stats"]["hits"] >= 1


class TestOptimizationConfig:
    """Test optimization configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OptimizationConfig()
        
        assert config.max_connections == 100
        assert config.connection_timeout == 30
        assert config.memory_threshold_mb == 512
        assert config.cpu_threshold_percent == 80.0
        assert config.cache_ttl_seconds == 300
        assert config.batch_size == 10
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = OptimizationConfig(
            max_connections=50,
            memory_threshold_mb=256
        )
        
        assert config.max_connections == 50
        assert config.memory_threshold_mb == 256
        # Other values should remain default
        assert config.connection_timeout == 30


if __name__ == "__main__":
    pytest.main([__file__])