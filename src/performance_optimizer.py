"""
Performance Optimization Module

Provides performance optimization tools for RAG API including:
- Memory usage optimization
- Connection pooling optimization
- Response caching optimization
- Request batching optimization
- Database query optimization
- Concurrent processing optimization
"""

import asyncio
import logging
import time
import psutil
import gc
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from datetime import datetime, timedelta
import concurrent.futures
from functools import wraps
import threading
from collections import defaultdict, deque
import weakref


@dataclass
class PerformanceMetrics:
    """Container for performance optimization metrics."""
    timestamp: str
    memory_usage_mb: float
    cpu_usage_percent: float
    active_connections: int
    cache_hit_rate: float
    avg_response_time: float
    optimization_suggestions: List[str]
    performance_score: float


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    max_connections: int = 100
    connection_timeout: int = 30
    memory_threshold_mb: int = 512
    cpu_threshold_percent: float = 80.0
    cache_ttl_seconds: int = 300
    batch_size: int = 10
    gc_threshold: int = 1000
    monitor_interval: int = 60


class MemoryOptimizer:
    """Optimizes memory usage through intelligent garbage collection and monitoring."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._memory_history = deque(maxlen=100)
        self._gc_stats = {"collections": 0, "freed_objects": 0}
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self._memory_history.append((time.time(), memory_mb))
        return memory_mb
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization."""
        initial_memory = self.get_memory_usage()
        
        # Force garbage collection
        collected_counts = gc.collect()
        self._gc_stats["collections"] += 1
        self._gc_stats["freed_objects"] += sum(collected_counts)
        
        # Clear weak references
        weakref.finalize._registry.clear()
        
        final_memory = self.get_memory_usage()
        freed_mb = initial_memory - final_memory
        
        result = {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "freed_mb": freed_mb,
            "gc_collections": collected_counts,
            "total_gc_stats": self._gc_stats.copy()
        }
        
        if freed_mb > 0:
            self.logger.info(f"Memory optimization freed {freed_mb:.2f} MB")
        
        return result
    
    def get_memory_trend(self) -> Dict[str, float]:
        """Analyze memory usage trend."""
        if len(self._memory_history) < 2:
            return {"trend": 0.0, "avg_usage": 0.0}
        
        recent_usage = [usage for _, usage in self._memory_history]
        avg_usage = sum(recent_usage) / len(recent_usage)
        
        # Calculate trend (MB per minute)
        if len(self._memory_history) >= 2:
            time_diff = self._memory_history[-1][0] - self._memory_history[0][0]
            memory_diff = self._memory_history[-1][1] - self._memory_history[0][1]
            trend = (memory_diff / max(time_diff, 1)) * 60  # MB per minute
        else:
            trend = 0.0
        
        return {"trend": trend, "avg_usage": avg_usage}


class ConnectionPoolOptimizer:
    """Optimizes connection pooling for better resource utilization."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._active_connections = 0
        self._connection_history = deque(maxlen=1000)
        self._pool_stats = {
            "created": 0,
            "closed": 0,
            "reused": 0,
            "timeouts": 0
        }
        self._lock = threading.Lock()
    
    def track_connection(self, action: str):
        """Track connection pool events."""
        with self._lock:
            timestamp = time.time()
            
            if action == "create":
                self._active_connections += 1
                self._pool_stats["created"] += 1
            elif action == "close":
                self._active_connections = max(0, self._active_connections - 1)
                self._pool_stats["closed"] += 1
            elif action == "reuse":
                self._pool_stats["reused"] += 1
            elif action == "timeout":
                self._pool_stats["timeouts"] += 1
            
            self._connection_history.append((timestamp, action, self._active_connections))
    
    def get_pool_metrics(self) -> Dict[str, Any]:
        """Get connection pool metrics."""
        with self._lock:
            utilization = (self._active_connections / self.config.max_connections) * 100
            
            return {
                "active_connections": self._active_connections,
                "max_connections": self.config.max_connections,
                "utilization_percent": utilization,
                "pool_stats": self._pool_stats.copy(),
                "recommendations": self._get_pool_recommendations(utilization)
            }
    
    def _get_pool_recommendations(self, utilization: float) -> List[str]:
        """Generate pool optimization recommendations."""
        recommendations = []
        
        if utilization > 85:  # Changed from 90 to 85
            recommendations.append("Consider increasing max_connections")
        elif utilization < 20:
            recommendations.append("Consider decreasing max_connections to save memory")
        
        if self._pool_stats["timeouts"] > self._pool_stats["created"] * 0.1:
            recommendations.append("Consider increasing connection_timeout")
        
        reuse_rate = (self._pool_stats["reused"] / max(self._pool_stats["created"], 1)) * 100
        if reuse_rate < 50:
            recommendations.append("Low connection reuse rate - check connection lifecycle")
        
        return recommendations


class ResponseOptimizer:
    """Optimizes API response performance through caching and compression."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._cache = {}
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0
        }
        self._response_times = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def cache_response(self, key: str, response: Any, ttl: Optional[int] = None) -> bool:
        """Cache a response with TTL."""
        ttl = ttl or self.config.cache_ttl_seconds
        expiry = time.time() + ttl
        
        with self._lock:
            # Evict expired entries
            self._evict_expired()
            
            self._cache[key] = {
                "data": response,
                "expiry": expiry,
                "access_count": 0,
                "created": time.time()
            }
            self._cache_stats["size"] = len(self._cache)
            
        return True
    
    def get_cached_response(self, key: str) -> Optional[Any]:
        """Retrieve cached response."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                if time.time() < entry["expiry"]:
                    entry["access_count"] += 1
                    self._cache_stats["hits"] += 1
                    return entry["data"]
                else:
                    # Expired
                    del self._cache[key]
                    self._cache_stats["size"] = len(self._cache)
            
            self._cache_stats["misses"] += 1
            return None
    
    def _evict_expired(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time >= entry["expiry"]
        ]
        
        for key in expired_keys:
            del self._cache[key]
            self._cache_stats["evictions"] += 1
        
        self._cache_stats["size"] = len(self._cache)
    
    def track_response_time(self, response_time: float):
        """Track response time for optimization analysis."""
        self._response_times.append(response_time)
    
    def get_cache_metrics(self) -> Dict[str, Any]:
        """Get caching performance metrics."""
        with self._lock:
            total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
            hit_rate = (self._cache_stats["hits"] / max(total_requests, 1)) * 100
            
            avg_response_time = (
                sum(self._response_times) / len(self._response_times)
                if self._response_times else 0
            )
            
            return {
                "cache_stats": self._cache_stats.copy(),
                "hit_rate_percent": hit_rate,
                "avg_response_time": avg_response_time,
                "cache_size": len(self._cache),
                "recommendations": self._get_cache_recommendations(hit_rate)
            }
    
    def _get_cache_recommendations(self, hit_rate: float) -> List[str]:
        """Generate cache optimization recommendations."""
        recommendations = []
        
        if hit_rate < 30:
            recommendations.append("Low cache hit rate - consider increasing TTL")
        
        if len(self._cache) > 1000:
            recommendations.append("Large cache size - consider implementing LRU eviction")
        
        if self._cache_stats["evictions"] > self._cache_stats["hits"]:
            recommendations.append("High eviction rate - consider increasing cache size or TTL")
        
        return recommendations


class BatchProcessor:
    """Optimizes request processing through intelligent batching."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._pending_requests = []
        self._batch_stats = {
            "batches_processed": 0,
            "total_requests": 0,
            "avg_batch_size": 0,
            "processing_time": 0
        }
        self._lock = threading.Lock()
    
    async def add_to_batch(self, request_data: Any) -> Any:
        """Add request to batch for processing."""
        future = asyncio.Future()
        
        with self._lock:
            self._pending_requests.append({
                "data": request_data,
                "timestamp": time.time(),
                "future": future
            })
            should_process = len(self._pending_requests) >= self.config.batch_size
        
        # Process batch if it's full
        if should_process:
            await self._process_batch()
        
        return future
    
    async def _process_batch(self):
        """Process accumulated batch of requests."""
        with self._lock:
            if not self._pending_requests:
                return
            
            batch = self._pending_requests.copy()
            self._pending_requests.clear()
        
        start_time = time.time()
        
        try:
            # Simulate batch processing (replace with actual implementation)
            results = await self._batch_process_requests([req["data"] for req in batch])
            
            # Complete futures
            for i, request in enumerate(batch):
                result = results[i] if i < len(results) else None
                request["future"].set_result(result)
        
        except Exception as e:
            # Handle batch processing error
            for request in batch:
                request["future"].set_exception(e)
        
        # Update stats
        processing_time = time.time() - start_time
        batch_size = len(batch)
        
        with self._lock:
            self._batch_stats["batches_processed"] += 1
            self._batch_stats["total_requests"] += batch_size
            self._batch_stats["processing_time"] += processing_time
            
            self._batch_stats["avg_batch_size"] = (
                self._batch_stats["total_requests"] / 
                max(self._batch_stats["batches_processed"], 1)
            )
    
    async def _batch_process_requests(self, requests: List[Any]) -> List[Any]:
        """Process a batch of requests efficiently."""
        # Placeholder for actual batch processing logic
        # This would be replaced with domain-specific batch processing
        results = []
        for request in requests:
            # Simulate processing
            await asyncio.sleep(0.01)
            results.append(f"processed_{request}")
        return results
    
    def get_batch_metrics(self) -> Dict[str, Any]:
        """Get batch processing metrics."""
        with self._lock:
            avg_processing_time = (
                self._batch_stats["processing_time"] / 
                max(self._batch_stats["batches_processed"], 1)
            )
            
            # Calculate current avg_batch_size
            current_avg_batch_size = (
                self._batch_stats["total_requests"] / 
                max(self._batch_stats["batches_processed"], 1)
            )
            
            batch_stats_copy = self._batch_stats.copy()
            batch_stats_copy["avg_batch_size"] = current_avg_batch_size
            
            return {
                "batch_stats": batch_stats_copy,
                "avg_processing_time": avg_processing_time,
                "pending_requests": len(self._pending_requests),
                "recommendations": self._get_batch_recommendations()
            }
    
    def _get_batch_recommendations(self) -> List[str]:
        """Generate batch processing recommendations."""
        recommendations = []
        
        if self._batch_stats["avg_batch_size"] < self.config.batch_size * 0.5:
            recommendations.append("Consider decreasing batch_size for lower latency")
        
        avg_processing_time = (
            self._batch_stats["processing_time"] / 
            max(self._batch_stats["batches_processed"], 1)
        )
        
        if avg_processing_time > 1.0:
            recommendations.append("High batch processing time - consider optimization")
        
        return recommendations


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize optimizers
        self.memory_optimizer = MemoryOptimizer(self.config)
        self.connection_optimizer = ConnectionPoolOptimizer(self.config)
        self.response_optimizer = ResponseOptimizer(self.config)
        self.batch_processor = BatchProcessor(self.config)
        
        self._monitoring_active = False
        self._monitor_task = None
    
    def start_monitoring(self):
        """Start continuous performance monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitor_performance())
        self.logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self._monitoring_active = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Performance monitoring stopped")
    
    async def _monitor_performance(self):
        """Continuous performance monitoring loop."""
        while self._monitoring_active:
            try:
                metrics = await self.get_comprehensive_metrics()
                
                # Auto-optimize based on thresholds
                if metrics.memory_usage_mb > self.config.memory_threshold_mb:
                    self.memory_optimizer.optimize_memory()
                
                if metrics.cpu_usage_percent > self.config.cpu_threshold_percent:
                    self.logger.warning(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")
                
                await asyncio.sleep(self.config.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def get_comprehensive_metrics(self) -> PerformanceMetrics:
        """Get comprehensive performance metrics."""
        # System metrics
        memory_usage = self.memory_optimizer.get_memory_usage()
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Component metrics
        pool_metrics = self.connection_optimizer.get_pool_metrics()
        cache_metrics = self.response_optimizer.get_cache_metrics()
        batch_metrics = self.batch_processor.get_batch_metrics()
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(
            memory_usage, cpu_usage, cache_metrics["hit_rate_percent"]
        )
        
        # Generate optimization suggestions
        suggestions = []
        suggestions.extend(pool_metrics["recommendations"])
        suggestions.extend(cache_metrics["recommendations"])
        suggestions.extend(batch_metrics["recommendations"])
        
        if memory_usage > self.config.memory_threshold_mb:
            suggestions.append("Memory usage is high - consider optimization")
        
        if cpu_usage > self.config.cpu_threshold_percent:
            suggestions.append("CPU usage is high - consider load balancing")
        
        return PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            active_connections=pool_metrics["active_connections"],
            cache_hit_rate=cache_metrics["hit_rate_percent"],
            avg_response_time=cache_metrics["avg_response_time"],
            optimization_suggestions=suggestions,
            performance_score=performance_score
        )
    
    def _calculate_performance_score(self, memory_usage: float, cpu_usage: float, cache_hit_rate: float) -> float:
        """Calculate overall performance score (0-100)."""
        # Memory score (higher usage = lower score)
        memory_score = max(0, 100 - (memory_usage / self.config.memory_threshold_mb) * 100)
        
        # CPU score (higher usage = lower score)
        cpu_score = max(0, 100 - cpu_usage)
        
        # Cache score (higher hit rate = higher score)
        cache_score = cache_hit_rate
        
        # Weighted average
        overall_score = (memory_score * 0.3 + cpu_score * 0.3 + cache_score * 0.4)
        return round(overall_score, 2)
    
    async def optimize_all(self) -> Dict[str, Any]:
        """Perform comprehensive performance optimization."""
        results = {}
        
        # Memory optimization
        results["memory"] = self.memory_optimizer.optimize_memory()
        
        # Cache cleanup
        self.response_optimizer._evict_expired()
        
        # Get current metrics
        results["metrics"] = asdict(await self.get_comprehensive_metrics())
        
        self.logger.info("Comprehensive optimization completed")
        return results
    
    def save_metrics(self, metrics: PerformanceMetrics, output_file: Optional[Path] = None):
        """Save performance metrics to file."""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = Path("logs") / f"performance_metrics_{timestamp}.json"
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        self.logger.info(f"Performance metrics saved to {output_file}")


# Decorator for automatic performance optimization
def optimize_performance(cache_key: Optional[str] = None, batch_enabled: bool = False):
    """Decorator to automatically optimize function performance."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get optimizer instance (assumes it's available in context)
            optimizer = getattr(wrapper, '_optimizer', None)
            if not optimizer:
                return await func(*args, **kwargs)
            
            # Try cache first
            if cache_key:
                key = f"{cache_key}_{hash(str(args) + str(kwargs))}"
                cached_result = optimizer.response_optimizer.get_cached_response(key)
                if cached_result is not None:
                    return cached_result
            
            # Track start time
            start_time = time.time()
            
            try:
                # Execute function
                if batch_enabled:
                    result = await optimizer.batch_processor.add_to_batch((args, kwargs))
                else:
                    result = await func(*args, **kwargs)
                
                # Cache result
                if cache_key:
                    optimizer.response_optimizer.cache_response(key, result)
                
                return result
                
            finally:
                # Track response time
                response_time = time.time() - start_time
                optimizer.response_optimizer.track_response_time(response_time)
        
        return wrapper
    return decorator