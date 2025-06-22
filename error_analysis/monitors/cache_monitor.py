"""
Cache Error Monitor

Specialized monitoring for Redis cache operations including connection failures,
hit/miss rates, and memory usage patterns.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict

from .base_monitor import BaseErrorMonitor


class CacheMonitor(BaseErrorMonitor):
    """
    Specialized error monitor for cache operations.
    
    Monitors Redis cache performance, connection health,
    and cache effectiveness metrics.
    """
    
    def __init__(self, error_classifier, error_logger, failure_tracker):
        """Initialize cache monitor."""
        super().__init__("cache_manager", error_classifier, error_logger, failure_tracker)
        
        # Cache-specific alert thresholds
        self.alert_thresholds.update({
            'cache_hit_rate_threshold': 0.7,  # 70% hit rate
            'connection_failure_rate': 0.02,  # 2% connection failures
            'cache_miss_rate_threshold': 0.4,  # 40% miss rate
            'eviction_rate_threshold': 0.1,   # 10% eviction rate
            'memory_usage_threshold': 0.9     # 90% memory usage
        })
        
        # Cache-specific metrics
        self.cache_operations = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'gets': 0,
            'deletes': 0,
            'evictions': 0,
            'connection_failures': 0
        }
        
        self.memory_usage_history = []
        self.operation_times = defaultdict(list)
        
        self.logger = logging.getLogger(f"{__name__}.cache_manager")
    
    def _update_operation_counts(self, operation: str, success: bool, hit: Optional[bool], 
                               error_details: Optional[Dict]):
        """Update cache operation counts."""
        if operation not in self.cache_operations:
            return
            
        if success:
            self.cache_operations[operation] += 1
            self._track_hit_miss(operation, hit)
        else:
            self._track_operation_failure(error_details)
    
    def _track_hit_miss(self, operation: str, hit: Optional[bool]):
        """Track cache hits and misses for get operations."""
        if operation == 'gets' and hit is not None:
            if hit:
                self.cache_operations['hits'] += 1
            else:
                self.cache_operations['misses'] += 1
    
    def _track_operation_failure(self, error_details: Optional[Dict]):
        """Track operation failures."""
        if error_details and 'connection' in str(error_details).lower():
            self.cache_operations['connection_failures'] += 1
    
    def _track_operation_timing(self, operation: str, operation_time: float, timestamp: datetime):
        """Track operation timing and performance metrics."""
        # Track operation times
        self.operation_times[operation].append(operation_time)
        
        # Keep only recent times (last 1000 operations)
        if len(self.operation_times[operation]) > 1000:
            self.operation_times[operation] = self.operation_times[operation][-1000:]
        
        # Track performance
        self.failure_tracker.track_performance_metric(
            self.component_name,
            f"cache_{operation}_time",
            operation_time,
            timestamp
        )
        
        # Check for slow operations
        if operation_time > 0.1:  # More than 100ms is slow for cache
            self.logger.warning(f"Slow cache {operation}: {operation_time:.3f}s")

    def track_cache_operation(self,
                            operation: str,
                            success: bool,
                            operation_time: float,
                            hit: Optional[bool] = None,
                            error_details: Optional[Dict] = None):
        """
        Track cache operation metrics.
        
        Args:
            operation: Type of operation (get, set, delete, etc.)
            success: Whether operation was successful
            operation_time: Time taken for operation
            hit: Whether it was a cache hit (for get operations)
            error_details: Error details if operation failed
        """
        with self._lock:
            timestamp = datetime.now()
            
            self._update_operation_counts(operation, success, hit, error_details)
            self._track_operation_timing(operation, operation_time, timestamp)
            self._check_cache_effectiveness()
    
    def track_memory_usage(self, used_memory: float, max_memory: float):
        """
        Track cache memory usage.
        
        Args:
            used_memory: Currently used memory in bytes
            max_memory: Maximum available memory in bytes
        """
        with self._lock:
            timestamp = datetime.now()
            usage_ratio = used_memory / max_memory if max_memory > 0 else 0
            
            self.memory_usage_history.append({
                'timestamp': timestamp,
                'used_memory': used_memory,
                'max_memory': max_memory,
                'usage_ratio': usage_ratio
            })
            
            # Keep only recent history (last 24 hours worth)
            cutoff = timestamp - timedelta(hours=24)
            self.memory_usage_history = [
                entry for entry in self.memory_usage_history
                if entry['timestamp'] >= cutoff
            ]
            
            # Track with failure tracker
            self.failure_tracker.track_performance_metric(
                self.component_name,
                "memory_usage_ratio",
                usage_ratio,
                timestamp
            )
            
            # Check for high memory usage
            if usage_ratio > self.alert_thresholds['memory_usage_threshold']:
                self.logger.warning(f"High cache memory usage: {usage_ratio:.1%}")
    
    def track_eviction_event(self, evicted_keys: int, reason: str = "memory_pressure"):
        """
        Track cache eviction events.
        
        Args:
            evicted_keys: Number of keys evicted
            reason: Reason for eviction
        """
        with self._lock:
            self.cache_operations['evictions'] += evicted_keys
            
            self.logger.info(f"Cache eviction: {evicted_keys} keys evicted due to {reason}")
            
            # Check for excessive evictions
            total_operations = sum(self.cache_operations.values())
            eviction_rate = self.cache_operations['evictions'] / max(total_operations, 1)
            
            if eviction_rate > self.alert_thresholds['eviction_rate_threshold']:
                self.logger.warning(f"High eviction rate: {eviction_rate:.1%}")
    
    def _check_cache_effectiveness(self):
        """Check cache hit rate and effectiveness."""
        total_gets = self.cache_operations['gets']
        
        if total_gets >= 100:  # Need sufficient data
            hit_rate = self.cache_operations['hits'] / total_gets
            miss_rate = self.cache_operations['misses'] / total_gets
            
            if hit_rate < self.alert_thresholds['cache_hit_rate_threshold']:
                self.logger.warning(f"Low cache hit rate: {hit_rate:.1%}")
            
            if miss_rate > self.alert_thresholds['cache_miss_rate_threshold']:
                self.logger.warning(f"High cache miss rate: {miss_rate:.1%}")
    
    def get_cache_performance_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive cache performance metrics.
        
        Args:
            hours: Time period to analyze
            
        Returns:
            Cache performance metrics
        """
        total_operations = sum(self.cache_operations.values())
        total_gets = self.cache_operations['gets']
        
        # Calculate rates
        hit_rate = self.cache_operations['hits'] / max(total_gets, 1)
        miss_rate = self.cache_operations['misses'] / max(total_gets, 1)
        connection_failure_rate = self.cache_operations['connection_failures'] / max(total_operations, 1)
        eviction_rate = self.cache_operations['evictions'] / max(total_operations, 1)
        
        # Calculate average operation times
        avg_operation_times = {}
        for operation, times in self.operation_times.items():
            if times:
                avg_operation_times[operation] = {
                    'avg_time': sum(times) / len(times),
                    'max_time': max(times),
                    'min_time': min(times),
                    'operation_count': len(times)
                }
        
        # Memory usage analysis
        memory_analysis = self._analyze_memory_usage(hours)
        
        return {
            'analysis_period_hours': hours,
            'operation_counts': dict(self.cache_operations),
            'performance_rates': {
                'hit_rate': hit_rate,
                'miss_rate': miss_rate,
                'connection_failure_rate': connection_failure_rate,
                'eviction_rate': eviction_rate
            },
            'operation_performance': avg_operation_times,
            'memory_analysis': memory_analysis,
            'health_status': self._assess_cache_health(hit_rate, connection_failure_rate, memory_analysis),
            'recommendations': self._generate_cache_recommendations(hit_rate, eviction_rate, memory_analysis)
        }
    
    def _analyze_memory_usage(self, hours: int) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent_usage = [
            entry for entry in self.memory_usage_history
            if entry['timestamp'] >= cutoff
        ]
        
        if not recent_usage:
            return {
                'avg_usage_ratio': 0,
                'max_usage_ratio': 0,
                'min_usage_ratio': 0,
                'usage_trend': 'unknown'
            }
        
        usage_ratios = [entry['usage_ratio'] for entry in recent_usage]
        
        # Calculate trend (simple linear regression)
        if len(recent_usage) >= 2:
            x_values = list(range(len(recent_usage)))
            y_values = usage_ratios
            
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            
            # Slope of trend line
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if n * sum_x2 != sum_x * sum_x else 0
            
            if slope > 0.001:
                trend = 'increasing'
            elif slope < -0.001:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'avg_usage_ratio': sum(usage_ratios) / len(usage_ratios),
            'max_usage_ratio': max(usage_ratios),
            'min_usage_ratio': min(usage_ratios),
            'usage_trend': trend,
            'data_points': len(recent_usage)
        }
    
    def _assess_cache_health(self, hit_rate: float, connection_failure_rate: float, memory_analysis: Dict) -> str:
        """Assess overall cache health."""
        issues = []
        
        if hit_rate < 0.5:
            issues.append('low_hit_rate')
        if connection_failure_rate > 0.05:
            issues.append('connection_issues')
        if memory_analysis.get('max_usage_ratio', 0) > 0.95:
            issues.append('memory_pressure')
        if memory_analysis.get('usage_trend') == 'increasing':
            issues.append('increasing_memory_usage')
        
        if not issues:
            return 'excellent'
        elif len(issues) == 1:
            return 'good'
        elif len(issues) == 2:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_cache_recommendations(self, hit_rate: float, eviction_rate: float, memory_analysis: Dict) -> List[str]:
        """Generate cache optimization recommendations."""
        recommendations = []
        
        if hit_rate < 0.6:
            recommendations.append("Consider reviewing cache key strategies to improve hit rate")
            recommendations.append("Analyze access patterns to optimize TTL settings")
        
        if eviction_rate > 0.1:
            recommendations.append("High eviction rate detected - consider increasing cache memory")
            recommendations.append("Review cache size limits and eviction policies")
        
        if memory_analysis.get('usage_trend') == 'increasing':
            recommendations.append("Memory usage is trending upward - monitor for potential memory leaks")
        
        if memory_analysis.get('max_usage_ratio', 0) > 0.9:
            recommendations.append("Cache memory usage is high - consider scaling cache capacity")
        
        avg_get_time = 0
        if 'gets' in self.operation_times and self.operation_times['gets']:
            avg_get_time = sum(self.operation_times['gets']) / len(self.operation_times['gets'])
        
        if avg_get_time > 0.01:  # More than 10ms for cache get
            recommendations.append("Cache get operations are slow - check network latency and cache load")
        
        if not recommendations:
            recommendations.append("Cache performance is optimal")
        
        return recommendations
    
    def get_cache_key_analysis(self) -> Dict[str, Any]:
        """
        Analyze cache key patterns and usage.
        Note: This would require tracking individual keys, which is implemented as a placeholder.
            
        Returns:
            Cache key analysis
        """
        # This is a placeholder implementation
        # In a real system, you'd track individual cache keys and their patterns
        
        return {
            'analysis_note': 'Cache key analysis requires individual key tracking implementation',
            'total_unique_keys_estimated': 'not_tracked',
            'most_accessed_key_patterns': 'not_tracked',
            'key_expiration_analysis': 'not_tracked',
            'recommendations': [
                'Implement detailed cache key tracking for better analysis',
                'Monitor key namespace distribution',
                'Analyze key access patterns for optimization opportunities'
            ]
        }
    
    def reset_cache_metrics(self):
        """Reset cache-specific metrics."""
        with self._lock:
            self.cache_operations = {key: 0 for key in self.cache_operations}
            self.memory_usage_history.clear()
            self.operation_times.clear()
            self.logger.info("Reset cache metrics")