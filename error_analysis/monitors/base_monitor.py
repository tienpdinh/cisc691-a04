"""
Base Error Monitor for RAG Pipeline Components

Provides common error monitoring functionality that can be extended
by component-specific monitors.
"""

import logging
import time
import functools
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading

from ..core.error_classifier import ErrorClassifier, ErrorInfo
from ..core.error_logger import ErrorLogger
from ..core.failure_tracker import FailureTracker


class BaseErrorMonitor:
    """
    Base class for component-specific error monitors.
    
    Provides common monitoring functionality including error tracking,
    performance measurement, and health status monitoring.
    """
    
    def __init__(self, 
                 component_name: str,
                 error_classifier: ErrorClassifier,
                 error_logger: ErrorLogger,
                 failure_tracker: FailureTracker):
        """
        Initialize base error monitor.
        
        Args:
            component_name: Name of the component being monitored
            error_classifier: Error classification system
            error_logger: Error logging system
            failure_tracker: Failure tracking system
        """
        self.component_name = component_name
        self.error_classifier = error_classifier
        self.error_logger = error_logger
        self.failure_tracker = failure_tracker
        
        self.logger = logging.getLogger(f"{__name__}.{component_name}")
        
        # Performance tracking
        self.performance_metrics = defaultdict(deque)
        self.error_counts = defaultdict(int)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Alert thresholds
        self.alert_thresholds = {
            'error_rate_per_minute': 5,
            'avg_response_time_seconds': 5.0,
            'success_rate_threshold': 0.95
        }
        
        # Recent metrics for health calculation
        self.recent_requests = deque(maxlen=1000)
        self.recent_errors = deque(maxlen=100)
    
    def monitor_function(self, function_name: Optional[str] = None):
        """
        Decorator to monitor function calls for errors and performance.
        
        Args:
            function_name: Optional custom name for the function
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                func_name = function_name or f"{func.__module__}.{func.__name__}"
                
                try:
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Record successful execution
                    execution_time = time.time() - start_time
                    self._record_success(func_name, execution_time)
                    
                    return result
                    
                except Exception as e:
                    # Record error
                    execution_time = time.time() - start_time
                    self._record_error(e, func_name, execution_time, args, kwargs)
                    
                    # Re-raise the exception
                    raise
            
            return wrapper
        return decorator
    
    def _record_success(self, function_name: str, execution_time: float):
        """Record successful function execution."""
        with self._lock:
            timestamp = datetime.now()
            
            # Record performance metrics
            self.performance_metrics[f"{function_name}_response_time"].append(execution_time)
            self.performance_metrics[f"{function_name}_success_count"].append(timestamp)
            
            # Record for health calculation
            self.recent_requests.append({
                'timestamp': timestamp,
                'function': function_name,
                'success': True,
                'response_time': execution_time
            })
            
            # Track performance metrics with failure tracker
            self.failure_tracker.track_performance_metric(
                self.component_name,
                "response_time",
                execution_time,
                timestamp
            )
    
    def _record_error(self, 
                     exception: Exception, 
                     function_name: str,
                     execution_time: float,
                     args: tuple,
                     kwargs: dict):
        """Record error information."""
        with self._lock:
            timestamp = datetime.now()
            
            # Create context information
            context = {
                'function_name': function_name,
                'execution_time': execution_time,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys()),
                'component': self.component_name
            }
            
            # Classify error
            error_info = self.error_classifier.classify_exception(
                exception, 
                self.component_name,
                context
            )
            
            # Log error
            self.error_logger.log_error(error_info)
            
            # Track failure
            self.failure_tracker.track_error(error_info)
            
            # Record metrics
            self.error_counts[function_name] += 1
            self.recent_errors.append({
                'timestamp': timestamp,
                'function': function_name,
                'error_type': type(exception).__name__,
                'error_message': str(exception)
            })
            
            # Record for health calculation
            self.recent_requests.append({
                'timestamp': timestamp,
                'function': function_name,
                'success': False,
                'response_time': execution_time,
                'error_info': error_info
            })
            
            # Check for alert conditions
            self._check_alert_conditions()
    
    def _check_alert_conditions(self):
        """Check if any alert conditions are met."""
        # Check error rate
        recent_errors_count = len([
            req for req in self.recent_requests
            if not req['success'] and 
            req['timestamp'] > datetime.now() - timedelta(minutes=1)
        ])
        
        if recent_errors_count > self.alert_thresholds['error_rate_per_minute']:
            self.logger.error(f"High error rate alert: {recent_errors_count} errors in last minute")
        
        # Check average response time
        recent_times = [
            req['response_time'] for req in self.recent_requests
            if req['timestamp'] > datetime.now() - timedelta(minutes=5)
        ]
        
        if recent_times:
            avg_time = sum(recent_times) / len(recent_times)
            if avg_time > self.alert_thresholds['avg_response_time_seconds']:
                self.logger.warning(f"High response time alert: {avg_time:.2f}s average")
        
        # Check success rate
        recent_success_rate = self.get_success_rate(minutes=10)
        if recent_success_rate < self.alert_thresholds['success_rate_threshold']:
            self.logger.error(f"Low success rate alert: {recent_success_rate:.2%}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of the component."""
        with self._lock:
            now = datetime.now()
            
            # Calculate metrics for different time periods
            metrics_1h = self._calculate_metrics(hours=1)
            metrics_24h = self._calculate_metrics(hours=24)
            
            # Determine overall health
            health_score = self._calculate_health_score(metrics_1h, metrics_24h)
            health_status = self._determine_health_status(health_score)
            
            return {
                'component': self.component_name,
                'health_score': health_score,
                'status': health_status,
                'last_updated': now.isoformat(),
                'metrics_1h': metrics_1h,
                'metrics_24h': metrics_24h,
                'recent_errors': len(self.recent_errors),
                'total_requests': len(self.recent_requests),
                'alert_thresholds': self.alert_thresholds
            }
    
    def _calculate_metrics(self, hours: int) -> Dict[str, Any]:
        """Calculate metrics for specified time period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Filter recent requests
        period_requests = [
            req for req in self.recent_requests
            if req['timestamp'] >= cutoff
        ]
        
        if not period_requests:
            return {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'success_rate': 1.0,
                'error_rate': 0.0,
                'avg_response_time': 0.0,
                'max_response_time': 0.0,
                'min_response_time': 0.0
            }
        
        successful = [req for req in period_requests if req['success']]
        failed = [req for req in period_requests if not req['success']]
        
        response_times = [req['response_time'] for req in period_requests]
        
        return {
            'total_requests': len(period_requests),
            'successful_requests': len(successful),
            'failed_requests': len(failed),
            'success_rate': len(successful) / len(period_requests),
            'error_rate': len(failed) / len(period_requests),
            'avg_response_time': sum(response_times) / len(response_times),
            'max_response_time': max(response_times),
            'min_response_time': min(response_times)
        }
    
    def _calculate_health_score(self, metrics_1h: Dict, metrics_24h: Dict) -> float:
        """Calculate overall health score (0-100)."""
        score = 100.0
        
        # Deduct for poor success rate (weighted by recency)
        success_rate_1h = metrics_1h.get('success_rate', 1.0)
        success_rate_24h = metrics_24h.get('success_rate', 1.0)
        
        # Recent errors are weighted more heavily
        score -= (1.0 - success_rate_1h) * 60  # Up to 60 points for 1h
        score -= (1.0 - success_rate_24h) * 20  # Up to 20 points for 24h
        
        # Deduct for high response times
        avg_time_1h = metrics_1h.get('avg_response_time', 0.0)
        if avg_time_1h > 1.0:  # More than 1 second is concerning
            score -= min(avg_time_1h * 5, 20)  # Up to 20 points
        
        # Deduct for high error counts
        error_count_1h = metrics_1h.get('failed_requests', 0)
        if error_count_1h > 0:
            score -= min(error_count_1h * 2, 10)  # Up to 10 points
        
        return max(0.0, score)
    
    def _determine_health_status(self, health_score: float) -> str:
        """Determine health status based on score."""
        if health_score >= 90:
            return "excellent"
        elif health_score >= 75:
            return "good"
        elif health_score >= 50:
            return "degraded"
        elif health_score >= 25:
            return "poor"
        else:
            return "critical"
    
    def get_success_rate(self, minutes: int = 60) -> float:
        """Get success rate for specified time period."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        period_requests = [
            req for req in self.recent_requests
            if req['timestamp'] >= cutoff
        ]
        
        if not period_requests:
            return 1.0
        
        successful = len([req for req in period_requests if req['success']])
        return successful / len(period_requests)
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for specified time period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        period_errors = [
            err for err in self.recent_errors
            if err['timestamp'] >= cutoff
        ]
        
        if not period_errors:
            return {
                'total_errors': 0,
                'error_types': {},
                'error_functions': {},
                'error_rate_per_hour': 0.0
            }
        
        # Count by error type
        error_types = defaultdict(int)
        error_functions = defaultdict(int)
        
        for error in period_errors:
            error_types[error['error_type']] += 1
            error_functions[error['function']] += 1
        
        return {
            'total_errors': len(period_errors),
            'error_types': dict(error_types),
            'error_functions': dict(error_functions),
            'error_rate_per_hour': len(period_errors) / hours if hours > 0 else 0,
            'most_common_error': max(error_types, key=error_types.get) if error_types else None,
            'most_error_prone_function': max(error_functions, key=error_functions.get) if error_functions else None
        }
    
    def reset_metrics(self):
        """Reset all metrics and counters."""
        with self._lock:
            self.performance_metrics.clear()
            self.error_counts.clear()
            self.recent_requests.clear()
            self.recent_errors.clear()
            self.logger.info(f"Reset metrics for {self.component_name}")
    
    def update_alert_thresholds(self, thresholds: Dict[str, Any]):
        """Update alert thresholds."""
        self.alert_thresholds.update(thresholds)
        self.logger.info(f"Updated alert thresholds for {self.component_name}: {thresholds}")
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, List[Dict[str, Any]]]:
        """Get performance trends over time."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Group requests by hour
        hourly_data = defaultdict(list)
        
        for req in self.recent_requests:
            if req['timestamp'] >= cutoff:
                hour_key = req['timestamp'].strftime('%Y-%m-%d %H:00:00')
                hourly_data[hour_key].append(req)
        
        trends = []
        for hour, requests in sorted(hourly_data.items()):
            successful = len([r for r in requests if r['success']])
            total = len(requests)
            avg_time = sum(r['response_time'] for r in requests) / total if total > 0 else 0
            
            trends.append({
                'hour': hour,
                'total_requests': total,
                'successful_requests': successful,
                'success_rate': successful / total if total > 0 else 1.0,
                'avg_response_time': avg_time
            })
        
        return {
            'component': self.component_name,
            'period_hours': hours,
            'hourly_trends': trends
        }