"""
API Endpoint Error Monitor

Specialized monitoring for FastAPI endpoints including request validation,
response times, and HTTP error patterns.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict

from .base_monitor import BaseErrorMonitor


class APIEndpointMonitor(BaseErrorMonitor):
    """
    Specialized error monitor for API endpoints.
    
    Monitors HTTP requests, response codes, validation errors,
    and endpoint-specific performance metrics.
    """
    
    def __init__(self, error_classifier, error_logger, failure_tracker):
        """Initialize API endpoint monitor."""
        super().__init__("api_endpoints", error_classifier, error_logger, failure_tracker)
        
        # API-specific alert thresholds
        self.alert_thresholds.update({
            'http_4xx_rate': 0.05,  # 5% client errors
            'http_5xx_rate': 0.01,  # 1% server errors
            'validation_error_rate': 0.03,  # 3% validation errors
            'request_timeout_rate': 0.02,  # 2% timeouts
            'large_payload_threshold_mb': 10,  # 10MB payload warning
            'slow_endpoint_threshold_seconds': 10.0  # 10s response time
        })
        
        # API-specific metrics
        self.endpoint_stats = defaultdict(lambda: {
            'request_count': 0,
            'success_count': 0,
            'error_count': 0,
            'response_times': [],
            'status_codes': defaultdict(int),
            'error_types': defaultdict(int)
        })
        
        self.request_sizes = []
        self.response_sizes = []
        self.validation_errors = []
        self.rate_limit_hits = 0
        
        self.logger = logging.getLogger(f"{__name__}.api_endpoints")
    
    def track_request(self,
                     endpoint: str,
                     method: str,
                     status_code: int,
                     response_time: float,
                     request_size: Optional[int] = None,
                     response_size: Optional[int] = None,
                     user_agent: Optional[str] = None,
                     error_details: Optional[Dict] = None):
        """
        Track API request metrics.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            status_code: HTTP status code
            response_time: Response time in seconds
            request_size: Request payload size in bytes
            response_size: Response payload size in bytes
            user_agent: Client user agent
            error_details: Error details if request failed
        """
        with self._lock:
            timestamp = datetime.now()
            endpoint_key = f"{method} {endpoint}"
            
            # Update endpoint stats
            stats = self.endpoint_stats[endpoint_key]
            stats['request_count'] += 1
            stats['response_times'].append(response_time)
            stats['status_codes'][status_code] += 1
            
            # Track success/error
            is_success = 200 <= status_code < 400
            if is_success:
                stats['success_count'] += 1
            else:
                stats['error_count'] += 1
                
                # Categorize error type
                if 400 <= status_code < 500:
                    error_type = 'client_error'
                    if status_code == 422:
                        error_type = 'validation_error'
                        self.validation_errors.append({
                            'timestamp': timestamp,
                            'endpoint': endpoint_key,
                            'details': error_details
                        })
                    elif status_code == 429:
                        error_type = 'rate_limit'
                        self.rate_limit_hits += 1
                elif 500 <= status_code < 600:
                    error_type = 'server_error'
                else:
                    error_type = 'unknown_error'
                
                stats['error_types'][error_type] += 1
            
            # Track payload sizes
            if request_size is not None:
                self.request_sizes.append(request_size)
                
                # Check for large payloads
                if request_size > self.alert_thresholds['large_payload_threshold_mb'] * 1024 * 1024:
                    self.logger.warning(f"Large request payload: {request_size / (1024*1024):.2f}MB to {endpoint_key}")
            
            if response_size is not None:
                self.response_sizes.append(response_size)
            
            # Track performance
            self.failure_tracker.track_performance_metric(
                self.component_name,
                f"endpoint_{endpoint_key}_response_time",
                response_time,
                timestamp
            )
            
            # Check for slow responses
            if response_time > self.alert_thresholds['slow_endpoint_threshold_seconds']:
                self.logger.warning(f"Slow endpoint response: {endpoint_key} took {response_time:.2f}s")
            
            # Record in base monitor
            self.recent_requests.append({
                'timestamp': timestamp,
                'function': endpoint_key,
                'success': is_success,
                'response_time': response_time,
                'status_code': status_code,
                'request_size': request_size,
                'response_size': response_size,
                'user_agent': user_agent
            })
            
            # Check alert conditions
            self._check_api_alert_conditions(endpoint_key)
    
    def _check_api_alert_conditions(self, endpoint_key: str):
        """Check API-specific alert conditions."""
        stats = self.endpoint_stats[endpoint_key]
        
        if stats['request_count'] < 10:  # Need minimum requests for meaningful rates
            return
        
        # Check error rates
        client_errors = sum(count for code, count in stats['status_codes'].items() if 400 <= code < 500)
        server_errors = sum(count for code, count in stats['status_codes'].items() if 500 <= code < 600)
        
        client_error_rate = client_errors / stats['request_count']
        server_error_rate = server_errors / stats['request_count']
        
        if client_error_rate > self.alert_thresholds['http_4xx_rate']:
            self.logger.warning(f"High client error rate for {endpoint_key}: {client_error_rate:.1%}")
        
        if server_error_rate > self.alert_thresholds['http_5xx_rate']:
            self.logger.error(f"High server error rate for {endpoint_key}: {server_error_rate:.1%}")
        
        # Check validation error rate
        validation_errors = stats['error_types'].get('validation_error', 0)
        validation_rate = validation_errors / stats['request_count']
        
        if validation_rate > self.alert_thresholds['validation_error_rate']:
            self.logger.warning(f"High validation error rate for {endpoint_key}: {validation_rate:.1%}")
    
    def get_endpoint_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive endpoint analytics.
        
        Args:
            hours: Time period to analyze
            
        Returns:
            Endpoint analytics and performance metrics
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Filter recent requests
        recent_requests = [
            req for req in self.recent_requests
            if req['timestamp'] >= cutoff
        ]
        
        if not recent_requests:
            return {
                'analysis_period_hours': hours,
                'total_requests': 0,
                'endpoints_analyzed': 0
            }
        
        # Analyze by endpoint
        endpoint_analysis = {}
        for req in recent_requests:
            endpoint = req['function']
            
            if endpoint not in endpoint_analysis:
                endpoint_analysis[endpoint] = {
                    'total_requests': 0,
                    'successful_requests': 0,
                    'failed_requests': 0,
                    'avg_response_time': 0,
                    'status_codes': defaultdict(int),
                    'response_times': []
                }
            
            analysis = endpoint_analysis[endpoint]
            analysis['total_requests'] += 1
            analysis['response_times'].append(req['response_time'])
            analysis['status_codes'][req.get('status_code', 0)] += 1
            
            if req['success']:
                analysis['successful_requests'] += 1
            else:
                analysis['failed_requests'] += 1
        
        # Calculate metrics for each endpoint
        for endpoint, analysis in endpoint_analysis.items():
            if analysis['response_times']:
                analysis['avg_response_time'] = sum(analysis['response_times']) / len(analysis['response_times'])
                analysis['max_response_time'] = max(analysis['response_times'])
                analysis['min_response_time'] = min(analysis['response_times'])
            
            analysis['success_rate'] = analysis['successful_requests'] / analysis['total_requests']
            analysis['error_rate'] = analysis['failed_requests'] / analysis['total_requests']
            
            # Convert defaultdict to regular dict for JSON serialization
            analysis['status_codes'] = dict(analysis['status_codes'])
            del analysis['response_times']  # Remove raw data
        
        # Overall metrics
        total_requests = len(recent_requests)
        successful_requests = len([req for req in recent_requests if req['success']])
        
        # Most and least used endpoints
        endpoint_usage = [(endpoint, analysis['total_requests']) for endpoint, analysis in endpoint_analysis.items()]
        endpoint_usage.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'analysis_period_hours': hours,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'overall_success_rate': successful_requests / total_requests,
            'endpoints_analyzed': len(endpoint_analysis),
            'endpoint_details': endpoint_analysis,
            'most_used_endpoint': endpoint_usage[0][0] if endpoint_usage else None,
            'least_used_endpoint': endpoint_usage[-1][0] if endpoint_usage else None,
            'avg_requests_per_hour': total_requests / hours if hours > 0 else 0
        }
    
    def _get_recent_validation_errors(self, hours: int) -> list:
        """Get validation errors within the specified time period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            err for err in self.validation_errors
            if err['timestamp'] >= cutoff
        ]
    
    def _analyze_error_patterns(self, validation_errors: list) -> tuple:
        """Analyze error patterns and endpoint distribution."""
        error_patterns = defaultdict(int)
        endpoint_errors = defaultdict(int)
        
        for error in validation_errors:
            endpoint = error['endpoint']
            endpoint_errors[endpoint] += 1
            
            # Classify error type
            pattern = self._classify_validation_error(error)
            error_patterns[pattern] += 1
        
        return dict(error_patterns), dict(endpoint_errors)
    
    def _classify_validation_error(self, error: dict) -> str:
        """Classify validation error into pattern category."""
        details = error.get('details', {})
        if not isinstance(details, dict):
            return 'other_validation_errors'
        
        error_message = str(details).lower()
        
        if 'required' in error_message:
            return 'missing_required_fields'
        elif 'type' in error_message:
            return 'incorrect_field_types'
        elif 'format' in error_message:
            return 'invalid_field_formats'
        elif 'length' in error_message or 'size' in error_message:
            return 'field_size_violations'
        else:
            return 'other_validation_errors'
    
    def _generate_validation_recommendations(self, error_patterns: dict) -> list:
        """Generate recommendations based on validation error patterns."""
        recommendations = []
        
        if not error_patterns:
            return recommendations
        
        most_common_pattern = max(error_patterns, key=error_patterns.get)
        
        if most_common_pattern == 'missing_required_fields':
            recommendations.extend([
                "Review API documentation for required fields",
                "Implement client-side validation for required fields"
            ])
        elif most_common_pattern == 'incorrect_field_types':
            recommendations.extend([
                "Improve type validation in client applications",
                "Provide clear data type specifications in API docs"
            ])
        elif most_common_pattern == 'invalid_field_formats':
            recommendations.extend([
                "Add format validation examples to API documentation",
                "Implement client-side format validation"
            ])
        
        return recommendations
    
    def get_validation_error_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """
        Analyze validation errors and patterns.
        
        Args:
            hours: Time period to analyze
            
        Returns:
            Validation error analysis
        """
        recent_validation_errors = self._get_recent_validation_errors(hours)
        
        if not recent_validation_errors:
            return {
                'total_validation_errors': 0,
                'error_patterns': {},
                'affected_endpoints': {},
                'recommendations': []
            }
        
        error_patterns, endpoint_errors = self._analyze_error_patterns(recent_validation_errors)
        recommendations = self._generate_validation_recommendations(error_patterns)
        
        most_common_pattern = max(error_patterns, key=error_patterns.get) if error_patterns else None
        
        return {
            'analysis_period_hours': hours,
            'total_validation_errors': len(recent_validation_errors),
            'error_patterns': error_patterns,
            'affected_endpoints': endpoint_errors,
            'most_common_pattern': most_common_pattern,
            'most_error_prone_endpoint': max(endpoint_errors, key=endpoint_errors.get) if endpoint_errors else None,
            'recommendations': recommendations
        }
    
    def get_performance_bottlenecks(self, top_n: int = 5) -> Dict[str, Any]:
        """
        Identify performance bottlenecks in API endpoints.
        
        Args:
            top_n: Number of top bottlenecks to return
            
        Returns:
            Performance bottleneck analysis
        """
        bottlenecks = []
        
        for endpoint, stats in self.endpoint_stats.items():
            if stats['response_times'] and len(stats['response_times']) >= 10:
                avg_time = sum(stats['response_times']) / len(stats['response_times'])
                max_time = max(stats['response_times'])
                
                # Calculate 95th percentile
                sorted_times = sorted(stats['response_times'])
                p95_index = int(len(sorted_times) * 0.95)
                p95_time = sorted_times[p95_index] if p95_index < len(sorted_times) else max_time
                
                bottlenecks.append({
                    'endpoint': endpoint,
                    'avg_response_time': avg_time,
                    'max_response_time': max_time,
                    'p95_response_time': p95_time,
                    'total_requests': stats['request_count'],
                    'performance_score': avg_time * stats['request_count']  # Weighted by usage
                })
        
        # Sort by performance impact
        bottlenecks.sort(key=lambda x: x['performance_score'], reverse=True)
        
        return {
            'total_endpoints_analyzed': len(self.endpoint_stats),
            'top_bottlenecks': bottlenecks[:top_n],
            'recommendations': self._generate_performance_recommendations(bottlenecks[:top_n])
        }
    
    def _generate_performance_recommendations(self, bottlenecks: List[Dict]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        if not bottlenecks:
            return ["No significant performance bottlenecks detected"]
        
        worst_endpoint = bottlenecks[0]
        
        if worst_endpoint['avg_response_time'] > 5.0:
            recommendations.append(f"Optimize {worst_endpoint['endpoint']} - average response time {worst_endpoint['avg_response_time']:.2f}s")
        
        if worst_endpoint['max_response_time'] > 30.0:
            recommendations.append(f"Investigate timeout issues in {worst_endpoint['endpoint']}")
        
        # Check for endpoints with high usage and moderate slowness
        high_usage_slow = [b for b in bottlenecks if b['total_requests'] > 100 and b['avg_response_time'] > 2.0]
        if high_usage_slow:
            recommendations.append("Consider caching or optimization for high-traffic slow endpoints")
        
        if len(bottlenecks) > 3:
            recommendations.append("Multiple endpoints showing performance issues - consider system-wide optimization")
        
        return recommendations
    
    def get_status_code_distribution(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get HTTP status code distribution analysis.
        
        Args:
            hours: Time period to analyze
            
        Returns:
            Status code distribution and health indicators
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent_requests = [
            req for req in self.recent_requests
            if req['timestamp'] >= cutoff and 'status_code' in req
        ]
        
        if not recent_requests:
            return {
                'total_requests': 0,
                'status_distribution': {},
                'health_indicators': {}
            }
        
        # Count status codes
        status_counts = defaultdict(int)
        for req in recent_requests:
            status_counts[req['status_code']] += 1
        
        total_requests = len(recent_requests)
        
        # Calculate percentages and health indicators
        status_distribution = {}
        for code, count in status_counts.items():
            status_distribution[code] = {
                'count': count,
                'percentage': (count / total_requests) * 100
            }
        
        # Health indicators
        success_codes = sum(count for code, count in status_counts.items() if 200 <= code < 300)
        client_errors = sum(count for code, count in status_counts.items() if 400 <= code < 500)
        server_errors = sum(count for code, count in status_counts.items() if 500 <= code < 600)
        
        health_indicators = {
            'success_rate': (success_codes / total_requests) * 100,
            'client_error_rate': (client_errors / total_requests) * 100,
            'server_error_rate': (server_errors / total_requests) * 100,
            'overall_health': 'good' if server_errors == 0 and client_errors < total_requests * 0.05 else 'needs_attention'
        }
        
        return {
            'analysis_period_hours': hours,
            'total_requests': total_requests,
            'status_distribution': dict(status_distribution),
            'health_indicators': health_indicators,
            'most_common_status': max(status_counts, key=status_counts.get) if status_counts else None
        }