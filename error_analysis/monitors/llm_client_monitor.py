"""
LLM Client Error Monitor

Specialized monitoring for LLM client operations including connection failures,
model loading issues, and response generation problems.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from .base_monitor import BaseErrorMonitor


class LLMClientMonitor(BaseErrorMonitor):
    """
    Specialized error monitor for LLM client operations.
    
    Monitors Ollama/OpenAI API calls, model loading, prompt processing,
    and response generation for errors and performance issues.
    """
    
    def __init__(self, error_classifier, error_logger, failure_tracker):
        """Initialize LLM client monitor."""
        super().__init__("llm_client", error_classifier, error_logger, failure_tracker)
        
        # LLM-specific alert thresholds
        self.alert_thresholds.update({
            'model_loading_time_seconds': 30.0,
            'prompt_processing_time_seconds': 60.0,
            'empty_response_rate': 0.05,  # 5% empty responses
            'context_length_exceeded_rate': 0.1,  # 10% context length issues
            'connection_failure_rate': 0.02  # 2% connection failures
        })
        
        # LLM-specific metrics
        self.model_loading_times = []
        self.prompt_lengths = []
        self.response_lengths = []
        self.empty_responses = 0
        self.context_length_errors = 0
        self.connection_failures = 0
        
        self.logger = logging.getLogger(f"{__name__}.llm_client")
    
    def track_model_loading(self, model_name: str, loading_time: float, success: bool):
        """
        Track model loading events.
        
        Args:
            model_name: Name of the model being loaded
            loading_time: Time taken to load the model
            success: Whether loading was successful
        """
        with self._lock:
            timestamp = datetime.now()
            
            self.model_loading_times.append({
                'timestamp': timestamp,
                'model_name': model_name,
                'loading_time': loading_time,
                'success': success
            })
            
            # Track performance
            self.failure_tracker.track_performance_metric(
                self.component_name,
                "model_loading_time",
                loading_time,
                timestamp
            )
            
            # Check for slow loading
            if loading_time > self.alert_thresholds['model_loading_time_seconds']:
                self.logger.warning(f"Slow model loading: {model_name} took {loading_time:.2f}s")
            
            if not success:
                self.logger.error(f"Model loading failed: {model_name}")
    
    def track_prompt_processing(self, 
                              prompt: str,
                              response: str,
                              processing_time: float,
                              success: bool,
                              error_details: Optional[Dict] = None):
        """
        Track prompt processing events.
        
        Args:
            prompt: The input prompt
            response: Generated response
            processing_time: Time taken to process
            success: Whether processing was successful
            error_details: Details if processing failed
        """
        with self._lock:
            timestamp = datetime.now()
            
            prompt_length = len(prompt.split())
            response_length = len(response.split()) if response else 0
            
            self.prompt_lengths.append(prompt_length)
            self.response_lengths.append(response_length)
            
            # Track specific issues
            if success:
                if not response or response.strip() == "":
                    self.empty_responses += 1
                    self.logger.warning("Empty response generated")
            else:
                if error_details:
                    error_message = error_details.get('message', '').lower()
                    
                    # Categorize errors
                    if any(term in error_message for term in ['context', 'length', 'token', 'too long']):
                        self.context_length_errors += 1
                        self.logger.warning(f"Context length error: prompt length {prompt_length} words")
                    
                    elif any(term in error_message for term in ['connection', 'timeout', 'network', 'unreachable']):
                        self.connection_failures += 1
                        self.logger.error(f"Connection failure: {error_message}")
            
            # Track performance
            self.failure_tracker.track_performance_metric(
                self.component_name,
                "prompt_processing_time",
                processing_time,
                timestamp
            )
            
            # Check for slow processing
            if processing_time > self.alert_thresholds['prompt_processing_time_seconds']:
                self.logger.warning(f"Slow prompt processing: {processing_time:.2f}s for {prompt_length} word prompt")
    
    def _get_failed_requests(self, hours: int) -> list:
        """Get failed requests within the specified time period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            req for req in self.recent_requests
            if not req['success'] and req['timestamp'] >= cutoff
        ]
    
    def _extract_prompt_characteristics(self, failed_requests: list) -> tuple:
        """Extract prompt lengths and error patterns from failed requests."""
        prompt_lengths = []
        error_patterns = []
        
        for req in failed_requests:
            if 'error_info' in req:
                error_info = req['error_info']
                context = error_info.context or {}
                
                # Extract prompt characteristics from context
                if 'prompt_length' in context:
                    prompt_lengths.append(context['prompt_length'])
                
                # Analyze error messages for patterns
                pattern = self._classify_error_pattern(error_info.message)
                if pattern:
                    error_patterns.append(pattern)
        
        return prompt_lengths, error_patterns
    
    def _classify_error_pattern(self, message: str) -> Optional[str]:
        """Classify error message into pattern category."""
        message_lower = message.lower()
        
        if 'context' in message_lower and 'length' in message_lower:
            return 'context_length_exceeded'
        elif 'timeout' in message_lower:
            return 'processing_timeout'
        elif 'connection' in message_lower:
            return 'connection_failure'
        elif 'model' in message_lower and 'load' in message_lower:
            return 'model_loading_failure'
        
        return None
    
    def _generate_recommendations(self, pattern_counts: dict) -> list:
        """Generate recommendations based on error patterns."""
        recommendations = []
        
        if pattern_counts.get('context_length_exceeded', 0) > 0:
            recommendations.append("Consider implementing prompt truncation or chunking")
        if pattern_counts.get('processing_timeout', 0) > 0:
            recommendations.append("Increase timeout values or optimize prompts")
        if pattern_counts.get('connection_failure', 0) > 0:
            recommendations.append("Implement retry mechanisms and check network stability")
        
        return recommendations
    
    def analyze_prompt_patterns(self, hours: int = 24) -> Dict[str, Any]:
        """
        Analyze patterns in prompts that lead to errors.
        
        Args:
            hours: Time period to analyze
            
        Returns:
            Analysis of prompt patterns
        """
        failed_requests = self._get_failed_requests(hours)
        
        if not failed_requests:
            return {
                'total_failed_prompts': 0,
                'common_failure_patterns': [],
                'avg_failed_prompt_length': 0,
                'recommendations': []
            }
        
        prompt_lengths, error_patterns = self._extract_prompt_characteristics(failed_requests)
        
        # Count pattern frequencies
        pattern_counts = {}
        for pattern in error_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        recommendations = self._generate_recommendations(pattern_counts)
        
        return {
            'analysis_period_hours': hours,
            'total_failed_prompts': len(failed_requests),
            'common_failure_patterns': sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True),
            'avg_failed_prompt_length': sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0,
            'max_failed_prompt_length': max(prompt_lengths) if prompt_lengths else 0,
            'recommendations': recommendations
        }
    
    def get_llm_specific_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get LLM-specific performance metrics.
        
        Args:
            hours: Time period to analyze
            
        Returns:
            LLM-specific metrics
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Filter recent data
        recent_loading = [
            item for item in self.model_loading_times
            if item['timestamp'] >= cutoff
        ]
        
        recent_requests = [
            req for req in self.recent_requests
            if req['timestamp'] >= cutoff
        ]
        
        total_requests = len(recent_requests)
        
        # Calculate rates
        empty_response_rate = self.empty_responses / total_requests if total_requests > 0 else 0
        context_error_rate = self.context_length_errors / total_requests if total_requests > 0 else 0
        connection_failure_rate = self.connection_failures / total_requests if total_requests > 0 else 0
        
        # Model loading metrics
        successful_loadings = [item for item in recent_loading if item['success']]
        avg_loading_time = sum(item['loading_time'] for item in successful_loadings) / len(successful_loadings) if successful_loadings else 0
        
        # Response time analysis
        response_times = [req['response_time'] for req in recent_requests if req['success']]
        
        return {
            'analysis_period_hours': hours,
            'total_requests': total_requests,
            'model_loading_metrics': {
                'total_loadings': len(recent_loading),
                'successful_loadings': len(successful_loadings),
                'avg_loading_time': avg_loading_time,
                'loading_success_rate': len(successful_loadings) / len(recent_loading) if recent_loading else 1.0
            },
            'prompt_processing_metrics': {
                'avg_prompt_length': sum(self.prompt_lengths[-100:]) / len(self.prompt_lengths[-100:]) if self.prompt_lengths else 0,
                'avg_response_length': sum(self.response_lengths[-100:]) / len(self.response_lengths[-100:]) if self.response_lengths else 0,
                'avg_processing_time': sum(response_times) / len(response_times) if response_times else 0,
                'max_processing_time': max(response_times) if response_times else 0
            },
            'error_rates': {
                'empty_response_rate': empty_response_rate,
                'context_length_exceeded_rate': context_error_rate,
                'connection_failure_rate': connection_failure_rate
            },
            'alert_status': {
                'empty_responses_high': empty_response_rate > self.alert_thresholds['empty_response_rate'],
                'context_errors_high': context_error_rate > self.alert_thresholds['context_length_exceeded_rate'],
                'connection_failures_high': connection_failure_rate > self.alert_thresholds['connection_failure_rate']
            }
        }
    
    def get_model_performance_comparison(self) -> Dict[str, Any]:
        """
        Compare performance across different models.
        
        Returns:
            Performance comparison between models
        """
        model_stats = {}
        
        for loading in self.model_loading_times[-100:]:  # Last 100 loadings
            model = loading['model_name']
            if model not in model_stats:
                model_stats[model] = {
                    'loading_times': [],
                    'success_count': 0,
                    'failure_count': 0
                }
            
            model_stats[model]['loading_times'].append(loading['loading_time'])
            if loading['success']:
                model_stats[model]['success_count'] += 1
            else:
                model_stats[model]['failure_count'] += 1
        
        # Calculate model performance metrics
        model_performance = {}
        for model, stats in model_stats.items():
            total_attempts = stats['success_count'] + stats['failure_count']
            avg_loading_time = sum(stats['loading_times']) / len(stats['loading_times']) if stats['loading_times'] else 0
            
            model_performance[model] = {
                'total_attempts': total_attempts,
                'success_rate': stats['success_count'] / total_attempts if total_attempts > 0 else 0,
                'avg_loading_time': avg_loading_time,
                'reliability_score': (stats['success_count'] / total_attempts) * (1.0 / max(avg_loading_time, 0.1)) if total_attempts > 0 else 0
            }
        
        # Rank models by reliability
        ranked_models = sorted(
            model_performance.items(),
            key=lambda x: x[1]['reliability_score'],
            reverse=True
        )
        
        return {
            'models_analyzed': len(model_performance),
            'model_performance': model_performance,
            'best_performing_model': ranked_models[0][0] if ranked_models else None,
            'worst_performing_model': ranked_models[-1][0] if ranked_models else None,
            'model_rankings': [(model, stats['reliability_score']) for model, stats in ranked_models]
        }
    
    def detect_llm_anomalies(self) -> Dict[str, Any]:
        """
        Detect anomalies in LLM behavior.
        
        Returns:
            Detected anomalies and alerts
        """
        anomalies = []
        
        # Check for sudden increase in empty responses
        recent_empty_rate = self.empty_responses / max(len(self.recent_requests), 1)
        if recent_empty_rate > 0.1:  # More than 10% empty responses
            anomalies.append({
                'type': 'high_empty_response_rate',
                'severity': 'high',
                'description': f'Empty response rate: {recent_empty_rate:.1%}',
                'recommendation': 'Check model health and prompt quality'
            })
        
        # Check for unusual response times
        recent_times = [req['response_time'] for req in self.recent_requests[-50:] if req['success']]
        if recent_times:
            avg_time = sum(recent_times) / len(recent_times)
            if avg_time > 30:  # More than 30 seconds average
                anomalies.append({
                    'type': 'high_response_time',
                    'severity': 'medium',
                    'description': f'Average response time: {avg_time:.2f}s',
                    'recommendation': 'Check model performance and system resources'
                })
        
        # Check for connection stability
        recent_failures = len([req for req in self.recent_requests[-100:] if not req['success']])
        if recent_failures > 20:  # More than 20% failure rate
            anomalies.append({
                'type': 'high_failure_rate',
                'severity': 'critical',
                'description': f'High failure rate: {recent_failures}% in last 100 requests',
                'recommendation': 'Check LLM service connectivity and health'
            })
        
        return {
            'anomalies_detected': len(anomalies),
            'anomalies': anomalies,
            'last_check': datetime.now().isoformat(),
            'system_status': 'normal' if not anomalies else 'anomalies_detected'
        }