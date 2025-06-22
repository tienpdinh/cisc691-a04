"""
Error Analysis Manager

Central coordinator for all error analysis components in the RAG system.
Provides unified interface for error tracking, failure detection, and system health monitoring.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import threading

from .error_classifier import ErrorClassifier
from .error_logger import ErrorLogger
from .failure_tracker import FailureTracker
from ..recovery.error_recovery import ErrorRecoveryManager
from ..monitors import (
    DocumentProcessorMonitor,
    VectorStoreMonitor,
    LLMClientMonitor,
    APIEndpointMonitor,
    CacheMonitor
)


class ErrorAnalysisManager:
    """
    Central manager for comprehensive error analysis across the RAG pipeline.
    
    Coordinates error classification, logging, failure tracking, and component monitoring
    to provide unified system health and error analytics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize error analysis manager.
        
        Args:
            config: Configuration for error analysis components
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.error_classifier = ErrorClassifier()
        self.error_logger = ErrorLogger(
            log_directory=self.config.get('log_directory', 'logs/errors'),
            max_memory_entries=self.config.get('max_memory_entries', 1000)
        )
        self.failure_tracker = FailureTracker(
            max_events=self.config.get('max_failure_events', 10000)
        )
        self.error_recovery = ErrorRecoveryManager(
            config=self.config.get('error_recovery', {})
        )
        
        # Initialize component monitors
        self.monitors = {
            'document_processor': DocumentProcessorMonitor(
                self.error_classifier, self.error_logger, self.failure_tracker
            ),
            'vector_store': VectorStoreMonitor(
                self.error_classifier, self.error_logger, self.failure_tracker
            ),
            'llm_client': LLMClientMonitor(
                self.error_classifier, self.error_logger, self.failure_tracker
            ),
            'api_endpoints': APIEndpointMonitor(
                self.error_classifier, self.error_logger, self.failure_tracker
            ),
            'cache_manager': CacheMonitor(
                self.error_classifier, self.error_logger, self.failure_tracker
            )
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # System status tracking
        self._last_health_check = None
        self._system_alerts = []
        
        self.logger.info("Error Analysis Manager initialized with components: %s", list(self.monitors.keys()))
    
    def get_monitor(self, component_name: str):
        """
        Get monitor for specific component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Component monitor or None if not found
        """
        return self.monitors.get(component_name)
    
    def track_error(self, exception: Exception, component: str, context: Optional[Dict] = None):
        """
        Track an error across all analysis systems and attempt recovery.
        
        Args:
            exception: The exception that occurred
            component: Component where error occurred
            context: Additional context information
        """
        try:
            # Classify error
            error_info = self.error_classifier.classify_exception(exception, component, context)
            
            # Log error
            self.error_logger.log_error(error_info)
            
            # Track failure
            self.failure_tracker.track_error(error_info)
            
            # Attempt error recovery
            recovery_result = self.error_recovery.recover_from_error(error_info, component, context)
            
            # Update component monitor if available
            if component in self.monitors:
                # The monitor's decorator will handle the tracking
                pass
            
            self.logger.debug(f"Tracked error in {component}: {error_info.error_id}")
            
            return recovery_result
            
        except Exception as e:
            self.logger.error(f"Failed to track error: {e}")
            return None
    
    def get_system_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system health status.
        
        Returns:
            System health status across all components
        """
        with self._lock:
            timestamp = datetime.now()
            
            # Get overall failure tracker status
            system_health = self.failure_tracker.get_system_health_status()
            
            # Get component-specific health
            component_health = {}
            for name, monitor in self.monitors.items():
                try:
                    component_health[name] = monitor.get_health_status()
                except Exception as e:
                    self.logger.error(f"Failed to get health status for {name}: {e}")
                    component_health[name] = {
                        'status': 'unknown',
                        'error': str(e)
                    }
            
            # Get recent error summary
            error_summary = self.error_logger.get_error_summary(hours=1)
            
            # Get recovery status
            recovery_status = self.error_recovery.get_system_recovery_status()
            
            # Determine overall system status
            critical_components = [
                name for name, health in component_health.items()
                if health.get('status') in ['critical', 'poor']
            ]
            
            if critical_components:
                overall_status = 'critical'
            elif system_health.get('overall_status') == 'degraded':
                overall_status = 'degraded'
            elif error_summary.get('critical_errors', 0) > 0:
                overall_status = 'warning'
            else:
                overall_status = 'healthy'
            
            self._last_health_check = timestamp
            
            return {
                'overall_status': overall_status,
                'timestamp': timestamp.isoformat(),
                'system_health': system_health,
                'component_health': component_health,
                'error_summary': error_summary,
                'recovery_status': recovery_status,
                'critical_components': critical_components,
                'active_alerts': len(self._system_alerts),
                'last_health_check': self._last_health_check.isoformat() if self._last_health_check else None
            }
    
    def get_comprehensive_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive error and failure analysis.
        
        Args:
            hours: Time period to analyze
            
        Returns:
            Comprehensive analysis across all systems
        """
        analysis = {
            'analysis_period_hours': hours,
            'timestamp': datetime.now().isoformat(),
            'error_analysis': {},
            'failure_analysis': {},
            'component_analysis': {},
            'predictions': {},
            'recommendations': []
        }
        
        try:
            # Error analysis
            analysis['error_analysis'] = self.error_logger.get_error_summary(hours)
            
            # Failure analysis
            analysis['failure_analysis'] = self.failure_tracker.get_failure_analysis(hours)
            
            # Component-specific analysis
            for name, monitor in self.monitors.items():
                try:
                    if hasattr(monitor, 'get_error_summary'):
                        analysis['component_analysis'][name] = monitor.get_error_summary(hours)
                except Exception as e:
                    self.logger.error(f"Failed to get analysis for {name}: {e}")
                    analysis['component_analysis'][name] = {'error': str(e)}
            
            # Failure predictions
            analysis['predictions'] = self.failure_tracker.get_failure_predictions()
            
            # Recovery statistics
            analysis['recovery_statistics'] = self.error_recovery.get_recovery_statistics(hours)
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_system_recommendations(analysis)
            
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive analysis: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _generate_system_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate system-wide recommendations based on analysis."""
        recommendations = []
        
        # Check error rates
        error_analysis = analysis.get('error_analysis', {})
        if error_analysis.get('critical_errors', 0) > 0:
            recommendations.append("Critical errors detected - immediate investigation required")
        
        if error_analysis.get('error_rate_per_hour', 0) > 10:
            recommendations.append("High error rate detected - review system stability")
        
        # Check failure patterns
        failure_analysis = analysis.get('failure_analysis', {})
        if failure_analysis.get('cascade_failures', 0) > 0:
            recommendations.append("Cascade failures detected - implement circuit breakers")
        
        # Check predictions
        predictions = analysis.get('predictions', {})
        high_risk_components = predictions.get('high_risk_components', [])
        if high_risk_components:
            recommendations.append(f"High risk components detected: {[c['component'] for c in high_risk_components]}")
        
        # Component-specific recommendations
        component_analysis = analysis.get('component_analysis', {})
        for component, comp_analysis in component_analysis.items():
            if isinstance(comp_analysis, dict) and comp_analysis.get('error_rate_per_hour', 0) > 5:
                recommendations.append(f"High error rate in {component} - review component health")
        
        if not recommendations:
            recommendations.append("System is operating within normal parameters")
        
        return recommendations
    
    def add_system_alert(self, alert: Dict[str, Any]):
        """
        Add system-wide alert.
        
        Args:
            alert: Alert information
        """
        with self._lock:
            alert['timestamp'] = datetime.now().isoformat()
            self._system_alerts.append(alert)
            
            # Keep only recent alerts (last 100)
            if len(self._system_alerts) > 100:
                self._system_alerts = self._system_alerts[-100:]
            
            self.logger.warning(f"System alert added: {alert.get('message', 'Unknown alert')}")
    
    def get_system_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent system alerts.
        
        Args:
            hours: Time period to retrieve alerts for
            
        Returns:
            List of recent alerts
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff.isoformat()
        
        return [
            alert for alert in self._system_alerts
            if alert.get('timestamp', '') >= cutoff_str
        ]
    
    def clear_alerts(self, older_than_hours: int = 24):
        """
        Clear old alerts.
        
        Args:
            older_than_hours: Clear alerts older than this many hours
        """
        with self._lock:
            cutoff = datetime.now() - timedelta(hours=older_than_hours)
            cutoff_str = cutoff.isoformat()
            
            initial_count = len(self._system_alerts)
            self._system_alerts = [
                alert for alert in self._system_alerts
                if alert.get('timestamp', '') >= cutoff_str
            ]
            
            cleared_count = initial_count - len(self._system_alerts)
            if cleared_count > 0:
                self.logger.info(f"Cleared {cleared_count} old alerts")
    
    def export_analysis_data(self, hours: int = 24, format: str = "json") -> Dict[str, Any]:
        """
        Export comprehensive analysis data.
        
        Args:
            hours: Time period to export
            format: Export format
            
        Returns:
            Exported data
        """
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'export_period_hours': hours,
                'system_health': self.get_system_health_status(),
                'comprehensive_analysis': self.get_comprehensive_analysis(hours),
                'error_trends': self.error_logger.get_error_trends(hours),
                'system_alerts': self.get_system_alerts(hours)
            }
            
            # Add component-specific exports
            export_data['component_exports'] = {}
            for name, monitor in self.monitors.items():
                try:
                    if hasattr(monitor, 'get_performance_trends'):
                        export_data['component_exports'][name] = monitor.get_performance_trends(hours)
                except Exception as e:
                    export_data['component_exports'][name] = {'export_error': str(e)}
            
            return export_data
            
        except Exception as e:
            self.logger.error(f"Failed to export analysis data: {e}")
            return {'export_error': str(e)}
    
    def reset_all_metrics(self):
        """Reset all metrics across all components."""
        with self._lock:
            try:
                # Reset component monitors
                for monitor in self.monitors.values():
                    monitor.reset_metrics()
                
                # Reset system alerts
                self._system_alerts.clear()
                
                self.logger.info("Reset all error analysis metrics")
                
            except Exception as e:
                self.logger.error(f"Failed to reset metrics: {e}")
    
    def configure_alerts(self, component: str, thresholds: Dict[str, Any]):
        """
        Configure alert thresholds for a component.
        
        Args:
            component: Component name
            thresholds: Alert threshold configuration
        """
        if component in self.monitors:
            self.monitors[component].update_alert_thresholds(thresholds)
            self.logger.info(f"Updated alert thresholds for {component}")
        else:
            self.logger.warning(f"Component {component} not found for alert configuration")
    
    def get_recovery_manager(self) -> ErrorRecoveryManager:
        """
        Get the error recovery manager instance.
        
        Returns:
            ErrorRecoveryManager instance
        """
        return self.error_recovery
    
    def register_recovery_fallback(self, component: str, handler: callable):
        """
        Register a custom recovery fallback handler.
        
        Args:
            component: Component name
            handler: Fallback handler function
        """
        self.error_recovery.register_fallback_handler(component, handler)
        self.logger.info(f"Registered recovery fallback for {component}")
    
    def add_recovery_cache_fallback(self, key: str, value: Any):
        """
        Add a fallback value for recovery cache.
        
        Args:
            key: Cache key
            value: Fallback value
        """
        self.error_recovery.add_cache_fallback(key, value)


# Global instance for easy access
_global_error_manager = None


def get_error_analysis_manager(config: Optional[Dict[str, Any]] = None) -> ErrorAnalysisManager:
    """
    Get global error analysis manager instance.
    
    Args:
        config: Configuration for manager (used only on first call)
        
    Returns:
        ErrorAnalysisManager instance
    """
    global _global_error_manager
    
    if _global_error_manager is None:
        _global_error_manager = ErrorAnalysisManager(config)
    
    return _global_error_manager