"""
Failure Mode Tracking System for RAG Pipeline

Tracks different types of system failures, performance degradation patterns,
and provides failure prediction capabilities.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import statistics

from .error_classifier import ErrorInfo, ErrorSeverity, ErrorCategory


class FailureMode(Enum):
    """Types of failure modes in RAG system."""
    COMPLETE_SYSTEM_FAILURE = "complete_system_failure"
    COMPONENT_FAILURE = "component_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    QUALITY_DEGRADATION = "quality_degradation"
    CASCADE_FAILURE = "cascade_failure"
    SILENT_FAILURE = "silent_failure"
    INTERMITTENT_FAILURE = "intermittent_failure"


@dataclass
class FailureEvent:
    """Container for failure event information."""
    failure_id: str
    timestamp: datetime
    failure_mode: FailureMode
    component: str
    severity: ErrorSeverity
    description: str
    affected_components: List[str]
    root_cause: Optional[str] = None
    impact_metrics: Dict[str, float] = field(default_factory=dict)
    recovery_time_seconds: Optional[float] = None
    is_resolved: bool = False


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    component_name: str
    last_seen: datetime
    error_count_1h: int = 0
    error_count_24h: int = 0
    avg_response_time: float = 0.0
    success_rate: float = 100.0
    health_score: float = 100.0  # 0-100
    status: str = "healthy"  # healthy, degraded, critical, down


class FailureTracker:
    """
    Comprehensive failure mode tracking and analysis system.
    
    Monitors system health, detects failure patterns, and provides
    early warning for potential issues.
    """
    
    def __init__(self, max_events: int = 10000):
        """
        Initialize failure tracker.
        
        Args:
            max_events: Maximum failure events to keep in memory
        """
        self.max_events = max_events
        self.failure_events = deque(maxlen=max_events)
        self.component_health = {}
        
        # Thread-safe operations
        self._lock = threading.Lock()
        
        # Failure pattern tracking
        self.failure_patterns = defaultdict(list)
        self.cascade_detection = defaultdict(list)
        
        # Performance baselines
        self.performance_baselines = {}
        self.degradation_thresholds = {
            'response_time': 0.5,  # 50% increase
            'error_rate': 0.1,     # 10% error rate
            'success_rate': 0.95   # Below 95% success
        }
        
        self.logger = logging.getLogger(__name__)
    
    def track_error(self, error_info: ErrorInfo):
        """
        Track an error and analyze for failure patterns.
        
        Args:
            error_info: ErrorInfo object to analyze
        """
        with self._lock:
            # Update component health
            self._update_component_health(error_info)
            
            # Detect failure modes
            failure_mode = self._detect_failure_mode(error_info)
            
            if failure_mode:
                # Create failure event
                failure_event = FailureEvent(
                    failure_id=f"failure_{error_info.error_id}",
                    timestamp=error_info.timestamp,
                    failure_mode=failure_mode,
                    component=error_info.component,
                    severity=error_info.severity,
                    description=f"{failure_mode.value}: {error_info.message}",
                    affected_components=[error_info.component]
                )
                
                # Check for cascade failures
                affected_components = self._detect_cascade_failure(error_info)
                failure_event.affected_components = affected_components
                
                self.failure_events.append(failure_event)
                
                # Update failure patterns
                self._update_failure_patterns(failure_event)
                
                self.logger.warning(f"Failure detected: {failure_mode.value} in {error_info.component}")
    
    def track_performance_metric(self, 
                               component: str,
                               metric_name: str, 
                               value: float,
                               timestamp: Optional[datetime] = None):
        """
        Track performance metrics for failure prediction.
        
        Args:
            component: Component name
            metric_name: Name of the metric
            value: Metric value
            timestamp: When metric was recorded
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            # Initialize component health if needed
            if component not in self.component_health:
                self.component_health[component] = ComponentHealth(
                    component_name=component,
                    last_seen=timestamp
                )
            
            health = self.component_health[component]
            health.last_seen = timestamp
            
            # Update specific metrics
            if metric_name == "response_time":
                health.avg_response_time = value
            elif metric_name == "success_rate":
                health.success_rate = value
            
            # Check for performance degradation
            if self._is_performance_degraded(component, metric_name, value):
                self._create_degradation_failure(component, metric_name, value)
            
            # Update overall health score
            health.health_score = self._calculate_health_score(health)
            health.status = self._determine_health_status(health.health_score)
    
    def _update_component_health(self, error_info: ErrorInfo):
        """Update component health metrics based on error."""
        component = error_info.component
        
        if component not in self.component_health:
            self.component_health[component] = ComponentHealth(
                component_name=component,
                last_seen=error_info.timestamp
            )
        
        health = self.component_health[component]
        health.last_seen = error_info.timestamp
        
        # Count recent errors
        now = datetime.now()
        recent_errors_1h = self._count_recent_errors(component, hours=1)
        recent_errors_24h = self._count_recent_errors(component, hours=24)
        
        health.error_count_1h = recent_errors_1h
        health.error_count_24h = recent_errors_24h
        
        # Update health score
        health.health_score = self._calculate_health_score(health)
        health.status = self._determine_health_status(health.health_score)
    
    def _detect_failure_mode(self, error_info: ErrorInfo) -> Optional[FailureMode]:
        """Detect failure mode based on error characteristics."""
        # Complete system failure indicators
        if error_info.severity == ErrorSeverity.CRITICAL:
            if error_info.category in [ErrorCategory.SYSTEM_DOWN, ErrorCategory.DATA_CORRUPTION]:
                return FailureMode.COMPLETE_SYSTEM_FAILURE
        
        # Component-specific failures
        if error_info.category in [
            ErrorCategory.RETRIEVAL_FAILURE, 
            ErrorCategory.GENERATION_FAILURE,
            ErrorCategory.CACHE_ISSUES
        ]:
            return FailureMode.COMPONENT_FAILURE
        
        # Performance degradation
        if error_info.category == ErrorCategory.PERFORMANCE:
            return FailureMode.PERFORMANCE_DEGRADATION
        
        # Quality issues
        if error_info.category == ErrorCategory.QUALITY_DEGRADATION:
            return FailureMode.QUALITY_DEGRADATION
        
        # Silent failures (configuration or validation issues that might cause wrong results)
        if error_info.category in [ErrorCategory.VALIDATION_WARNINGS, ErrorCategory.CONFIGURATION]:
            return FailureMode.SILENT_FAILURE
        
        return None
    
    def _detect_cascade_failure(self, error_info: ErrorInfo) -> List[str]:
        """Detect if this error is part of a cascade failure."""
        affected_components = [error_info.component]
        
        # Look for recent failures in related components
        recent_window = datetime.now() - timedelta(minutes=10)
        
        for event in reversed(self.failure_events):
            if event.timestamp < recent_window:
                break
            
            # If we see failures in multiple components within a short time
            if event.component != error_info.component:
                if event.component not in affected_components:
                    affected_components.append(event.component)
        
        # If multiple components are affected, it's likely a cascade
        if len(affected_components) > 1:
            return affected_components
        
        return [error_info.component]
    
    def _is_performance_degraded(self, component: str, metric_name: str, value: float) -> bool:
        """Check if performance metric indicates degradation."""
        # Get baseline for this metric
        baseline_key = f"{component}_{metric_name}"
        
        if baseline_key not in self.performance_baselines:
            # First measurement becomes baseline
            self.performance_baselines[baseline_key] = value
            return False
        
        baseline = self.performance_baselines[baseline_key]
        
        # Check for degradation based on metric type
        if metric_name == "response_time":
            # Response time increased significantly
            return value > baseline * (1 + self.degradation_thresholds['response_time'])
        
        elif metric_name == "success_rate":
            # Success rate dropped below threshold
            return value < self.degradation_thresholds['success_rate']
        
        elif metric_name == "error_rate":
            # Error rate increased above threshold
            return value > self.degradation_thresholds['error_rate']
        
        return False
    
    def _create_degradation_failure(self, component: str, metric_name: str, value: float):
        """Create a failure event for performance degradation."""
        failure_event = FailureEvent(
            failure_id=f"degradation_{component}_{metric_name}_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            failure_mode=FailureMode.PERFORMANCE_DEGRADATION,
            component=component,
            severity=ErrorSeverity.MEDIUM,
            description=f"Performance degradation detected: {metric_name} = {value}",
            affected_components=[component],
            impact_metrics={metric_name: value}
        )
        
        self.failure_events.append(failure_event)
        self.logger.warning(f"Performance degradation in {component}: {metric_name} = {value}")
    
    def _calculate_health_score(self, health: ComponentHealth) -> float:
        """Calculate overall health score for a component."""
        score = 100.0
        
        # Deduct for recent errors
        score -= min(health.error_count_1h * 10, 50)  # Max 50 point deduction for 1h errors
        score -= min(health.error_count_24h * 2, 30)   # Max 30 point deduction for 24h errors
        
        # Deduct for poor success rate
        if health.success_rate < 100:
            score -= (100 - health.success_rate) * 2
        
        # Deduct for high response time (if available)
        if health.avg_response_time > 1.0:  # More than 1 second
            score -= min(health.avg_response_time * 10, 20)
        
        return max(0.0, score)
    
    def _determine_health_status(self, health_score: float) -> str:
        """Determine health status based on score."""
        if health_score >= 80:
            return "healthy"
        elif health_score >= 60:
            return "degraded"
        elif health_score >= 30:
            return "critical"
        else:
            return "down"
    
    def _count_recent_errors(self, component: str, hours: int) -> int:
        """Count errors for a component in recent hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        count = 0
        
        for event in reversed(self.failure_events):
            if event.timestamp < cutoff:
                break
            if event.component == component:
                count += 1
        
        return count
    
    def _update_failure_patterns(self, failure_event: FailureEvent):
        """Update failure pattern tracking."""
        pattern_key = f"{failure_event.component}_{failure_event.failure_mode.value}"
        self.failure_patterns[pattern_key].append(failure_event.timestamp)
        
        # Keep only recent patterns (last 7 days)
        cutoff = datetime.now() - timedelta(days=7)
        self.failure_patterns[pattern_key] = [
            ts for ts in self.failure_patterns[pattern_key] if ts >= cutoff
        ]
    
    def get_system_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        with self._lock:
            total_components = len(self.component_health)
            if total_components == 0:
                return {
                    'overall_status': 'unknown',
                    'total_components': 0,
                    'healthy_components': 0,
                    'degraded_components': 0,
                    'critical_components': 0,
                    'down_components': 0,
                    'system_health_score': 0
                }
            
            status_counts = defaultdict(int)
            health_scores = []
            
            for health in self.component_health.values():
                status_counts[health.status] += 1
                health_scores.append(health.health_score)
            
            avg_health_score = statistics.mean(health_scores) if health_scores else 0
            
            # Determine overall status
            if status_counts['down'] > 0:
                overall_status = 'critical'
            elif status_counts['critical'] > 0:
                overall_status = 'critical'
            elif status_counts['degraded'] > 0:
                overall_status = 'degraded'
            else:
                overall_status = 'healthy'
            
            return {
                'overall_status': overall_status,
                'total_components': total_components,
                'healthy_components': status_counts['healthy'],
                'degraded_components': status_counts['degraded'],
                'critical_components': status_counts['critical'],
                'down_components': status_counts['down'],
                'system_health_score': round(avg_health_score, 2),
                'component_details': {
                    name: {
                        'status': health.status,
                        'health_score': health.health_score,
                        'last_seen': health.last_seen.isoformat(),
                        'error_count_1h': health.error_count_1h,
                        'success_rate': health.success_rate
                    }
                    for name, health in self.component_health.items()
                }
            }
    
    def get_failure_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Get failure analysis for specified time period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_failures = [
            event for event in self.failure_events 
            if event.timestamp >= cutoff
        ]
        
        if not recent_failures:
            return {
                'total_failures': 0,
                'failure_modes': {},
                'affected_components': {},
                'cascade_failures': 0,
                'mean_time_to_failure': None,
                'most_frequent_failure_mode': None,
                'most_affected_component': None
            }
        
        # Analyze failure modes
        failure_mode_counts = defaultdict(int)
        component_failure_counts = defaultdict(int)
        cascade_count = 0
        
        for event in recent_failures:
            failure_mode_counts[event.failure_mode.value] += 1
            component_failure_counts[event.component] += 1
            
            if len(event.affected_components) > 1:
                cascade_count += 1
        
        # Calculate mean time between failures
        if len(recent_failures) > 1:
            time_diffs = []
            for i in range(1, len(recent_failures)):
                diff = (recent_failures[i].timestamp - recent_failures[i-1].timestamp).total_seconds()
                time_diffs.append(diff)
            mtbf = statistics.mean(time_diffs) if time_diffs else None
        else:
            mtbf = None
        
        return {
            'analysis_period_hours': hours,
            'total_failures': len(recent_failures),
            'failure_modes': dict(failure_mode_counts),
            'affected_components': dict(component_failure_counts),
            'cascade_failures': cascade_count,
            'mean_time_between_failures_seconds': mtbf,
            'most_frequent_failure_mode': max(failure_mode_counts, key=failure_mode_counts.get) if failure_mode_counts else None,
            'most_affected_component': max(component_failure_counts, key=component_failure_counts.get) if component_failure_counts else None,
            'failure_rate_per_hour': len(recent_failures) / hours if hours > 0 else 0
        }
    
    def get_failure_predictions(self) -> Dict[str, Any]:
        """Predict potential failures based on patterns."""
        predictions = []
        
        with self._lock:
            for component, health in self.component_health.items():
                risk_score = 0
                risk_factors = []
                
                # High error rate
                if health.error_count_1h > 5:
                    risk_score += 30
                    risk_factors.append(f"High error rate: {health.error_count_1h} errors in 1h")
                
                # Poor success rate
                if health.success_rate < 95:
                    risk_score += 25
                    risk_factors.append(f"Low success rate: {health.success_rate}%")
                
                # High response time
                if health.avg_response_time > 2.0:
                    risk_score += 20
                    risk_factors.append(f"High response time: {health.avg_response_time}s")
                
                # Declining health score
                if health.health_score < 70:
                    risk_score += 25
                    risk_factors.append(f"Poor health score: {health.health_score}")
                
                # Check for failure patterns
                pattern_key = f"{component}_component_failure"
                if pattern_key in self.failure_patterns:
                    recent_failures = len(self.failure_patterns[pattern_key])
                    if recent_failures > 3:
                        risk_score += 20
                        risk_factors.append(f"Recurring failures: {recent_failures} in past week")
                
                if risk_score > 30:  # Threshold for prediction
                    predictions.append({
                        'component': component,
                        'risk_score': min(risk_score, 100),
                        'risk_level': 'high' if risk_score > 70 else 'medium',
                        'risk_factors': risk_factors,
                        'predicted_failure_mode': 'component_failure',
                        'recommendation': self._get_recommendation(component, risk_factors)
                    })
        
        return {
            'total_predictions': len(predictions),
            'high_risk_components': [p for p in predictions if p['risk_level'] == 'high'],
            'medium_risk_components': [p for p in predictions if p['risk_level'] == 'medium'],
            'all_predictions': predictions
        }
    
    def _get_recommendation(self, component: str, risk_factors: List[str]) -> str:
        """Get recommendation based on risk factors."""
        if any('error rate' in factor for factor in risk_factors):
            return f"Monitor {component} closely and investigate error patterns"
        elif any('response time' in factor for factor in risk_factors):
            return f"Check {component} performance and consider scaling"
        elif any('success rate' in factor for factor in risk_factors):
            return f"Review {component} reliability and implement retry mechanisms"
        else:
            return f"Perform health check on {component} and review recent changes"
    
    def reset_baseline(self, component: str, metric_name: str):
        """Reset performance baseline for a component metric."""
        baseline_key = f"{component}_{metric_name}"
        if baseline_key in self.performance_baselines:
            del self.performance_baselines[baseline_key]
            self.logger.info(f"Reset baseline for {baseline_key}")
    
    def mark_failure_resolved(self, failure_id: str, recovery_time_seconds: float):
        """Mark a failure as resolved and record recovery time."""
        with self._lock:
            for event in self.failure_events:
                if event.failure_id == failure_id:
                    event.is_resolved = True
                    event.recovery_time_seconds = recovery_time_seconds
                    self.logger.info(f"Marked failure {failure_id} as resolved (recovery time: {recovery_time_seconds}s)")
                    break