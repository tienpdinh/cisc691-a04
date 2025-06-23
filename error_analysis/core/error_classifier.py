"""
Error Classification System for RAG Pipeline

Provides systematic error categorization, severity assessment, and pattern recognition
for all components in the RAG system.
"""

import logging
import traceback
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
import re


class ErrorSeverity(Enum):
    """Error severity levels for RAG system errors."""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"


class ErrorCategory(Enum):
    """Error categories for different RAG pipeline components."""
    SYSTEM_DOWN = "system_down"
    DATA_CORRUPTION = "data_corruption"
    SECURITY = "security"
    RETRIEVAL_FAILURE = "retrieval_failure"
    GENERATION_FAILURE = "generation_failure"
    PERFORMANCE = "performance"
    QUALITY_DEGRADATION = "quality_degradation"
    PROCESSING_ERRORS = "processing_errors"
    CACHE_ISSUES = "cache_issues"
    VALIDATION_WARNINGS = "validation_warnings"
    CONFIGURATION = "configuration"
    MONITORING = "monitoring"


@dataclass
class ErrorInfo:
    """Container for detailed error information."""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    error_type: str
    message: str
    details: Dict[str, Any]
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    recovery_suggested: Optional[str] = None


class ErrorClassifier:
    """
    Comprehensive error classification system for RAG pipeline.
    
    Analyzes exceptions, system states, and performance metrics to classify
    errors by severity and category for proper handling and reporting.
    """
    
    def __init__(self):
        """Initialize error classifier with predefined patterns."""
        self.logger = logging.getLogger(__name__)
        self._init_error_patterns()
        self._init_component_mapping()
    
    def _init_error_patterns(self):
        """Initialize error pattern matching rules."""
        self.error_patterns = {
            # Critical system errors
            ErrorCategory.SYSTEM_DOWN: {
                'patterns': [
                    r'connection.*refused',
                    r'service.*unavailable',
                    r'database.*down',
                    r'api.*unreachable',
                    r'timeout.*exceeded',
                    r'network.*error'
                ],
                'severity': ErrorSeverity.CRITICAL,
                'keywords': ['connection', 'timeout', 'unavailable', 'refused', 'unreachable']
            },
            
            # Data integrity issues
            ErrorCategory.DATA_CORRUPTION: {
                'patterns': [
                    r'index.*corrupt',
                    r'embedding.*invalid',
                    r'vector.*malformed',
                    r'cache.*corrupt',
                    r'serialization.*failed'
                ],
                'severity': ErrorSeverity.CRITICAL,
                'keywords': ['corrupt', 'malformed', 'invalid', 'serialization']
            },
            
            # Security issues
            ErrorCategory.SECURITY: {
                'patterns': [
                    r'authentication.*failed',
                    r'unauthorized.*access',
                    r'permission.*denied',
                    r'api.*key.*invalid',
                    r'security.*violation'
                ],
                'severity': ErrorSeverity.CRITICAL,
                'keywords': ['authentication', 'unauthorized', 'permission', 'security']
            },
            
            # Retrieval system errors
            ErrorCategory.RETRIEVAL_FAILURE: {
                'patterns': [
                    r'no.*documents.*found',
                    r'similarity.*search.*failed',
                    r'embedding.*generation.*failed',
                    r'vector.*store.*error',
                    r'retrieval.*timeout'
                ],
                'severity': ErrorSeverity.HIGH,
                'keywords': ['retrieval', 'similarity', 'embedding', 'vector', 'documents']
            },
            
            # LLM generation errors
            ErrorCategory.GENERATION_FAILURE: {
                'patterns': [
                    r'llm.*response.*empty',
                    r'model.*loading.*failed',
                    r'context.*too.*long',
                    r'generation.*timeout',
                    r'prompt.*processing.*error'
                ],
                'severity': ErrorSeverity.HIGH,
                'keywords': ['llm', 'model', 'generation', 'prompt', 'response']
            },
            
            # Performance issues
            ErrorCategory.PERFORMANCE: {
                'patterns': [
                    r'memory.*exhausted',
                    r'rate.*limit.*exceeded',
                    r'processing.*slow',
                    r'latency.*high',
                    r'throughput.*low'
                ],
                'severity': ErrorSeverity.HIGH,
                'keywords': ['memory', 'rate', 'latency', 'throughput', 'performance']
            },
            
            # Quality degradation
            ErrorCategory.QUALITY_DEGRADATION: {
                'patterns': [
                    r'relevance.*score.*low',
                    r'response.*incoherent',
                    r'hallucination.*detected',
                    r'factual.*error',
                    r'quality.*metrics.*poor'
                ],
                'severity': ErrorSeverity.MEDIUM,
                'keywords': ['relevance', 'quality', 'coherent', 'factual', 'hallucination']
            },
            
            # Processing errors
            ErrorCategory.PROCESSING_ERRORS: {
                'patterns': [
                    r'file.*format.*unsupported',
                    r'text.*extraction.*failed',
                    r'chunking.*error',
                    r'preprocessing.*failed',
                    r'parsing.*error'
                ],
                'severity': ErrorSeverity.MEDIUM,
                'keywords': ['file', 'text', 'chunking', 'preprocessing', 'parsing']
            },
            
            # Cache issues
            ErrorCategory.CACHE_ISSUES: {
                'patterns': [
                    r'cache.*miss',
                    r'redis.*connection.*failed',
                    r'cache.*eviction',
                    r'ttl.*expired',
                    r'cache.*corruption'
                ],
                'severity': ErrorSeverity.MEDIUM,
                'keywords': ['cache', 'redis', 'eviction', 'ttl', 'miss']
            },
            
            # Validation warnings
            ErrorCategory.VALIDATION_WARNINGS: {
                'patterns': [
                    r'input.*sanitized',
                    r'parameter.*adjusted',
                    r'default.*value.*used',
                    r'validation.*warning',
                    r'format.*corrected'
                ],
                'severity': ErrorSeverity.LOW,
                'keywords': ['sanitized', 'adjusted', 'default', 'validation', 'warning']
            },
            
            # Configuration issues
            ErrorCategory.CONFIGURATION: {
                'patterns': [
                    r'config.*missing',
                    r'setting.*not.*found',
                    r'environment.*variable.*missing',
                    r'parameter.*deprecated',
                    r'configuration.*invalid'
                ],
                'severity': ErrorSeverity.LOW,
                'keywords': ['config', 'setting', 'environment', 'parameter', 'deprecated']
            },
            
            # Monitoring issues
            ErrorCategory.MONITORING: {
                'patterns': [
                    r'metric.*collection.*failed',
                    r'log.*rotation',
                    r'cleanup.*performed',
                    r'monitoring.*error',
                    r'instrumentation.*failed'
                ],
                'severity': ErrorSeverity.LOW,
                'keywords': ['metric', 'log', 'cleanup', 'monitoring', 'instrumentation']
            }
        }
    
    def _init_component_mapping(self):
        """Initialize component-specific error mapping."""
        self.component_keywords = {
            'document_processor': ['pdf', 'docx', 'text', 'extraction', 'parsing', 'chunking'],
            'vector_store': ['chromadb', 'embedding', 'vector', 'similarity', 'index'],
            'llm_client': ['ollama', 'openai', 'model', 'generation', 'prompt', 'response'],
            'api_endpoints': ['fastapi', 'endpoint', 'request', 'response', 'validation'],
            'cache_manager': ['redis', 'cache', 'ttl', 'eviction', 'serialization'],
            'rag_processor': ['retrieval', 'augmentation', 'context', 'pipeline']
        }
    
    def classify_exception(self, 
                         exception: Exception, 
                         component: str = "unknown",
                         context: Optional[Dict[str, Any]] = None) -> ErrorInfo:
        """
        Classify an exception into error category and severity.
        
        Args:
            exception: The exception to classify
            component: Component where error occurred
            context: Additional context information
            
        Returns:
            ErrorInfo object with classification details
        """
        error_message = str(exception).lower()
        error_type = type(exception).__name__
        stack_trace = traceback.format_exc()
        
        # Generate unique error ID
        error_id = self._generate_error_id(error_message, error_type, component)
        
        # Classify error
        category, severity = self._classify_error_text(error_message, error_type)
        
        # Determine component if not provided
        if component == "unknown":
            component = self._detect_component(error_message, stack_trace)
        
        # Suggest recovery action
        recovery_suggestion = self._suggest_recovery(category, error_type)
        
        return ErrorInfo(
            error_id=error_id,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            component=component,
            error_type=error_type,
            message=str(exception),
            details={
                'error_class': error_type,
                'error_args': getattr(exception, 'args', []),
                'classification_confidence': self._calculate_confidence(error_message, category)
            },
            stack_trace=stack_trace,
            context=context or {},
            recovery_suggested=recovery_suggestion
        )
    
    def classify_performance_issue(self,
                                 metric_name: str,
                                 metric_value: float,
                                 threshold: float,
                                 component: str) -> ErrorInfo:
        """
        Classify performance-related issues.
        
        Args:
            metric_name: Name of the performance metric
            metric_value: Current metric value
            threshold: Expected threshold
            component: Component being monitored
            
        Returns:
            ErrorInfo for performance issue
        """
        # Determine severity based on threshold deviation
        deviation = abs(metric_value - threshold) / threshold
        
        if deviation > 0.5:  # 50% deviation
            severity = ErrorSeverity.HIGH
        elif deviation > 0.25:  # 25% deviation
            severity = ErrorSeverity.MEDIUM
        else:
            severity = ErrorSeverity.LOW
        
        error_id = self._generate_error_id(f"performance_{metric_name}", component, str(metric_value))
        
        return ErrorInfo(
            error_id=error_id,
            timestamp=datetime.now(),
            severity=severity,
            category=ErrorCategory.PERFORMANCE,
            component=component,
            error_type="PerformanceThresholdExceeded",
            message=f"{metric_name} ({metric_value}) exceeded threshold ({threshold})",
            details={
                'metric_name': metric_name,
                'current_value': metric_value,
                'threshold': threshold,
                'deviation_percentage': deviation * 100
            },
            recovery_suggested=f"Monitor {metric_name} and consider scaling {component}"
        )
    
    def _classify_error_text(self, error_message: str, error_type: str) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify error based on message text and type."""
        best_match = None
        best_score = 0
        
        for category, config in self.error_patterns.items():
            score = 0
            
            # Check pattern matches
            for pattern in config['patterns']:
                if re.search(pattern, error_message, re.IGNORECASE):
                    score += 3
            
            # Check keyword matches
            for keyword in config['keywords']:
                if keyword.lower() in error_message:
                    score += 1
            
            # Bonus for error type matches
            if error_type.lower() in error_message:
                score += 2
            
            if score > best_score:
                best_score = score
                best_match = category
        
        # Default classification if no patterns match
        if best_match is None:
            best_match = ErrorCategory.PROCESSING_ERRORS
            severity = ErrorSeverity.MEDIUM
        else:
            severity = self.error_patterns[best_match]['severity']
        
        return best_match, severity
    
    def _detect_component(self, error_message: str, stack_trace: str) -> str:
        """Detect component from error message and stack trace."""
        combined_text = f"{error_message} {stack_trace}".lower()
        
        component_scores = {}
        for component, keywords in self.component_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                component_scores[component] = score
        
        if component_scores:
            return max(component_scores, key=component_scores.get)
        
        return "unknown"
    
    def _suggest_recovery(self, category: ErrorCategory, error_type: str) -> str:
        """Suggest recovery action based on error category."""
        recovery_suggestions = {
            ErrorCategory.SYSTEM_DOWN: "Check service availability and restart if necessary",
            ErrorCategory.DATA_CORRUPTION: "Verify data integrity and restore from backup",
            ErrorCategory.SECURITY: "Review authentication credentials and permissions",
            ErrorCategory.RETRIEVAL_FAILURE: "Check vector store connection and rebuild index if needed",
            ErrorCategory.GENERATION_FAILURE: "Verify LLM service status and retry with different parameters",
            ErrorCategory.PERFORMANCE: "Monitor resource usage and consider scaling",
            ErrorCategory.QUALITY_DEGRADATION: "Review input data quality and model parameters",
            ErrorCategory.PROCESSING_ERRORS: "Validate input format and preprocessing pipeline",
            ErrorCategory.CACHE_ISSUES: "Check cache service and clear if corrupted",
            ErrorCategory.VALIDATION_WARNINGS: "Review input validation rules",
            ErrorCategory.CONFIGURATION: "Verify configuration settings and environment",
            ErrorCategory.MONITORING: "Check monitoring service configuration"
        }
        
        return recovery_suggestions.get(category, "Review error details and contact support")
    
    def _generate_error_id(self, *components) -> str:
        """Generate unique error ID from components."""
        combined = "_".join(str(c) for c in components)
        return hashlib.md5(combined.encode()).hexdigest()[:12]
    
    def _calculate_confidence(self, error_message: str, category: ErrorCategory) -> float:
        """Calculate classification confidence score."""
        if category not in self.error_patterns:
            return 0.5
        
        config = self.error_patterns[category]
        matches = 0
        total_checks = len(config['patterns']) + len(config['keywords'])
        
        # Count pattern matches
        for pattern in config['patterns']:
            if re.search(pattern, error_message, re.IGNORECASE):
                matches += 1
        
        # Count keyword matches
        for keyword in config['keywords']:
            if keyword.lower() in error_message:
                matches += 1
        
        return min(1.0, matches / total_checks) if total_checks > 0 else 0.5
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error classification statistics."""
        return {
            'total_categories': len(ErrorCategory),
            'total_severity_levels': len(ErrorSeverity),
            'patterns_configured': sum(len(config['patterns']) for config in self.error_patterns.values()),
            'components_monitored': len(self.component_keywords)
        }