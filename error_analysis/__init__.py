"""
Error Analysis Module for RAG System

Provides comprehensive error tracking, failure mode detection, and recovery mechanisms
for the RAG pipeline components.
"""

from .core.error_classifier import ErrorClassifier, ErrorSeverity, ErrorCategory
from .core.failure_tracker import FailureTracker, FailureMode
from .core.error_logger import ErrorLogger
from .recovery.error_recovery import ErrorRecoveryManager, RecoveryStrategy, with_error_recovery
from .core.error_analysis_manager import ErrorAnalysisManager, get_error_analysis_manager

__all__ = [
    'ErrorClassifier',
    'ErrorSeverity', 
    'ErrorCategory',
    'FailureTracker',
    'FailureMode',
    'ErrorLogger',
    'ErrorRecoveryManager',
    'RecoveryStrategy',
    'with_error_recovery',
    'ErrorAnalysisManager',
    'get_error_analysis_manager'
]