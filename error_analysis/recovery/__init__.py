"""
Error Recovery Mechanisms

Automatic error recovery capabilities including retry strategies,
circuit breakers, fallback mechanisms, and graceful degradation.
"""

from .error_recovery import (
    ErrorRecoveryManager, 
    RecoveryStrategy, 
    with_error_recovery,
    CircuitBreaker,
    CircuitState,
    RetryConfig,
    CircuitBreakerConfig
)

__all__ = [
    'ErrorRecoveryManager',
    'RecoveryStrategy',
    'with_error_recovery',
    'CircuitBreaker',
    'CircuitState',
    'RetryConfig',
    'CircuitBreakerConfig'
]