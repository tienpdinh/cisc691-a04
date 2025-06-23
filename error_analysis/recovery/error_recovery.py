"""
Error Recovery System

Provides automatic error recovery mechanisms for common failure scenarios
in the RAG pipeline, including retry strategies, fallback mechanisms,
and circuit breaker patterns.
"""

import time
import logging
import functools
from typing import Optional, Dict, Any, Callable, List, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

from ..core.error_classifier import ErrorInfo, ErrorSeverity, ErrorCategory


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CACHE_FALLBACK = "cache_fallback"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry strategy."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    request_volume_threshold: int = 10


@dataclass
class RecoveryAction:
    """Represents a recovery action taken."""
    timestamp: datetime
    strategy: RecoveryStrategy
    component: str
    error_info: ErrorInfo
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    """Circuit breaker implementation for error recovery."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.success_count = 0
        self.request_count = 0
        
    def can_execute(self) -> bool:
        """Check if request can be executed."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if (self.last_failure_time and 
                datetime.now() - self.last_failure_time > 
                timedelta(seconds=self.config.recovery_timeout)):
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful execution."""
        self.request_count += 1
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed execution."""
        self.request_count += 1
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
        elif (self.state == CircuitState.CLOSED and 
              self.failure_count >= self.config.failure_threshold and
              self.request_count >= self.config.request_volume_threshold):
            self.state = CircuitState.OPEN
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'request_count': self.request_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class ErrorRecoveryManager:
    """Manages error recovery mechanisms for the RAG system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.recovery_actions: List[RecoveryAction] = []
        self.fallback_handlers: Dict[str, Callable] = {}
        self.cache_fallbacks: Dict[str, Any] = {}
        
        # Default recovery strategies per component
        self.component_strategies = {
            'llm_client': [RecoveryStrategy.RETRY, RecoveryStrategy.CACHE_FALLBACK, RecoveryStrategy.FALLBACK],
            'vector_store': [RecoveryStrategy.RETRY, RecoveryStrategy.CIRCUIT_BREAKER],
            'cache_manager': [RecoveryStrategy.RETRY, RecoveryStrategy.GRACEFUL_DEGRADATION],
            'api_endpoints': [RecoveryStrategy.RETRY, RecoveryStrategy.GRACEFUL_DEGRADATION],
            'document_processor': [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK]
        }
        
        self._setup_circuit_breakers()
        self._setup_default_fallbacks()
    
    def _setup_circuit_breakers(self):
        """Initialize circuit breakers for components."""
        components = ['llm_client', 'vector_store', 'cache_manager', 'api_endpoints']
        
        for component in components:
            config = CircuitBreakerConfig(
                failure_threshold=self.config.get('circuit_breaker', {}).get('failure_threshold', 5),
                recovery_timeout=self.config.get('circuit_breaker', {}).get('recovery_timeout', 60.0),
                success_threshold=self.config.get('circuit_breaker', {}).get('success_threshold', 3),
                request_volume_threshold=self.config.get('circuit_breaker', {}).get('request_volume_threshold', 10)
            )
            self.circuit_breakers[component] = CircuitBreaker(component, config)
    
    def _setup_default_fallbacks(self):
        """Setup default fallback handlers."""
        self.fallback_handlers.update({
            'llm_client': self._llm_fallback,
            'vector_store': self._vector_store_fallback,
            'cache_manager': self._cache_fallback,
            'document_processor': self._document_processor_fallback
        })
    
    def recover_from_error(self, error_info: ErrorInfo, component: str, 
                          context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Attempt to recover from an error using appropriate strategies.
        
        Args:
            error_info: Information about the error
            component: Component that experienced the error
            context: Additional context for recovery
            
        Returns:
            Result of recovery attempt or None if no recovery possible
        """
        strategies = self.component_strategies.get(component, [RecoveryStrategy.RETRY])
        
        for strategy in strategies:
            try:
                result = self._apply_strategy(strategy, error_info, component, context)
                if result is not None:
                    self._record_recovery_action(strategy, error_info, component, True, 
                                               {'result': 'success'})
                    return result
            except Exception as recovery_error:
                self._record_recovery_action(strategy, error_info, component, False,
                                           {'recovery_error': str(recovery_error)})
                continue
        
        # No recovery strategy worked
        return None
    
    def _apply_strategy(self, strategy: RecoveryStrategy, error_info: ErrorInfo, 
                       component: str, context: Optional[Dict[str, Any]]) -> Any:
        """Apply a specific recovery strategy."""
        if strategy == RecoveryStrategy.RETRY:
            return self._apply_retry_strategy(error_info, component, context)
        elif strategy == RecoveryStrategy.FALLBACK:
            return self._apply_fallback_strategy(error_info, component, context)
        elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return self._apply_circuit_breaker_strategy(error_info, component, context)
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return self._apply_graceful_degradation(error_info, component, context)
        elif strategy == RecoveryStrategy.CACHE_FALLBACK:
            return self._apply_cache_fallback(error_info, component, context)
        
        return None
    
    def _apply_retry_strategy(self, error_info: ErrorInfo, component: str, 
                             context: Optional[Dict[str, Any]]) -> Any:
        """Apply retry strategy with exponential backoff."""
        if not self._should_retry(error_info):
            return None
        
        config = RetryConfig(
            max_attempts=self.config.get('retry', {}).get('max_attempts', 3),
            base_delay=self.config.get('retry', {}).get('base_delay', 1.0),
            max_delay=self.config.get('retry', {}).get('max_delay', 60.0)
        )
        
        original_function = context.get('function') if context else None
        if not original_function:
            return None
        
        for attempt in range(config.max_attempts):
            try:
                delay = min(
                    config.base_delay * (config.exponential_base ** attempt),
                    config.max_delay
                )
                
                if attempt > 0:
                    time.sleep(delay)
                
                # Retry the original function
                args = context.get('args', ())
                kwargs = context.get('kwargs', {})
                return original_function(*args, **kwargs)
                
            except Exception:
                if attempt == config.max_attempts - 1:
                    raise
                continue
        
        return None
    
    def _apply_fallback_strategy(self, error_info: ErrorInfo, component: str,
                                context: Optional[Dict[str, Any]]) -> Any:
        """Apply fallback strategy."""
        fallback_handler = self.fallback_handlers.get(component)
        if fallback_handler:
            return fallback_handler(error_info, context)
        return None
    
    def _apply_circuit_breaker_strategy(self, error_info: ErrorInfo, component: str,
                                       context: Optional[Dict[str, Any]]) -> Any:
        """Apply circuit breaker strategy."""
        circuit_breaker = self.circuit_breakers.get(component)
        if not circuit_breaker or not circuit_breaker.can_execute():
            return None
        
        try:
            # Execute original function
            original_function = context.get('function') if context else None
            if not original_function:
                return None
            
            args = context.get('args', ())
            kwargs = context.get('kwargs', {})
            result = original_function(*args, **kwargs)
            
            circuit_breaker.record_success()
            return result
            
        except Exception:
            circuit_breaker.record_failure()
            raise
    
    def _apply_graceful_degradation(self, error_info: ErrorInfo, component: str,
                                   context: Optional[Dict[str, Any]]) -> Any:
        """Apply graceful degradation strategy."""
        if component == 'cache_manager':
            # Return None to indicate cache miss, continue without cache
            return {'status': 'degraded', 'cache_disabled': True}
        elif component == 'api_endpoints':
            # Return simplified response
            return {'status': 'degraded', 'message': 'Service temporarily degraded'}
        
        return None
    
    def _apply_cache_fallback(self, error_info: ErrorInfo, component: str,
                             context: Optional[Dict[str, Any]]) -> Any:
        """Apply cache fallback strategy."""
        if component == 'llm_client':
            query = context.get('query') if context else None
            if query and query in self.cache_fallbacks:
                return self.cache_fallbacks[query]
        
        return None
    
    def _should_retry(self, error_info: ErrorInfo) -> bool:
        """Determine if error should be retried."""
        # Don't retry validation errors or security issues
        non_retryable_categories = {
            ErrorCategory.SECURITY,
            ErrorCategory.VALIDATION_WARNINGS,
            ErrorCategory.DATA_CORRUPTION
        }
        
        return error_info.category not in non_retryable_categories
    
    def _record_recovery_action(self, strategy: RecoveryStrategy, error_info: ErrorInfo,
                               component: str, success: bool, details: Dict[str, Any]):
        """Record a recovery action."""
        action = RecoveryAction(
            timestamp=datetime.now(),
            strategy=strategy,
            component=component,
            error_info=error_info,
            success=success,
            details=details
        )
        self.recovery_actions.append(action)
        
        # Keep only recent actions (last 1000)
        if len(self.recovery_actions) > 1000:
            self.recovery_actions = self.recovery_actions[-1000:]
    
    # Default fallback handlers
    def _llm_fallback(self, error_info: ErrorInfo, context: Optional[Dict[str, Any]]) -> Any:
        """Fallback for LLM client errors."""
        return {
            'response': 'I apologize, but I am experiencing technical difficulties. Please try again later.',
            'fallback': True,
            'error_category': error_info.category.value
        }
    
    def _vector_store_fallback(self, error_info: ErrorInfo, context: Optional[Dict[str, Any]]) -> Any:
        """Fallback for vector store errors."""
        return {
            'documents': [],
            'fallback': True,
            'message': 'Document retrieval temporarily unavailable'
        }
    
    def _cache_fallback(self, error_info: ErrorInfo, context: Optional[Dict[str, Any]]) -> Any:
        """Fallback for cache errors."""
        return None  # Indicate cache miss, proceed without cache
    
    def _document_processor_fallback(self, error_info: ErrorInfo, context: Optional[Dict[str, Any]]) -> Any:
        """Fallback for document processor errors."""
        return {
            'status': 'failed',
            'message': 'Document processing failed, please try a different format',
            'fallback': True
        }
    
    def register_fallback_handler(self, component: str, handler: Callable):
        """Register a custom fallback handler for a component."""
        self.fallback_handlers[component] = handler
    
    def add_cache_fallback(self, key: str, value: Any):
        """Add a fallback value to cache."""
        self.cache_fallbacks[key] = value
    
    def get_recovery_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get recovery statistics for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_actions = [
            action for action in self.recovery_actions 
            if action.timestamp >= cutoff_time
        ]
        
        total_attempts = len(recent_actions)
        successful_recoveries = len([a for a in recent_actions if a.success])
        
        strategy_stats = {}
        component_stats = {}
        
        for action in recent_actions:
            # Strategy statistics
            strategy_name = action.strategy.value
            if strategy_name not in strategy_stats:
                strategy_stats[strategy_name] = {'attempts': 0, 'successes': 0}
            strategy_stats[strategy_name]['attempts'] += 1
            if action.success:
                strategy_stats[strategy_name]['successes'] += 1
            
            # Component statistics
            if action.component not in component_stats:
                component_stats[action.component] = {'attempts': 0, 'successes': 0}
            component_stats[action.component]['attempts'] += 1
            if action.success:
                component_stats[action.component]['successes'] += 1
        
        # Calculate success rates
        for stats in strategy_stats.values():
            stats['success_rate'] = stats['successes'] / stats['attempts'] if stats['attempts'] > 0 else 0
        
        for stats in component_stats.values():
            stats['success_rate'] = stats['successes'] / stats['attempts'] if stats['attempts'] > 0 else 0
        
        return {
            'time_period_hours': hours,
            'total_recovery_attempts': total_attempts,
            'successful_recoveries': successful_recoveries,
            'overall_success_rate': successful_recoveries / total_attempts if total_attempts > 0 else 0,
            'strategy_breakdown': strategy_stats,
            'component_breakdown': component_stats,
            'circuit_breaker_status': {
                name: cb.get_status() for name, cb in self.circuit_breakers.items()
            }
        }
    
    def get_system_recovery_status(self) -> Dict[str, Any]:
        """Get current system recovery status."""
        return {
            'active_circuit_breakers': len([
                cb for cb in self.circuit_breakers.values() 
                if cb.state != CircuitState.CLOSED
            ]),
            'circuit_breakers': {
                name: cb.get_status() for name, cb in self.circuit_breakers.items()
            },
            'recent_recovery_attempts': len([
                action for action in self.recovery_actions
                if action.timestamp >= datetime.now() - timedelta(minutes=5)
            ]),
            'fallback_handlers_registered': len(self.fallback_handlers),
            'cache_fallbacks_available': len(self.cache_fallbacks)
        }


def with_error_recovery(component: str, recovery_manager: Optional[ErrorRecoveryManager] = None):
    """
    Decorator to add error recovery to functions.
    
    Args:
        component: Component name for recovery strategies
        recovery_manager: Recovery manager instance (optional)
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if recovery_manager:
                    from ..core.error_classifier import ErrorClassifier
                    
                    # Classify the error
                    classifier = ErrorClassifier()
                    error_info = classifier.classify_exception(e, component)
                    
                    # Attempt recovery
                    context = {
                        'function': func,
                        'args': args,
                        'kwargs': kwargs
                    }
                    
                    recovery_result = recovery_manager.recover_from_error(
                        error_info, component, context
                    )
                    
                    if recovery_result is not None:
                        return recovery_result
                
                # Re-raise if no recovery possible
                raise
        
        return wrapper
    return decorator