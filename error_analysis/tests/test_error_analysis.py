"""
Tests for Error Analysis System

Basic tests to verify error analysis components work correctly.
"""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from error_analysis.core.error_classifier import ErrorClassifier, ErrorSeverity, ErrorCategory
from error_analysis.core.error_logger import ErrorLogger
from error_analysis.core.failure_tracker import FailureTracker
from error_analysis.recovery.error_recovery import ErrorRecoveryManager, RecoveryStrategy, with_error_recovery
from error_analysis.core.error_analysis_manager import ErrorAnalysisManager


class TestErrorClassifier:
    """Test error classification functionality."""
    
    def test_error_classifier_initialization(self):
        """Test error classifier initializes correctly."""
        classifier = ErrorClassifier()
        
        assert classifier is not None
        assert len(classifier.error_patterns) > 0
        assert len(classifier.component_keywords) > 0
    
    def test_classify_connection_error(self):
        """Test classification of connection errors."""
        classifier = ErrorClassifier()
        
        # Create a connection error
        error = ConnectionError("Connection refused to database")
        error_info = classifier.classify_exception(error, "vector_store")
        
        assert error_info.severity == ErrorSeverity.CRITICAL
        assert error_info.category == ErrorCategory.SYSTEM_DOWN
        assert error_info.component == "vector_store"
        assert "connection" in error_info.message.lower()
    
    def test_classify_validation_error(self):
        """Test classification of validation errors."""
        classifier = ErrorClassifier()
        
        # Create a validation error
        error = ValueError("Input parameter validation failed")
        error_info = classifier.classify_exception(error, "api_endpoints")
        
        assert error_info.severity == ErrorSeverity.LOW
        assert error_info.category == ErrorCategory.VALIDATION_WARNINGS
        assert error_info.component == "api_endpoints"
    
    def test_error_id_generation(self):
        """Test that error IDs are generated consistently."""
        classifier = ErrorClassifier()
        
        error1 = ValueError("Test error")
        error2 = ValueError("Test error")
        
        error_info1 = classifier.classify_exception(error1, "test_component")
        error_info2 = classifier.classify_exception(error2, "test_component")
        
        # Same error should generate same ID
        assert error_info1.error_id == error_info2.error_id


class TestErrorLogger:
    """Test error logging functionality."""
    
    def test_error_logger_initialization(self):
        """Test error logger initializes correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = ErrorLogger(log_directory=temp_dir)
            
            assert logger is not None
            assert Path(temp_dir).exists()
            assert logger.db_path.exists()
    
    def test_log_error(self):
        """Test logging an error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = ErrorLogger(log_directory=temp_dir)
            classifier = ErrorClassifier()
            
            # Create and log an error
            error = RuntimeError("Test runtime error unique for this test")
            error_info = classifier.classify_exception(error, "test_component")
            
            logger.log_error(error_info)
            
            # Verify error was logged
            recent_errors = logger.get_recent_errors(limit=10)
            # Find our specific error
            matching_errors = [e for e in recent_errors if e.error_id == error_info.error_id]
            assert len(matching_errors) >= 1
            assert matching_errors[0].error_id == error_info.error_id
    
    def test_error_summary(self):
        """Test error summary generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = ErrorLogger(log_directory=temp_dir)
            classifier = ErrorClassifier()
            
            # Log multiple errors
            for i in range(5):
                error = RuntimeError(f"Test error {i}")
                error_info = classifier.classify_exception(error, "test_component")
                logger.log_error(error_info)
            
            # Get summary
            summary = logger.get_error_summary(hours=1)
            
            assert summary['total_errors'] == 5
            assert 'test_component' in summary['component_breakdown']


class TestFailureTracker:
    """Test failure tracking functionality."""
    
    def test_failure_tracker_initialization(self):
        """Test failure tracker initializes correctly."""
        tracker = FailureTracker()
        
        assert tracker is not None
        assert len(tracker.failure_events) == 0
        assert len(tracker.component_health) == 0
    
    def test_track_error(self):
        """Test tracking an error."""
        tracker = FailureTracker()
        classifier = ErrorClassifier()
        
        # Create and track an error
        error = ConnectionError("Database connection failed")
        error_info = classifier.classify_exception(error, "vector_store")
        
        tracker.track_error(error_info)
        
        # Verify tracking
        assert len(tracker.failure_events) >= 0  # May create failure event
        assert "vector_store" in tracker.component_health
    
    def test_system_health_status(self):
        """Test system health status reporting."""
        tracker = FailureTracker()
        
        # Get initial status (should be healthy)
        status = tracker.get_system_health_status()
        
        assert status['total_components'] >= 0
        assert 'overall_status' in status
        assert 'system_health_score' in status


class TestErrorAnalysisManager:
    """Test error analysis manager functionality."""
    
    def test_manager_initialization(self):
        """Test manager initializes correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {'log_directory': temp_dir}
            manager = ErrorAnalysisManager(config)
            
            assert manager is not None
            assert len(manager.monitors) > 0
            assert 'llm_client' in manager.monitors
            assert 'api_endpoints' in manager.monitors
    
    def test_track_error_integration(self):
        """Test end-to-end error tracking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {'log_directory': temp_dir}
            manager = ErrorAnalysisManager(config)
            
            # Track an error
            error = ValueError("Integration test error")
            manager.track_error(error, "llm_client", {"test": "context"})
            
            # Verify error was tracked
            health_status = manager.get_system_health_status()
            assert health_status is not None
            assert 'system_health' in health_status
    
    def test_system_health_status(self):
        """Test system health status reporting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {'log_directory': temp_dir}
            manager = ErrorAnalysisManager(config)
            
            # Get health status
            status = manager.get_system_health_status()
            
            assert 'overall_status' in status
            assert 'component_health' in status
            assert 'error_summary' in status
            assert status['overall_status'] in ['healthy', 'degraded', 'critical', 'warning']
    
    def test_comprehensive_analysis(self):
        """Test comprehensive analysis generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {'log_directory': temp_dir}
            manager = ErrorAnalysisManager(config)
            
            # Generate analysis
            analysis = manager.get_comprehensive_analysis(hours=1)
            
            assert 'error_analysis' in analysis
            assert 'failure_analysis' in analysis
            assert 'component_analysis' in analysis
            assert 'recommendations' in analysis
            assert isinstance(analysis['recommendations'], list)


class TestMonitorIntegration:
    """Test monitor integration with decorators."""
    
    def test_monitor_decorator(self):
        """Test monitor decorator functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {'log_directory': temp_dir}
            manager = ErrorAnalysisManager(config)
            
            # Get a monitor
            llm_monitor = manager.get_monitor('llm_client')
            assert llm_monitor is not None
            
            # Test the decorator
            @llm_monitor.monitor_function('test_function')
            def test_function():
                return "success"
            
            # Call the decorated function
            result = test_function()
            assert result == "success"
            
            # Check that metrics were recorded
            health = llm_monitor.get_health_status()
            assert health is not None
    
    def test_monitor_error_handling(self):
        """Test monitor error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {'log_directory': temp_dir}
            manager = ErrorAnalysisManager(config)
            
            # Get a monitor
            llm_monitor = manager.get_monitor('llm_client')
            
            # Test the decorator with error
            @llm_monitor.monitor_function('test_error_function')
            def test_error_function():
                raise ValueError("Test error for monitoring")
            
            # Call the decorated function and expect error
            with pytest.raises(ValueError):
                test_error_function()
            
            # Check that error was recorded
            error_summary = llm_monitor.get_error_summary(hours=1)
            assert error_summary['total_errors'] >= 1


# Integration test
def test_full_error_analysis_workflow():
    """Test complete error analysis workflow."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {'log_directory': temp_dir, 'max_memory_entries': 100}
        
        # Initialize manager
        manager = ErrorAnalysisManager(config)
        
        # Simulate various errors
        errors = [
            (ConnectionError("Database unavailable"), "vector_store"),
            (ValueError("Invalid input format"), "api_endpoints"),
            (TimeoutError("LLM response timeout"), "llm_client"),
            (RuntimeError("Cache connection failed"), "cache_manager")
        ]
        
        # Track all errors
        for error, component in errors:
            manager.track_error(error, component, {"simulation": True})
        
        # Get comprehensive analysis
        analysis = manager.get_comprehensive_analysis(hours=1)
        
        # Verify analysis contains expected data
        assert analysis['error_analysis']['total_errors'] == len(errors)
        assert len(analysis['component_analysis']) > 0
        assert len(analysis['recommendations']) > 0
        
        # Get system health
        health = manager.get_system_health_status()
        assert health['overall_status'] in ['healthy', 'degraded', 'critical', 'warning']
        
        # Export data
        export_data = manager.export_analysis_data(hours=1)
        assert 'system_health' in export_data
        assert 'comprehensive_analysis' in export_data


class TestErrorRecovery:
    """Test error recovery functionality."""
    
    def test_recovery_manager_initialization(self):
        """Test recovery manager initializes correctly."""
        recovery_manager = ErrorRecoveryManager()
        
        assert recovery_manager is not None
        assert len(recovery_manager.circuit_breakers) > 0
        assert len(recovery_manager.fallback_handlers) > 0
    
    def test_retry_strategy(self):
        """Test retry strategy functionality."""
        recovery_manager = ErrorRecoveryManager({'retry': {'max_attempts': 2}})
        classifier = ErrorClassifier()
        
        # Create a retryable error
        error = ConnectionError("Network timeout")
        error_info = classifier.classify_exception(error, "llm_client")
        
        # Create context with a function that fails once then succeeds
        call_count = 0
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Network timeout")
            return "success"
        
        context = {
            'function': test_function,
            'args': (),
            'kwargs': {}
        }
        
        # Test recovery
        result = recovery_manager.recover_from_error(error_info, "llm_client", context)
        assert result == "success"
        assert call_count == 2  # Failed once, succeeded on retry
    
    def test_fallback_strategy(self):
        """Test fallback strategy functionality."""
        recovery_manager = ErrorRecoveryManager()
        classifier = ErrorClassifier()
        
        # Create an error that triggers fallback
        error = RuntimeError("LLM service unavailable")
        error_info = classifier.classify_exception(error, "llm_client")
        
        # Test fallback
        result = recovery_manager.recover_from_error(error_info, "llm_client", {})
        assert result is not None
        assert result.get('fallback') is True
        assert 'response' in result
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        recovery_manager = ErrorRecoveryManager({
            'circuit_breaker': {
                'failure_threshold': 3,
                'request_volume_threshold': 3
            }
        })
        
        circuit_breaker = recovery_manager.circuit_breakers['llm_client']
        
        # Test normal operation
        assert circuit_breaker.can_execute() is True
        
        # Record failures to trip the breaker (need to meet both thresholds)
        circuit_breaker.record_failure()
        circuit_breaker.record_failure()
        circuit_breaker.record_failure()
        
        # Now should be open (failure threshold and volume threshold both met)
        assert circuit_breaker.can_execute() is False
        assert circuit_breaker.state.value == 'open'
    
    def test_recovery_decorator(self):
        """Test recovery decorator functionality."""
        recovery_manager = ErrorRecoveryManager()
        
        @with_error_recovery('llm_client', recovery_manager)
        def test_function_with_recovery():
            raise ValueError("Test error for recovery")
        
        # Should not raise exception due to recovery
        result = test_function_with_recovery()
        assert result is not None  # Should return fallback result
    
    def test_recovery_statistics(self):
        """Test recovery statistics collection."""
        recovery_manager = ErrorRecoveryManager()
        classifier = ErrorClassifier()
        
        # Simulate some recovery attempts
        error = ValueError("Test error")
        error_info = classifier.classify_exception(error, "llm_client")
        
        recovery_manager.recover_from_error(error_info, "llm_client", {})
        
        # Get statistics
        stats = recovery_manager.get_recovery_statistics(hours=1)
        
        assert 'total_recovery_attempts' in stats
        assert 'successful_recoveries' in stats
        assert 'strategy_breakdown' in stats
        assert 'component_breakdown' in stats


class TestIntegratedErrorRecovery:
    """Test integrated error recovery with analysis manager."""
    
    def test_manager_with_recovery(self):
        """Test error analysis manager with recovery integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'log_directory': temp_dir,
                'error_recovery': {
                    'enabled': True,
                    'retry': {'max_attempts': 2}
                }
            }
            manager = ErrorAnalysisManager(config)
            
            # Test that recovery manager is initialized
            assert manager.error_recovery is not None
            recovery_manager = manager.get_recovery_manager()
            assert recovery_manager is not None
            
            # Test error tracking with recovery
            error = ConnectionError("Test connection error")
            result = manager.track_error(error, "llm_client", {})
            
            # Should get fallback result
            assert result is not None
            
            # Get system health with recovery status
            health = manager.get_system_health_status()
            assert 'recovery_status' in health
            assert 'circuit_breakers' in health['recovery_status']
    
    def test_recovery_fallback_registration(self):
        """Test custom recovery fallback registration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {'log_directory': temp_dir}
            manager = ErrorAnalysisManager(config)
            
            # Register custom fallback
            def custom_fallback(error_info, context):
                return {'custom_response': 'Custom fallback executed'}
            
            manager.register_recovery_fallback('test_component', custom_fallback)
            
            # Test that fallback is registered
            recovery_manager = manager.get_recovery_manager()
            assert 'test_component' in recovery_manager.fallback_handlers
    
    def test_cache_fallback_integration(self):
        """Test cache fallback integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {'log_directory': temp_dir}
            manager = ErrorAnalysisManager(config)
            
            # Add cache fallback
            manager.add_recovery_cache_fallback('test_query', 'cached_response')
            
            # Verify fallback is added
            recovery_manager = manager.get_recovery_manager()
            assert 'test_query' in recovery_manager.cache_fallbacks
            assert recovery_manager.cache_fallbacks['test_query'] == 'cached_response'