#!/usr/bin/env python3
"""
Error Analysis Test Runner

Script to test error analysis functionality and generate sample data.
Similar to benchmarks/scripts/test_benchmarks.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add the project root to the path to enable imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from error_analysis.core.error_analysis_manager import get_error_analysis_manager


def generate_test_errors(manager, count=10):
    """Generate test errors for demonstration."""
    test_errors = [
        (ConnectionError("Database connection timeout"), "vector_store"),
        (ValueError("Invalid query format"), "api_endpoints"),
        (TimeoutError("LLM response timeout"), "llm_client"),
        (RuntimeError("Cache server unavailable"), "cache_manager"),
        (FileNotFoundError("Document not found"), "document_processor"),
        (MemoryError("Insufficient memory for embeddings"), "vector_store"),
        (KeyError("Missing configuration key"), "api_endpoints"),
        (ImportError("Missing dependency"), "llm_client"),
        (PermissionError("Access denied to file"), "document_processor"),
        (OSError("Network unreachable"), "cache_manager")
    ]
    
    print("\n🧪 GENERATING TEST ERRORS")
    print("=" * 50)
    print(f"Generating {min(count, len(test_errors))} test errors for analysis...")
    print()
    
    for i in range(min(count, len(test_errors))):
        error, component = test_errors[i]
        context = {
            'test_mode': True,
            'error_index': i,
            'timestamp': time.time()
        }
        
        print(f"   [{i+1:2d}] {error.__class__.__name__}")
        print(f"        Message: {error}")
        print(f"        Component: {component}")
        
        manager.track_error(error, component, context)
        
        # Small delay to spread out timestamps
        time.sleep(0.1)
    
    print(f"\n✅ Successfully generated {min(count, len(test_errors))} test errors")


def check_monitoring_components(manager):
    """Test component monitoring functionality."""
    print("\n🔍 TESTING COMPONENT MONITORING")
    print("=" * 50)
    print("Testing decorator functionality for each monitor...")
    print()
    
    # Test each monitor
    for component_name, monitor in manager.monitors.items():
        print(f"📊 Testing {component_name.replace('_', ' ').title()} Monitor")
        print(f"    Component: {component_name}")
        
        # Test decorator functionality
        @monitor.monitor_function(f'test_{component_name}_function')
        def test_function():
            if component_name == 'api_endpoints':
                # Simulate API call
                time.sleep(0.01)
                return {'status': 'success', 'data': 'test'}
            elif component_name == 'llm_client':
                # Simulate LLM call
                time.sleep(0.05)
                return 'Test LLM response'
            elif component_name == 'cache_manager':
                # Simulate cache operation
                return 'cached_value'
            else:
                # Generic operation
                time.sleep(0.02)
                return 'success'
        
        # Call the decorated function
        result = test_function()
        
        # Get health status
        health = monitor.get_health_status()
        print(f"    Status: {health.get('status', 'unknown').upper()}")
        print()


def check_error_recovery(manager):
    """Test error recovery mechanisms."""
    print("\n🔄 TESTING ERROR RECOVERY")
    print("=" * 50)
    print("Testing recovery mechanisms and circuit breakers...")
    print()
    
    recovery_manager = manager.get_recovery_manager()
    
    # Test circuit breaker
    print("🔧 Circuit Breaker Test")
    print("    Testing circuit breaker functionality...")
    circuit_breaker = recovery_manager.circuit_breakers.get('llm_client')
    if circuit_breaker:
        initial_state = circuit_breaker.state
        print(f"    Initial State: {initial_state.value.upper()}")
        
        # Simulate some failures
        print("    Simulating failures...")
        for i in range(3):
            circuit_breaker.record_failure()
        
        print(f"    State After Failures: {circuit_breaker.state.value.upper()}")
    else:
        print("    No circuit breaker found for llm_client")
    
    print()
    
    # Test recovery with fallback
    print("🛠️  Recovery Fallback Test")
    print("    Testing recovery fallback mechanisms...")
    try:
        # Simulate an error that should trigger recovery
        error = ConnectionError("Test connection error for recovery")
        result = manager.track_error(error, 'llm_client', {'test_recovery': True})
        
        if result:
            print(f"    Recovery Status: SUCCESS")
            print(f"    Recovery Type: {type(result).__name__}")
            if isinstance(result, dict) and 'fallback' in result:
                print(f"    Fallback Triggered: {result.get('fallback', False)}")
        else:
            print("    Recovery Status: NO RESULT")
    except Exception as e:
        print(f"    Recovery Status: ERROR - {e}")
    
    print()


def check_comprehensive_analysis(manager):
    """Test comprehensive analysis generation."""
    print("\n📊 TESTING COMPREHENSIVE ANALYSIS")
    print("=" * 50)
    print("Generating comprehensive error analysis report...")
    print()
    
    # Generate analysis
    analysis = manager.get_comprehensive_analysis(hours=1)
    
    # Print key metrics
    error_analysis = analysis.get('error_analysis', {})
    print("📈 Error Analysis Metrics")
    print(f"    Total Errors Analyzed: {error_analysis.get('total_errors', 0)}")
    print(f"    Critical Errors: {error_analysis.get('critical_errors', 0)}")
    print(f"    Error Rate (per hour): {error_analysis.get('error_rate_per_hour', 0):.2f}")
    print()
    
    recovery_stats = analysis.get('recovery_statistics', {})
    print("🔄 Recovery Statistics")
    print(f"    Recovery Attempts: {recovery_stats.get('total_recovery_attempts', 0)}")
    print(f"    Successful Recoveries: {recovery_stats.get('successful_recoveries', 0)}")
    print(f"    Success Rate: {recovery_stats.get('overall_success_rate', 0):.2%}")
    print()
    
    recommendations = analysis.get('recommendations', [])
    print("💡 Recommendations Generated")
    print(f"    Total Recommendations: {len(recommendations)}")
    if recommendations:
        for i, rec in enumerate(recommendations[:3], 1):  # Show first 3
            print(f"    [{i}] {rec}")
    print()


def run_full_test(config_path=None):
    """Run comprehensive error analysis test."""
    print("\n" + "=" * 70)
    print("🚀 ERROR ANALYSIS SYSTEM - COMPREHENSIVE TEST")
    print("=" * 70)
    print("Testing all components of the error analysis system...")
    print()
    
    # Load configuration
    config = {}
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f).get('error_analysis', {})
        print(f"📋 Configuration loaded from: {config_path}")
    else:
        print("📋 Using default configuration")
    
    print()
    
    # Initialize manager
    print("⚙️  INITIALIZATION")
    print("-" * 30)
    print("Initializing error analysis manager...")
    manager = get_error_analysis_manager(config)
    print(f"✅ Manager initialized successfully")
    print(f"📊 Available monitors: {len(manager.monitors)}")
    print()
    
    # Test 1: Generate test errors
    generate_test_errors(manager, count=8)
    
    # Test 2: Test monitoring
    check_monitoring_components(manager)
    
    # Test 3: Test recovery
    check_error_recovery(manager)
    
    # Test 4: Test analysis
    check_comprehensive_analysis(manager)
    
    # Test 5: Get system health
    print("\n🏥 FINAL SYSTEM HEALTH CHECK")
    print("=" * 50)
    print("Checking overall system status after tests...")
    print()
    
    health = manager.get_system_health_status()
    print("📊 System Status Summary")
    print(f"    Overall Status: {health.get('overall_status', 'unknown').upper()}")
    print(f"    Components Monitored: {len(health.get('component_health', {}))}")
    print(f"    Critical Components: {len(health.get('critical_components', []))}")
    print(f"    Active Alerts: {health.get('active_alerts', 0)}")
    
    # Show component breakdown
    print("\n📋 Component Health Breakdown")
    for component, comp_health in health.get('component_health', {}).items():
        status = comp_health.get('status', 'unknown')
        print(f"    {component.replace('_', ' ').title()}: {status.upper()}")
    
    print("\n" + "=" * 70)
    print("✅ ERROR ANALYSIS SYSTEM TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)
    
    return {
        'system_health': health,
        'test_timestamp': time.time(),
        'test_status': 'completed'
    }


def run_quick_test():
    """Run a quick test of basic functionality."""
    print("\n" + "=" * 60)
    print("⚡ ERROR ANALYSIS SYSTEM - QUICK TEST")
    print("=" * 60)
    print("Running basic functionality test...")
    print()
    
    # Initialize with minimal config
    print("⚙️  Initializing system...")
    manager = get_error_analysis_manager()
    print("✅ System initialized")
    print()
    
    # Generate a few test errors
    print("🧪 Generating test errors...")
    test_errors = [
        (ConnectionError("Quick test connection error"), "vector_store"),
        (ValueError("Quick test validation error"), "api_endpoints"),
        (RuntimeError("Quick test runtime error"), "llm_client")
    ]
    
    for i, (error, component) in enumerate(test_errors, 1):
        print(f"    [{i}] {error.__class__.__name__} → {component}")
        manager.track_error(error, component, {'quick_test': True})
    
    print()
    
    # Get basic health status
    print("🏥 Checking system health...")
    health = manager.get_system_health_status()
    
    print("📊 Results Summary")
    print(f"    System Status: {health.get('overall_status', 'unknown').upper()}")
    print(f"    Components Monitored: {len(health.get('component_health', {}))}")
    print(f"    Errors Generated: {len(test_errors)}")
    
    print("\n" + "=" * 60)
    print("✅ QUICK TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description='Test error analysis functionality')
    
    parser.add_argument(
        '--config',
        type=str,
        default='error_analysis/config/error_analysis_config.json',
        help='Path to error analysis configuration file'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test only'
    )
    
    parser.add_argument(
        '--errors',
        type=int,
        default=8,
        help='Number of test errors to generate (default: 8)'
    )
    
    parser.add_argument(
        '--no-recovery',
        action='store_true',
        help='Skip recovery testing'
    )
    
    args = parser.parse_args()
    
    try:
        if args.quick:
            run_quick_test()
        else:
            results = run_full_test(config_path=args.config)
            
            # Print final summary
            print("\n📋 Test Summary:")
            health = results.get('system_health', {})
            print(f"   Final System Status: {health.get('overall_status', 'unknown').upper()}")
            
            if health.get('critical_components'):
                print(f"   Critical Components: {len(health['critical_components'])}")
            
            if health.get('recovery_status'):
                recovery_status = health['recovery_status']
                print(f"   Active Circuit Breakers: {recovery_status.get('active_circuit_breakers', 0)}")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()