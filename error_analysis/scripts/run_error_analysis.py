#!/usr/bin/env python3
"""
Error Analysis Runner

Script to run comprehensive error analysis on the RAG system.
Similar to benchmarks/scripts/run_benchmarks.py
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the project root to the path to enable imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from error_analysis.core.error_analysis_manager import get_error_analysis_manager


def run_error_analysis(config_path=None, hours=24, output_dir=None):
    """
    Run comprehensive error analysis.
    
    Args:
        config_path: Path to error analysis config file
        hours: Number of hours to analyze
        output_dir: Directory to save results
    """
    print("\n" + "=" * 70)
    print("🔍 ERROR ANALYSIS RUNNER")
    print("=" * 70)
    print(f"Analyzing system errors from the last {hours} hour{'s' if hours != 1 else ''}")
    print()
    
    start_time = time.time()
    
    # Load configuration
    print("📋 CONFIGURATION")
    print("-" * 30)
    config = {}
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f).get('error_analysis', {})
        print(f"✅ Configuration loaded from: {config_path}")
    else:
        print("✅ Using default configuration")
    print()
    
    # Initialize error analysis manager
    print("⚙️  INITIALIZATION")
    print("-" * 30)
    print("Initializing error analysis manager...")
    manager = get_error_analysis_manager(config)
    print("✅ Manager initialized successfully")
    print()
    
    print("📊 ANALYSIS GENERATION")
    print("-" * 30)
    print("Generating comprehensive analysis report...")
    
    # Get system health status
    health_status = manager.get_system_health_status()
    print(f"✅ System health checked")
    print(f"📊 Current Status: {health_status['overall_status'].upper()}")
    
    # Get comprehensive analysis
    analysis = manager.get_comprehensive_analysis(hours=hours)
    print(f"✅ Error analysis completed")
    
    # Get recovery statistics
    recovery_manager = manager.get_recovery_manager()
    recovery_stats = recovery_manager.get_recovery_statistics(hours=hours)
    print(f"✅ Recovery statistics generated")
    print()
    
    # Compile results
    results = {
        'timestamp': datetime.now().isoformat(),
        'analysis_period_hours': hours,
        'system_health': health_status,
        'comprehensive_analysis': analysis,
        'recovery_statistics': recovery_stats,
        'summary': {
            'total_errors': analysis['error_analysis'].get('total_errors', 0),
            'critical_errors': analysis['error_analysis'].get('critical_errors', 0),
            'overall_status': health_status['overall_status'],
            'recovery_success_rate': recovery_stats.get('overall_success_rate', 0),
            'components_monitored': len(health_status['component_health']),
            'recommendations_count': len(analysis.get('recommendations', []))
        }
    }
    
    # Print summary
    print("📈 ANALYSIS SUMMARY")
    print("=" * 50)
    print("Key metrics from the analysis:")
    print()
    print("📊 Error Statistics")
    print(f"    Total Errors: {results['summary']['total_errors']}")
    print(f"    Critical Errors: {results['summary']['critical_errors']}")
    print(f"    Error Rate: {analysis['error_analysis'].get('error_rate_per_hour', 0):.2f}/hour")
    print()
    print("🏥 System Health")
    print(f"    Overall Status: {results['summary']['overall_status'].upper()}")
    print(f"    Components Monitored: {results['summary']['components_monitored']}")
    print(f"    Critical Components: {len(health_status.get('critical_components', []))}")
    print()
    print("🔄 Recovery Performance") 
    print(f"    Recovery Success Rate: {results['summary']['recovery_success_rate']:.2%}")
    print(f"    Total Recovery Attempts: {recovery_stats.get('total_recovery_attempts', 0)}")
    print()
    print("💡 Recommendations")
    print(f"    Total Generated: {results['summary']['recommendations_count']}")
    
    # Print recommendations
    if analysis.get('recommendations'):
        print("\n📋 Detailed Recommendations")
        print("-" * 40)
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"    [{i}] {rec}")
    print()
    
    # Save results if output directory specified
    if output_dir:
        print("💾 SAVING RESULTS")
        print("-" * 30)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"error_analysis_{timestamp}.json"
        filepath = output_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Results saved to: {filepath}")
        print(f"📁 Output directory: {output_path}")
        print()
    
    execution_time = time.time() - start_time
    print("=" * 70)
    print(f"✅ ERROR ANALYSIS COMPLETED IN {execution_time:.2f} SECONDS")
    print("=" * 70)
    
    return results


def main():
    """Main entry point for error analysis runner."""
    parser = argparse.ArgumentParser(description='Run comprehensive error analysis')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='error_analysis/config/error_analysis_config.json',
        help='Path to error analysis configuration file'
    )
    
    parser.add_argument(
        '--hours',
        type=int,
        default=24,
        help='Number of hours to analyze (default: 24)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='error_analysis/results',
        help='Output directory for results (default: error_analysis/results)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick analysis (last 1 hour only)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to file'
    )
    
    args = parser.parse_args()
    
    # Adjust hours for quick mode
    if args.quick:
        hours = 1
    else:
        hours = args.hours
    
    # Set output directory
    output_dir = None if args.no_save else args.output
    
    try:
        results = run_error_analysis(
            config_path=args.config,
            hours=hours,
            output_dir=output_dir
        )
        
        # Exit with appropriate code based on system status
        status = results['summary']['overall_status']
        if status == 'critical':
            sys.exit(2)
        elif status in ['degraded', 'warning']:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Error analysis failed: {e}")
        sys.exit(3)


if __name__ == '__main__':
    main()