#!/usr/bin/env python3
"""
Error Analysis Results Viewer

Script to view and analyze error analysis results.
Similar to benchmarks/scripts/view_results.py
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the path to enable imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_result_file(filepath):
    """Load and parse a result file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None


def display_summary(results):
    """Display a summary of error analysis results."""
    print("\n" + "=" * 70)
    print("📊 ERROR ANALYSIS RESULTS - SUMMARY")
    print("=" * 70)
    
    timestamp = results.get('timestamp', 'Unknown')
    hours = results.get('analysis_period_hours', 'Unknown')
    
    print("📅 Analysis Information")
    print(f"    Timestamp: {timestamp}")
    print(f"    Analysis Period: {hours} hour{'s' if hours != 1 else ''}")
    print()
    
    summary = results.get('summary', {})
    print("📊 Error Statistics")
    print(f"    Total Errors: {summary.get('total_errors', 0)}")
    print(f"    Critical Errors: {summary.get('critical_errors', 0)}")
    print()
    print("🏥 System Health")
    print(f"    Overall Status: {summary.get('overall_status', 'Unknown').upper()}")
    print(f"    Components Monitored: {summary.get('components_monitored', 0)}")
    print()
    print("🔄 Recovery Performance")
    print(f"    Success Rate: {summary.get('recovery_success_rate', 0):.2%}")
    print()
    print("💡 Recommendations")
    print(f"    Total Generated: {summary.get('recommendations_count', 0)}")


def display_health_status(results):
    """Display system health status."""
    health = results.get('system_health', {})
    if not health:
        print("\n⚠️  No health status data available")
        return
    
    print("\n🏥 SYSTEM HEALTH STATUS")
    print("=" * 50)
    
    print("📊 Overall System Status")
    print(f"    Status: {health.get('overall_status', 'Unknown').upper()}")
    print(f"    Critical Components: {len(health.get('critical_components', []))}")
    print(f"    Active Alerts: {health.get('active_alerts', 0)}")
    
    # Component health breakdown
    component_health = health.get('component_health', {})
    if component_health:
        print("\n📋 Component Health Breakdown")
        for component, comp_health in component_health.items():
            status = comp_health.get('status', 'unknown')
            component_name = component.replace('_', ' ').title()
            print(f"    {component_name}: {status.upper()}")
    else:
        print("\n⚠️  No component health data available")


def display_error_analysis(results):
    """Display error analysis details."""
    analysis = results.get('comprehensive_analysis', {})
    error_analysis = analysis.get('error_analysis', {})
    
    if not error_analysis:
        print("\n⚠️  No error analysis data available")
        return
    
    print("\n🚨 ERROR ANALYSIS DETAILS")
    print("=" * 50)
    
    print("📊 Error Statistics")
    print(f"    Total Errors: {error_analysis.get('total_errors', 0)}")
    print(f"    Critical Errors: {error_analysis.get('critical_errors', 0)}")
    print(f"    Error Rate: {error_analysis.get('error_rate_per_hour', 0):.2f}/hour")
    
    # Component breakdown
    component_breakdown = error_analysis.get('component_breakdown', {})
    if component_breakdown:
        print("\n📋 Errors by Component")
        for component, count in component_breakdown.items():
            component_name = component.replace('_', ' ').title()
            print(f"    {component_name}: {count}")
    
    # Severity breakdown
    severity_breakdown = error_analysis.get('severity_breakdown', {})
    if severity_breakdown:
        print("\n⚠️  Errors by Severity Level")
        for severity, count in severity_breakdown.items():
            print(f"    {severity.title()}: {count}")


def display_recovery_stats(results):
    """Display recovery statistics."""
    recovery_stats = results.get('recovery_statistics', {})
    
    if not recovery_stats:
        print("\n⚠️  No recovery statistics available")
        return
    
    print("\n🔄 RECOVERY STATISTICS")
    print("=" * 50)
    
    print("📊 Recovery Performance")
    print(f"    Total Attempts: {recovery_stats.get('total_recovery_attempts', 0)}")
    print(f"    Successful Recoveries: {recovery_stats.get('successful_recoveries', 0)}")
    print(f"    Overall Success Rate: {recovery_stats.get('overall_success_rate', 0):.2%}")
    
    # Strategy breakdown
    strategy_breakdown = recovery_stats.get('strategy_breakdown', {})
    if strategy_breakdown:
        print("\n🛠️  Recovery by Strategy")
        for strategy, stats in strategy_breakdown.items():
            success_rate = stats.get('success_rate', 0)
            attempts = stats.get('attempts', 0)
            strategy_name = strategy.replace('_', ' ').title()
            print(f"    {strategy_name}: {success_rate:.2%} ({attempts} attempts)")
    
    # Component breakdown
    component_breakdown = recovery_stats.get('component_breakdown', {})
    if component_breakdown:
        print("\n📋 Recovery by Component")
        for component, stats in component_breakdown.items():
            success_rate = stats.get('success_rate', 0)
            attempts = stats.get('attempts', 0)
            component_name = component.replace('_', ' ').title()
            print(f"    {component_name}: {success_rate:.2%} ({attempts} attempts)")


def display_recommendations(results):
    """Display recommendations."""
    analysis = results.get('comprehensive_analysis', {})
    recommendations = analysis.get('recommendations', [])
    
    if not recommendations:
        print("\n✅ No recommendations needed - system operating optimally")
        return
    
    print("\n💡 SYSTEM RECOMMENDATIONS")
    print("=" * 50)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"    [{i}] {rec}")
    print()


def compare_results(filepaths):
    """Compare multiple result files."""
    results = []
    for filepath in filepaths:
        result = load_result_file(filepath)
        if result:
            results.append((filepath, result))
    
    if len(results) < 2:
        print("Need at least 2 valid result files to compare")
        return
    
    print("📈 Results Comparison")
    print("=" * 50)
    
    # Compare summaries
    print("File\t\t\tErrors\tCritical\tStatus\t\tRecovery Rate")
    print("-" * 80)
    
    for filepath, result in results:
        filename = Path(filepath).name
        summary = result.get('summary', {})
        
        total_errors = summary.get('total_errors', 0)
        critical_errors = summary.get('critical_errors', 0)
        status = summary.get('overall_status', 'unknown')[:8]
        recovery_rate = summary.get('recovery_success_rate', 0)
        
        print(f"{filename[:20]:<20}\t{total_errors}\t{critical_errors}\t\t{status}\t\t{recovery_rate:.1%}")


def list_result_files(directory):
    """List available result files in directory."""
    results_dir = Path(directory)
    if not results_dir.exists():
        print(f"Results directory {directory} does not exist")
        return
    
    result_files = list(results_dir.glob("error_analysis_*.json"))
    
    if not result_files:
        print(f"No error analysis result files found in {directory}")
        return
    
    print(f"📁 Available Result Files in {directory}")
    print("=" * 50)
    
    # Sort by modification time (newest first)
    result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    for i, filepath in enumerate(result_files, 1):
        # Get file timestamp
        timestamp = datetime.fromtimestamp(filepath.stat().st_mtime)
        
        # Try to load summary info
        result = load_result_file(filepath)
        if result and 'summary' in result:
            summary = result['summary']
            total_errors = summary.get('total_errors', 0)
            status = summary.get('overall_status', 'unknown')
            info = f"({total_errors} errors, {status})"
        else:
            info = "(invalid file)"
        
        print(f"{i:2d}. {filepath.name} - {timestamp.strftime('%Y-%m-%d %H:%M:%S')} {info}")


def main():
    """Main entry point for results viewer."""
    parser = argparse.ArgumentParser(description='View error analysis results')
    
    parser.add_argument(
        'files',
        nargs='*',
        help='Result file(s) to view'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available result files'
    )
    
    parser.add_argument(
        '--directory',
        type=str,
        default='error_analysis/results',
        help='Results directory (default: error_analysis/results)'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple result files'
    )
    
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Show only summary information'
    )
    
    parser.add_argument(
        '--latest',
        action='store_true',
        help='View the latest result file'
    )
    
    args = parser.parse_args()
    
    # List files mode
    if args.list:
        list_result_files(args.directory)
        return
    
    # Find latest file if requested
    if args.latest:
        results_dir = Path(args.directory)
        if results_dir.exists():
            result_files = list(results_dir.glob("error_analysis_*.json"))
            if result_files:
                # Get the most recent file
                latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
                args.files = [str(latest_file)]
            else:
                print("No result files found")
                return
        else:
            print(f"Results directory {args.directory} does not exist")
            return
    
    # No files specified
    if not args.files:
        print("No files specified. Use --list to see available files or --latest for the most recent.")
        return
    
    # Compare mode
    if args.compare and len(args.files) > 1:
        compare_results(args.files)
        return
    
    # View single file
    if len(args.files) == 1:
        result = load_result_file(args.files[0])
        if not result:
            return
        
        print(f"📄 Viewing: {args.files[0]}")
        print()
        
        display_summary(result)
        
        if not args.summary_only:
            display_health_status(result)
            display_error_analysis(result)
            display_recovery_stats(result)
            display_recommendations(result)
    
    # Multiple files without compare
    elif len(args.files) > 1:
        print("Multiple files specified. Use --compare to compare them.")
        for filepath in args.files:
            result = load_result_file(filepath)
            if result:
                print(f"\n📄 {filepath}:")
                display_summary(result)


if __name__ == '__main__':
    main()