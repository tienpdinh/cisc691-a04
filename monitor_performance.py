#!/usr/bin/env python3
"""
Performance monitoring script for RAG API
"""

import asyncio
import json
import time
import argparse
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from performance_optimizer import PerformanceOptimizer, OptimizationConfig


async def monitor_performance(duration: int = 300, interval: int = 30, output_file: str = None):
    """
    Monitor performance for a specified duration.
    
    Args:
        duration: Monitoring duration in seconds
        interval: Check interval in seconds
        output_file: Optional output file for results
    """
    print(f"Starting performance monitoring for {duration} seconds...")
    print(f"Checking every {interval} seconds")
    
    # Initialize optimizer
    config = OptimizationConfig(monitor_interval=interval)
    optimizer = PerformanceOptimizer(config)
    
    results = []
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            # Get current metrics
            metrics = await optimizer.get_comprehensive_metrics()
            
            # Display current status
            print(f"\n--- Performance Check at {metrics.timestamp} ---")
            print(f"Performance Score: {metrics.performance_score}/100")
            print(f"Memory Usage: {metrics.memory_usage_mb:.1f} MB")
            print(f"CPU Usage: {metrics.cpu_usage_percent:.1f}%")
            print(f"Cache Hit Rate: {metrics.cache_hit_rate:.1f}%")
            print(f"Active Connections: {metrics.active_connections}")
            print(f"Avg Response Time: {metrics.avg_response_time:.3f}s")
            
            if metrics.optimization_suggestions:
                print("Suggestions:")
                for suggestion in metrics.optimization_suggestions:
                    print(f"  - {suggestion}")
            
            # Auto-optimize if performance is poor
            if metrics.performance_score < 60:
                print("Performance score is low - triggering optimization...")
                optimization_results = await optimizer.optimize_all()
                print(f"Optimization completed: freed {optimization_results.get('memory', {}).get('freed_mb', 0):.1f} MB")
            
            # Store results
            results.append({
                "timestamp": metrics.timestamp,
                "performance_score": metrics.performance_score,
                "memory_usage_mb": metrics.memory_usage_mb,
                "cpu_usage_percent": metrics.cpu_usage_percent,
                "cache_hit_rate": metrics.cache_hit_rate,
                "active_connections": metrics.active_connections,
                "avg_response_time": metrics.avg_response_time,
                "optimization_suggestions": metrics.optimization_suggestions
            })
            
            # Wait for next check
            await asyncio.sleep(interval)
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    
    # Generate summary
    if results:
        print("\n=== Performance Summary ===")
        avg_score = sum(r["performance_score"] for r in results) / len(results)
        avg_memory = sum(r["memory_usage_mb"] for r in results) / len(results)
        avg_cpu = sum(r["cpu_usage_percent"] for r in results) / len(results)
        avg_cache_hit = sum(r["cache_hit_rate"] for r in results) / len(results)
        
        print(f"Average Performance Score: {avg_score:.1f}/100")
        print(f"Average Memory Usage: {avg_memory:.1f} MB")
        print(f"Average CPU Usage: {avg_cpu:.1f}%")
        print(f"Average Cache Hit Rate: {avg_cache_hit:.1f}%")
        
        # Save results if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump({
                    "summary": {
                        "avg_performance_score": avg_score,
                        "avg_memory_usage_mb": avg_memory,
                        "avg_cpu_usage_percent": avg_cpu,
                        "avg_cache_hit_rate": avg_cache_hit,
                        "total_checks": len(results)
                    },
                    "detailed_results": results
                }, f, indent=2)
            
            print(f"Results saved to {output_path}")


async def performance_benchmark(queries: list = None):
    """
    Run a performance benchmark with sample queries.
    
    Args:
        queries: List of test queries
    """
    if not queries:
        queries = [
            "What were the Q4 2023 sales results?",
            "Compare Q1 2024 to Q1 2023 performance",
            "What are the key retail trends in 2024?",
            "Show me the quarterly growth rates",
            "What was the best performing quarter?"
        ]
    
    print("Running performance benchmark...")
    
    optimizer = PerformanceOptimizer()
    
    # Simulate some load and measure performance
    start_time = time.time()
    
    # Get baseline metrics
    baseline_metrics = await optimizer.get_comprehensive_metrics()
    print(f"Baseline Performance Score: {baseline_metrics.performance_score}/100")
    
    # Simulate query processing (in real scenario, this would be actual API calls)
    for i, query in enumerate(queries):
        print(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Track response time
        response_time = 0.1 + (i * 0.02)  # Simulate increasing response time
        optimizer.response_optimizer.track_response_time(response_time)
        
        # Simulate caching some responses
        if i % 2 == 0:
            cache_key = f"query_{hash(query)}"
            optimizer.response_optimizer.cache_response(cache_key, f"response_for_{query}")
    
    # Get final metrics
    final_metrics = await optimizer.get_comprehensive_metrics()
    
    print(f"\n=== Benchmark Results ===")
    print(f"Total Time: {time.time() - start_time:.2f} seconds")
    print(f"Final Performance Score: {final_metrics.performance_score}/100")
    print(f"Cache Hit Rate: {final_metrics.cache_hit_rate:.1f}%")
    print(f"Average Response Time: {final_metrics.avg_response_time:.3f}s")
    
    if final_metrics.optimization_suggestions:
        print("\nOptimization Suggestions:")
        for suggestion in final_metrics.optimization_suggestions:
            print(f"  - {suggestion}")


def main():
    parser = argparse.ArgumentParser(description="Performance monitoring for RAG API")
    parser.add_argument("--monitor", "-m", action="store_true", help="Start monitoring mode")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run performance benchmark")
    parser.add_argument("--duration", "-d", type=int, default=300, help="Monitoring duration in seconds")
    parser.add_argument("--interval", "-i", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--output", "-o", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    if args.benchmark:
        asyncio.run(performance_benchmark())
    elif args.monitor:
        asyncio.run(monitor_performance(args.duration, args.interval, args.output))
    else:
        print("Please specify --monitor or --benchmark")
        parser.print_help()


if __name__ == "__main__":
    main()