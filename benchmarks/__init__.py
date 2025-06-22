"""
Performance Benchmarking System for RAG API

This module provides comprehensive benchmarking tools for evaluating:
- Response latency and throughput
- Retrieval accuracy and precision
- Response quality and relevance
- Baseline LLM comparisons

Usage:
    from benchmarks import BenchmarkRunner
    
    runner = BenchmarkRunner("http://localhost:8001")
    report = await runner.run_full_benchmark_suite()

Or run from command line:
    python benchmarks/scripts/run_benchmarks.py --url http://localhost:8001
"""

try:
    from .core.latency_benchmark import LatencyBenchmark
    from .core.accuracy_benchmark import AccuracyBenchmark
    from .core.quality_benchmark import QualityBenchmark
    from .core.baseline_comparison import BaselineComparison
    from .core.benchmark_runner import BenchmarkRunner
    
    __all__ = [
        'LatencyBenchmark',
        'AccuracyBenchmark', 
        'QualityBenchmark',
        'BaselineComparison',
        'BenchmarkRunner'
    ]
except ImportError as e:
    import logging
    logging.warning(f"Some benchmark modules could not be imported: {e}")
    logging.warning("Install benchmark dependencies with: pip install -r benchmarks/requirements_benchmarks.txt")
    
    __all__ = []