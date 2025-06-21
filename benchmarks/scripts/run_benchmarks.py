#!/usr/bin/env python3
"""
Simple script to run RAG system benchmarks.

Usage:
    python run_benchmarks.py
    python run_benchmarks.py --url http://localhost:8001
    python run_benchmarks.py --quick  # Run faster, less comprehensive benchmarks
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from benchmarks.core import BenchmarkRunner
except ImportError as e:
    print(f"Error importing benchmarks: {e}")
    print("Please install benchmark dependencies:")
    print("pip install -r benchmarks/requirements_benchmarks.txt")
    sys.exit(1)

async def main():
    parser = argparse.ArgumentParser(description="Run RAG System Benchmarks")
    parser.add_argument("--url", default="http://localhost:8001", help="RAG API URL")
    parser.add_argument("--timeout", type=int, default=15, help="Request timeout (seconds)")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks (faster, less comprehensive)")
    parser.add_argument("--output-dir", type=Path, default="benchmarks/results", help="Output directory for results")
    
    args = parser.parse_args()
    
    print("🚀 Starting RAG System Benchmarks...")
    print(f"📡 API URL: {args.url}")
    print(f"⏱️  Timeout: {args.timeout}s")
    print(f"📁 Output Directory: {args.output_dir}")
    
    if args.quick:
        print("⚡ Quick mode enabled (reduced test scope)")
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(args.url, args.timeout)
    runner.setup_logging("INFO")
    
    # Create quick configuration if requested
    if args.quick:
        quick_config = {
            "latency": {
                "concurrent_users": [1, 3],
                "requests_per_user": 3,
                "test_queries": [
                    "What were the retail e-commerce sales trends in Q1 2024?",
                    "How did Q4 2023 retail sales compare to Q4 2024?",
                    "What are the key metrics from the Q2 2024 e-commerce report?"
                ]
            },
            "accuracy": {
                "top_k_values": [1, 3, 5],
                "ground_truth_file": "benchmarks/data/ground_truth.json"
            },
            "quality": {
                "test_cases_file": "benchmarks/data/quality_test_cases.json",
                "evaluation_metrics": ["rouge", "semantic_similarity", "coherence"]
            },
            "comparison": {
                "test_queries_file": "benchmarks/data/comparison_queries.json"
            }
        }
        runner.benchmark_config = quick_config
    
    try:
        print("\\n📊 Running benchmark suite...")
        report = await runner.run_full_benchmark_suite()
        
        print("\\n✅ Benchmarks completed successfully!")
        print(f"📋 Results saved in: {args.output_dir}/")
        
        # Print summary
        print("\\n📈 BENCHMARK SUMMARY:")
        print("=" * 50)
        
        if "summary" in report:
            summary = report["summary"]
            
            # Latency results
            if "latency" in summary:
                lat = summary["latency"]
                print(f"🏃 LATENCY:")
                print(f"  • RAG Query: {lat.get('query_with_rag_avg_latency', 0):.3f}s")
                print(f"  • Direct LLM: {lat.get('query_without_rag_avg_latency', 0):.3f}s")
                print(f"  • Retrieval: {lat.get('retrieve_avg_latency', 0):.3f}s")
                print(f"  • Throughput: {lat.get('rag_throughput_rps', 0):.2f} req/s")
            
            # Accuracy results
            if "accuracy" in summary:
                acc = summary["accuracy"]
                print(f"\\n🎯 ACCURACY:")
                print(f"  • Precision@5: {acc.get('average_precision_at_5', 0):.3f}")
                print(f"  • Recall@5: {acc.get('average_recall_at_5', 0):.3f}")
                print(f"  • Mean Reciprocal Rank: {acc.get('average_mrr', 0):.3f}")
                print(f"  • Queries Tested: {acc.get('total_queries_tested', 0)}")
            
            # Quality results
            if "quality" in summary:
                qual = summary["quality"]
                print(f"\\n⭐ QUALITY:")
                print(f"  • Overall Score: {qual.get('average_overall_quality', 0):.3f}")
                print(f"  • ROUGE-1: {qual.get('average_rouge1', 0):.3f}")
                print(f"  • Coherence: {qual.get('average_coherence', 0):.3f}")
                print(f"  • Test Cases: {qual.get('total_test_cases', 0)}")
            
            # Comparison results
            if "comparison" in summary:
                comp = summary["comparison"]
                total = comp.get('total_queries', 1)
                rag_wins = comp.get('rag_wins', 0)
                baseline_wins = comp.get('baseline_wins', 0)
                
                print(f"\\n🥊 RAG vs BASELINE:")
                print(f"  • RAG Wins: {rag_wins}/{total} ({rag_wins/total*100:.1f}%)")
                print(f"  • Baseline Wins: {baseline_wins}/{total} ({baseline_wins/total*100:.1f}%)")
                print(f"  • Quality Improvement: {comp.get('quality_improvement_percentage', 0):.1f}%")
                print(f"  • Improvement Score: {comp.get('average_improvement_score', 0):.3f}")
        
        print("\\n" + "=" * 50)
        print("🎉 Benchmark suite completed!")
        
        # Provide next steps
        print("\\n📝 NEXT STEPS:")
        print(f"  • View detailed results: {args.output_dir}/consolidated_report_*.json")
        print(f"  • Read summary: {args.output_dir}/consolidated_report_*.txt")
        print(f"  • Check logs: {args.output_dir}/benchmark_*.log")
        
        return 0
        
    except Exception as e:
        print(f"\\n❌ Benchmark failed: {e}")
        print("\\n🔧 TROUBLESHOOTING:")
        print("  • Ensure RAG API is running at the specified URL")
        print("  • Check API endpoint accessibility")
        print("  • Verify all dependencies are installed")
        print("  • Check logs for detailed error information")
        return 1

def check_api_availability(url: str) -> bool:
    """Check if the RAG API is accessible."""
    import httpx
    
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(f"{url}/health")
            return response.status_code == 200
    except Exception:
        return False

if __name__ == "__main__":
    # Quick API availability check
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8001")
    temp_args, _ = parser.parse_known_args()
    
    print("🔍 Checking API availability...")
    if not check_api_availability(temp_args.url):
        print(f"⚠️  Warning: RAG API at {temp_args.url} is not responding")
        print("   Make sure the API is running before starting benchmarks")
        print("   Start with: docker compose up -d  or  python main.py")
        response = input("\\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(1)
    else:
        print("✅ API is accessible")
    
    # Run benchmarks
    exit_code = asyncio.run(main())
    sys.exit(exit_code)