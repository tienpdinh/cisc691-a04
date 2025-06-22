#!/usr/bin/env python3
"""
Results Viewer for RAG Benchmark System

This script helps you view and analyze benchmark results in various formats.
"""

import json
import glob
from pathlib import Path
from datetime import datetime
import argparse

def list_all_results():
    """List all available result files."""
    results_dir = Path("benchmarks/results")
    
    if not results_dir.exists():
        print("❌ No results directory found. Run benchmarks first!")
        return []
    
    # Find all result files
    json_files = list(results_dir.glob("*.json"))
    txt_files = list(results_dir.glob("*.txt"))
    log_files = list(results_dir.glob("*.log"))
    
    print("📁 Available Results:")
    print("=" * 50)
    
    if json_files:
        print("📊 JSON Results:")
        for file in sorted(json_files):
            print(f"  • {file.name}")
    
    if txt_files:
        print("\\n📝 Text Summaries:")
        for file in sorted(txt_files):
            print(f"  • {file.name}")
    
    if log_files:
        print("\\n📋 Log Files:")
        for file in sorted(log_files):
            print(f"  • {file.name}")
    
    print("\\n" + "=" * 50)
    return json_files

def view_summary(file_path: Path):
    """View a summary of benchmark results."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"\\n📊 BENCHMARK RESULTS: {file_path.name}")
        print("=" * 60)
        
        # Benchmark info
        if "benchmark_info" in data:
            info = data["benchmark_info"]
            print(f"🕐 Timestamp: {info.get('timestamp', 'Unknown')}")
            print(f"🌐 API URL: {info.get('base_url', info.get('api_url', 'Unknown'))}")
            print(f"⏱️  Timeout: {info.get('timeout', 'Unknown')}s")
        
        # Summary statistics
        if "summary" in data:
            summary = data["summary"]
            
            # Latency results
            if "latency" in summary:
                print("\\n🏃 LATENCY PERFORMANCE:")
                lat = summary["latency"]
                print(f"  • RAG Query Latency: {lat.get('query_with_rag_avg_latency', lat.get('avg_query_latency', 0)):.3f}s")
                print(f"  • Direct LLM Latency: {lat.get('query_without_rag_avg_latency', 0):.3f}s")
                print(f"  • Retrieval Latency: {lat.get('retrieve_avg_latency', 0):.3f}s")
                print(f"  • Throughput: {lat.get('rag_throughput_rps', lat.get('throughput_rps', 0)):.2f} req/s")
                print(f"  • Success Rate: {lat.get('success_rate', 100):.1f}%")
            
            # Accuracy results
            if "accuracy" in summary:
                print("\\n🎯 RETRIEVAL ACCURACY:")
                acc = summary["accuracy"]
                print(f"  • Precision@5: {acc.get('average_precision_at_5', 0):.3f}")
                print(f"  • Recall@5: {acc.get('average_recall_at_5', 0):.3f}")
                print(f"  • Mean Reciprocal Rank: {acc.get('average_mrr', 0):.3f}")
                print(f"  • Queries Tested: {acc.get('total_queries_tested', 0)}")
            
            # Quality results
            if "quality" in summary:
                print("\\n⭐ RESPONSE QUALITY:")
                qual = summary["quality"]
                print(f"  • Overall Score: {qual.get('average_overall_quality', qual.get('avg_quality_score', 0)):.3f}")
                print(f"  • ROUGE-1: {qual.get('average_rouge1', 0):.3f}")
                print(f"  • Coherence: {qual.get('average_coherence', qual.get('avg_coherence', 0)):.3f}")
                print(f"  • Test Cases: {qual.get('total_test_cases', 0)}")
            
            # Comparison results
            if "comparison" in summary:
                print("\\n🥊 RAG vs BASELINE:")
                comp = summary["comparison"]
                total = comp.get('total_queries', 1)
                rag_wins = comp.get('rag_wins', 0)
                baseline_wins = comp.get('baseline_wins', 0)
                
                print(f"  • RAG Wins: {rag_wins}/{total} ({rag_wins/total*100:.1f}%)")
                print(f"  • Baseline Wins: {baseline_wins}/{total} ({baseline_wins/total*100:.1f}%)")
                print(f"  • Quality Improvement: {comp.get('quality_improvement_percentage', 0):.1f}%")
                print(f"  • Improvement Score: {comp.get('average_improvement_score', comp.get('improvement_score', 0)):.3f}")
        
        # Individual results count
        if "individual_results" in data:
            count = len(data["individual_results"])
            print(f"\\n📋 Individual Results: {count} entries")
        elif "results" in data:
            if isinstance(data["results"], list):
                count = len(data["results"])
                print(f"\\n📋 Individual Results: {count} entries")
            elif isinstance(data["results"], dict):
                count = len(data["results"])
                print(f"\\n📋 Result Categories: {count}")
        
        print("\\n" + "=" * 60)
        
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")

def view_detailed_results(file_path: Path, limit: int = 5):
    """View detailed individual results."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"\\n🔍 DETAILED RESULTS: {file_path.name}")
        print("=" * 60)
        
        # Individual results
        individual_results = []
        if "individual_results" in data:
            individual_results = data["individual_results"]
        elif "results" in data and isinstance(data["results"], list):
            individual_results = data["results"]
        
        if individual_results:
            print(f"\\n📝 Individual Results (showing first {limit}):")
            for i, result in enumerate(individual_results[:limit]):
                print(f"\\n  Result {i+1}:")
                
                # Query info
                if "query" in result:
                    print(f"    Query: {result['query'][:100]}...")
                
                # Latency info
                if "latency" in result:
                    print(f"    Latency: {result['latency']:.3f}s")
                if "mean_latency" in result:
                    print(f"    Mean Latency: {result['mean_latency']:.3f}s")
                if "throughput_rps" in result:
                    print(f"    Throughput: {result['throughput_rps']:.2f} req/s")
                
                # Quality info
                if "overall_quality_score" in result:
                    print(f"    Quality Score: {result['overall_quality_score']:.3f}")
                if "rouge_scores" in result:
                    rouge = result["rouge_scores"]
                    print(f"    ROUGE-1: {rouge.get('rouge1', 0):.3f}")
                
                # Accuracy info
                if "precision_at_k" in result:
                    precision = result["precision_at_k"]
                    print(f"    Precision@5: {precision.get('5', precision.get(5, 0)):.3f}")
                
                # Comparison info
                if "improvement_score" in result:
                    print(f"    Improvement: {result['improvement_score']:.3f}")
                
            if len(individual_results) > limit:
                print(f"\\n    ... and {len(individual_results) - limit} more results")
        
        print("\\n" + "=" * 60)
        
    except Exception as e:
        print(f"❌ Error reading detailed results from {file_path}: {e}")

def compare_results():
    """Compare multiple benchmark results."""
    results_dir = Path("benchmarks/results")
    json_files = list(results_dir.glob("*results*.json"))
    
    if len(json_files) < 2:
        print("⚠️  Need at least 2 result files to compare")
        return
    
    print("\\n📊 RESULTS COMPARISON")
    print("=" * 50)
    
    comparison_data = []
    
    for file in json_files[-3:]:  # Compare last 3 files
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            summary = data.get("summary", {})
            comparison_data.append({
                "file": file.name,
                "timestamp": data.get("benchmark_info", {}).get("timestamp", "Unknown"),
                "latency": summary.get("latency", {}).get("query_with_rag_avg_latency", 0),
                "quality": summary.get("quality", {}).get("average_overall_quality", 0),
                "improvement": summary.get("comparison", {}).get("quality_improvement_percentage", 0)
            })
        except:
            continue
    
    if comparison_data:
        print(f"{'File':<30} {'Latency':<10} {'Quality':<10} {'Improvement':<12}")
        print("-" * 65)
        for item in comparison_data:
            print(f"{item['file']:<30} {item['latency']:<10.3f} {item['quality']:<10.3f} {item['improvement']:<12.1f}%")

def export_to_csv(file_path: Path):
    """Export results to CSV format."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        csv_file = file_path.with_suffix('.csv')
        
        # Export summary to CSV
        if "summary" in data:
            import csv
            
            with open(csv_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(['Metric', 'Value'])
                
                # Write summary data
                summary = data["summary"]
                for category, metrics in summary.items():
                    if isinstance(metrics, dict):
                        for metric, value in metrics.items():
                            writer.writerow([f"{category}_{metric}", value])
                    else:
                        writer.writerow([category, metrics])
        
        print(f"✅ Exported to: {csv_file}")
        
    except Exception as e:
        print(f"❌ Error exporting to CSV: {e}")

def main():
    parser = argparse.ArgumentParser(description="View RAG Benchmark Results")
    parser.add_argument("--list", "-l", action="store_true", help="List all available results")
    parser.add_argument("--view", "-v", type=str, help="View specific result file")
    parser.add_argument("--detailed", "-d", type=str, help="View detailed results from file")
    parser.add_argument("--compare", "-c", action="store_true", help="Compare multiple results")
    parser.add_argument("--export", "-e", type=str, help="Export results to CSV")
    parser.add_argument("--latest", action="store_true", help="View latest results")
    
    args = parser.parse_args()
    
    if args.list or not any(vars(args).values()):
        json_files = list_all_results()
        
        if json_files and not args.list:
            print("\\n💡 Usage examples:")
            print(f"  python view_results.py --view {json_files[0].name}")
            print(f"  python view_results.py --detailed {json_files[0].name}")
            print("  python view_results.py --compare")
            print("  python view_results.py --latest")
    
    elif args.view:
        file_path = Path("benchmarks/results") / args.view
        if file_path.exists():
            view_summary(file_path)
        else:
            print(f"❌ File not found: {file_path}")
    
    elif args.detailed:
        file_path = Path("benchmarks/results") / args.detailed
        if file_path.exists():
            view_detailed_results(file_path)
        else:
            print(f"❌ File not found: {file_path}")
    
    elif args.compare:
        compare_results()
    
    elif args.export:
        file_path = Path("benchmarks/results") / args.export
        if file_path.exists():
            export_to_csv(file_path)
        else:
            print(f"❌ File not found: {file_path}")
    
    elif args.latest:
        results_dir = Path("benchmarks/results")
        json_files = sorted(results_dir.glob("*.json"), key=lambda x: x.stat().st_mtime)
        if json_files:
            view_summary(json_files[-1])
        else:
            print("❌ No result files found")

if __name__ == "__main__":
    main()