"""
Main Benchmark Runner

Orchestrates all benchmarking components and generates comprehensive reports.
"""

import asyncio
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import argparse

from .latency_benchmark import LatencyBenchmark
from .accuracy_benchmark import AccuracyBenchmark
from .quality_benchmark import QualityBenchmark
from .baseline_comparison import BaselineComparison

class BenchmarkRunner:
    """
    Main orchestrator for comprehensive RAG system benchmarking.
    
    Runs all benchmark types and generates consolidated reports.
    """
    
    def __init__(self, base_url: str = "http://localhost:8001", timeout: int = 15):
        """
        Initialize benchmark runner.
        
        Args:
            base_url: Base URL of the RAG API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        # Initialize benchmark components
        self.latency_benchmark = LatencyBenchmark(base_url, timeout)
        self.accuracy_benchmark = AccuracyBenchmark(base_url, timeout)
        self.quality_benchmark = QualityBenchmark(base_url, timeout)
        self.baseline_comparison = BaselineComparison(base_url, timeout)
        
        # Results storage
        self.results = {}
        self.benchmark_config = {}
        
    def setup_logging(self, log_level: str = "INFO"):
        """
        Set up logging for benchmark runner.
        
        Args:
            log_level: Logging level
        """
        log_dir = Path("benchmarks/results")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="[%(asctime)s] %(levelname)s %(name)s:%(lineno)d - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            ]
        )
    
    def load_benchmark_config(self, config_file: Path):
        """
        Load benchmark configuration from file.
        
        Args:
            config_file: Path to configuration JSON file
        """
        try:
            with open(config_file, 'r') as f:
                self.benchmark_config = json.load(f)
            self.logger.info(f"Loaded benchmark configuration from {config_file}")
        except Exception as e:
            self.logger.error(f"Error loading config from {config_file}: {e}")
            self.benchmark_config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default benchmark configuration."""
        return {
            "latency": {
                "concurrent_users": [1, 3, 5, 10],
                "requests_per_user": 5,
                "test_queries": [
                    "What were the retail e-commerce sales trends in Q1 2024?",
                    "How did Q4 2023 retail sales compare to Q4 2024?",
                    "What are the key metrics from the Q2 2024 e-commerce report?",
                    "Show me the sales performance across all quarters in 2023",
                    "What insights can be found in the Q3 2024 retail sales data?"
                ]
            },
            "accuracy": {
                "top_k_values": [1, 3, 5, 10],
                "ground_truth_file": "benchmarks/data/ground_truth.json"
            },
            "quality": {
                "test_cases_file": "benchmarks/data/quality_test_cases.json",
                "evaluation_metrics": ["rouge", "bert_score", "semantic_similarity", "coherence"]
            },
            "comparison": {
                "test_queries_file": "benchmarks/data/comparison_queries.json"
            }
        }
    
    async def run_latency_benchmark(self) -> Dict[str, Any]:
        """
        Run latency benchmark tests.
        
        Returns:
            Dictionary with latency benchmark results
        """
        self.logger.info("Starting latency benchmark...")
        
        config = self.benchmark_config.get("latency", {})
        test_queries = config.get("test_queries", ["What were the retail e-commerce sales trends in Q1 2024?"])
        
        # Run comprehensive latency tests
        latency_results = await self.latency_benchmark.run_full_benchmark(test_queries)
        
        # Store results
        self.results["latency"] = {
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "results": {endpoint: result.__dict__ for endpoint, result in latency_results.items()}
        }
        
        self.logger.info("Latency benchmark completed")
        return self.results["latency"]
    
    async def run_accuracy_benchmark(self) -> Dict[str, Any]:
        """
        Run accuracy benchmark tests.
        
        Returns:
            Dictionary with accuracy benchmark results
        """
        self.logger.info("Starting accuracy benchmark...")
        
        config = self.benchmark_config.get("accuracy", {})
        ground_truth_file = Path(config.get("ground_truth_file", "benchmarks/data/ground_truth.json"))
        
        # Create sample ground truth if file doesn't exist
        if not ground_truth_file.exists():
            self.logger.info("Creating sample ground truth data...")
            self.accuracy_benchmark.create_sample_ground_truth(ground_truth_file)
        
        # Run accuracy tests
        accuracy_results = await self.accuracy_benchmark.run_full_benchmark(ground_truth_file)
        
        # Store results
        self.results["accuracy"] = {
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "results": [result.__dict__ for result in accuracy_results]
        }
        
        self.logger.info("Accuracy benchmark completed")
        return self.results["accuracy"]
    
    async def run_quality_benchmark(self) -> Dict[str, Any]:
        """
        Run quality benchmark tests.
        
        Returns:
            Dictionary with quality benchmark results
        """
        self.logger.info("Starting quality benchmark...")
        
        config = self.benchmark_config.get("quality", {})
        test_cases_file = Path(config.get("test_cases_file", "benchmarks/data/quality_test_cases.json"))
        
        # Create sample test cases if file doesn't exist
        if not test_cases_file.exists():
            self.logger.info("Creating sample quality test cases...")
            self.quality_benchmark.create_sample_test_cases(test_cases_file)
        
        # Run quality tests
        quality_results = await self.quality_benchmark.run_full_benchmark(test_cases_file)
        
        # Store results
        self.results["quality"] = {
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "results": [result.__dict__ for result in quality_results]
        }
        
        self.logger.info("Quality benchmark completed")
        return self.results["quality"]
    
    async def run_comparison_benchmark(self) -> Dict[str, Any]:
        """
        Run baseline comparison tests.
        
        Returns:
            Dictionary with comparison benchmark results
        """
        self.logger.info("Starting baseline comparison...")
        
        config = self.benchmark_config.get("comparison", {})
        test_queries_file = Path(config.get("test_queries_file", "benchmarks/data/comparison_queries.json"))
        
        # Create sample test queries if file doesn't exist
        if not test_queries_file.exists():
            self.logger.info("Creating sample comparison queries...")
            self.baseline_comparison.create_sample_test_queries(test_queries_file)
        
        # Run comparison tests
        comparison_results, summary = await self.baseline_comparison.run_full_comparison(test_queries_file)
        
        # Store results
        self.results["comparison"] = {
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "summary": summary.__dict__,
            "results": [result.__dict__ for result in comparison_results]
        }
        
        self.logger.info("Baseline comparison completed")
        return self.results["comparison"]
    
    def generate_consolidated_report(self) -> Dict[str, Any]:
        """
        Generate a consolidated benchmark report.
        
        Returns:
            Dictionary with consolidated results and analysis
        """
        self.logger.info("Generating consolidated report...")
        
        report = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "base_url": self.base_url,
                "timeout": self.timeout,
                "config": self.benchmark_config
            },
            "summary": {},
            "detailed_results": self.results
        }
        
        # Extract key metrics for summary
        try:
            # Latency summary
            if "latency" in self.results:
                latency_data = self.results["latency"]["results"]
                report["summary"]["latency"] = {
                    "query_with_rag_avg_latency": latency_data.get("query_with_rag", {}).get("mean_latency", 0),
                    "query_without_rag_avg_latency": latency_data.get("query_without_rag", {}).get("mean_latency", 0),
                    "retrieve_avg_latency": latency_data.get("retrieve", {}).get("mean_latency", 0),
                    "rag_throughput_rps": latency_data.get("query_with_rag", {}).get("throughput_rps", 0)
                }
            
            # Accuracy summary
            if "accuracy" in self.results:
                accuracy_results = self.results["accuracy"]["results"]
                if accuracy_results:
                    avg_precision_5 = sum(r.get("precision_at_k", {}).get(5, 0) for r in accuracy_results) / len(accuracy_results)
                    avg_recall_5 = sum(r.get("recall_at_k", {}).get(5, 0) for r in accuracy_results) / len(accuracy_results)
                    avg_mrr = sum(r.get("mean_reciprocal_rank", 0) for r in accuracy_results) / len(accuracy_results)
                    
                    report["summary"]["accuracy"] = {
                        "average_precision_at_5": avg_precision_5,
                        "average_recall_at_5": avg_recall_5,
                        "average_mrr": avg_mrr,
                        "total_queries_tested": len(accuracy_results)
                    }
            
            # Quality summary
            if "quality" in self.results:
                quality_results = self.results["quality"]["results"]
                if quality_results:
                    avg_overall_quality = sum(r.get("overall_quality_score", 0) for r in quality_results) / len(quality_results)
                    avg_rouge1 = sum(r.get("rouge_scores", {}).get("rouge1", 0) for r in quality_results) / len(quality_results)
                    avg_coherence = sum(r.get("coherence_score", 0) for r in quality_results) / len(quality_results)
                    
                    report["summary"]["quality"] = {
                        "average_overall_quality": avg_overall_quality,
                        "average_rouge1": avg_rouge1,
                        "average_coherence": avg_coherence,
                        "total_test_cases": len(quality_results)
                    }
            
            # Comparison summary
            if "comparison" in self.results:
                comparison_summary = self.results["comparison"]["summary"]
                report["summary"]["comparison"] = {
                    "rag_wins": comparison_summary.get("rag_wins", 0),
                    "baseline_wins": comparison_summary.get("baseline_wins", 0),
                    "total_queries": comparison_summary.get("total_queries", 0),
                    "quality_improvement_percentage": comparison_summary.get("quality_improvement_percentage", 0),
                    "average_improvement_score": comparison_summary.get("average_improvement", 0)
                }
                
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            report["summary"]["error"] = str(e)
        
        return report
    
    def save_consolidated_report(self, report: Dict[str, Any], output_file: Optional[Path] = None):
        """
        Save consolidated report to file.
        
        Args:
            report: Consolidated report dictionary
            output_file: Output file path
        """
        if output_file is None:
            output_file = Path("benchmarks/results") / f"consolidated_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Consolidated report saved to {output_file}")
        
        # Also save a summary text file
        summary_file = output_file.with_suffix('.txt')
        self._save_text_summary(report, summary_file)
    
    def _save_text_summary(self, report: Dict[str, Any], output_file: Path):
        """Save human-readable text summary."""
        try:
            with open(output_file, 'w') as f:
                f.write("RAG SYSTEM BENCHMARK REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Benchmark Date: {report['benchmark_info']['timestamp']}\n")
                f.write(f"API Endpoint: {report['benchmark_info']['base_url']}\n\n")
                
                # Latency Summary
                if "latency" in report["summary"]:
                    f.write("LATENCY PERFORMANCE\n")
                    f.write("-" * 20 + "\n")
                    latency = report["summary"]["latency"]
                    f.write(f"RAG Query Latency: {latency.get('query_with_rag_avg_latency', 0):.3f}s\n")
                    f.write(f"Direct LLM Latency: {latency.get('query_without_rag_avg_latency', 0):.3f}s\n")
                    f.write(f"Retrieval Latency: {latency.get('retrieve_avg_latency', 0):.3f}s\n")
                    f.write(f"Throughput: {latency.get('rag_throughput_rps', 0):.2f} requests/second\n\n")
                
                # Accuracy Summary
                if "accuracy" in report["summary"]:
                    f.write("RETRIEVAL ACCURACY\n")
                    f.write("-" * 18 + "\n")
                    accuracy = report["summary"]["accuracy"]
                    f.write(f"Precision@5: {accuracy.get('average_precision_at_5', 0):.3f}\n")
                    f.write(f"Recall@5: {accuracy.get('average_recall_at_5', 0):.3f}\n")
                    f.write(f"Mean Reciprocal Rank: {accuracy.get('average_mrr', 0):.3f}\n")
                    f.write(f"Queries Tested: {accuracy.get('total_queries_tested', 0)}\n\n")
                
                # Quality Summary
                if "quality" in report["summary"]:
                    f.write("RESPONSE QUALITY\n")
                    f.write("-" * 16 + "\n")
                    quality = report["summary"]["quality"]
                    f.write(f"Overall Quality Score: {quality.get('average_overall_quality', 0):.3f}\n")
                    f.write(f"ROUGE-1 Score: {quality.get('average_rouge1', 0):.3f}\n")
                    f.write(f"Coherence Score: {quality.get('average_coherence', 0):.3f}\n")
                    f.write(f"Test Cases: {quality.get('total_test_cases', 0)}\n\n")
                
                # Comparison Summary
                if "comparison" in report["summary"]:
                    f.write("RAG vs BASELINE COMPARISON\n")
                    f.write("-" * 27 + "\n")
                    comparison = report["summary"]["comparison"]
                    f.write(f"RAG Wins: {comparison.get('rag_wins', 0)}\n")
                    f.write(f"Baseline Wins: {comparison.get('baseline_wins', 0)}\n")
                    f.write(f"Total Comparisons: {comparison.get('total_queries', 0)}\n")
                    f.write(f"Quality Improvement: {comparison.get('quality_improvement_percentage', 0):.1f}%\n")
                    f.write(f"Average Improvement Score: {comparison.get('average_improvement_score', 0):.3f}\n\n")
                
                f.write("=" * 50 + "\n")
                f.write("End of Report\n")
                
        except Exception as e:
            self.logger.error(f"Error saving text summary: {e}")
    
    async def run_full_benchmark_suite(self, config_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run the complete benchmark suite.
        
        Args:
            config_file: Optional configuration file path
            
        Returns:
            Consolidated benchmark report
        """
        self.logger.info("Starting full benchmark suite...")
        
        # Load configuration
        if config_file and config_file.exists():
            self.load_benchmark_config(config_file)
        else:
            self.benchmark_config = self._get_default_config()
        
        try:
            # Run all benchmarks
            await self.run_latency_benchmark()
            await self.run_accuracy_benchmark()
            await self.run_quality_benchmark()
            await self.run_comparison_benchmark()
            
            # Generate consolidated report
            report = self.generate_consolidated_report()
            
            # Save report
            self.save_consolidated_report(report)
            
            self.logger.info("Full benchmark suite completed successfully")
            return report
            
        except Exception as e:
            self.logger.error(f"Error running benchmark suite: {e}")
            raise

def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(description="RAG System Benchmark Runner")
    parser.add_argument("--url", default="http://localhost:8001", help="Base URL of RAG API")
    parser.add_argument("--timeout", type=int, default=15, help="Request timeout in seconds")
    parser.add_argument("--config", type=Path, help="Path to benchmark configuration file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = BenchmarkRunner(args.url, args.timeout)
    runner.setup_logging(args.log_level)
    
    # Run benchmarks
    try:
        report = asyncio.run(runner.run_full_benchmark_suite(args.config))
        print(f"\\nBenchmark completed successfully!")
        print(f"Results saved in: benchmarks/results/")
        
        # Print summary
        if "summary" in report:
            print("\\nSUMMARY:")
            if "latency" in report["summary"]:
                print(f"  RAG Latency: {report['summary']['latency'].get('query_with_rag_avg_latency', 0):.3f}s")
            if "accuracy" in report["summary"]:
                print(f"  Precision@5: {report['summary']['accuracy'].get('average_precision_at_5', 0):.3f}")
            if "quality" in report["summary"]:
                print(f"  Quality Score: {report['summary']['quality'].get('average_overall_quality', 0):.3f}")
            if "comparison" in report["summary"]:
                print(f"  Quality Improvement: {report['summary']['comparison'].get('quality_improvement_percentage', 0):.1f}%")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())