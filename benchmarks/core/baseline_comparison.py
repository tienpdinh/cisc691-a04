"""
Baseline LLM Comparison Module

Compares RAG system performance against baseline LLM responses.
Evaluates the improvement gained by using retrieval-augmented generation.
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import httpx
import statistics

@dataclass
class ComparisonMetrics:
    """Container for baseline comparison results."""
    query: str
    rag_response: str
    baseline_response: str
    rag_latency: float
    baseline_latency: float
    improvement_score: float
    rag_quality_score: float
    baseline_quality_score: float
    rag_relevance_score: float
    baseline_relevance_score: float
    rag_factual_accuracy: float
    baseline_factual_accuracy: float
    response_length_rag: int
    response_length_baseline: int
    timestamp: str

@dataclass
class ComparisonSummary:
    """Summary statistics for baseline comparison."""
    total_queries: int
    rag_wins: int
    baseline_wins: int
    ties: int
    average_improvement: float
    average_rag_quality: float
    average_baseline_quality: float
    average_latency_rag: float
    average_latency_baseline: float
    quality_improvement_percentage: float
    relevance_improvement_percentage: float
    accuracy_improvement_percentage: float

class BaselineComparison:
    """
    Comprehensive comparison between RAG and baseline LLM responses.
    
    Compares:
    - Response quality and accuracy
    - Relevance to queries
    - Response latency
    - Factual accuracy
    - Overall improvement metrics
    """
    
    def __init__(self, base_url: str = "http://localhost:8001", timeout: int = 15):
        """
        Initialize baseline comparison.
        
        Args:
            base_url: Base URL of the RAG API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        # Initialize quality evaluator (reuse from quality_benchmark)
        try:
            from .quality_benchmark import QualityBenchmark
            self.quality_evaluator = QualityBenchmark(base_url, timeout)
            self.logger.info("Initialized quality evaluator for comparison")
        except ImportError:
            self.logger.warning("Quality benchmark not available for comparison")
            self.quality_evaluator = None
    
    async def generate_rag_response(self, query: str) -> Tuple[str, float]:
        """
        Generate response using RAG system.
        
        Args:
            query: Input query
            
        Returns:
            Tuple of (response_text, latency_seconds)
        """
        import time
        
        url = f"{self.base_url}/query"
        payload = {"query": query, "use_rag": True}
        
        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                end_time = time.perf_counter()
                
                data = response.json()
                response_text = data.get('response', '')
                latency = end_time - start_time
                
                return response_text, latency
                
            except Exception as e:
                end_time = time.perf_counter()
                latency = end_time - start_time
                self.logger.error(f"Error generating RAG response: {e}")
                return "", latency
    
    async def generate_baseline_response(self, query: str) -> Tuple[str, float]:
        """
        Generate response using baseline LLM (no RAG).
        
        Args:
            query: Input query
            
        Returns:
            Tuple of (response_text, latency_seconds)
        """
        import time
        
        url = f"{self.base_url}/query"
        payload = {"query": query, "use_rag": False}
        
        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                end_time = time.perf_counter()
                
                data = response.json()
                response_text = data.get('response', '')
                latency = end_time - start_time
                
                return response_text, latency
                
            except Exception as e:
                end_time = time.perf_counter()
                latency = end_time - start_time
                self.logger.error(f"Error generating baseline response: {e}")
                return "", latency
    
    def calculate_improvement_score(self, rag_quality: float, baseline_quality: float) -> float:
        """
        Calculate improvement score from RAG vs baseline.
        
        Args:
            rag_quality: Quality score for RAG response
            baseline_quality: Quality score for baseline response
            
        Returns:
            Improvement score (-1.0 to 1.0)
        """
        if baseline_quality == 0:
            return 1.0 if rag_quality > 0 else 0.0
        
        improvement = (rag_quality - baseline_quality) / baseline_quality
        return max(-1.0, min(1.0, improvement))
    
    async def evaluate_response_quality(self, query: str, response: str, reference: str = "") -> float:
        """
        Evaluate response quality using available metrics.
        
        Args:
            query: Original query
            response: Response to evaluate
            reference: Reference answer (if available)
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        if not self.quality_evaluator or not response:
            return 0.0
        
        try:
            # Use semantic similarity with query as a proxy for quality
            if hasattr(self.quality_evaluator, 'calculate_semantic_similarity'):
                relevance = self.quality_evaluator.calculate_semantic_similarity(query, response)
            else:
                relevance = 0.5
            
            # Use coherence score
            if hasattr(self.quality_evaluator, 'calculate_coherence_score'):
                coherence = self.quality_evaluator.calculate_coherence_score(response)
            else:
                coherence = 0.5
            
            # Combine metrics (simple average for now)
            quality_score = (relevance + coherence) / 2
            return quality_score
            
        except Exception as e:
            self.logger.error(f"Error evaluating response quality: {e}")
            return 0.0
    
    def calculate_relevance_score(self, query: str, response: str) -> float:
        """
        Calculate relevance of response to query.
        
        Args:
            query: Original query
            response: Response to evaluate
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not self.quality_evaluator:
            return 0.0
        
        return self.quality_evaluator.calculate_semantic_similarity(query, response)
    
    def calculate_factual_accuracy(self, response: str, context_docs: List[str] = None) -> float:
        """
        Calculate factual accuracy of response.
        
        Args:
            response: Response to evaluate
            context_docs: Context documents for verification
            
        Returns:
            Factual accuracy score (0.0 to 1.0)
        """
        if not self.quality_evaluator or not context_docs:
            # Simple heuristic: longer responses are assumed more detailed/accurate
            if response:
                words = len(response.split())
                return min(1.0, words / 100)  # Normalize by 100 words
            return 0.0
        
        return self.quality_evaluator.calculate_factual_accuracy_score(response, context_docs)
    
    async def compare_responses(self, query: str, context_docs: List[str] = None) -> ComparisonMetrics:
        """
        Compare RAG and baseline responses for a single query.
        
        Args:
            query: Query to test
            context_docs: Context documents for accuracy evaluation
            
        Returns:
            ComparisonMetrics with comparison results
        """
        self.logger.info(f"Comparing responses for query: '{query}'")
        
        # Generate both responses
        rag_response, rag_latency = await self.generate_rag_response(query)
        baseline_response, baseline_latency = await self.generate_baseline_response(query)
        
        # Evaluate quality
        rag_quality = await self.evaluate_response_quality(query, rag_response)
        baseline_quality = await self.evaluate_response_quality(query, baseline_response)
        
        # Calculate relevance
        rag_relevance = self.calculate_relevance_score(query, rag_response)
        baseline_relevance = self.calculate_relevance_score(query, baseline_response)
        
        # Calculate factual accuracy
        rag_accuracy = self.calculate_factual_accuracy(rag_response, context_docs)
        baseline_accuracy = self.calculate_factual_accuracy(baseline_response, context_docs)
        
        # Calculate improvement
        improvement = self.calculate_improvement_score(rag_quality, baseline_quality)
        
        metrics = ComparisonMetrics(
            query=query,
            rag_response=rag_response,
            baseline_response=baseline_response,
            rag_latency=rag_latency,
            baseline_latency=baseline_latency,
            improvement_score=improvement,
            rag_quality_score=rag_quality,
            baseline_quality_score=baseline_quality,
            rag_relevance_score=rag_relevance,
            baseline_relevance_score=baseline_relevance,
            rag_factual_accuracy=rag_accuracy,
            baseline_factual_accuracy=baseline_accuracy,
            response_length_rag=len(rag_response.split()),
            response_length_baseline=len(baseline_response.split()),
            timestamp=datetime.now().isoformat()
        )
        
        self.logger.info(f"Comparison completed. Improvement: {improvement:.3f}, "
                        f"RAG quality: {rag_quality:.3f}, Baseline quality: {baseline_quality:.3f}")
        
        return metrics
    
    def calculate_summary_statistics(self, comparison_results: List[ComparisonMetrics]) -> ComparisonSummary:
        """
        Calculate summary statistics from comparison results.
        
        Args:
            comparison_results: List of ComparisonMetrics
            
        Returns:
            ComparisonSummary with aggregated statistics
        """
        if not comparison_results:
            return ComparisonSummary(
                total_queries=0, rag_wins=0, baseline_wins=0, ties=0,
                average_improvement=0.0, average_rag_quality=0.0, average_baseline_quality=0.0,
                average_latency_rag=0.0, average_latency_baseline=0.0,
                quality_improvement_percentage=0.0, relevance_improvement_percentage=0.0,
                accuracy_improvement_percentage=0.0
            )
        
        total_queries = len(comparison_results)
        rag_wins = sum(1 for r in comparison_results if r.rag_quality_score > r.baseline_quality_score)
        baseline_wins = sum(1 for r in comparison_results if r.baseline_quality_score > r.rag_quality_score)
        ties = total_queries - rag_wins - baseline_wins
        
        # Calculate averages
        avg_improvement = statistics.mean([r.improvement_score for r in comparison_results])
        avg_rag_quality = statistics.mean([r.rag_quality_score for r in comparison_results])
        avg_baseline_quality = statistics.mean([r.baseline_quality_score for r in comparison_results])
        avg_rag_latency = statistics.mean([r.rag_latency for r in comparison_results])
        avg_baseline_latency = statistics.mean([r.baseline_latency for r in comparison_results])
        
        # Calculate improvement percentages
        quality_improvement = ((avg_rag_quality - avg_baseline_quality) / avg_baseline_quality * 100) if avg_baseline_quality > 0 else 0
        
        avg_rag_relevance = statistics.mean([r.rag_relevance_score for r in comparison_results])
        avg_baseline_relevance = statistics.mean([r.baseline_relevance_score for r in comparison_results])
        relevance_improvement = ((avg_rag_relevance - avg_baseline_relevance) / avg_baseline_relevance * 100) if avg_baseline_relevance > 0 else 0
        
        avg_rag_accuracy = statistics.mean([r.rag_factual_accuracy for r in comparison_results])
        avg_baseline_accuracy = statistics.mean([r.baseline_factual_accuracy for r in comparison_results])
        accuracy_improvement = ((avg_rag_accuracy - avg_baseline_accuracy) / avg_baseline_accuracy * 100) if avg_baseline_accuracy > 0 else 0
        
        return ComparisonSummary(
            total_queries=total_queries,
            rag_wins=rag_wins,
            baseline_wins=baseline_wins,
            ties=ties,
            average_improvement=avg_improvement,
            average_rag_quality=avg_rag_quality,
            average_baseline_quality=avg_baseline_quality,
            average_latency_rag=avg_rag_latency,
            average_latency_baseline=avg_baseline_latency,
            quality_improvement_percentage=quality_improvement,
            relevance_improvement_percentage=relevance_improvement,
            accuracy_improvement_percentage=accuracy_improvement
        )
    
    def load_test_queries(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load test queries from JSON file.
        
        Args:
            file_path: Path to test queries JSON file
            
        Returns:
            List of test query dictionaries
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            self.logger.info(f"Loaded {len(data)} test queries for comparison")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading test queries from {file_path}: {e}")
            return []
    
    def save_results(self, 
                    comparison_results: List[ComparisonMetrics], 
                    summary: ComparisonSummary,
                    output_file: Optional[Path] = None):
        """
        Save comparison results to file.
        
        Args:
            comparison_results: List of ComparisonMetrics
            summary: ComparisonSummary statistics
            output_file: Output file path
        """
        if output_file is None:
            output_file = Path("benchmarks/results") / f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            'summary': asdict(summary),
            'individual_comparisons': [asdict(metrics) for metrics in comparison_results],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"Comparison results saved to {output_file}")
    
    async def run_full_comparison(self, test_queries_file: Path) -> Tuple[List[ComparisonMetrics], ComparisonSummary]:
        """
        Run comprehensive baseline comparison.
        
        Args:
            test_queries_file: Path to test queries JSON file
            
        Returns:
            Tuple of (comparison_results, summary_statistics)
        """
        self.logger.info("Starting full baseline comparison...")
        
        # Load test queries
        test_queries = self.load_test_queries(test_queries_file)
        
        if not test_queries:
            self.logger.error("No test queries available")
            return [], ComparisonSummary(
                total_queries=0, rag_wins=0, baseline_wins=0, ties=0,
                average_improvement=0.0, average_rag_quality=0.0, average_baseline_quality=0.0,
                average_latency_rag=0.0, average_latency_baseline=0.0,
                quality_improvement_percentage=0.0, relevance_improvement_percentage=0.0,
                accuracy_improvement_percentage=0.0
            )
        
        # Run comparisons concurrently
        comparison_tasks = []
        valid_queries = []
        
        for query_data in test_queries:
            query = query_data.get('query', query_data.get('text', ''))
            context_docs = query_data.get('context_documents', [])
            
            if query:
                valid_queries.append(query_data)
                comparison_tasks.append(self.compare_responses(query, context_docs))
        
        # Execute all comparisons concurrently
        comparison_results = []
        if comparison_tasks:
            try:
                results = await asyncio.gather(*comparison_tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Error comparing responses for query '{valid_queries[i].get('query', '')}': {result}")
                    else:
                        comparison_results.append(result)
                        
            except Exception as e:
                self.logger.error(f"Error in concurrent comparison execution: {e}")
        
        # Calculate summary statistics
        summary = self.calculate_summary_statistics(comparison_results)
        
        # Save results
        self.save_results(comparison_results, summary)
        
        self.logger.info(f"Baseline comparison completed. RAG wins: {summary.rag_wins}/{summary.total_queries}, "
                        f"Average improvement: {summary.average_improvement:.3f}")
        
        return comparison_results, summary
    
    def create_sample_test_queries(self, output_file: Path):
        """
        Create sample test queries for comparison.
        
        Args:
            output_file: Path where to save the sample file
        """
        sample_queries = [
            {
                "query": "What are the key success factors in project management?",
                "context_documents": [
                    "Successful project management requires clear objectives and scope definition.",
                    "Effective communication and stakeholder management are crucial for project success."
                ],
                "category": "factual",
                "difficulty": "easy"
            },
            {
                "query": "How does artificial intelligence transform business operations?",
                "context_documents": [
                    "AI automates repetitive tasks and enables data-driven decision making.",
                    "Machine learning algorithms can optimize business processes and reduce costs."
                ],
                "category": "analytical",
                "difficulty": "medium"
            },
            {
                "query": "What are the main challenges in implementing RAG systems in production?",
                "context_documents": [
                    "RAG systems face challenges in retrieval accuracy and response quality.",
                    "Production deployment requires careful consideration of latency and scalability."
                ],
                "category": "technical",
                "difficulty": "hard"
            },
            {
                "query": "Explain the benefits of using vector databases for similarity search.",
                "context_documents": [
                    "Vector databases enable efficient similarity search using embeddings.",
                    "They provide fast retrieval for high-dimensional data and semantic search."
                ],
                "category": "technical",
                "difficulty": "medium"
            },
            {
                "query": "What is the difference between supervised and unsupervised learning?",
                "context_documents": [
                    "Supervised learning uses labeled data to train models for prediction.",
                    "Unsupervised learning finds patterns in data without explicit labels."
                ],
                "category": "educational",
                "difficulty": "easy"
            }
        ]
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(sample_queries, f, indent=2)
        
        self.logger.info(f"Sample test queries created at {output_file}")