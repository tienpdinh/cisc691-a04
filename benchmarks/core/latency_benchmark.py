"""
Latency Benchmarking Module

Measures response times, throughput, and performance under load for RAG API endpoints.
"""

import time
import asyncio
import statistics
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import httpx
import concurrent.futures
from datetime import datetime
import logging

@dataclass
class LatencyMetrics:
    """Container for latency measurement results."""
    endpoint: str
    request_count: int
    mean_latency: float
    median_latency: float
    p95_latency: float
    p99_latency: float
    min_latency: float
    max_latency: float
    throughput_rps: float
    total_time: float
    success_rate: float
    error_count: int
    timestamp: str

class LatencyBenchmark:
    """
    Comprehensive latency benchmarking for RAG API endpoints.
    
    Measures:
    - Single request latency
    - Concurrent request performance
    - Throughput under load
    - Statistical distributions
    """
    
    def __init__(self, base_url: str = "http://localhost:8001", timeout: int = 15):
        """
        Initialize latency benchmark.
        
        Args:
            base_url: Base URL of the RAG API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
    async def measure_single_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Measure latency of a single request.
        
        Args:
            endpoint: API endpoint to test
            payload: Request payload
            
        Returns:
            Dictionary with timing results
        """
        url = f"{self.base_url}{endpoint}"
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            start_time = time.perf_counter()
            try:
                if endpoint == "/upload-document":
                    # Handle file upload
                    response = await client.post(url, files=payload)
                else:
                    # Handle JSON requests
                    response = await client.post(url, json=payload)
                
                end_time = time.perf_counter()
                latency = end_time - start_time
                
                return {
                    'latency': latency,
                    'status_code': response.status_code,
                    'success': response.status_code == 200,
                    'response_size': len(response.content),
                    'error': None
                }
                
            except Exception as e:
                end_time = time.perf_counter()
                latency = end_time - start_time
                
                return {
                    'latency': latency,
                    'status_code': 0,
                    'success': False,
                    'response_size': 0,
                    'error': str(e)
                }
    
    async def measure_concurrent_requests(self, 
                                        endpoint: str, 
                                        payload: Dict[str, Any], 
                                        concurrent_users: int = 10,
                                        requests_per_user: int = 5) -> LatencyMetrics:
        """
        Measure latency under concurrent load.
        
        Args:
            endpoint: API endpoint to test
            payload: Request payload
            concurrent_users: Number of concurrent users
            requests_per_user: Requests per user
            
        Returns:
            LatencyMetrics object with results
        """
        self.logger.info(f"Starting concurrent test: {concurrent_users} users, {requests_per_user} requests each")
        
        total_requests = concurrent_users * requests_per_user
        tasks = []
        
        # Create tasks for concurrent execution
        for _ in range(total_requests):
            task = self.measure_single_request(endpoint, payload)
            tasks.append(task)
        
        # Execute all requests concurrently
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        
        # Process results
        latencies = []
        successful_requests = 0
        error_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                error_count += 1
                continue
                
            latencies.append(result['latency'])
            if result['success']:
                successful_requests += 1
            else:
                error_count += 1
        
        # Calculate metrics
        if latencies:
            metrics = LatencyMetrics(
                endpoint=endpoint,
                request_count=total_requests,
                mean_latency=statistics.mean(latencies),
                median_latency=statistics.median(latencies),
                p95_latency=self._percentile(latencies, 95),
                p99_latency=self._percentile(latencies, 99),
                min_latency=min(latencies),
                max_latency=max(latencies),
                throughput_rps=successful_requests / total_time,
                total_time=total_time,
                success_rate=successful_requests / total_requests * 100,
                error_count=error_count,
                timestamp=datetime.now().isoformat()
            )
        else:
            # All requests failed
            metrics = LatencyMetrics(
                endpoint=endpoint,
                request_count=total_requests,
                mean_latency=0,
                median_latency=0,
                p95_latency=0,
                p99_latency=0,
                min_latency=0,
                max_latency=0,
                throughput_rps=0,
                total_time=total_time,
                success_rate=0,
                error_count=error_count,
                timestamp=datetime.now().isoformat()
            )
        
        self.logger.info(f"Completed concurrent test: {metrics.success_rate:.1f}% success rate, "
                        f"{metrics.throughput_rps:.2f} RPS")
        
        return metrics
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        lower = int(index)
        upper = min(lower + 1, len(sorted_data) - 1)
        weight = index - lower
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight
    
    async def benchmark_query_endpoint(self, 
                                     queries: List[str],
                                     concurrent_users: int = 5,
                                     use_rag: bool = True) -> LatencyMetrics:
        """
        Benchmark the /query endpoint with different queries.
        
        Args:
            queries: List of test queries
            concurrent_users: Number of concurrent users
            use_rag: Whether to use RAG or direct LLM
            
        Returns:
            LatencyMetrics for query endpoint
        """
        # Use first query for payload template
        payload = {
            "query": queries[0] if queries else "What is this document about?",
            "use_rag": use_rag
        }
        
        return await self.measure_concurrent_requests(
            "/query", 
            payload, 
            concurrent_users=concurrent_users,
            requests_per_user=len(queries) if queries else 1
        )
    
    async def benchmark_retrieve_endpoint(self, 
                                        queries: List[str],
                                        concurrent_users: int = 5) -> LatencyMetrics:
        """
        Benchmark the /retrieve endpoint.
        
        Args:
            queries: List of test queries
            concurrent_users: Number of concurrent users
            
        Returns:
            LatencyMetrics for retrieve endpoint
        """
        payload = {
            "query": queries[0] if queries else "artificial intelligence",
            "top_k": 5
        }
        
        return await self.measure_concurrent_requests(
            "/retrieve", 
            payload, 
            concurrent_users=concurrent_users,
            requests_per_user=len(queries) if queries else 1
        )
    
    def save_results(self, metrics: LatencyMetrics, output_file: Optional[Path] = None):
        """
        Save benchmark results to file.
        
        Args:
            metrics: LatencyMetrics to save
            output_file: Output file path
        """
        if output_file is None:
            output_file = Path("benchmarks/results") / f"latency_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        self.logger.info(f"Latency results saved to {output_file}")
    
    async def run_full_benchmark(self, queries: List[str]) -> Dict[str, LatencyMetrics]:
        """
        Run comprehensive latency benchmark across all endpoints.
        
        Args:
            queries: List of test queries
            
        Returns:
            Dictionary of endpoint -> LatencyMetrics
        """
        results = {}
        
        self.logger.info("Starting full latency benchmark...")
        
        # Benchmark query endpoint with RAG
        results['query_with_rag'] = await self.benchmark_query_endpoint(
            queries, concurrent_users=3, use_rag=True
        )
        
        # Benchmark query endpoint without RAG
        results['query_without_rag'] = await self.benchmark_query_endpoint(
            queries, concurrent_users=3, use_rag=False
        )
        
        # Benchmark retrieve endpoint
        results['retrieve'] = await self.benchmark_retrieve_endpoint(
            queries, concurrent_users=3
        )
        
        # Save all results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for endpoint, metrics in results.items():
            output_file = Path("benchmarks/results") / f"latency_{endpoint}_{timestamp}.json"
            self.save_results(metrics, output_file)
        
        self.logger.info("Full latency benchmark completed")
        
        return results