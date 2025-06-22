#!/usr/bin/env python3
"""
Live RAG System Monitoring

Monitors live RAG API endpoints, captures real errors, and tracks performance
when rag-api, chromadb, redis containers and ollama are running.
"""

import asyncio
import aiohttp
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import redis
import requests

# Add the project root to the path to enable imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from error_analysis.core.error_analysis_manager import get_error_analysis_manager


class LiveRAGMonitor:
    """
    Monitor live RAG system components and capture real errors.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize live RAG monitor."""
        self.error_manager = get_error_analysis_manager(config or {})
        
        # RAG API configuration (based on your docker setup)
        self.rag_api_base = "http://localhost:8001"  # rag-api mapped to 8001:8000
        self.redis_host = "localhost"
        self.redis_port = 6380  # redis mapped to 6380:6379
        self.chromadb_host = "localhost" 
        self.chromadb_port = 8000  # chromadb mapped to 8000:8000
        self.ollama_host = "localhost"
        self.ollama_port = 11434  # ollama running locally
        
        # Test queries for monitoring (based on retail e-commerce sales data)
        self.test_queries = [
            "What were the retail e-commerce sales trends in Q1 2024?",
            "How did Q4 2023 sales compare to Q4 2024?",
            "What are the key retail sales metrics for 2023?",
            "Show me the quarterly sales performance for 2024.",
            "What was the growth rate between Q2 2023 and Q2 2024?",
            "Which quarter had the highest e-commerce sales in 2024?",
            "What are the main trends in retail e-commerce sales?",
            "How did sales perform across all quarters in 2023?",
            "What factors influenced Q3 2024 sales performance?",
            "Compare the year-over-year growth for Q1 periods.",
            "What seasonal patterns exist in the e-commerce data?",
            "Which quarters showed the strongest sales growth?"
        ]
        
        # Monitoring results
        self.monitoring_results = {
            'api_calls': [],
            'errors': [],
            'performance_metrics': [],
            'component_health': {},
            'start_time': None,
            'end_time': None
        }
    
    async def check_service_availability(self) -> Dict[str, bool]:
        """Check if all services are available."""
        print("\n🔍 CHECKING SERVICE AVAILABILITY")
        print("=" * 50)
        
        services = {}
        
        # Check RAG API
        try:
            response = requests.get(f"{self.rag_api_base}/health", timeout=30)
            services['rag_api'] = response.status_code == 200
            print(f"    RAG API (port {self.rag_api_base.split(':')[-1]}): {'✅ Available' if services['rag_api'] else '❌ Unavailable'}")
        except Exception as e:
            services['rag_api'] = False
            print(f"    RAG API (port {self.rag_api_base.split(':')[-1]}): ❌ Unavailable - {e}")
            self.error_manager.track_error(e, 'api_endpoints', {'service': 'rag_api', 'check': 'availability'})
        
        # Check Redis
        try:
            r = redis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=True)
            r.ping()
            services['redis'] = True
            print(f"    Redis (port {self.redis_port}): ✅ Available")
        except Exception as e:
            services['redis'] = False
            print(f"    Redis (port {self.redis_port}): ❌ Unavailable - {e}")
            self.error_manager.track_error(e, 'cache_manager', {'service': 'redis', 'check': 'availability'})
        
        # Check ChromaDB using v2 API (proper client method)
        try:
            import chromadb
            # Use ChromaDB v2 client like in src folder
            client = chromadb.HttpClient(host=self.chromadb_host, port=self.chromadb_port)
            # Test connection by listing collections (v2 API method)
            collections = client.list_collections()
            services['chromadb'] = True
            print(f"    ChromaDB (port {self.chromadb_port}): ✅ Available (v2 API, {len(collections)} collections)")
        except Exception as e:
            services['chromadb'] = False
            print(f"    ChromaDB (port {self.chromadb_port}): ❌ Unavailable - {e}")
            self.error_manager.track_error(e, 'vector_store', {'service': 'chromadb', 'check': 'availability'})
        
        # Check Ollama
        try:
            response = requests.get(f"http://{self.ollama_host}:{self.ollama_port}/api/tags", timeout=10)
            services['ollama'] = response.status_code == 200
            print(f"    Ollama (port {self.ollama_port}): {'✅ Available' if services['ollama'] else '❌ Unavailable'}")
        except Exception as e:
            services['ollama'] = False
            print(f"    Ollama (port {self.ollama_port}): ❌ Unavailable - {e}")
            self.error_manager.track_error(e, 'llm_client', {'service': 'ollama', 'check': 'availability'})
        
        print()
        return services
    
    async def monitor_rag_query(self, query: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Monitor a single RAG query and capture errors/performance."""
        start_time = time.time()
        
        try:
            # Make RAG API call
            async with session.post(
                f"{self.rag_api_base}/query",
                json={"query": query},
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                response_time = time.time() - start_time
                
                # Check response status
                if response.status == 200:
                    response_data = await response.json()
                    
                    # Track successful API call
                    api_monitor = self.error_manager.get_monitor('api_endpoints')
                    if api_monitor:
                        api_monitor.track_request(
                            endpoint="/query",
                            method="POST",
                            status_code=200,
                            response_time=response_time,
                            request_size=len(query.encode('utf-8')),
                            user_agent="LiveRAGMonitor/1.0"
                        )
                    
                    return {
                        'query': query,
                        'status': 'success',
                        'response_time': response_time,
                        'response_data': response_data,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    # Handle HTTP errors
                    error_text = await response.text()
                    error = Exception(f"HTTP {response.status}: {error_text}")
                    
                    # Track failed API call
                    api_monitor = self.error_manager.get_monitor('api_endpoints')
                    if api_monitor:
                        api_monitor.track_request(
                            endpoint="/query",
                            method="POST",
                            status_code=response.status,
                            response_time=response_time,
                            error_details={'http_error': error_text}
                        )
                    
                    # Track error in error analysis
                    self.error_manager.track_error(error, 'api_endpoints', {
                        'query': query,
                        'status_code': response.status,
                        'response_time': response_time
                    })
                    
                    return {
                        'query': query,
                        'status': 'error',
                        'response_time': response_time,
                        'error': str(error),
                        'status_code': response.status,
                        'timestamp': datetime.now().isoformat()
                    }
                    
        except asyncio.TimeoutError as e:
            response_time = time.time() - start_time
            
            # Track timeout error
            self.error_manager.track_error(e, 'api_endpoints', {
                'query': query,
                'error_type': 'timeout',
                'response_time': response_time
            })
            
            return {
                'query': query,
                'status': 'timeout',
                'response_time': response_time,
                'error': 'Request timeout',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # Track general error
            self.error_manager.track_error(e, 'api_endpoints', {
                'query': query,
                'error_type': type(e).__name__,
                'response_time': response_time
            })
            
            return {
                'query': query,
                'status': 'error',
                'response_time': response_time,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def monitor_document_ingestion(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Monitor document ingestion endpoint."""
        print("📄 Testing document ingestion...")
        
        # Test document upload (if endpoint exists)
        try:
            # Check if ingestion endpoint exists
            async with session.get(f"{self.rag_api_base}/docs", timeout=120) as response:
                if response.status == 200:
                    return {'ingestion_status': 'available', 'error': None}
                else:
                    error = f"Ingestion endpoint returned {response.status}"
                    self.error_manager.track_error(Exception(error), 'document_processor', {
                        'endpoint': '/docs',
                        'status_code': response.status
                    })
                    return {'ingestion_status': 'error', 'error': error}
                    
        except Exception as e:
            self.error_manager.track_error(e, 'document_processor', {
                'endpoint': '/docs',
                'operation': 'check_availability'
            })
            return {'ingestion_status': 'unavailable', 'error': str(e)}
    
    def monitor_redis_operations(self) -> Dict[str, Any]:
        """Monitor Redis cache operations."""
        print("🗄️  Testing Redis cache operations...")
        
        cache_monitor = self.error_manager.get_monitor('cache_manager')
        results = []
        
        try:
            r = redis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=True)
            
            # Test SET operation
            start_time = time.time()
            test_key = f"monitor_test_{int(time.time())}"
            test_value = "test_value_for_monitoring"
            
            r.set(test_key, test_value, ex=60)  # Expire in 60 seconds
            set_time = time.time() - start_time
            
            if cache_monitor:
                cache_monitor.track_cache_operation(
                    operation='sets',
                    success=True,
                    operation_time=set_time
                )
            
            results.append({
                'operation': 'set',
                'success': True,
                'time': set_time,
                'key': test_key
            })
            
            # Test GET operation
            start_time = time.time()
            retrieved_value = r.get(test_key)
            get_time = time.time() - start_time
            
            hit = retrieved_value is not None
            if cache_monitor:
                cache_monitor.track_cache_operation(
                    operation='gets',
                    success=True,
                    operation_time=get_time,
                    hit=hit
                )
            
            results.append({
                'operation': 'get',
                'success': True,
                'time': get_time,
                'hit': hit,
                'value_match': retrieved_value == test_value
            })
            
            # Test DELETE operation
            start_time = time.time()
            r.delete(test_key)
            delete_time = time.time() - start_time
            
            if cache_monitor:
                cache_monitor.track_cache_operation(
                    operation='deletes',
                    success=True,
                    operation_time=delete_time
                )
            
            results.append({
                'operation': 'delete',
                'success': True,
                'time': delete_time
            })
            
            return {'redis_operations': results, 'overall_status': 'success'}
            
        except Exception as e:
            self.error_manager.track_error(e, 'cache_manager', {
                'operation': 'redis_monitoring',
                'error_type': type(e).__name__
            })
            return {'redis_operations': results, 'overall_status': 'error', 'error': str(e)}
    
    def monitor_chromadb_operations(self) -> Dict[str, Any]:
        """Monitor ChromaDB vector operations."""
        print("🔍 Testing ChromaDB vector operations...")
        
        vector_monitor = self.error_manager.get_monitor('vector_store')
        
        try:
            import chromadb
            import time
            
            # Test ChromaDB connection and operations using proper v2 API
            start_time = time.time()
            client = chromadb.HttpClient(host=self.chromadb_host, port=self.chromadb_port)
            
            # List collections using v2 API
            collections = client.list_collections()
            operation_time = time.time() - start_time
            
            # Track successful vector store operation
            if vector_monitor:
                vector_monitor.track_query(
                    query_type='list_collections',
                    query_time=operation_time,
                    result_count=len(collections),
                    success=True
                )
            
            # Try to get collection details if any exist
            collection_details = []
            for collection in collections[:3]:  # Check first 3 collections
                try:
                    count = collection.count()
                    collection_details.append({
                        'name': collection.name,
                        'count': count
                    })
                except Exception:
                    # Skip if count fails
                    collection_details.append({
                        'name': collection.name,
                        'count': 'unknown'
                    })
            
            return {
                'chromadb_status': 'success',
                'collections_count': len(collections),
                'collection_details': collection_details,
                'response_time': operation_time,
                'api_version': 'v2'
            }
                
        except Exception as e:
            self.error_manager.track_error(e, 'vector_store', {
                'operation': 'chromadb_monitoring'
            })
            return {'chromadb_status': 'error', 'error': str(e)}
    
    def monitor_ollama_operations(self) -> Dict[str, Any]:
        """Monitor Ollama LLM operations."""
        print("🤖 Testing Ollama LLM operations...")
        
        llm_monitor = self.error_manager.get_monitor('llm_client')
        
        try:
            # Check available models
            start_time = time.time()
            response = requests.get(f"http://{self.ollama_host}:{self.ollama_port}/api/tags", timeout=10)
            model_check_time = time.time() - start_time
            
            if response.status_code == 200:
                models = response.json()
                
                # Track successful LLM operation
                if llm_monitor:
                    llm_monitor.track_model_loading(
                        model_name="model_check",
                        loading_time=model_check_time,
                        success=True
                    )
                
                return {
                    'ollama_status': 'success',
                    'available_models': len(models.get('models', [])),
                    'model_check_time': model_check_time,
                    'models': [model.get('name', 'unknown') for model in models.get('models', [])]
                }
            else:
                error = f"Ollama returned status {response.status_code}"
                self.error_manager.track_error(Exception(error), 'llm_client', {
                    'operation': 'list_models',
                    'status_code': response.status_code
                })
                return {'ollama_status': 'error', 'error': error}
                
        except Exception as e:
            self.error_manager.track_error(e, 'llm_client', {
                'operation': 'ollama_monitoring'
            })
            return {'ollama_status': 'error', 'error': str(e)}
    
    async def run_comprehensive_monitoring(self, duration_minutes: int = 5, query_interval: float = 2.0):
        """Run comprehensive monitoring of the live RAG system."""
        print("\n" + "=" * 80)
        print("🚀 LIVE RAG SYSTEM MONITORING")
        print("=" * 80)
        print(f"⏱️  Duration: {duration_minutes} minutes")
        print(f"🔄 Query interval: {query_interval} seconds")
        print()
        
        self.monitoring_results['start_time'] = datetime.now().isoformat()
        
        # Check service availability first
        services = await self.check_service_availability()
        unavailable_services = [service for service, available in services.items() if not available]
        
        if unavailable_services:
            print(f"⚠️  Warning: Some services are unavailable: {', '.join(unavailable_services)}")
            print("Continuing with available services...")
            print()
        
        # Monitor individual components
        print("🔧 COMPONENT MONITORING")
        print("=" * 50)
        
        # Monitor Redis if available
        if services.get('redis', False):
            redis_results = self.monitor_redis_operations()
            self.monitoring_results['component_health']['redis'] = redis_results
        
        # Monitor ChromaDB if available
        if services.get('chromadb', False):
            chromadb_results = self.monitor_chromadb_operations()
            self.monitoring_results['component_health']['chromadb'] = chromadb_results
        
        # Monitor Ollama if available
        if services.get('ollama', False):
            ollama_results = self.monitor_ollama_operations()
            self.monitoring_results['component_health']['ollama'] = ollama_results
        
        print()
        
        # Monitor RAG API calls if available
        if services.get('rag_api', False):
            print("📡 RAG API MONITORING")
            print("=" * 50)
            print(f"Testing {len(self.test_queries)} queries over {duration_minutes} minutes...")
            print()
            
            end_time = time.time() + (duration_minutes * 60)
            query_count = 0
            
            async with aiohttp.ClientSession() as session:
                # Monitor document ingestion
                ingestion_result = await self.monitor_document_ingestion(session)
                self.monitoring_results['component_health']['document_ingestion'] = ingestion_result
                
                # Monitor queries
                while time.time() < end_time:
                    query = self.test_queries[query_count % len(self.test_queries)]
                    
                    print(f"🔍 Query {query_count + 1}: {query[:50]}{'...' if len(query) > 50 else ''}")
                    
                    result = await self.monitor_rag_query(query, session)
                    self.monitoring_results['api_calls'].append(result)
                    
                    # Log result
                    if result['status'] == 'success':
                        print(f"    ✅ Success ({result['response_time']:.2f}s)")
                    elif result['status'] == 'timeout':
                        print(f"    ⏰ Timeout ({result['response_time']:.2f}s)")
                        self.monitoring_results['errors'].append(result)
                    else:
                        print(f"    ❌ Error: {result.get('error', 'Unknown error')}")
                        self.monitoring_results['errors'].append(result)
                    
                    query_count += 1
                    
                    # Wait before next query
                    await asyncio.sleep(query_interval)
        else:
            print("❌ RAG API unavailable - skipping API monitoring")
        
        self.monitoring_results['end_time'] = datetime.now().isoformat()
        
        # Generate final report
        await self.generate_monitoring_report()
    
    async def generate_monitoring_report(self):
        """Generate comprehensive monitoring report."""
        print("\n" + "=" * 80)
        print("📊 MONITORING REPORT")
        print("=" * 80)
        
        # Summary statistics
        total_calls = len(self.monitoring_results['api_calls'])
        successful_calls = len([call for call in self.monitoring_results['api_calls'] if call['status'] == 'success'])
        error_calls = len(self.monitoring_results['errors'])
        
        print("📈 API Call Summary")
        print(f"    Total API Calls: {total_calls}")
        print(f"    Successful Calls: {successful_calls}")
        print(f"    Failed Calls: {error_calls}")
        
        if total_calls > 0:
            success_rate = (successful_calls / total_calls) * 100
            print(f"    Success Rate: {success_rate:.1f}%")
            
            # Response time statistics
            response_times = [call['response_time'] for call in self.monitoring_results['api_calls']]
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                max_response_time = max(response_times)
                min_response_time = min(response_times)
                
                print(f"    Avg Response Time: {avg_response_time:.2f}s")
                print(f"    Max Response Time: {max_response_time:.2f}s")
                print(f"    Min Response Time: {min_response_time:.2f}s")
        
        print()
        
        # Component health summary
        print("🏥 Component Health Summary")
        for component, health in self.monitoring_results['component_health'].items():
            status = health.get('overall_status', health.get('chromadb_status', health.get('ollama_status', 'unknown')))
            print(f"    {component.title()}: {status.upper()}")
        
        print()
        
        # Error analysis
        if self.monitoring_results['errors']:
            print("🚨 Error Analysis")
            error_types = {}
            for error in self.monitoring_results['errors']:
                error_type = error.get('error', 'Unknown')[:50]
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            for error_type, count in error_types.items():
                print(f"    {error_type}: {count} occurrences")
        else:
            print("✅ No errors detected during monitoring")
        
        print()
        
        # Get comprehensive error analysis
        print("📋 COMPREHENSIVE ERROR ANALYSIS")
        print("=" * 50)
        
        analysis = self.error_manager.get_comprehensive_analysis(hours=1)
        health_status = self.error_manager.get_system_health_status()
        
        print(f"📊 Error Statistics (Last Hour)")
        print(f"    Total Errors: {analysis['error_analysis']['total_errors']}")
        print(f"    Critical Errors: {analysis['error_analysis']['critical_errors']}")
        print(f"    Error Rate: {analysis['error_analysis']['error_rate_per_hour']:.2f}/hour")
        
        print(f"\n🏥 System Health")
        print(f"    Overall Status: {health_status['overall_status'].upper()}")
        print(f"    Components Monitored: {len(health_status['component_health'])}")
        
        # Recommendations
        if analysis.get('recommendations'):
            print(f"\n💡 Recommendations")
            for i, rec in enumerate(analysis['recommendations'][:5], 1):
                print(f"    [{i}] {rec}")
        
        print("\n" + "=" * 80)
        print("✅ LIVE MONITORING COMPLETED")
        print("=" * 80)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"error_analysis/results/live_monitoring_{timestamp}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(self.monitoring_results, f, indent=2, default=str)
            print(f"📁 Results saved to: {results_file}")
        except Exception as e:
            print(f"⚠️  Failed to save results: {e}")


async def main():
    """Main entry point for live RAG monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor live RAG system')
    parser.add_argument('--duration', type=int, default=5, help='Monitoring duration in minutes (default: 5)')
    parser.add_argument('--interval', type=float, default=2.0, help='Query interval in seconds (default: 2.0)')
    parser.add_argument('--config', type=str, help='Error analysis config file path')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f).get('error_analysis', {})
        except Exception as e:
            print(f"⚠️  Failed to load config: {e}")
    
    # Create and run monitor
    monitor = LiveRAGMonitor(config)
    
    try:
        await monitor.run_comprehensive_monitoring(
            duration_minutes=args.duration,
            query_interval=args.interval
        )
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitoring failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())