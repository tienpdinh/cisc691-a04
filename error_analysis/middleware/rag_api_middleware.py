"""
RAG API Middleware for Error Analysis Integration

Middleware to integrate error analysis monitoring with the RAG API endpoints.
Automatically captures errors, tracks performance, and monitors component health.
"""

from __future__ import annotations

import time
import json
import traceback
from typing import Dict, Any, Optional
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.error_analysis_manager import get_error_analysis_manager


class RAGErrorAnalysisMiddleware(BaseHTTPMiddleware):
    """
    Middleware to integrate error analysis with RAG API.
    
    Automatically tracks:
    - API request/response metrics
    - Error occurrences and classifications
    - Component performance
    - System health status
    """
    
    def __init__(self, app, config: Optional[Dict] = None):
        """Initialize middleware with error analysis manager."""
        super().__init__(app)
        self.error_manager = get_error_analysis_manager(config or {})
        self.api_monitor = self.error_manager.get_monitor('api_endpoints')
        
        # Track middleware initialization
        print("✅ RAG Error Analysis Middleware initialized")
    
    async def dispatch(self, request: Request, call_next):
        """Process request and track errors/performance."""
        start_time = time.time()
        
        # Extract request information
        endpoint = str(request.url.path)
        method = request.method
        request_size = int(request.headers.get('content-length', 0))
        user_agent = request.headers.get('user-agent', 'Unknown')
        
        # Read request body for context (if it's JSON)
        request_context = {}
        try:
            if request.headers.get('content-type') == 'application/json':
                body = await request.body()
                if body:
                    request_context = json.loads(body.decode('utf-8'))
        except Exception:
            # Ignore request body parsing errors
            pass
        
        # Create a new request object with the body
        async def receive():
            return {"type": "http.request", "body": await request.body()}
        
        # Set up the request for processing
        request._receive = receive
        
        try:
            # Process the request
            response = await call_next(request)
            response_time = time.time() - start_time
            
            # Track successful request
            if self.api_monitor:
                self.api_monitor.track_request(
                    endpoint=endpoint,
                    method=method,
                    status_code=response.status_code,
                    response_time=response_time,
                    request_size=request_size,
                    user_agent=user_agent
                )
            
            # Add performance headers
            response.headers["X-Response-Time"] = f"{response_time:.3f}s"
            response.headers["X-Error-Analysis"] = "monitored"
            
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # Determine status code
            status_code = getattr(e, 'status_code', 500)
            
            # Track failed request
            if self.api_monitor:
                self.api_monitor.track_request(
                    endpoint=endpoint,
                    method=method,
                    status_code=status_code,
                    response_time=response_time,
                    request_size=request_size,
                    user_agent=user_agent,
                    error_details={
                        'exception': str(e),
                        'type': type(e).__name__,
                        'traceback': traceback.format_exc()
                    }
                )
            
            # Track error in error analysis system
            error_context = {
                'endpoint': endpoint,
                'method': method,
                'response_time': response_time,
                'request_context': request_context,
                'user_agent': user_agent
            }
            
            self.error_manager.track_error(e, 'api_endpoints', error_context)
            
            # Return appropriate error response
            return JSONResponse(
                status_code=status_code,
                content={
                    "error": str(e),
                    "detail": "An error occurred while processing your request",
                    "error_id": getattr(e, 'error_id', None),
                    "timestamp": time.time()
                },
                headers={
                    "X-Response-Time": f"{response_time:.3f}s",
                    "X-Error-Analysis": "error-tracked"
                }
            )


def add_rag_monitoring_decorators(app, error_manager):
    """
    Add monitoring decorators to RAG components.
    
    This function should be called during app initialization to set up
    monitoring for various RAG components.
    """
    
    # Get component monitors
    llm_monitor = error_manager.get_monitor('llm_client')
    vector_monitor = error_manager.get_monitor('vector_store')
    cache_monitor = error_manager.get_monitor('cache_manager')
    doc_monitor = error_manager.get_monitor('document_processor')
    
    # LLM Client Monitoring
    def monitor_llm_call(original_func):
        """Decorator to monitor LLM calls."""
        if llm_monitor:
            return llm_monitor.monitor_function('llm_query')(original_func)
        return original_func
    
    # Vector Store Monitoring
    def monitor_vector_search(original_func):
        """Decorator to monitor vector search operations."""
        if vector_monitor:
            return vector_monitor.monitor_function('vector_search')(original_func)
        return original_func
    
    # Cache Monitoring
    def monitor_cache_operation(original_func):
        """Decorator to monitor cache operations."""
        if cache_monitor:
            return cache_monitor.monitor_function('cache_operation')(original_func)
        return original_func
    
    # Document Processing Monitoring
    def monitor_document_processing(original_func):
        """Decorator to monitor document processing."""
        if doc_monitor:
            return doc_monitor.monitor_function('document_processing')(original_func)
        return original_func
    
    # Store decorators in app state for use in route handlers
    app.state.error_analysis = {
        'manager': error_manager,
        'decorators': {
            'llm': monitor_llm_call,
            'vector': monitor_vector_search,
            'cache': monitor_cache_operation,
            'document': monitor_document_processing
        }
    }
    
    print("✅ RAG monitoring decorators added to app state")


def track_component_error(app, component: str, error: Exception, context: Dict[str, Any] = None):
    """
    Helper function to manually track component errors.
    
    Args:
        app: FastAPI app instance
        component: Component name (llm_client, vector_store, cache_manager, etc.)
        error: Exception that occurred
        context: Additional context information
    """
    if hasattr(app.state, 'error_analysis'):
        error_manager = app.state.error_analysis['manager']
        error_manager.track_error(error, component, context or {})


def get_system_health(app) -> Dict[str, Any]:
    """
    Get current system health status.
    
    Args:
        app: FastAPI app instance
        
    Returns:
        System health status dictionary
    """
    if hasattr(app.state, 'error_analysis'):
        error_manager = app.state.error_analysis['manager']
        return error_manager.get_system_health_status()
    
    return {'error': 'Error analysis not initialized'}


def get_error_analysis_report(app, hours: int = 24) -> Dict[str, Any]:
    """
    Get comprehensive error analysis report.
    
    Args:
        app: FastAPI app instance
        hours: Number of hours to analyze
        
    Returns:
        Comprehensive analysis report
    """
    if hasattr(app.state, 'error_analysis'):
        error_manager = app.state.error_analysis['manager']
        return error_manager.get_comprehensive_analysis(hours=hours)
    
    return {'error': 'Error analysis not initialized'}


# Example integration code for your RAG API
def integrate_error_analysis_with_rag_api(app):
    """
    Complete integration example for RAG API.
    
    Add this to your main.py or wherever you initialize your FastAPI app.
    """
    
    # Initialize error analysis configuration
    error_config = {
        'enabled': True,
        'log_directory': 'logs/errors',
        'max_memory_entries': 1000,
        'alert_thresholds': {
            'error_rate_per_minute': 5,
            'avg_response_time_seconds': 10.0,
            'success_rate_threshold': 0.90
        }
    }
    
    # Add middleware
    app.add_middleware(RAGErrorAnalysisMiddleware, config=error_config)
    
    # Add monitoring decorators
    error_manager = get_error_analysis_manager(error_config)
    add_rag_monitoring_decorators(app, error_manager)
    
    # Add health check endpoints
    @app.get("/health/error-analysis")
    async def error_analysis_health():
        """Get error analysis system health."""
        return get_system_health(app)
    
    @app.get("/health/error-analysis/comprehensive")
    async def comprehensive_error_analysis(hours: int = 24):
        """Get comprehensive error analysis."""
        return get_error_analysis_report(app, hours)
    
    @app.get("/health/error-analysis/component/{component_name}")
    async def component_health(component_name: str):
        """Get health status for specific component."""
        if hasattr(app.state, 'error_analysis'):
            error_manager = app.state.error_analysis['manager']
            monitor = error_manager.get_monitor(component_name)
            if monitor:
                return monitor.get_health_status()
        return {"error": "Component not found or error analysis not initialized"}
    
    print("✅ Complete RAG API error analysis integration ready")
    return app


# Usage example for route handlers
"""
Example of how to use in your RAG API route handlers:

@app.post("/query")
async def rag_query(request: QueryRequest):
    # Get monitoring decorators from app state
    decorators = request.app.state.error_analysis['decorators']
    
    try:
        # Monitor LLM call
        @decorators['llm']
        def call_llm(prompt):
            # Your LLM call here
            return ollama_client.generate(prompt)
        
        # Monitor vector search
        @decorators['vector']
        def search_vectors(query_embedding):
            # Your vector search here
            return chroma_client.query(query_embeddings=[query_embedding])
        
        # Monitor cache operations
        @decorators['cache']
        def cache_get(key):
            # Your cache get here
            return redis_client.get(key)
        
        # Use decorated functions
        response = call_llm(prompt)
        results = search_vectors(embedding)
        cached_data = cache_get(cache_key)
        
        return {"response": response}
        
    except Exception as e:
        # Error is automatically tracked by middleware
        # But you can add additional context if needed
        track_component_error(request.app, 'rag_pipeline', e, {
            'query': request.query,
            'additional_context': 'any extra info'
        })
        raise
"""