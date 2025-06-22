"""
Error Analysis Integration Example

Demonstrates how to integrate the error analysis system with existing RAG components.
"""

import json
from pathlib import Path
from .error_analysis_manager import get_error_analysis_manager


def setup_error_analysis_for_rag_system():
    """
    Set up error analysis for the RAG system.
    
    Returns:
        Configured ErrorAnalysisManager instance
    """
    # Load configuration
    config_path = Path(__file__).parent / "config" / "error_analysis_config.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Config file not found at {config_path}, using defaults")
        config = {}
    
    # Initialize error analysis manager
    error_manager = get_error_analysis_manager(config.get('error_analysis', {}))
    
    return error_manager


def integrate_with_api_endpoints(app, error_manager):
    """
    Example of integrating error analysis with FastAPI endpoints.
    
    Args:
        app: FastAPI application instance
        error_manager: ErrorAnalysisManager instance
    """
    from fastapi import Request, HTTPException
    import time
    
    # Get API monitor
    api_monitor = error_manager.get_monitor('api_endpoints')
    
    @app.middleware("http")
    async def error_analysis_middleware(request: Request, call_next):
        """Middleware to track API requests and errors."""
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Track successful request
            if api_monitor:
                api_monitor.track_request(
                    endpoint=str(request.url.path),
                    method=request.method,
                    status_code=response.status_code,
                    response_time=response_time,
                    request_size=int(request.headers.get('content-length', 0)),
                    user_agent=request.headers.get('user-agent')
                )
            
            return response
            
        except Exception as e:
            # Calculate response time for failed request
            response_time = time.time() - start_time
            
            # Determine status code
            if isinstance(e, HTTPException):
                status_code = e.status_code
            else:
                status_code = 500
            
            # Track failed request
            if api_monitor:
                api_monitor.track_request(
                    endpoint=str(request.url.path),
                    method=request.method,
                    status_code=status_code,
                    response_time=response_time,
                    error_details={'exception': str(e), 'type': type(e).__name__}
                )
            
            # Track error in error analysis system
            error_manager.track_error(e, 'api_endpoints', {
                'endpoint': str(request.url.path),
                'method': request.method,
                'response_time': response_time
            })
            
            # Re-raise the exception
            raise


def integrate_with_llm_client(llm_client, error_manager):
    """
    Example of integrating error analysis with LLM client.
    
    Args:
        llm_client: LLM client instance
        error_manager: ErrorAnalysisManager instance
    """
    # Get LLM monitor
    llm_monitor = error_manager.get_monitor('llm_client')
    
    # Add monitoring decorator to LLM methods
    if llm_monitor and hasattr(llm_client, 'query'):
        original_query = llm_client.query
        
        @llm_monitor.monitor_function('llm_query')
        def monitored_query(*args, **kwargs):
            return original_query(*args, **kwargs)
        
        llm_client.query = monitored_query
    
    # Add specific LLM tracking if available
    if hasattr(llm_monitor, 'track_prompt_processing'):
        def track_llm_call(prompt, response, processing_time, success, error_details=None):
            llm_monitor.track_prompt_processing(
                prompt=prompt,
                response=response,
                processing_time=processing_time,
                success=success,
                error_details=error_details
            )
        
        # This would be called from within the LLM client
        llm_client.track_call = track_llm_call


def integrate_with_cache_manager(cache_manager, error_manager):
    """
    Example of integrating error analysis with cache manager.
    
    Args:
        cache_manager: Cache manager instance
        error_manager: ErrorAnalysisManager instance
    """
    # Get cache monitor
    cache_monitor = error_manager.get_monitor('cache_manager')
    
    if cache_monitor and hasattr(cache_monitor, 'track_cache_operation'):
        # Wrap cache operations
        if hasattr(cache_manager, 'get'):
            original_get = cache_manager.get
            
            def monitored_get(key):
                import time
                start_time = time.time()
                
                try:
                    result = original_get(key)
                    operation_time = time.time() - start_time
                    
                    # Track cache get operation
                    cache_monitor.track_cache_operation(
                        operation='gets',
                        success=True,
                        operation_time=operation_time,
                        cache_key=key,
                        hit=result is not None
                    )
                    
                    return result
                    
                except Exception as e:
                    operation_time = time.time() - start_time
                    
                    cache_monitor.track_cache_operation(
                        operation='gets',
                        success=False,
                        operation_time=operation_time,
                        cache_key=key,
                        error_details={'exception': str(e)}
                    )
                    
                    # Track error
                    error_manager.track_error(e, 'cache_manager', {'operation': 'get', 'key': key})
                    raise
            
            cache_manager.get = monitored_get


def add_health_check_endpoint(app, error_manager):
    """
    Add error analysis health check endpoint to FastAPI app.
    
    Args:
        app: FastAPI application instance
        error_manager: ErrorAnalysisManager instance
    """
    @app.get("/health/error-analysis")
    async def error_analysis_health():
        """Get error analysis system health status."""
        try:
            health_status = error_manager.get_system_health_status()
            return {
                "status": "success",
                "data": health_status
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    @app.get("/health/error-analysis/comprehensive")
    async def comprehensive_error_analysis():
        """Get comprehensive error analysis (last 24 hours)."""
        try:
            analysis = error_manager.get_comprehensive_analysis(hours=24)
            return {
                "status": "success",
                "data": analysis
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }


# Example usage function
def setup_complete_error_analysis_integration(app, llm_client, cache_manager):
    """
    Complete example of setting up error analysis integration.
    
    Args:
        app: FastAPI application
        llm_client: LLM client instance
        cache_manager: Cache manager instance
    """
    # Set up error analysis
    error_manager = setup_error_analysis_for_rag_system()
    
    # Integrate with components
    integrate_with_api_endpoints(app, error_manager)
    integrate_with_llm_client(llm_client, error_manager)
    integrate_with_cache_manager(cache_manager, error_manager)
    
    # Add health check endpoints
    add_health_check_endpoint(app, error_manager)
    
    print("\n" + "=" * 60)
    print("✅ ERROR ANALYSIS INTEGRATION COMPLETE")
    print("=" * 60)
    print("\n📡 Available API Endpoints")
    print("-" * 30)
    print("    GET /health/error-analysis")
    print("        → System health status")
    print()
    print("    GET /health/error-analysis/comprehensive")
    print("        → Detailed error analysis (24 hours)")
    print("\n" + "=" * 60)
    
    return error_manager