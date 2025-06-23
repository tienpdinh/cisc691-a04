from fastapi import APIRouter, UploadFile, File, Request, Depends
from .api_models import QueryRequest, QueryResponse, UploadResponse, HealthResponse, RetrieveRequest, RetrieveResponse
from .api_endpoints import query_rag, upload_document, health_check, retrieve_chunks

def create_router() -> APIRouter:
    """Create API router with all endpoints."""
    router = APIRouter()
    
    @router.post("/query", response_model=QueryResponse)
    async def query_endpoint(request: QueryRequest, app_request: Request):
        return await query_rag(request, app_request)
    
    @router.post("/upload-document", response_model=UploadResponse)
    async def upload_endpoint(file: UploadFile = File(...), app_request: Request = None):
        return await upload_document(file, app_request)
    
    @router.post("/retrieve", response_model=RetrieveResponse)
    async def retrieve_endpoint(request: RetrieveRequest, app_request: Request):
        return await retrieve_chunks(request, app_request)
    
    @router.get("/health", response_model=HealthResponse)
    async def health_endpoint():
        return await health_check()
    
    @router.get("/performance")
    async def performance_endpoint(app_request: Request):
        """Get current performance metrics."""
        optimizer = app_request.app.state.performance_optimizer
        metrics = await optimizer.get_comprehensive_metrics()
        return {
            "performance_score": metrics.performance_score,
            "memory_usage_mb": metrics.memory_usage_mb,
            "cpu_usage_percent": metrics.cpu_usage_percent,
            "cache_hit_rate": metrics.cache_hit_rate,
            "active_connections": metrics.active_connections,
            "avg_response_time": metrics.avg_response_time,
            "optimization_suggestions": metrics.optimization_suggestions,
            "timestamp": metrics.timestamp
        }
    
    @router.post("/performance/optimize")
    async def optimize_endpoint(app_request: Request):
        """Trigger performance optimization."""
        optimizer = app_request.app.state.performance_optimizer
        results = await optimizer.optimize_all()
        return {
            "message": "Performance optimization completed",
            "results": results
        }
    
    return router