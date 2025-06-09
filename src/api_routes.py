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
    
    return router