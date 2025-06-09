from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str
    use_rag: bool = True

class QueryResponse(BaseModel):
    query: str
    response: str
    use_rag: bool

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 3

class RetrieveResult(BaseModel):
    id: str
    score: float
    text: str
    context: str

class RetrieveResponse(BaseModel):
    query: str
    results: List[RetrieveResult]
    count: int

class UploadResponse(BaseModel):
    message: str
    filename: str
    steps_completed: List[str]
    status: str

class HealthResponse(BaseModel):
    status: str
    message: str