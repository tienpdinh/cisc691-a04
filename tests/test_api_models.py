import pytest
from pydantic import ValidationError
from src.api_models import (
    QueryRequest, QueryResponse, RetrieveRequest, RetrieveResponse, 
    RetrieveResult, UploadResponse, HealthResponse
)

class TestQueryRequest:
    def test_valid_query_request(self):
        request = QueryRequest(query="What is AI?", use_rag=True)
        assert request.query == "What is AI?"
        assert request.use_rag is True

    def test_query_request_default_use_rag(self):
        request = QueryRequest(query="What is AI?")
        assert request.use_rag is True

    def test_missing_query_fails(self):
        with pytest.raises(ValidationError):
            QueryRequest(use_rag=True)

class TestQueryResponse:
    def test_valid_query_response(self):
        response = QueryResponse(query="What is AI?", response="AI is...", use_rag=True)
        assert response.query == "What is AI?"
        assert response.response == "AI is..."
        assert response.use_rag is True

class TestRetrieveRequest:
    def test_valid_retrieve_request(self):
        request = RetrieveRequest(query="test query", top_k=5)
        assert request.query == "test query"
        assert request.top_k == 5

    def test_default_top_k(self):
        request = RetrieveRequest(query="test query")
        assert request.top_k == 3

class TestRetrieveResult:
    def test_valid_retrieve_result(self):
        result = RetrieveResult(
            id="doc_1",
            score=0.85,
            text="Sample text content",
            context="Sample context"
        )
        assert result.id == "doc_1"
        assert result.score == 0.85
        assert result.text == "Sample text content"
        assert result.context == "Sample context"

class TestRetrieveResponse:
    def test_valid_retrieve_response(self):
        results = [
            RetrieveResult(id="doc_1", score=0.85, text="Text 1", context="Context 1"),
            RetrieveResult(id="doc_2", score=0.92, text="Text 2", context="Context 2")
        ]
        response = RetrieveResponse(query="test", results=results, count=2)
        assert response.query == "test"
        assert len(response.results) == 2
        assert response.count == 2

    def test_empty_results(self):
        response = RetrieveResponse(query="test", results=[], count=0)
        assert len(response.results) == 0
        assert response.count == 0

class TestUploadResponse:
    def test_valid_upload_response(self):
        response = UploadResponse(
            message="Success",
            filename="test.pdf",
            steps_completed=["ingest", "embed", "store"],
            status="success"
        )
        assert response.message == "Success"
        assert response.filename == "test.pdf"
        assert response.steps_completed == ["ingest", "embed", "store"]
        assert response.status == "success"

class TestHealthResponse:
    def test_valid_health_response(self):
        response = HealthResponse(status="healthy", message="API is running")
        assert response.status == "healthy"
        assert response.message == "API is running"