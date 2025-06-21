import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
import io
from src.api_app import create_app
from src.api_routes import create_router
from src.config_manager import ConfigManager

@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    config = Mock(spec=ConfigManager)
    config.get.side_effect = lambda key, default=None: {
        "embedding_model_name": "test-model",
        "collection_name": "test_collection",
        "retriever_min_score_threshold": "0.5",
        "raw_input_directory": "/tmp/raw_input",
        "cleaned_text_directory": "/tmp/cleaned_text",
        "embeddings_directory": "/tmp/embeddings",
        "chromadb_host": "localhost",
        "chromadb_port": 8000,
        "llm_api_url": "http://test-llm",
        "llm_model_name": "test-llm-model",
        "llm_provider": "ollama",
        "project_id": None,
        "location": None
    }.get(key, default)
    return config

@pytest.fixture
def app_with_routes(mock_config):
    """Create FastAPI app with routes for testing."""
    app = create_app(mock_config)
    router = create_router()
    app.include_router(router)
    return app

@pytest.fixture
def client(app_with_routes):
    """Create test client."""
    return TestClient(app_with_routes)

class TestHealthEndpoint:
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["message"] == "LangChain RAG API is running"

class TestQueryEndpoint:
    @patch('src.api_endpoints._get_or_create_rag_processor')
    def test_query_success(self, mock_get_processor, client):
        """Test successful query."""
        # Mock the RAG processor
        mock_processor = Mock()
        mock_processor.query = AsyncMock(return_value={"response": "Test response"})
        mock_get_processor.return_value = mock_processor
        
        response = client.post("/query", json={
            "query": "What is AI?",
            "use_rag": True
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is AI?"
        assert data["response"] == "Test response"
        assert data["use_rag"] is True

    def test_query_invalid_request(self, client):
        """Test query with invalid request."""
        response = client.post("/query", json={})
        assert response.status_code == 422  # Validation error

class TestRetrieveEndpoint:
    @patch('src.api_endpoints._get_or_create_rag_processor')
    def test_retrieve_success(self, mock_get_processor, client):
        """Test successful retrieve."""
        # Mock the RAG processor
        mock_processor = Mock()
        mock_processor.retrieve_documents = AsyncMock(return_value=[
            {
                "content": "Sample text",
                "metadata": {"source": "doc_1"},
                "score": 0.85
            }
        ])
        mock_get_processor.return_value = mock_processor
        
        response = client.post("/retrieve", json={
            "query": "test query",
            "top_k": 3
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert data["count"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["id"] == "doc_1"

    @patch('src.api_endpoints._get_or_create_rag_processor')
    def test_retrieve_no_results(self, mock_get_processor, client):
        """Test retrieve with no results."""
        mock_processor = Mock()
        mock_processor.retrieve_documents = AsyncMock(return_value=[])
        mock_get_processor.return_value = mock_processor
        
        response = client.post("/retrieve", json={
            "query": "nonexistent query"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert len(data["results"]) == 0

class TestUploadEndpoint:
    @patch('src.api_endpoints.LangChainDocumentProcessor')
    @patch('src.api_endpoints._get_or_create_vector_store')
    @patch('src.api_endpoints.shutil.copyfileobj')
    @patch('builtins.open', new_callable=MagicMock)
    def test_upload_success(self, mock_open, mock_copyfile, mock_get_store, mock_processor_class, client):
        """Test successful file upload."""
        # Mock document processor
        mock_processor = Mock()
        mock_documents = [Mock(), Mock(), Mock()]  # 3 mock documents
        mock_processor.process_files.return_value = mock_documents
        mock_processor_class.return_value = mock_processor
        
        # Mock vector store
        mock_store = Mock()
        mock_store.add_documents.return_value = ["id1", "id2", "id3"]
        mock_get_store.return_value = mock_store
        
        # Create test file
        test_file_content = b"Test PDF content"
        test_file = io.BytesIO(test_file_content)
        
        response = client.post(
            "/upload-document",
            files={"file": ("test.pdf", test_file, "application/pdf")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Document processed successfully with LangChain"
        assert data["filename"] == "test.pdf"
        assert data["steps_completed"] == ["langchain_document_processing", "langchain_vector_storage"]
        assert data["status"] == "success"
        assert data["document_chunks"] == 3

    def test_upload_invalid_file_type(self, client):
        """Test upload with invalid file type."""
        test_file_content = b"Test content"
        test_file = io.BytesIO(test_file_content)
        
        response = client.post(
            "/upload-document",
            files={"file": ("test.xyz", test_file, "application/octet-stream")}
        )
        
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_upload_no_file(self, client):
        """Test upload with no file."""
        response = client.post("/upload-document")
        assert response.status_code == 422  # Validation error