"""
Tests for API endpoints and business logic.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import HTTPException, UploadFile
from pathlib import Path
from src.api_endpoints import (
    query_rag, upload_document, retrieve_chunks, health_check,
    _get_or_create_vector_store, _get_or_create_rag_processor
)
from src.api_models import QueryRequest, RetrieveRequest


class TestAPIEndpoints:
    """Test cases for API endpoint functions."""

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request object."""
        mock_request = Mock()
        mock_request.app.state.config = {
            'collection_name': 'test_collection',
            'embedding_model_name': 'test-model',
            'chromadb_host': 'localhost',
            'chromadb_port': 8000,
            'vectordb_directory': '/tmp/test',
            'llm_provider': 'ollama',
            'llm_model_name': 'llama3.1',
            'llm_api_url': 'http://localhost:11434/api/generate',
            'openai_api_key': None,
            'retriever_min_score_threshold': '0.5',
            'raw_input_directory': '/tmp/raw',
            'cache': {
                'enabled': True,
                'redis_host': 'localhost',
                'redis_port': 6379
            }
        }
        return mock_request

    @pytest.fixture
    def mock_rag_processor(self):
        """Mock RAG processor."""
        mock_processor = Mock()
        mock_processor.query = AsyncMock(return_value={
            'response': 'Test response',
            'context_used': True,
            'retrieved_docs': ['doc1', 'doc2'],
            'scores': [0.9, 0.8]
        })
        mock_processor.retrieve_documents = AsyncMock(return_value=[
            {'content': 'Test content', 'metadata': {'source': 'test.txt'}, 'score': 0.9}
        ])
        mock_processor.cache_manager = Mock()
        mock_processor.cache_manager.health_check = AsyncMock(return_value={
            'status': 'healthy',
            'enabled': True,
            'stats': {'hits': 10, 'misses': 5}
        })
        return mock_processor

    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store."""
        mock_store = Mock()
        mock_store.get_collection_info.return_value = {
            'collection_name': 'test_collection',
            'document_count': 100
        }
        return mock_store

    @pytest.mark.asyncio
    async def test_query_rag_success(self, mock_request, mock_rag_processor):
        """Test successful RAG query."""
        with patch('src.api_endpoints._get_or_create_rag_processor', return_value=mock_rag_processor):
            query_request = QueryRequest(query="What is AI?", use_rag=True)
            
            response = await query_rag(query_request, mock_request)
            
            assert response.query == "What is AI?"
            assert response.response == "Test response"
            assert response.use_rag is True
            mock_rag_processor.query.assert_called_once_with(
                question="What is AI?",
                use_rag=True
            )

    @pytest.mark.asyncio
    async def test_query_rag_without_rag(self, mock_request, mock_rag_processor):
        """Test query without RAG."""
        with patch('src.api_endpoints._get_or_create_rag_processor', return_value=mock_rag_processor):
            mock_rag_processor.query.return_value = {
                'response': 'Direct LLM response',
                'context_used': False,
                'retrieved_docs': [],
                'scores': []
            }
            
            query_request = QueryRequest(query="What is AI?", use_rag=False)
            
            response = await query_rag(query_request, mock_request)
            
            assert response.query == "What is AI?"
            assert response.response == "Direct LLM response"
            assert response.use_rag is False

    @pytest.mark.asyncio
    async def test_query_rag_error(self, mock_request, mock_rag_processor):
        """Test query with error."""
        with patch('src.api_endpoints._get_or_create_rag_processor', return_value=mock_rag_processor):
            mock_rag_processor.query.side_effect = Exception("RAG processing error")
            
            query_request = QueryRequest(query="What is AI?", use_rag=True)
            
            with pytest.raises(HTTPException) as exc_info:
                await query_rag(query_request, mock_request)
            
            assert exc_info.value.status_code == 500
            assert "Error processing query" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_upload_document_success(self, mock_request, mock_vector_store, tmp_path):
        """Test successful document upload."""
        with patch('src.api_endpoints._get_or_create_vector_store', return_value=mock_vector_store):
            with patch('src.api_endpoints.LangChainDocumentProcessor') as mock_processor_class:
                with patch('src.api_endpoints.shutil.copyfileobj'):
                    # Create temporary directory for the test
                    temp_dir = tmp_path / "raw_input"
                    temp_dir.mkdir()
                    
                    # Update mock request config to use temp directory
                    mock_request.app.state.config['raw_input_directory'] = str(temp_dir)
                    
                    # Mock file
                    mock_file = Mock(spec=UploadFile)
                    mock_file.filename = "test.pdf"
                    mock_file.file = Mock()
                    
                    # Mock processor
                    mock_processor = Mock()
                    mock_processor.process_files.return_value = [
                        Mock(page_content="Test content", metadata={"source": "test.pdf"})
                    ]
                    mock_processor_class.return_value = mock_processor
                    
                    # Mock vector store response
                    mock_vector_store.add_documents.return_value = ["doc_id_1"]
                    
                    response = await upload_document(mock_file, mock_request)
                    
                    assert response.filename == "test.pdf"
                    assert response.status == "success"
                    assert "processed successfully" in response.message
                    assert response.document_chunks == 1

    @pytest.mark.asyncio
    async def test_upload_document_invalid_extension(self, mock_request):
        """Test upload with invalid file extension."""
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.xyz"
        
        with pytest.raises(HTTPException) as exc_info:
            await upload_document(mock_file, mock_request)
        
        assert exc_info.value.status_code == 400
        assert "Unsupported file type" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_upload_document_processing_error(self, mock_request, mock_vector_store, tmp_path):
        """Test document upload with processing error."""
        with patch('src.api_endpoints._get_or_create_vector_store', return_value=mock_vector_store):
            with patch('src.api_endpoints.LangChainDocumentProcessor') as mock_processor_class:
                with patch('src.api_endpoints.shutil.copyfileobj'):
                    # Create temporary directory for the test
                    temp_dir = tmp_path / "raw_input"
                    temp_dir.mkdir()
                    
                    # Update mock request config to use temp directory
                    mock_request.app.state.config['raw_input_directory'] = str(temp_dir)
                    
                    # Mock file
                    mock_file = Mock(spec=UploadFile)
                    mock_file.filename = "test.pdf"
                    mock_file.file = Mock()
                    
                    # Mock processor error
                    mock_processor = Mock()
                    mock_processor.process_files.side_effect = Exception("Processing error")
                    mock_processor_class.return_value = mock_processor
                    
                    with pytest.raises(HTTPException) as exc_info:
                        await upload_document(mock_file, mock_request)
                    
                    assert exc_info.value.status_code == 500
                    assert "Error processing document" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_retrieve_chunks_success(self, mock_request, mock_rag_processor):
        """Test successful chunk retrieval."""
        with patch('src.api_endpoints._get_or_create_rag_processor', return_value=mock_rag_processor):
            retrieve_request = RetrieveRequest(query="test query", top_k=3)
            
            response = await retrieve_chunks(retrieve_request, mock_request)
            
            assert response.query == "test query"
            assert response.count == 1
            assert len(response.results) == 1
            assert response.results[0].text == "Test content"
            assert abs(response.results[0].score - 0.9) < 1e-10

    @pytest.mark.asyncio
    async def test_retrieve_chunks_error(self, mock_request, mock_rag_processor):
        """Test chunk retrieval with error."""
        with patch('src.api_endpoints._get_or_create_rag_processor', return_value=mock_rag_processor):
            mock_rag_processor.retrieve_documents.side_effect = Exception("Retrieval error")
            
            retrieve_request = RetrieveRequest(query="test query", top_k=3)
            
            with pytest.raises(HTTPException) as exc_info:
                await retrieve_chunks(retrieve_request, mock_request)
            
            assert exc_info.value.status_code == 500
            assert "Error retrieving chunks" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_vector_store, mock_rag_processor):
        """Test successful health check."""
        with patch('src.api_endpoints._vector_store', mock_vector_store):
            with patch('src.api_endpoints._rag_processor', mock_rag_processor):
                response = await health_check()
                
                assert response.status == "healthy"
                assert response.message == "LangChain RAG API is running"
                assert "vector_store" in response.components
                assert "rag_processor" in response.components
                assert "cache" in response.components

    @pytest.mark.asyncio
    async def test_health_check_uninitialized(self):
        """Test health check with uninitialized components."""
        with patch('src.api_endpoints._vector_store', None):
            with patch('src.api_endpoints._rag_processor', None):
                response = await health_check()
                
                assert response.status == "healthy"
                assert response.components["vector_store"] == "not_initialized"
                assert response.components["rag_processor"] == "not_initialized"

    @pytest.mark.asyncio
    async def test_health_check_error(self, mock_vector_store):
        """Test health check with error."""
        with patch('src.api_endpoints._vector_store', mock_vector_store):
            with patch('src.api_endpoints._rag_processor', None):
                mock_vector_store.get_collection_info.side_effect = Exception("Collection error")
                
                response = await health_check()
                
                assert response.status == "healthy"  # Should still be healthy overall
                assert "error" in response.components["collection_info"]


class TestComponentCreation:
    """Test cases for component creation functions."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'collection_name': 'test_collection',
            'embedding_model_name': 'test-model',
            'chromadb_host': 'localhost',
            'chromadb_port': 8000,
            'vectordb_directory': '/tmp/test',
            'llm_provider': 'ollama',
            'llm_model_name': 'llama3.1',
            'llm_api_url': 'http://localhost:11434/api/generate',
            'openai_api_key': None,
            'cache': {
                'enabled': True,
                'redis_host': 'localhost',
                'redis_port': 6379
            }
        }

    def test_get_or_create_vector_store_new(self, sample_config):
        """Test creating new vector store."""
        with patch('src.api_endpoints.get_cache_manager') as mock_get_cache:
            with patch('src.api_endpoints.LangChainVectorStore') as mock_vector_store_class:
                with patch('src.api_endpoints._vector_store', None):
                    mock_cache = Mock()
                    mock_get_cache.return_value = mock_cache
                    mock_vector_store = Mock()
                    mock_vector_store_class.return_value = mock_vector_store
                    
                    result = _get_or_create_vector_store(sample_config)
                    
                    assert result is mock_vector_store
                    mock_vector_store_class.assert_called_once()
                    mock_get_cache.assert_called_once_with(sample_config)

    def test_get_or_create_vector_store_existing(self, sample_config):
        """Test getting existing vector store."""
        mock_existing_store = Mock()
        with patch('src.api_endpoints._vector_store', mock_existing_store):
            result = _get_or_create_vector_store(sample_config)
            
            assert result is mock_existing_store

    def test_get_or_create_rag_processor_new(self, sample_config):
        """Test creating new RAG processor."""
        with patch('src.api_endpoints._get_or_create_vector_store') as mock_get_vector:
            with patch('src.api_endpoints.LangChainRAGProcessor') as mock_rag_class:
                with patch('src.api_endpoints._rag_processor', None):
                    mock_vector_store = Mock()
                    mock_get_vector.return_value = mock_vector_store
                    mock_rag_processor = Mock()
                    mock_rag_class.return_value = mock_rag_processor
                    
                    result = _get_or_create_rag_processor(sample_config)
                    
                    assert result is mock_rag_processor
                    mock_rag_class.assert_called_once()
                    # Verify config was passed
                    call_kwargs = mock_rag_class.call_args[1]
                    assert call_kwargs['config'] == sample_config

    def test_get_or_create_rag_processor_existing(self, sample_config):
        """Test getting existing RAG processor."""
        mock_existing_processor = Mock()
        with patch('src.api_endpoints._rag_processor', mock_existing_processor):
            result = _get_or_create_rag_processor(sample_config)
            
            assert result is mock_existing_processor

    def test_get_or_create_rag_processor_with_openai(self, sample_config):
        """Test creating RAG processor with OpenAI configuration."""
        sample_config['llm_provider'] = 'openai'
        sample_config['openai_api_key'] = 'test-key'
        
        with patch('src.api_endpoints._get_or_create_vector_store') as mock_get_vector:
            with patch('src.api_endpoints.LangChainRAGProcessor') as mock_rag_class:
                with patch('src.api_endpoints._rag_processor', None):
                    mock_vector_store = Mock()
                    mock_get_vector.return_value = mock_vector_store
                    mock_rag_processor = Mock()
                    mock_rag_class.return_value = mock_rag_processor
                    
                    result = _get_or_create_rag_processor(sample_config)
                    
                    assert result is mock_rag_processor
                    call_kwargs = mock_rag_class.call_args[1]
                    assert call_kwargs['llm_provider'] == 'openai'
                    assert call_kwargs['openai_api_key'] == 'test-key'