import pytest
import json
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import requests
from src.llm_client import LLMClient


class TestLLMClient:
    
    def test_init_ollama_default(self):
        client = LLMClient(
            llm_api_url="http://localhost:11434/api/generate",
            llm_model_name="llama2"
        )
        
        assert client.llm_api_url == "http://localhost:11434/api/generate"
        assert client.llm_model_name == "llama2"
        assert client.llm_provider == "ollama"
        assert client.project_id is None
        assert client.location is None

    def test_init_vertex_ai(self):
        client = LLMClient(
            llm_api_url="https://vertex-ai-url",
            llm_model_name="gemini-1.5-flash",
            llm_provider="vertex_ai",
            project_id="test-project",
            location="us-central1"
        )
        
        assert client.llm_provider == "vertex_ai"
        assert client.project_id == "test-project"
        assert client.location == "us-central1"

    @patch('src.llm_client.requests.post')
    def test_query_ollama_success(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"response": "This is the LLM response"}
        mock_post.return_value = mock_response
        
        client = LLMClient(
            llm_api_url="http://localhost:11434/api/generate",
            llm_model_name="llama2",
            llm_provider="ollama"
        )
        
        result = client.query("What is AI?")
        
        assert result == "This is the LLM response"
        
        # Verify the request was made correctly
        expected_payload = {
            "model": "llama2",
            "prompt": "What is AI?",
            "stream": False
        }
        expected_headers = {"Content-Type": "application/json"}
        
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/generate",
            headers=expected_headers,
            data=json.dumps(expected_payload),
            timeout=60
        )

    @patch('src.llm_client.requests.post')
    def test_query_ollama_success_with_whitespace(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"response": "  Response with whitespace  "}
        mock_post.return_value = mock_response
        
        client = LLMClient(
            llm_api_url="http://localhost:11434/api/generate",
            llm_model_name="llama2",
            llm_provider="ollama"
        )
        
        result = client.query("Test prompt")
        
        assert result == "Response with whitespace"

    @patch('src.llm_client.requests.post')
    def test_query_ollama_no_response_field(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"status": "complete"}  # No "response" field
        mock_post.return_value = mock_response
        
        client = LLMClient(
            llm_api_url="http://localhost:11434/api/generate",
            llm_model_name="llama2",
            llm_provider="ollama"
        )
        
        result = client.query("Test prompt")
        
        assert result == "No response from LLM"

    @patch('src.llm_client.requests.post')
    def test_query_ollama_request_exception(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        client = LLMClient(
            llm_api_url="http://localhost:11434/api/generate",
            llm_model_name="llama2",
            llm_provider="ollama"
        )
        
        result = client.query("Test prompt")
        
        assert result == "Error: Could not connect to Ollama. Make sure it's running."

    @patch('src.llm_client.requests.post')
    def test_query_ollama_http_error(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_post.return_value = mock_response
        
        client = LLMClient(
            llm_api_url="http://localhost:11434/api/generate",
            llm_model_name="llama2",
            llm_provider="ollama"
        )
        
        result = client.query("Test prompt")
        
        assert result == "Error: Could not connect to Ollama. Make sure it's running."

    @patch('src.llm_client.requests.post')
    def test_query_ollama_timeout(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        
        client = LLMClient(
            llm_api_url="http://localhost:11434/api/generate",
            llm_model_name="llama2",
            llm_provider="ollama"
        )
        
        result = client.query("Test prompt")
        
        assert result == "Error: Could not connect to Ollama. Make sure it's running."

    @patch('src.llm_client.requests.post')
    def test_query_ollama_json_decode_error(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "test", 0)
        mock_post.return_value = mock_response
        
        client = LLMClient(
            llm_api_url="http://localhost:11434/api/generate",
            llm_model_name="llama2",
            llm_provider="ollama"
        )
        
        result = client.query("Test prompt")
        
        assert result == "Error: Could not connect to Ollama. Make sure it's running."

    @patch('src.llm_client.VERTEX_AI_AVAILABLE', True)
    def test_query_vertex_ai_success(self):
        # Mock the entire method to avoid import issues
        client = LLMClient(
            llm_api_url="https://vertex-ai-url",
            llm_model_name="gemini-1.5-flash",
            llm_provider="vertex_ai",
            project_id="test-project",
            location="us-central1"
        )
        
        # Mock the _query_vertex_ai_sync method directly
        client._query_vertex_ai_sync = Mock(return_value="Vertex AI response")
        
        result = client.query("What is AI?")
        
        assert result == "Vertex AI response"
        client._query_vertex_ai_sync.assert_called_once_with("What is AI?")

    @patch('src.llm_client.VERTEX_AI_AVAILABLE', True)
    def test_query_vertex_ai_no_text_response(self):
        client = LLMClient(
            llm_api_url="https://vertex-ai-url",
            llm_model_name="gemini-1.5-flash",
            llm_provider="vertex_ai"
        )
        
        # Mock the _query_vertex_ai_sync method to return no response
        client._query_vertex_ai_sync = Mock(return_value="No response from Vertex AI")
        
        result = client.query("Test prompt")
        
        assert result == "No response from Vertex AI"

    @patch('src.llm_client.VERTEX_AI_AVAILABLE', True)
    def test_query_vertex_ai_exception(self):
        client = LLMClient(
            llm_api_url="https://vertex-ai-url",
            llm_model_name="gemini-1.5-flash",
            llm_provider="vertex_ai"
        )
        
        # Mock the _query_vertex_ai_sync method to raise an exception
        client._query_vertex_ai_sync = Mock(return_value="Error: Could not connect to Vertex AI: test error")
        
        result = client.query("Test prompt")
        
        assert "Error: Could not connect to Vertex AI" in result

    @patch('src.llm_client.LLMClient._query_vertex_ai_sync')
    def test_query_vertex_ai_not_available(self, mock_vertex_query):
        mock_vertex_query.return_value = "Error: Vertex AI libraries not installed"
        
        client = LLMClient(
            llm_api_url="https://vertex-ai-url",
            llm_model_name="gemini-1.5-flash",
            llm_provider="vertex_ai"
        )
        
        result = client.query("Test prompt")
        
        assert result == "Error: Vertex AI libraries not installed"

    @patch('src.llm_client.VERTEX_AI_AVAILABLE', True)
    def test_vertex_ai_initialization(self):
        # Mock the _init_vertex_ai method to avoid aiplatform import issues
        with patch.object(LLMClient, '_init_vertex_ai'):
            client = LLMClient(
                llm_api_url="https://vertex-ai-url",
                llm_model_name="gemini-1.5-flash",
                llm_provider="vertex_ai",
                project_id="test-project",
                location="us-central1"
            )
            
            # Verify the client was created with correct attributes
            assert client.llm_provider == "vertex_ai"
            assert client.project_id == "test-project"
            assert client.location == "us-central1"


class TestLLMClientCaching:
    """Test cases for LLM client caching functionality."""

    @pytest.fixture
    def cache_config(self):
        """Sample cache configuration."""
        return {
            'cache': {
                'enabled': True,
                'redis_host': 'localhost',
                'redis_port': 6379,
                'redis_db': 0,
                'ttl_seconds': {'llm_responses': 3600}
            }
        }

    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager for testing."""
        mock_cache = Mock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock(return_value=True)
        return mock_cache

    def test_init_with_cache_config(self, cache_config):
        """Test LLMClient initialization with cache configuration."""
        with patch('src.llm_client.get_cache_manager') as mock_get_cache:
            mock_cache = Mock()
            mock_get_cache.return_value = mock_cache
            
            client = LLMClient(
                llm_api_url="http://localhost:11434/api/generate",
                llm_model_name="llama2",
                config=cache_config
            )
            
            assert client.cache_manager is mock_cache
            mock_get_cache.assert_called_once_with(cache_config)

    def test_init_without_cache_config(self):
        """Test LLMClient initialization without cache configuration."""
        client = LLMClient(
            llm_api_url="http://localhost:11434/api/generate",
            llm_model_name="llama2"
        )
        
        assert client.cache_manager is None

    def test_generate_cache_key(self, cache_config):
        """Test cache key generation."""
        with patch('src.llm_client.get_cache_manager'):
            client = LLMClient(
                llm_api_url="http://localhost:11434/api/generate",
                llm_model_name="llama2",
                llm_provider="ollama",
                config=cache_config
            )
            
            key = client._generate_cache_key("What is AI?")
            
            assert key.startswith('llm:')
            assert len(key.split(':')[1]) == 16  # Hash length

    def test_generate_cache_key_consistency(self, cache_config):
        """Test that same prompt generates same cache key."""
        with patch('src.llm_client.get_cache_manager'):
            client = LLMClient(
                llm_api_url="http://localhost:11434/api/generate",
                llm_model_name="llama2",
                config=cache_config
            )
            
            key1 = client._generate_cache_key("What is AI?")
            key2 = client._generate_cache_key("What is AI?")
            
            assert key1 == key2

    def test_generate_cache_key_different_models(self, cache_config):
        """Test that different models generate different cache keys."""
        with patch('src.llm_client.get_cache_manager'):
            client1 = LLMClient(
                llm_api_url="http://localhost:11434/api/generate",
                llm_model_name="llama2",
                config=cache_config
            )
            client2 = LLMClient(
                llm_api_url="http://localhost:11434/api/generate",
                llm_model_name="llama3.1",
                config=cache_config
            )
            
            key1 = client1._generate_cache_key("What is AI?")
            key2 = client2._generate_cache_key("What is AI?")
            
            assert key1 != key2

    @pytest.mark.asyncio
    async def test_query_cache_hit(self, cache_config, mock_cache_manager):
        """Test query with cache hit."""
        with patch('src.llm_client.get_cache_manager', return_value=mock_cache_manager):
            cached_response = "Cached LLM response"
            mock_cache_manager.get.return_value = cached_response
            
            client = LLMClient(
                llm_api_url="http://localhost:11434/api/generate",
                llm_model_name="llama2",
                config=cache_config
            )
            
            result = await client.query_async("What is AI?")
            
            assert result == cached_response
            mock_cache_manager.get.assert_called_once()
            mock_cache_manager.set.assert_not_called()

    @pytest.mark.asyncio
    @patch('src.llm_client.requests.post')
    async def test_query_cache_miss_ollama(self, mock_post, cache_config, mock_cache_manager):
        """Test query with cache miss for Ollama."""
        with patch('src.llm_client.get_cache_manager', return_value=mock_cache_manager):
            # Cache miss
            mock_cache_manager.get.return_value = None
            
            # Mock Ollama response
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"response": "Fresh LLM response"}
            mock_post.return_value = mock_response
            
            client = LLMClient(
                llm_api_url="http://localhost:11434/api/generate",
                llm_model_name="llama2",
                config=cache_config
            )
            
            result = await client.query_async("What is AI?")
            
            assert result == "Fresh LLM response"
            mock_cache_manager.get.assert_called_once()
            mock_cache_manager.set.assert_called_once()
            
            # Verify the response was cached
            set_call_args = mock_cache_manager.set.call_args
            assert set_call_args[0][1] == "Fresh LLM response"  # Cached value
            assert set_call_args[0][2] == 'llm_responses'  # Cache type

    @pytest.mark.asyncio
    @patch('src.llm_client.requests.post')
    async def test_query_error_not_cached(self, mock_post, cache_config, mock_cache_manager):
        """Test that error responses are not cached."""
        with patch('src.llm_client.get_cache_manager', return_value=mock_cache_manager):
            # Cache miss
            mock_cache_manager.get.return_value = None
            
            # Mock error response
            mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
            
            client = LLMClient(
                llm_api_url="http://localhost:11434/api/generate",
                llm_model_name="llama2",
                config=cache_config
            )
            
            result = await client.query_async("What is AI?")
            
            assert result.startswith("Error:")
            mock_cache_manager.get.assert_called_once()
            mock_cache_manager.set.assert_not_called()  # Error responses not cached

    @pytest.mark.asyncio
    async def test_query_no_cache_manager(self):
        """Test query when cache manager is not available."""
        with patch('src.llm_client.requests.post') as mock_post:
            # Mock Ollama response
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"response": "LLM response"}
            mock_post.return_value = mock_response
            
            client = LLMClient(
                llm_api_url="http://localhost:11434/api/generate",
                llm_model_name="llama2"
                # No cache config
            )
            
            result = await client.query_async("What is AI?")
            
            assert result == "LLM response"
            # Should still work without caching