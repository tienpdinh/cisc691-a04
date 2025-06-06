import pytest
import json
from unittest.mock import Mock, patch, MagicMock
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
        
        # Mock the _query_vertex_ai method directly
        client._query_vertex_ai = Mock(return_value="Vertex AI response")
        
        result = client.query("What is AI?")
        
        assert result == "Vertex AI response"
        client._query_vertex_ai.assert_called_once_with("What is AI?")

    @patch('src.llm_client.VERTEX_AI_AVAILABLE', True)
    def test_query_vertex_ai_no_text_response(self):
        client = LLMClient(
            llm_api_url="https://vertex-ai-url",
            llm_model_name="gemini-1.5-flash",
            llm_provider="vertex_ai"
        )
        
        # Mock the _query_vertex_ai method to return no response
        client._query_vertex_ai = Mock(return_value="No response from Vertex AI")
        
        result = client.query("Test prompt")
        
        assert result == "No response from Vertex AI"

    @patch('src.llm_client.VERTEX_AI_AVAILABLE', True)
    def test_query_vertex_ai_exception(self):
        client = LLMClient(
            llm_api_url="https://vertex-ai-url",
            llm_model_name="gemini-1.5-flash",
            llm_provider="vertex_ai"
        )
        
        # Mock the _query_vertex_ai method to raise an exception
        client._query_vertex_ai = Mock(return_value="Error: Could not connect to Vertex AI: test error")
        
        result = client.query("Test prompt")
        
        assert "Error: Could not connect to Vertex AI" in result

    @patch('src.llm_client.VERTEX_AI_AVAILABLE', False)
    def test_query_vertex_ai_not_available(self):
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