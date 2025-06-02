"""Tests for LLMClient class."""
import pytest
import responses
import requests
from classes.llm_client import LLMClient

class TestLLMClient:
    """Test cases for LLMClient."""
    
    @pytest.fixture
    def llm_client(self):
        """Create LLMClient instance for testing."""
        return LLMClient(
            llm_api_url="http://localhost:11434/api/generate",
            llm_model_name="test-model"
        )
    
    def test_init(self, llm_client):
        """Test LLMClient initialization."""
        assert llm_client.llm_api_url == "http://localhost:11434/api/generate"
        assert llm_client.llm_model_name == "test-model"
    
    @responses.activate
    def test_query_success(self, llm_client):
        """Test successful LLM query."""
        responses.add(
            responses.POST,
            "http://localhost:11434/api/generate",
            json={"response": "This is a test response from the LLM."},
            status=200
        )
        
        result = llm_client.query("Test prompt")
        
        assert result == "This is a test response from the LLM."
        assert len(responses.calls) == 1
        
        # Check request payload
        request_data = responses.calls[0].request.body.decode()
        assert '"model": "test-model"' in request_data
        assert '"prompt": "Test prompt"' in request_data
        assert '"stream": false' in request_data
    
    @responses.activate
    def test_query_no_response_field(self, llm_client):
        """Test LLM query with missing response field."""
        responses.add(
            responses.POST,
            "http://localhost:11434/api/generate",
            json={"error": "No response field"},
            status=200
        )
        
        result = llm_client.query("Test prompt")
        
        assert result == "No response from LLM"
    
    @responses.activate
    def test_query_http_error(self, llm_client):
        """Test LLM query with HTTP error."""
        responses.add(
            responses.POST,
            "http://localhost:11434/api/generate",
            status=500
        )
        
        result = llm_client.query("Test prompt")
        
        assert "Error: Could not connect to the LLM" in result
        assert "Make sure Ollama is running" in result
    
    @responses.activate
    def test_query_connection_error(self, llm_client):
        """Test LLM query with connection error."""
        responses.add(
            responses.POST,
            "http://localhost:11434/api/generate",
            body=requests.exceptions.ConnectionError("Connection failed")
        )
        
        result = llm_client.query("Test prompt")
        
        assert "Error: Could not connect to the LLM" in result
    
    @responses.activate
    def test_query_timeout(self, llm_client):
        """Test LLM query with timeout."""
        responses.add(
            responses.POST,
            "http://localhost:11434/api/generate",
            body=requests.exceptions.Timeout("Request timed out")
        )
        
        result = llm_client.query("Test prompt")
        
        assert "Error: Could not connect to the LLM" in result
    
    @responses.activate
    def test_query_whitespace_handling(self, llm_client):
        """Test that response whitespace is properly stripped."""
        responses.add(
            responses.POST,
            "http://localhost:11434/api/generate",
            json={"response": "  \n  Response with whitespace  \n  "},
            status=200
        )
        
        result = llm_client.query("Test prompt")
        
        assert result == "Response with whitespace"
    
    @responses.activate
    def test_query_empty_response(self, llm_client):
        """Test LLM query with empty response."""
        responses.add(
            responses.POST,
            "http://localhost:11434/api/generate",
            json={"response": ""},
            status=200
        )
        
        result = llm_client.query("Test prompt")
        
        assert result == ""