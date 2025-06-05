import pytest
import json
from unittest.mock import Mock, patch
import requests
from classes.llm_client import LLMClient


class TestLLMClient:
    
    def test_init(self):
        client = LLMClient(
            llm_api_url="http://localhost:11434/api/generate",
            llm_model_name="llama2"
        )
        
        assert client.llm_api_url == "http://localhost:11434/api/generate"
        assert client.llm_model_name == "llama2"
    
    @patch('classes.llm_client.requests.post')
    def test_query_success(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"response": "This is the LLM response"}
        mock_post.return_value = mock_response
        
        client = LLMClient(
            llm_api_url="http://localhost:11434/api/generate",
            llm_model_name="llama2"
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
    
    @patch('classes.llm_client.requests.post')
    def test_query_success_with_whitespace(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"response": "  Response with whitespace  "}
        mock_post.return_value = mock_response
        
        client = LLMClient(
            llm_api_url="http://localhost:11434/api/generate",
            llm_model_name="llama2"
        )
        
        result = client.query("Test prompt")
        
        assert result == "Response with whitespace"
    
    @patch('classes.llm_client.requests.post')
    def test_query_no_response_field(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"status": "complete"}  # No "response" field
        mock_post.return_value = mock_response
        
        client = LLMClient(
            llm_api_url="http://localhost:11434/api/generate",
            llm_model_name="llama2"
        )
        
        result = client.query("Test prompt")
        
        assert result == "No response from LLM"
    
    @patch('classes.llm_client.requests.post')
    def test_query_request_exception(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        client = LLMClient(
            llm_api_url="http://localhost:11434/api/generate",
            llm_model_name="llama2"
        )
        
        result = client.query("Test prompt")
        
        assert result == "Error: Could not connect to the LLM. Make sure Ollama is running."
    
    @patch('classes.llm_client.requests.post')
    def test_query_http_error(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_post.return_value = mock_response
        
        client = LLMClient(
            llm_api_url="http://localhost:11434/api/generate",
            llm_model_name="llama2"
        )
        
        result = client.query("Test prompt")
        
        assert result == "Error: Could not connect to the LLM. Make sure Ollama is running."
    
    @patch('classes.llm_client.requests.post')
    def test_query_timeout(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        
        client = LLMClient(
            llm_api_url="http://localhost:11434/api/generate",
            llm_model_name="llama2"
        )
        
        result = client.query("Test prompt")
        
        assert result == "Error: Could not connect to the LLM. Make sure Ollama is running."
    
    @patch('classes.llm_client.requests.post')
    def test_query_json_decode_error(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "test", 0)
        mock_post.return_value = mock_response
        
        client = LLMClient(
            llm_api_url="http://localhost:11434/api/generate",
            llm_model_name="llama2"
        )
        
        result = client.query("Test prompt")
        
        assert result == "Error: Could not connect to the LLM. Make sure Ollama is running."