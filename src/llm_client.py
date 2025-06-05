import requests
import json
import logging
import os
from typing import Optional

try:
    from google.cloud import aiplatform
    from google.auth import default
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False


class LLMClient:
    """
    Handles interactions with LLM APIs (Ollama, Vertex AI, etc.)
    """
    def __init__(self,
                 llm_api_url: str,
                 llm_model_name: str,
                 llm_provider: Optional[str] = "ollama",
                 project_id: Optional[str] = None,
                 location: Optional[str] = None):

        self.llm_api_url = llm_api_url
        self.llm_model_name = llm_model_name
        self.llm_provider = llm_provider or "ollama"
        self.project_id = project_id
        self.location = location

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized LLMClient: provider: {self.llm_provider}, "
                        f"model_name: {self.llm_model_name}")

        # Initialize Vertex AI if needed
        if self.llm_provider == "vertex_ai" and VERTEX_AI_AVAILABLE:
            self._init_vertex_ai()

    def _init_vertex_ai(self):
        """Initialize Vertex AI client"""
        try:
            if self.project_id and self.location:
                aiplatform.init(project=self.project_id, location=self.location)
                self.logger.info(f"Initialized Vertex AI: {self.project_id}/{self.location}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Vertex AI: {e}")

    def query(self, prompt: str):
        """
        Sends a query to the configured LLM API.
        :param prompt: User query string
        :return: LLM response text
        """
        if self.llm_provider == "vertex_ai":
            return self._query_vertex_ai(prompt)
        else:
            return self._query_ollama(prompt)

    def _query_vertex_ai(self, prompt: str):
        """Query Vertex AI Gemini model"""
        if not VERTEX_AI_AVAILABLE:
            return "Error: Vertex AI libraries not installed"

        try:
            from vertexai.generative_models import GenerativeModel
            
            model = GenerativeModel(self.llm_model_name)
            response = model.generate_content(prompt)
            
            if response.text:
                return response.text.strip()
            else:
                return "No response from Vertex AI"
                
        except Exception as e:
            self.logger.error(f"Error querying Vertex AI: {e}")
            return f"Error: Could not connect to Vertex AI: {str(e)}"

    def _query_ollama(self, prompt: str):
        """Query Ollama API"""
        payload = {
            "model": self.llm_model_name,
            "prompt": prompt,
            "stream": False
        }
        
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.llm_api_url,
                                     headers=headers,
                                     data=json.dumps(payload),
                                     timeout=60)
            response.raise_for_status()
            
            # Handle Ollama response format
            response_data = response.json()
            if "response" in response_data:
                return response_data["response"].strip()
            else:
                return "No response from LLM"
                
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            self.logger.error(f"Error querying Ollama: {e}")
            return f"Error: Could not connect to Ollama. Make sure it's running."