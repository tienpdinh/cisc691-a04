import requests
import json
import logging
import hashlib
from typing import Optional, Dict, Any
from .cache_manager import get_cache_manager

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
                 location: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):

        self.llm_api_url = llm_api_url
        self.llm_model_name = llm_model_name
        self.llm_provider = llm_provider or "ollama"
        self.project_id = project_id
        self.location = location
        self.config = config or {}

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized LLMClient: provider: {self.llm_provider}, "
                        f"model_name: {self.llm_model_name}")

        # Initialize cache manager
        self.cache_manager = get_cache_manager(self.config) if self.config else None

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
        Sends a query to the configured LLM API (synchronous version).
        :param prompt: User query string
        :return: LLM response text
        """
        if self.llm_provider == "vertex_ai":
            response = self._query_vertex_ai_sync(prompt)
        else:
            response = self._query_ollama_sync(prompt)
        return response

    async def query_async(self, prompt: str):
        """
        Sends a query to the configured LLM API with caching (async version).
        :param prompt: User query string
        :return: LLM response text
        """
        # Check cache first
        cache_key = self._generate_cache_key(prompt)
        if self.cache_manager:
            cached_response = await self.cache_manager.get(cache_key, 'llm_responses')
            if cached_response:
                self.logger.debug(f"Cache hit for LLM query: {prompt[:50]}...")
                return cached_response

        # Query LLM if not cached
        if self.llm_provider == "vertex_ai":
            response = self._query_vertex_ai_sync(prompt)
        else:
            response = self._query_ollama_sync(prompt)

        # Cache the response
        if self.cache_manager and response and not response.startswith("Error:"):
            await self.cache_manager.set(cache_key, response, 'llm_responses')
            self.logger.debug(f"Cached LLM response for query: {prompt[:50]}...")

        return response

    def _generate_cache_key(self, prompt: str) -> str:
        """Generate cache key for LLM query including model info."""
        key_data = {
            'prompt': prompt,
            'model': self.llm_model_name,
            'provider': self.llm_provider
        }
        return f"llm:{hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:16]}"

    def _query_vertex_ai_sync(self, prompt: str):
        """Query Vertex AI Gemini model"""
        if not VERTEX_AI_AVAILABLE:
            return "Error: Vertex AI libraries not installed"

        try:
            import vertexai
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

    def _query_ollama_sync(self, prompt: str):
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