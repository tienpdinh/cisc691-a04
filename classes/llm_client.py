import requests
import json
import logging


class LLMClient:
    """
    Handles direct interactions with a locally running LLM API.
    """
    def __init__(self,
                 llm_api_url: str,
                 llm_model_name: str):

        self.llm_api_url = llm_api_url
        self.llm_model_name = llm_model_name

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized LLMClient: llm_api_url: {self.llm_api_url}, "
                        f"model_name: {self.llm_model_name}")

    def query(self, prompt: str):
        """
        Sends a query to the local LLM API.
        :param prompt: User query string
        :return: LLM response text
        """
        # For Ollama API format
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
            self.logger.error(f"Error querying LLM: {e}")
            return f"Error: Could not connect to the LLM. Make sure Ollama is running."