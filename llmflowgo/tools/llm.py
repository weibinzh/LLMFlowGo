from MEoh.base.sample import LLM as MEOH_LLM
from MEoh.tools.llm.llm_api_https import HttpsApi
from typing import Dict, Any

# This is a simple wrapper to match the expected format.
# MEoh's LLM API is already quite flexible.
class ConfigurableLLM(MEOH_LLM):
    def __init__(self, api_key: str, base_url: str, model_name: str, **kwargs):
        # Parse base_url to extract host
        from urllib.parse import urlparse
        
        parsed_url = urlparse(base_url)
        host = parsed_url.hostname
        
        if not host:
            raise ValueError(f"Invalid base_url: {base_url} - could not extract hostname")
        
        print(f"[DEBUG] Parsed URL - host: {host}, path: {parsed_url.path}")
        print(f"[DEBUG] Using HttpsApi with host: {host}")
        
        # Create HttpsApi instance for Claude model
        self._https_api = HttpsApi(
            host=host,
            key=api_key,
            model=model_name,
            timeout=180, 
            **kwargs
        )
        print(f"Initialized ConfigurableLLM for Claude model: {model_name} at {base_url}")

    def draw_sample(self, prompt: str) -> str:
        """
        Draws a sample from the configured LLM.
        """
        # Use the HttpsApi instance to make the actual API call
        return self._https_api.draw_sample(prompt)
    
    def close(self):
        # This method is required by the MEoh framework interface.
        # Since we are not maintaining a persistent connection, we can just pass.
        pass

def get_llm_from_config(llm_config: Dict[str, Any]) -> ConfigurableLLM:
    """
    Factory function to create a ConfigurableLLM instance from a config dictionary.
    """
    if not all(k in llm_config for k in ['apiKey', 'baseUrl', 'modelName']):
        raise ValueError("LLM configuration is missing one or more required keys: apiKey, baseUrl, modelName")

    return ConfigurableLLM(
        api_key=llm_config['apiKey'],
        base_url=llm_config['baseUrl'],
        model_name=llm_config['modelName']
    )
