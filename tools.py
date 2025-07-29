import os
import logging
from langchain_core.tools import tool
from openai import OpenAI
from prompt_loader import prompt_loader

logger = logging.getLogger(__name__)

class PerplexityClient:
    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            logger.error("Perplexity API key is required")
            raise ValueError("Perplexity API key is required")
        
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.perplexity.ai")
    
    def search(self, query: str, model: str, prompt_type: str) -> dict:
        """Perform search with specified model and prompt type."""
        system_prompt = prompt_loader.get_prompt("tools", prompt_type)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        logger.debug(f"Perplexity {prompt_type} input: query={query}, messages={messages}")
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
            )
            content = response.choices[0].message.content
            logger.debug(f"Perplexity {prompt_type} raw response: {content}")
            logger.info(f"Search completed for query: {query} using model: {model}")
            
            return {
                "raw_content": content,
                "query": query,
                "model": model
            }
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return {"error": str(e), "query": query}

# Lazy initialization - only create client when tools are actually called
_perplexity_client = None

def _get_perplexity_client():
    global _perplexity_client
    if _perplexity_client is None:
        _perplexity_client = PerplexityClient()
    return _perplexity_client

@tool
def initial_search(query: str) -> dict:
    """Perform initial fast search using sonar-pro mode."""
    return _get_perplexity_client().search(query, 'sonar-pro', 'initial_search_system')

@tool
def deep_search(query: str) -> dict:
    """Perform deep verification search using sonar-deep-research mode."""
    return _get_perplexity_client().search(query, 'sonar-deep-research', 'deep_search_system')

tools = [initial_search, deep_search] 