import os
import getpass
import logging
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

class Config:
    @staticmethod
    def load():
        load_dotenv()
        
        if not os.environ.get("AZURE_OPENAI_ENDPOINT"):
            os.environ["AZURE_OPENAI_ENDPOINT"] = "https://lxproductinternal.openai.azure.com/openai/deployments/openai-gpt-4.1-deployment/chat/completions?api-version=2025-01-01-preview"
        
        logger.info("Environment variables loaded successfully")

    @staticmethod
    def get_llm():
        from langchain_openai import AzureChatOpenAI
        
        llm = AzureChatOpenAI(
            azure_deployment="openai-gpt-4.1-deployment",
            api_version="2025-01-01-preview",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=3,
        )
        logger.info("Azure OpenAI LLM initialized")
        return llm