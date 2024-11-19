import os
from typing import AsyncIterator
import logging
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger('LLMClient')

class LLMClient:
    def __init__(self):
        if os.path.exists(".env"):
            load_dotenv()
        
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        os.environ["GROQ_API_KEY"] = self.groq_api_key
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key

    def get_llm(self, llm_name: str = 'gpt-4o', is_pro: bool = False):
        if not is_pro:
            return ChatGroq(
                model="llama-3.1-70b-versatile",
                temperature=0.7,
                max_tokens=1000,
                timeout=None,
                max_retries=3,
            )

        llm_configs = {
            'gpt-4o': lambda: ChatOpenAI(
                api_key=self.openai_api_key,
                model_name="gpt-4o",
                max_retries=3,
                timeout=None,
                temperature=0.4,
                max_tokens=1800,
            ),
            'sonnet-3.5': lambda: ChatAnthropic(
                api_key=self.anthropic_api_key,
                model_name="claude-3-5-sonnet-20241022",
                max_tokens=3000,
                temperature=1,
                timeout=None,
                max_retries=3,
            ),
            'llama-3.1-70b': lambda: ChatGroq(
                model="llama-3.1-70b-versatile",
                temperature=0.7,
                max_tokens=1800,
                timeout=None,
                max_retries=3,
            )
        }
        
        return llm_configs.get(llm_name, llm_configs['gpt-4o'])()
