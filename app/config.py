# app/config.py
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "RAG Service"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    
    # Pinecone Settings
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "majorproject2"
    
    # Jina AI Settings
    JINA_AI_BASE_URL_SEGMENTATION: str = "https://segment.jina.ai/"
    JINA_AI_BASE_URL_EMBEDDING: str = "https://api.jina.ai/v1/embeddings"
    TOTAL_JINA_AI_API_KEYS: int
    JINA_API_KEY_1: str
    JINA_API_KEY_2: str
    MAX_CHUNK_LENGTH: int = 1000

    # LLM Settings
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str
    GROQ_API_KEY: str
    DEFAULT_LLM_MODEL: str = "gpt-4"
    DEFAULT_LLM_MAX_TOKENS: int = 1000
    DEFAULT_LLM_TEMPERATURE: float = 0.7
    
    # Voyage Settings
    VOYAGE_API_KEY: str

    @property
    def jina_api_keys(self) -> List[str]:
        """Returns list of all configured Jina API keys"""
        keys = []
        for i in range(1, self.TOTAL_JINA_AI_API_KEYS + 1):
            key = getattr(self, f"JINA_API_KEY_{i}")
            if key:
                keys.append(key)
        return keys

    class Config:
        env_file = ".env"

settings = Settings()