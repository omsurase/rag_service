from typing import List, Dict
import requests
import random
import logging
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JinaAIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        # Hardcoded API keys
        self.api_keys = [
            "jina_d5e680c565de427db01e8a350f5fd4d3X4nxUjhM2rklBvZ672SyR12cByC8",
            "jina_08948b0b5da34b22b606192c42f7ce2e77qCG9xzZnpnU4lv0-6xftqmxQNK"
        ]
        logger.info(f"Initialized JinaAIClient with {len(self.api_keys)} hardcoded API keys")

    def _get_random_header(self) -> Dict:
        key = random.choice(self.api_keys)
        logger.debug("Selected random API key for request")
        
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }

    def post(self, data: Dict) -> Dict:
        headers = self._get_random_header()
        logger.info(f"Sending POST request to {self.base_url}")
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            logger.info(f"Received response with status code: {response.status_code}")
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

def segment_text(text: str) -> List[str]:
    logger.info(f"Starting text segmentation for text with length: {len(text)}")
    
    client = JinaAIClient(settings.JINA_AI_BASE_URL_SEGMENTATION)
    
    # Clean and normalize text
    text = text.replace('\n', ' ')
    logger.debug("Normalized text, removing newlines")
    
    chunks = []
    MAX_CHAR_LENGTH = 100
    
    logger.info(f"Segmenting text into chunks of max {MAX_CHAR_LENGTH} characters")
    
    for i in range(0, len(text), MAX_CHAR_LENGTH):
        current_chunk = text[i:i + MAX_CHAR_LENGTH]
        body = {
            'content': current_chunk,
            "tokenizer": "o200k_base",
            "max_chunk_length": str(settings.MAX_CHUNK_LENGTH),
            "return_chunks": "true"
        }
        
        try:
            response = client.post(data=body)
            
            if response and "chunks" in response:
                chunk_count = len(response["chunks"])
                logger.debug(f"Received {chunk_count} chunks for current segment")
                chunks.extend(response["chunks"])
            else:
                logger.warning(f"No chunks found in response for segment: {current_chunk}")
                
        except Exception as e:
            logger.error(f"Segmentation error for chunk: {current_chunk}. Error: {e}")
            continue
    
    logger.info(f"Text segmentation complete. Total chunks: {len(chunks)}")
    return chunks