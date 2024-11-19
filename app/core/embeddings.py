import time
import voyageai
from typing import List
import os

class EmbeddingGenerator:
    def __init__(self):
        self.client = voyageai.Client(api_key="pa-aNAbHnDwy8lMFC8LAO_sFgTF-SGd_W0ZknOxt1FAXVA")
        self.batch_size = 128
        
    def generate(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        try:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = self.client.embed(
                    batch, 
                    model="voyage-3"
                ).embeddings
                embeddings.extend(batch_embeddings)
                
                if (len(texts) / self.batch_size) > 4:
                    time.sleep(1)
                    
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []