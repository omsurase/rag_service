from pinecone import Pinecone
from typing import List, Dict
from app.config import settings

class PineconeClient:
    def __init__(self):
        self.client = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME
        self.index = None

    def get_index(self):
        if self.index is None:
            self.index = self.client.Index(name=self.index_name)
        return self.index

    async def upsert(self, vectors: List[Dict]):
        try:
            index = self.get_index()
            valid_vectors = [
                v for v in vectors 
                if isinstance(v, dict) and 'id' in v and 'values' in v and v['values'] is not None
            ]
            if valid_vectors:
                return index.upsert(vectors=valid_vectors)
        except Exception as e:
            print(f"Error upserting vectors: {e}")
            return None

    async def query(self, query_vector: List[float], top_k: int):
        try:
            index = self.get_index()
            return index.query(
                vector=query_vector,  # Changed from query_vector to vector
                top_k=top_k,
                include_values=True,
                include_metadata=True
            )
        except Exception as e:
            print(f"Error querying index: {e}")
            raise