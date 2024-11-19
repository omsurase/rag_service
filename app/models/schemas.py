from pydantic import BaseModel
from typing import List, Optional, Dict


class Document(BaseModel):
    id: str
    content: str
    metadata: Optional[Dict] = None

class DocumentList(BaseModel):
    documents: List[Document]

class Query(BaseModel):
    text: str
    top_k: int = 3

class Response(BaseModel):
    answer: str
    sources: List[Document]