from fastapi import APIRouter, HTTPException
from typing import List
from app.models.schemas import Document, DocumentList, Query, Response
from app.core.rag_engine import RAGEngine
from app.config import settings

router = APIRouter()
rag_engine = RAGEngine()
FINANCIAL_STATEMENT_TEXT = """The condensed consolidated financial statements include the accounts of Apple Inc. and its wholly owned subsidiaries (collectively "Apple" or the "Company"). In the opinion of the Company's management, the condensed consolidated financial statements reflect all adjustments, which are normal and recurring in nature, necessary for fair financial statement presentation. The preparation of these condensed consolidated financial statements and accompanying notes in conformity with U.S. generally accepted accounting principles ("GAAP") requires the use of management estimates. These condensed consolidated financial statements and accompanying notes should be read in conjunction with the Company's annual consolidated financial statements and accompanying notes included in its Annual Report on Form 10-K for the fiscal year ended September 30, 2023 (the "2023 Form 10-K")."""


@router.post("/documents", status_code=201)
async def index_documents(request: DocumentList):
    try:
        # Pass the list of documents directly to process_documents
        await rag_engine.process_documents(request.documents)
        return {"message": f"Successfully indexed {len(request.documents)} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test", status_code=200)
async def test_post(request_data: dict):
    try:
        return {
            "message": "POST request received successfully",
            "received_data": request_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/query", response_model=Response)
async def query(query: Query):
    try:
        return await rag_engine.query(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/test-config")
async def test_config():
    return {
        "pinecone": "configured" if settings.PINECONE_API_KEY else "missing",
        "jina_keys": len(settings.jina_api_keys),
        "voyage": "configured" if settings.VOYAGE_API_KEY else "missing"
    }