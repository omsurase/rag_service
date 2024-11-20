import logging
from fastapi import APIRouter, HTTPException
from typing import List
from app.models.schemas import Document, DocumentList, Query, Response
from app.core.rag_engine import RAGEngine
from app.config import settings

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()
rag_engine = RAGEngine()
FINANCIAL_STATEMENT_TEXT = """The condensed consolidated financial statements include the accounts of Apple Inc. and its wholly owned subsidiaries (collectively "Apple" or the "Company"). In the opinion of the Company's management, the condensed consolidated financial statements reflect all adjustments, which are normal and recurring in nature, necessary for fair financial statement presentation. The preparation of these condensed consolidated financial statements and accompanying notes in conformity with U.S. generally accepted accounting principles ("GAAP") requires the use of management estimates. These condensed consolidated financial statements and accompanying notes should be read in conjunction with the Company's annual consolidated financial statements and accompanying notes included in its Annual Report on Form 10-K for the fiscal year ended September 30, 2023 (the "2023 Form 10-K")."""


@router.post("/documents", status_code=201)
async def index_documents(request: DocumentList):
    try:
        # Log the incoming request details
        logger.info(f"Received request to index {len(request.documents)} documents")
        print(f"DEBUG: Indexing {len(request.documents)} documents")

        # Log document details for debugging
        for i, doc in enumerate(request.documents, 1):
            logger.debug(f"Document {i}: ID={doc.id}, Content Length={len(doc.content)}")
            print(f"DEBUG: Document {i} - ID: {doc.id}, Content Length: {len(doc.content)}")

        # Process documents
        indexed_count = await rag_engine.process_documents(request.documents)
        
        # Log successful indexing
        logger.info(f"Successfully indexed {indexed_count} document chunks")
        print(f"DEBUG: Successfully indexed {indexed_count} document chunks")

        return {
            "message": f"Successfully indexed {len(request.documents)} documents", 
            "indexed_chunks": indexed_count
        }
    except Exception as e:
        # Log the full error for debugging
        logger.error(f"Error indexing documents: {str(e)}", exc_info=True)
        print(f"ERROR: Failed to index documents - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test", status_code=200)
async def test_post(request_data: dict):
    try:
        # Log incoming request details
        logger.info("Received test POST request")
        print("DEBUG: Test POST request received")
        logger.debug(f"Request data: {request_data}")
        print(f"DEBUG: Request data: {request_data}")

        return {
            "message": "POST request received successfully",
            "received_data": request_data
        }
    except Exception as e:
        # Log the error
        logger.error(f"Error in test POST: {str(e)}", exc_info=True)
        print(f"ERROR: Test POST failed - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/query", response_model=Response)
async def query(query: Query):
    try:
        # Log query details
        logger.info(f"Received query: {query.text[:100]}...")
        print(f"DEBUG: Query received - first 100 chars: {query.text[:100]}")
        
        # Log query parameters
        logger.debug(f"Query details: top_k={query.top_k}")
        print(f"DEBUG: Query top_k: {query.top_k}")

        # Execute query
        response = await rag_engine.query(query)
        
        # Log query results
        logger.info(f"Query processed. Found {len(response.sources)} sources")
        print(f"DEBUG: Query processed. Sources found: {len(response.sources)}")

        return response
    except Exception as e:
        # Log the full error
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        print(f"ERROR: Query processing failed - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/test-config")
async def test_config():
    try:
        # Log configuration check
        logger.info("Checking application configuration")
        print("DEBUG: Testing application configuration")

        config_status = {
            "pinecone": "configured" if settings.PINECONE_API_KEY else "missing",
            "jina_keys": len(settings.jina_api_keys),
            "voyage": "configured" if settings.VOYAGE_API_KEY else "missing"
        }

        # Log configuration details
        logger.info(f"Configuration status: {config_status}")
        print(f"DEBUG: Configuration status - {config_status}")

        return config_status
    except Exception as e:
        # Log any configuration checking errors
        logger.error(f"Error checking configuration: {str(e)}", exc_info=True)
        print(f"ERROR: Configuration check failed - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))