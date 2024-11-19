import logging
from typing import List
from app.models.schemas import Document, Query, Response
from app.core.jina_client import segment_text
from app.core.embeddings import EmbeddingGenerator
from app.services.pinecone_client import PineconeClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_engine.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('RAGEngine')

class RAGEngine:
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = PineconeClient()
        logger.info("RAGEngine initialized with embedding generator and vector store")

    async def process_documents(self, documents: List[Document]):
        try:
            logger.info(f"Starting to process {len(documents)} documents")
            all_chunks = []

            # Segment documents
            for doc in documents:
                logger.info(f"Processing document with ID: {doc.id}")
                chunks = segment_text(doc.content)
                logger.info(f"Document {doc.id} segmented into {len(chunks)} chunks")

                for chunk in chunks:
                    chunk_id = f"{doc.id}-{len(all_chunks)}"
                    all_chunks.append({
                        "id": chunk_id,
                        "text": chunk,
                        "metadata": {
                            "document_id": doc.id,
                            "content": chunk,
                            **doc.metadata
                        } if doc.metadata else {"document_id": doc.id, "content": chunk}
                    })
                logger.debug(f"Created chunks with IDs: {[chunk['id'] for chunk in all_chunks[-len(chunks):]]}")

                logger.info(f"Generated total of {len(all_chunks)} chunks across all documents")

                # Generate embeddings
                logger.info("Generating embeddings for chunks")
                embeddings = self.embedding_generator.generate([chunk["text"] for chunk in all_chunks])
                logger.info(f"Generated {len(embeddings)} embeddings")

                # Prepare vectors for storage
                vectors = [
                    {
                        "id": chunk["id"],
                    "values": embedding,
                    "metadata": chunk["metadata"]
                }
                for chunk, embedding in zip(all_chunks, embeddings)
            ]
            logger.info(f"Prepared {len(vectors)} vectors for storage")

                # Store vectors
            logger.info("Storing vectors in vector database")
            await self.vector_store.upsert(vectors)
            logger.info("Successfully stored all vectors")

            return len(vectors)

        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}", exc_info=True)
            raise

    async def query(self, query: Query) -> Response:
        try:
            logger.info(f"Processing query: {query.text[:100]}...")
            
            # Generate query embedding
            logger.info("Generating query embedding")
            query_embedding = self.embedding_generator.generate([query.text])[0]
            logger.info("Query embedding generated")

            # Search similar vectors
            logger.info(f"Searching for top {query.top_k} similar vectors")
            results = await self.vector_store.query(query_embedding, query.top_k)

            if not results:
                logger.info("No results found for query")
                return Response(answer="No relevant information found.", sources=[])

            # Process results
            sources = []
            logger.info(f"Found {len(results.matches)} matches")
            
            for match in results.matches:
                doc_id = match.metadata["document_id"]
                sources.append(Document(
                    id=doc_id,
                    content=match.metadata.get("content", ""),
                    metadata=match.metadata
                ))
                logger.debug(f"Added source document: {doc_id}")

            response = Response(
                answer=f"Found {len(sources)} relevant documents",
                sources=sources
            )
            logger.info("Query processing completed successfully")
            return response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise