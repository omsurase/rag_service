import logging
from typing import List
from app.models.schemas import Document, Query, Response
from app.core.jina_client import segment_text
from app.core.embeddings import EmbeddingGenerator
from app.services.pinecone_client import PineconeClient
from app.core.llm_client import LLMClient  # Import the LLM client
from langchain_core.messages import HumanMessage, SystemMessage

# Set up logging
logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, llm_name: str = 'sonnet-3.5', is_pro: bool = False):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = PineconeClient()
        self.llm_client = LLMClient()
        self.llm = self.llm_client.get_llm(llm_name, is_pro)
        logger.info(f"RAGEngine initialized with {llm_name} LLM")

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
                            **(doc.metadata or {})
                        }
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

            if not results or len(results.matches) == 0:
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

            # Generate response using LLM
            logger.info("Generating natural language response with LLM")
            llm_response = await self.generate_llm_response(query.text, sources)
            
            logger.info("Query processing completed successfully")
            return Response(
                answer=llm_response,
                sources=sources
            )

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise

    async def generate_llm_response(self, query: str, sources: List[Document]) -> str:
        """
        Generate a natural language response using the LLM based on the query and retrieved sources.
        
        Args:
            query (str): The original user query
            sources (List[Document]): List of relevant source documents
        
        Returns:
            str: Natural language response generated by the LLM
        """
        try:
            # Combine source document contents
            context = "\n\n".join([
                f"Document {doc.id}: {doc.content}" + 
                (f" (Source: {doc.metadata.get('source', 'Unknown')})" if doc.metadata else "")
                for doc in sources
            ])
            
            # Prepare full prompt
            full_prompt = f"""
            You are an AI assistant helping to answer a specific query based on provided context.

            Query: {query}

            Context:
            {context}

            Guidelines:
            1. Carefully analyze the provided context
            2. Directly answer the query using only the information in the context
            3. If the context does not contain sufficient information, clearly state that
            4. Be concise and precise
            5. If possible, cite the source documents
            """
            
            # Generate response
            response = await self.llm.ainvoke(full_prompt)
            
            logger.info("LLM response generated successfully")
            return response.content if response.content else "No relevant information found."

        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}", exc_info=True)
            return "I apologize, but I couldn't generate a response based on the available information."