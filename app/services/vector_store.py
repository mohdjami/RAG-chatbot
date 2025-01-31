# app/services/vector_store.py
from typing import List, Dict, Any, Optional, Tuple
from pinecone import Pinecone, PodSpec
from app.core.config import settings
from app.utils.error_handlers import VectorStoreError
from loguru import logger
import uuid
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

class VectorStoreService:
    """Service for managing vector storage and retrieval using Pinecone."""
    
    def __init__(self):
        try:
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self.index_name = settings.PINECONE_INDEX_NAME
            self._ensure_index_exists()
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise VectorStoreError(f"Vector store initialization failed: {str(e)}")

    def _ensure_index_exists(self):
        """Ensure the required Pinecone index exists, create if it doesn't."""
        try:
            existing_indexes = self.pc.list_indexes()
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=settings.VECTOR_DIMENSION,
                    metric="cosine",
                    spec=PodSpec(
                        environment=settings.PINECONE_ENVIRONMENT,
                        pod_type="p1.x1"  # Adjust based on your needs
                    )
                )
                logger.info("Index created successfully")
        except Exception as e:
            logger.error(f"Failed to create Pinecone index: {str(e)}")
            raise VectorStoreError(f"Index creation failed: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def store_embeddings(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Store document chunks and their embeddings in Pinecone.
        
        Args:
            chunks: List of dictionaries containing text, embeddings, and metadata
            
        Returns:
            List of vector IDs
        """
        try:
            vectors = []
            ids = []
            
            for chunk in chunks:
                vector_id = str(uuid.uuid4())
                vectors.append({
                    "id": vector_id,
                    "values": chunk["embedding"],
                    "metadata": {
                        "text": chunk["text"],
                        **chunk["metadata"]
                    }
                })
                ids.append(vector_id)
            
            # Batch upsert in chunks of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                await asyncio.to_thread(
                    self.index.upsert,
                    vectors=batch
                )
                
            logger.info(f"Successfully stored {len(vectors)} vectors")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to store vectors: {str(e)}")
            raise VectorStoreError(f"Vector storage failed: {str(e)}")

    async def query_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Query for similar vectors in Pinecone.
        
        Args:
            query_embedding: Query vector
            top_k: Number of similar vectors to retrieve
            filter: Optional metadata filter
            
        Returns:
            List of similar documents with scores
        """
        try:
            response = await asyncio.to_thread(
                self.index.query,
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter
            )
            
            results = []
            for match in response.matches:
                results.append({
                    "score": match.score,
                    "text": match.metadata["text"],
                    "metadata": {
                        k: v for k, v in match.metadata.items()
                        if k != "text"
                    }
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to query vectors: {str(e)}")
            raise VectorStoreError(f"Vector query failed: {str(e)}")

    async def delete_vectors(
        self,
        filter: Dict[str, Any]
    ) -> None:
        """Delete vectors based on metadata filter."""
        try:
            await asyncio.to_thread(
                self.index.delete,
                filter=filter
            )
            logger.info(f"Successfully deleted vectors with filter: {filter}")
        except Exception as e:
            logger.error(f"Failed to delete vectors: {str(e)}")
            raise VectorStoreError(f"Vector deletion failed: {str(e)}")

