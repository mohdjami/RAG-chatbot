
# app/services/embedding_service.py
from typing import List, Dict, Any
import asyncio
from loguru import logger
from jina import Client
from app.core.config import settings
from app.utils.error_handlers import DocumentProcessingError

class EmbeddingService:
    """Service for generating embeddings using Jina AI."""
    
    def __init__(self):
        self.client = Client(
            host=settings.JINA_API_ENDPOINT,
            api_key=settings.JINA_API_KEY
        )
    
    async def generate_embeddings(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            texts: List of text chunks to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Process in batches to avoid overwhelming the API
            batch_size = 20
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = await self._process_batch(batch)
                embeddings.extend(batch_embeddings)
                
                logger.debug(
                    f"Processed embedding batch {i//batch_size + 1}/"
                    f"{(len(texts) + batch_size - 1)//batch_size}"
                )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise DocumentProcessingError(
                f"Embedding generation failed: {str(e)}"
            )
    
    async def _process_batch(self, texts: List[str]) -> List[List[float]]:
        """Process a batch of texts to generate embeddings."""
        try:
            response = await asyncio.to_thread(
                self.client.encode,
                texts
            )
            return response.embeddings
        except Exception as e:
            logger.error(f"Error processing embedding batch: {str(e)}")
            raise DocumentProcessingError(
                f"Batch embedding processing failed: {str(e)}"
            )

