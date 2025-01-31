# app/services/document_processor.py
from typing import List, Optional, Dict, Any
from pathlib import Path
import asyncio
from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from app.core.config import settings
from app.utils.error_handlers import DocumentProcessingError

class DocumentProcessor:
    """Document processing service that handles various document types and chunking."""
    
    SUPPORTED_FORMATS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': Docx2txtLoader,
    }
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = ["\n\n", "\n", " ", ""]
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )
    
    async def process_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a document and return chunked text with metadata.
        
        Args:
            file_path: Path to the document
            metadata: Additional metadata to attach to chunks
            
        Returns:
            List of dictionaries containing text chunks and metadata
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise DocumentProcessingError(f"File not found: {file_path}")
            
            # Get appropriate loader
            loader_cls = self.SUPPORTED_FORMATS.get(path.suffix.lower())
            if not loader_cls:
                raise DocumentProcessingError(
                    f"Unsupported file format: {path.suffix}"
                )
            
            logger.info(f"Processing document: {path.name}")
            
            # Load document
            loader = loader_cls(file_path)
            docs = await asyncio.to_thread(loader.load)
            
            # Split text into chunks
            chunks = await self._split_documents(docs)
            
            # Add metadata
            base_metadata = {
                "source": path.name,
                "file_type": path.suffix.lower(),
                **metadata or {}
            }
            
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    **base_metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                processed_chunks.append({
                    "text": chunk.page_content,
                    "metadata": chunk_metadata
                })
            
            logger.info(
                f"Successfully processed {path.name}: "
                f"{len(processed_chunks)} chunks created"
            )
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise DocumentProcessingError(f"Document processing failed: {str(e)}")
    
    async def _split_documents(self, docs: List[Any]) -> List[Any]:
        """Split documents into chunks using text splitter."""
        try:
            return await asyncio.to_thread(
                self.text_splitter.split_documents,
                docs
            )
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise DocumentProcessingError(f"Document splitting failed: {str(e)}")
