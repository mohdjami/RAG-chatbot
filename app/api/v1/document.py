# Example usage in API endpoint
# app/api/v1/document.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService

router = APIRouter()

@router.post("/process")
async def process_document(
    file: UploadFile = File(...),
    metadata: Optional[Dict[str, Any]] = None
):
    try:
        # Save uploaded file temporarily
        temp_path = f"temp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document
        processor = DocumentProcessor()
        chunks = await processor.process_document(temp_path, metadata)
        
        # Generate embeddings
        embedding_service = EmbeddingService()
        texts = [chunk["text"] for chunk in chunks]
        embeddings = await embedding_service.generate_embeddings(texts)
        
        # Combine chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
        
        return {"status": "success", "chunks": chunks}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )