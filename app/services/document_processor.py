from typing import List, Optional, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Doc2txtLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    
    SPPORTED_FORMATS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': Doc2txtLoader,
    }
    def __init__(
        self,
        chunk_size=1000,
        chunk_overlap=100,
        separators: List[str] = ["\n\n", "\n", " ", ""]
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.text_splitter =  RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )


    def process_document(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        # do some processing
        return self.document