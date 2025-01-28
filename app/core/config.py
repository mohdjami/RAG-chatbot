from typing import Any, Dict, Optional
from pydantic_settings import BaseSettings
from pydantic import validator
import secrets

class Settings(BaseSettings):
    PROJECT_NAME: str = "RAG Chatbot"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # Database
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    SQLALCHEMY_DATABASE_URI: Optional[str] = None

    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return f"postgresql+asyncpg://{values.get('POSTGRES_USER')}:{values.get('POSTGRES_PASSWORD')}@{values.get('POSTGRES_SERVER')}/{values.get('POSTGRES_DB')}"
    
    # External Services
    GROQ_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str
    JINA_API_KEY: str
    
    # Vector Store
    VECTOR_DIMENSION: int = 768  # Default for many embedding models
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()