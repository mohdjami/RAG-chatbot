# app/services/chat_service.py
from typing import Dict, List, Optional, Set
from fastapi import WebSocket
import asyncio
import json
from datetime import datetime
from loguru import logger
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.llm_service import LLMService
from app.utils.error_handlers import ChatError
from app.models.chat import ConversationModel, MessageModel
from sqlalchemy.ext.asyncio import AsyncSession

class ChatService:
    """
    Chat service that orchestrates the RAG pipeline and manages conversations.
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreService()
        self.llm_service = LLMService()
        self.active_connections: Dict[str, WebSocket] = {}
        self._rate_limits: Dict[str, List[datetime]] = {}
        
    async def connect_websocket(self, websocket: WebSocket, client_id: str):
        """Handle new WebSocket connections."""
        try:
            await websocket.accept()
            self.active_connections[client_id] = websocket
            logger.info(f"WebSocket connected for client: {client_id}")
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {str(e)}")
            raise ChatError(f"Connection failed: {str(e)}")

    async def process_message(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        db_session: Optional[AsyncSession] = None
    ) -> Dict:
        """
        Process a user message through the RAG pipeline.
        
        Args:
            message: User's message
            conversation_id: Optional conversation ID for context
            metadata: Optional metadata for the message
            db_session: Database session for persistence
            
        Returns:
            Generated response with sources and metadata
        """
        try:
            # Rate limiting check
            if not self._check_rate_limit(conversation_id or "anonymous"):
                raise ChatError("Rate limit exceeded. Please wait before sending more messages.")

            # Generate embedding for query
            query_embedding = await self.embedding_service.generate_embeddings([message])
            
            # Retrieve relevant context
            context = await self.vector_store.query_similar(
                query_embedding[0],
                top_k=5,
                filter=metadata
            )
            
            # Get conversation history if conversation_id provided
            history = []
            if conversation_id and db_session:
                history = await self._get_conversation_history(
                    conversation_id,
                    db_session
                )
            
            # Generate response using LLM
            response = await self.llm_service.generate_response(
                query=message,
                context=context,
                conversation_history=history
            )
            
            # Generate follow-up questions
            followup_questions = await self.llm_service.generate_followup_questions(
                context="\n".join([c["text"] for c in context]),
                current_question=message
            )
            
            # Save conversation if db_session provided
            if db_session:
                await self._save_conversation(
                    message=message,
                    response=response["response"],
                    conversation_id=conversation_id,
                    metadata=metadata,
                    db_session=db_session
                )
            
            return {
                "response": response["response"],
                "sources": response["sources"],
                "followup_questions": followup_questions,
                "conversation_id": conversation_id,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Message processing failed: {str(e)}")
            raise ChatError(f"Failed to process message: {str(e)}")

    async def _get_conversation_history(
        self,
        conversation_id: str,
        db_session: AsyncSession
    ) -> List[Dict]:
        """Retrieve conversation history from database."""
        try:
            conversation = await ConversationModel.get_by_id(
                conversation_id,
                db_session
            )
            if not conversation:
                return []
                
            messages = await MessageModel.get_conversation_messages(
                conversation_id,
                db_session,
                limit=10  # Last 10 messages
            )
            
            return [
                {
                    "role": "user" if msg.is_user else "assistant",
                    "content": msg.content
                }
                for msg in messages
            ]
            
        except Exception as e:
            logger.error(f"Failed to retrieve conversation history: {str(e)}")
            return []

    async def _save_conversation(
        self,
        message: str,
        response: str,
        conversation_id: Optional[str],
        metadata: Optional[Dict],
        db_session: AsyncSession
    ):
        """Save conversation messages to database."""
        try:
            if not conversation_id:
                conversation = ConversationModel(metadata=metadata)
                db_session.add(conversation)
                await db_session.flush()
                conversation_id = conversation.id
            
            # Save user message
            user_message = MessageModel(
                conversation_id=conversation_id,
                content=message,
                is_user=True
            )
            db_session.add(user_message)
            
            # Save assistant response
            assistant_message = MessageModel(
                conversation_id=conversation_id,
                content=response,
                is_user=False
            )
            db_session.add(assistant_message)
            
            await db_session.commit()
            
        except Exception as e:
            logger.error(f"Failed to save conversation: {str(e)}")
            await db_session.rollback()

    def _check_rate_limit(self, client_id: str, limit: int = 10, window: int = 60) -> bool:
        """Check if client has exceeded rate limit."""
        now = datetime.now()
        if client_id not in self._rate_limits:
            self._rate_limits[client_id] = []
            
        # Remove old timestamps
        self._rate_limits[client_id] = [
            ts for ts in self._rate_limits[client_id]
            if (now - ts).seconds < window
        ]
        
        # Check limit
        if len(self._rate_limits[client_id]) >= limit:
            return False
            
        # Add new timestamp
        self._rate_limits[client_id].append(now)
        return True

