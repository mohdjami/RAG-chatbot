# app/models/chat.py
from sqlalchemy import Column, String, Boolean, ForeignKey, JSON, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from uuid import uuid4
from app.db.base_class import Base

class ConversationModel(Base):
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON)
    
    messages = relationship("MessageModel", back_populates="conversation")
    
    @classmethod
    async def get_by_id(cls, id: str, db_session):
        return await db_session.get(cls, id)

class MessageModel(Base):
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"))
    content = Column(String)
    is_user = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("ConversationModel", back_populates="messages")
    
    @classmethod
    async def get_conversation_messages(cls, conversation_id: str, db_session, limit: int = 10):
        query = await db_session.execute(
            select(cls)
            .where(cls.conversation_id == conversation_id)
            .order_by(cls.created_at.desc())
            .limit(limit)
        )
        return query.scalars().all()