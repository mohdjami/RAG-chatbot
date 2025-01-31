# app/services/llm_service.py
from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from app.core.config import settings
from app.utils.error_handlers import LLMError
from loguru import logger
import json

class LLMService:
    """Service for interacting with Groq LLM."""
    
    def __init__(self):
        try:
            self.llm = ChatGroq(
                groq_api_key=settings.GROQ_API_KEY,
                model_name="mixtral-8x7b-32768",  # or your preferred model
                temperature=0.7,
                max_tokens=4096
            )
            logger.info("Initialized Groq LLM service")
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {str(e)}")
            raise LLMError(f"LLM initialization failed: {str(e)}")

    async def generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using Groq LLM with context.
        
        Args:
            query: User's question
            context: Retrieved similar documents
            conversation_history: Optional previous conversation
            
        Returns:
            Generated response with metadata
        """
        try:
            # Prepare context
            context_str = "\n\n".join([
                f"Context {i+1}:\n{doc['text']}"
                for i, doc in enumerate(context)
            ])
            
            # Prepare conversation history
            history_str = ""
            if conversation_history:
                history_str = "\n".join([
                    f"{msg['role']}: {msg['content']}"
                    for msg in conversation_history[-5:]  # Last 5 messages
                ])
            
            # Prepare system prompt
            system_prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context.
If the context doesn't contain enough information to answer the question, say so.
Always maintain a professional and helpful tone. If you use information from the context, cite it naturally in your response.
Base your response primarily on the provided context rather than your general knowledge."""

            messages = [
                SystemMessage(content=system_prompt),
            ]
            
            if history_str:
                messages.append(SystemMessage(
                    content=f"Previous conversation:\n{history_str}"
                ))
            
            messages.append(SystemMessage(
                content=f"Context for the current question:\n{context_str}"
            ))
            
            messages.append(HumanMessage(content=query))
            
            # Generate response
            response = await self.llm.ainvoke(messages)
            
            # Extract sources
            sources = [
                {
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "similarity_score": doc["score"]
                }
                for doc in context
                if doc["score"] > 0.7  # Adjust threshold as needed
            ]
            
            return {
                "response": response.content,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Failed to generate LLM response: {str(e)}")
            raise LLMError(f"Response generation failed: {str(e)}")

    async def generate_followup_questions(
        self,
        context: str,
        current_question: str,
        max_questions: int = 3
    ) -> List[str]:
        """Generate follow-up questions based on context and current question."""
        try:
            prompt = f"""Based on the following context and current question, generate {max_questions} relevant follow-up questions that would help explore the topic further.
            
Context: {context}

Current Question: {current_question}

Generate exactly {max_questions} follow-up questions in JSON format like this:
{{"questions": ["question1", "question2", "question3"]}}"""

            messages = [
                SystemMessage(content="You are a helpful AI that generates relevant follow-up questions."),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse JSON response
            questions = json.loads(response.content)["questions"]
            return questions[:max_questions]
            
        except Exception as e:
            logger.error(f"Failed to generate follow-up questions: {str(e)}")
            raise LLMError(f"Follow-up question generation failed: {str(e)}")