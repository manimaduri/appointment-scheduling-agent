from fastapi import APIRouter, HTTPException, Depends
from backend.models.schemas import (
    ChatRequest,
    ChatResponse,
    FAQRequest,
    FAQResponse,
    ErrorResponse
)
from backend.agent.scheduling_agent import SchedulingAgent
from backend.rag.faq_rag import FAQRAG
from datetime import datetime

router = APIRouter(prefix="/api", tags=["Chat"])

# Global instances (will be set in main.py)
scheduling_agent: SchedulingAgent = None
faq_rag: FAQRAG = None


def set_agent(agent: SchedulingAgent):
    """Set the global scheduling agent instance."""
    global scheduling_agent
    scheduling_agent = agent


def set_faq_rag(rag: FAQRAG):
    """Set the global FAQ RAG instance."""
    global faq_rag
    faq_rag = rag


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for conversational appointment scheduling.
    
    Handles:
    - Appointment booking conversations
    - FAQ questions
    - Availability checks
    """
    if not scheduling_agent:
        raise HTTPException(
            status_code=500,
            detail="Scheduling agent not initialized"
        )
    
    try:
        response = await scheduling_agent.chat(
            message=request.message,
            session_id=request.session_id,
            model=request.model.value
        )
        
        return ChatResponse(
            response=response,
            session_id=request.session_id,
            timestamp=datetime.now(),
            model_used=request.model.value
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat: {str(e)}"
        )


@router.post("/ask-faq", response_model=FAQResponse)
async def ask_faq(request: FAQRequest):
    """
    Direct FAQ endpoint for asking clinic-related questions.
    
    Uses RAG to retrieve relevant information and generate answers.
    """
    if not faq_rag:
        raise HTTPException(
            status_code=500,
            detail="FAQ system not initialized"
        )
    
    try:
        result = faq_rag.ask(
            question=request.question,
            model=request.model.value
        )
        
        return FAQResponse(
            answer=result['answer'],
            sources=result['sources'],
            confidence=result['confidence'],
            model_used=request.model.value
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing FAQ: {str(e)}"
        )


@router.delete("/chat/{session_id}")
async def clear_session(session_id: str):
    """
    Clear chat history for a session.
    """
    if not scheduling_agent:
        raise HTTPException(
            status_code=500,
            detail="Scheduling agent not initialized"
        )
    
    try:
        scheduling_agent.clear_session(session_id)
        return {"message": f"Session {session_id} cleared successfully"}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing session: {str(e)}"
        )


@router.get("/chat/{session_id}/info")
async def get_session_info(session_id: str):
    """
    Get information about a chat session.
    """
    if not scheduling_agent:
        raise HTTPException(
            status_code=500,
            detail="Scheduling agent not initialized"
        )
    
    try:
        info = scheduling_agent.get_session_info(session_id)
        return info
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting session info: {str(e)}"
        )