import pytest
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag.vector_store import VectorStore
from backend.rag.faq_rag import FAQRAG
from backend.tools.availability_tool import AvailabilityTool
from backend.tools.booking_tool import BookingTool
from backend.agent.scheduling_agent import SchedulingAgent


@pytest.fixture
def vector_store():
    """Create a test vector store."""
    store = VectorStore(
        collection_name="test_clinic_faq",
        persist_directory="./test_vectordb"
    )
    
    # Add test documents
    test_docs = [
        "Question: What are clinic hours?\nAnswer: 9 AM to 5 PM Monday to Friday",
        "Question: Where is the clinic?\nAnswer: 123 Medical Center Drive"
    ]
    test_metadata = [
        {"question": "What are clinic hours?", "answer": "9 AM to 5 PM Monday to Friday", "category": "general"},
        {"question": "Where is the clinic?", "answer": "123 Medical Center Drive", "category": "location"}
    ]
    
    store.add_documents(test_docs, test_metadata)
    
    yield store
    
    # Cleanup
    try:
        store.delete_collection()
    except:
        pass


@pytest.fixture
def faq_rag(vector_store):
    """Create a test FAQ RAG instance."""
    return FAQRAG(vector_store)


@pytest.fixture
def availability_tool():
    """Create availability tool instance."""
    return AvailabilityTool("http://localhost:8000")


@pytest.fixture
def booking_tool():
    """Create booking tool instance."""
    return BookingTool("http://localhost:8000")


@pytest.fixture
def scheduling_agent(faq_rag, availability_tool, booking_tool):
    """Create scheduling agent instance."""
    return SchedulingAgent(faq_rag, availability_tool, booking_tool)


def test_vector_store_initialization(vector_store):
    """Test vector store initialization."""
    assert vector_store is not None
    assert vector_store.get_collection_count() > 0


def test_vector_store_search(vector_store):
    """Test vector store similarity search."""
    results = vector_store.similarity_search("clinic hours", k=1)
    
    assert len(results) > 0
    assert "clinic hours" in results[0]['document'].lower()


def test_faq_rag_answer(faq_rag):
    """Test FAQ RAG answer generation."""
    result = faq_rag.ask("What are your hours?")
    
    assert result is not None
    assert 'answer' in result
    assert 'confidence' in result
    assert len(result['answer']) > 0


@pytest.mark.asyncio
async def test_availability_tool():
    """Test availability tool."""
    tool = AvailabilityTool("http://localhost:8000")
    
    # Get tomorrow's date
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    result = await tool.check_availability(
        date=tomorrow,
        appointment_type="Consultation"
    )
    
    # Note: This test assumes the API is running
    # In a real test, you'd mock the HTTP calls
    assert result is not None


@pytest.mark.asyncio
async def test_scheduling_agent_faq(scheduling_agent):
    """Test scheduling agent FAQ handling."""
    response = await scheduling_agent.chat(
        message="What are your clinic hours?",
        session_id="test_session_1"
    )
    
    assert response is not None
    assert len(response) > 0


@pytest.mark.asyncio
async def test_scheduling_agent_greeting(scheduling_agent):
    """Test scheduling agent greeting."""
    response = await scheduling_agent.chat(
        message="Hello",
        session_id="test_session_2"
    )
    
    assert response is not None
    assert len(response) > 0


@pytest.mark.asyncio
async def test_scheduling_agent_booking_intent(scheduling_agent):
    """Test scheduling agent booking intent."""
    response = await scheduling_agent.chat(
        message="I want to book an appointment",
        session_id="test_session_3"
    )
    
    assert response is not None
    assert len(response) > 0


def test_session_management(scheduling_agent):
    """Test session management."""
    session_id = "test_session_4"
    
    info_before = scheduling_agent.get_session_info(session_id)
    assert info_before['message_count'] == 0
    
    # Clear session
    scheduling_agent.clear_session(session_id)
    
    info_after = scheduling_agent.get_session_info(session_id)
    assert info_after['message_count'] == 0


def test_tool_descriptions(availability_tool, booking_tool):
    """Test tool descriptions."""
    avail_desc = availability_tool.get_tool_description()
    booking_desc = booking_tool.get_tool_description()
    
    assert avail_desc is not None
    assert 'function' in avail_desc
    assert avail_desc['function']['name'] == 'check_availability'
    
    assert booking_desc is not None
    assert 'function' in booking_desc
    assert booking_desc['function']['name'] == 'book_appointment'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])