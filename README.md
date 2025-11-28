# Medical Appointment Scheduling Agent - Backend

## Overview

This is a **Medical Appointment Scheduling Agent** backend system that uses AI to help patients book medical appointments through natural conversation. The system leverages Retrieval-Augmented Generation (RAG) to provide accurate information about clinic services, schedules, and policies.

## Project Architecture

### Directory Structure

```
backend/
├── agent/              # Core AI agent logic
│   ├── prompts.py     # Agent instruction templates
│   └── scheduling_agent.py  # Main agent implementation
├── api/               # External service integrations
│   ├── calendly_integration.py  # Scheduling platform API
│   └── chat.py        # Conversational interface
├── data/              # Static information storage
│   ├── clinic_info.json      # Clinic details (location, hours, services)
│   └── doctor_schedule.json  # Doctor availability data
├── models/            # Data validation schemas
│   └── schemas.py     # Pydantic models for type safety
├── rag/               # Knowledge retrieval system
│   ├── embeddings.py  # Text vectorization
│   ├── faq_rag.py     # FAQ retrieval logic
│   └── vector_store.py # ChromaDB interface
├── tools/             # Agent action capabilities
│   ├── availability_tool.py  # Check appointment slots
│   └── booking_tool.py       # Create appointments
├── tests/             # Unit tests
└── vectordb/          # Persistent vector database (generated)
```

## How It Works

### 1. **Knowledge Base (RAG System)**
The system stores clinic FAQs in a vector database (`vectordb/chroma.sqlite3`). When users ask questions:
- Questions are converted to embeddings using `HFEmbeddings`
- Similar Q&A pairs are retrieved from `VectorStore`
- Relevant information is provided to the AI agent

### 2. **Conversational Agent**
The `agent/scheduling_agent.py` coordinates:
- Understanding user intent
- Retrieving clinic information via `rag/faq_rag.py`
- Checking availability with `tools/availability_tool.py`
- Booking appointments through `tools/booking_tool.py`

### 3. **Data Management**
- `data/clinic_info.json`: Services, hours, contact details
- `data/doctor_schedule.json`: Doctor availability
- `models/schemas.py`: Ensures data consistency

## Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository** (if not already done)

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   Create a `.env` file in the backend directory:
   ```env
   COLLECTION_NAME=clinic_faq
   VECTOR_DB_PATH=./vectordb
   GROQ_API_KEY=your_groq_key_here
   OPENAI_API_KEY=your_openai_key_here
   CALENDLY_API_TOKEN=your_calendly_token
   ```

## Setting Up the Vector Database

The vector database must be generated from your FAQ data before running the application.

### Step 1: Prepare FAQ Data

Create a JSON file at `data/clinic_faq.json` with your clinic FAQs:

```json
[
  {
    "question": "What are your operating hours?",
    "answer": "We're open Monday to Friday from 8:00 AM to 6:00 PM, and Saturday from 9:00 AM to 2:00 PM. We're closed on Sundays.",
    "category": "hours"
  },
  {
    "question": "Do you accept insurance?",
    "answer": "Yes, we accept most major insurance providers including Blue Cross, Aetna, and UnitedHealthcare.",
    "category": "insurance"
  },
  {
    "question": "What should I bring to my first appointment?",
    "answer": "Please bring a valid ID, insurance card, list of current medications, and any relevant medical records.",
    "category": "first_visit"
  }
]
```

### Step 2: Create Population Script

Create `scripts/populate_vector_db.py`:

```python
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rag.vector_store import get_vector_store

def populate_database():
    """Populate vector database with FAQ data"""
    
    # Load FAQ data
    faq_path = Path(__file__).parent.parent / 'data' / 'clinic_faq.json'
    
    if not faq_path.exists():
        print(f"Error: FAQ file not found at {faq_path}")
        return
    
    with open(faq_path, 'r', encoding='utf-8') as file:
        faqs = json.load(file)
    
    print(f"Loading {len(faqs)} FAQs into vector database...")
    
    # Initialize vector store
    vector_store = get_vector_store()
    
    # Clear existing data (optional)
    try:
        vector_store.collection.delete(where={})
        print("Cleared existing data from vector store")
    except Exception as e:
        print(f"Note: {e}")
    
    # Populate with FAQs
    for idx, item in enumerate(faqs):
        doc_text = f"Question: {item['question']}\nAnswer: {item['answer']}"
        
        vector_store.add_documents(
            documents=[doc_text],
            ids=[f"faq_{idx}"],
            metadatas=[{
                "question": item['question'],
                "answer": item['answer'],
                "category": item.get('category', 'general')
            }]
        )
    
    print(f"✓ Successfully populated {len(faqs)} FAQ entries into vector database")

if __name__ == "__main__":
    populate_database()
```

### Step 3: Generate the Vector Database

```bash
python scripts/populate_vector_db.py
```

This will create the `vectordb/` directory with ChromaDB files.

### Updating the Database

To add new FAQs or update existing ones:
1. Modify `data/clinic_faq.json`
2. Re-run the population script: `python scripts/populate_vector_db.py`
3. The vector database will be rebuilt with the new data

## Running the Application

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing

Run the test suite:

```bash
pytest tests/
```

Run specific test file:

```bash
pytest tests/test_agent.py -v
```

## Key Components

### RAG System
- **`rag/vector_store.py`**: Manages ChromaDB for semantic search
- **`rag/embeddings.py`**: Converts text to numerical vectors using sentence-transformers
- **`rag/faq_rag.py`**: Retrieves relevant FAQs for user queries

### Agent System
- **`agent/scheduling_agent.py`**: Main AI agent orchestration
- **`agent/prompts.py`**: System prompts and instructions

### Tools
- **`tools/availability_tool.py`**: Checks doctor availability
- **`tools/booking_tool.py`**: Creates appointments

### API Integration
- **`api/calendly_integration.py`**: Integrates with Calendly for scheduling
- **`api/chat.py`**: Handles conversational interactions

## Technology Stack

- **FastAPI**: Web framework
- **Groq/OpenAI**: LLM providers
- **ChromaDB**: Vector database for semantic search
- **Sentence Transformers**: Text embeddings
- **Pydantic**: Data validation
- **Pytest**: Testing framework

## Project Notes

### Why `vectordb/` is Git-Ignored

The `vectordb/` directory is excluded from version control because:
- Vector databases can be large (several MB)
- They can be regenerated from source FAQ data
- Different environments may need custom databases
- Reduces repository size

Each developer/environment should generate their own vector database using the populate script.

### Environment Variables

Required environment variables:
- `GROQ_API_KEY` or `OPENAI_API_KEY`: For LLM access
- `CALENDLY_API_TOKEN`: For appointment booking (optional for testing)
- `COLLECTION_NAME`: ChromaDB collection name (default: `clinic_faq`)
- `VECTOR_DB_PATH`: Path to vector database (default: `./vectordb`)

## Development Workflow

1. Set up Python virtual environment
2. Install dependencies
3. Configure `.env` file
4. Prepare FAQ data in `data/clinic_faq.json`
5. Generate vector database
6. Run tests to verify setup
7. Start development server

## Troubleshooting

### Vector Database Issues
If you encounter vector database errors:
```bash
# Delete existing database
rm -rf vectordb/

# Regenerate
python scripts/populate_vector_db.py
```

### Import Errors
Ensure you're running commands from the backend directory and the virtual environment is activated.

### API Key Issues
Verify your `.env` file has valid API keys and is in the correct location.

## Contact

mail : manikantamaduri2023@gmail.com
