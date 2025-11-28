import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.api import chat, calendly_integration
from backend.rag.vector_store import get_vector_store
from backend.rag.faq_rag import get_faq_rag
from backend.tools.availability_tool import get_availability_tool
from backend.tools.booking_tool import get_booking_tool
from backend.agent.scheduling_agent import get_scheduling_agent

load_dotenv()

# ======================================
# 🚀 CREATE ONE SINGLE FASTAPI APP
# ======================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Medical Appointment Scheduling Agent...")

    global vector_store, faq_rag, scheduling_agent

    try:
        # Vector store
        print("📊 Initializing vector store...")
        vector_store = get_vector_store()

        clinic_info_path = os.path.join("data", "clinic_info.json")
        if os.path.exists(clinic_info_path):
            vector_store.initialize_from_json(clinic_info_path)

        # FAQ RAG
        print("🤖 Initializing FAQ RAG system...")
        faq_rag = get_faq_rag(vector_store)

        # Tools
        print("🔧 Initializing tools...")
        base_url = f"http://{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', 8000)}"
        availability_tool = get_availability_tool(base_url)
        booking_tool = get_booking_tool(base_url)

        # Agent
        print("🎯 Initializing scheduling agent...")
        scheduling_agent = get_scheduling_agent(
            faq_rag=faq_rag,
            availability_tool=availability_tool,
            booking_tool=booking_tool
        )

        # Inject into routers
        chat.set_agent(scheduling_agent)
        chat.set_faq_rag(faq_rag)

        print("✅ All systems initialized successfully!")
        print(f"📍 Server running on {base_url}")
        print(f"📚 API docs at {base_url}/docs")

    except Exception as e:
        print(f"❌ Error during startup: {e}")
        raise

    yield

    print("👋 Shutting down...")

# Main app
app = FastAPI(
    title="Medical Appointment Scheduling Agent",
    description="Conversational AI agent with scheduling + FAQ",
    version="1.0.0",
    lifespan=lifespan
)

# ======================================
# 🌍 ENABLE FULL CORS (on the correct app)
# ======================================

cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================
# 📌 INCLUDE ROUTERS
# ======================================

app.include_router(chat.router)
app.include_router(calendly_integration.router)

@app.get("/")
async def root():
    return {
        "message": "Medical Appointment Scheduling Agent API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vector_store": True,
        "faq_rag": True,
        "scheduling_agent": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
