from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add the services directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
services_dir = os.path.join(current_dir, '..', 'services')
sys.path.insert(0, services_dir)

from advanced_rag_llm_system import AdvancedRAGEnhancedLLM  # Ensure advanced_rag_llm_system.py exists in ../services/

app = FastAPI(title="ARGO FloatChat API")

# CORS for Next.js development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system with error handling
try:
    rag_system = AdvancedRAGEnhancedLLM()
    print("RAG system initialized successfully")
except Exception as e:
    print(f"Failed to initialize RAG system: {e}")
    rag_system = None

class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
async def process_query(request: QueryRequest):
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        result = rag_system.process_advanced_query(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_system_status": "initialized" if rag_system else "failed"
    }