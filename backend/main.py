# main.py - FastAPI Entry Point for FloatChat
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import json
import logging
from datetime import datetime
import traceback
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# Import your existing advanced RAG system
from src.services.advanced_rag_llm_system import AdvancedRAGEnhancedLLM
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="FloatChat API",
    description="Advanced oceanographic data analysis using ARGO float data with RAG-enhanced LLM",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
rag_system = AdvancedRAGEnhancedLLM()
engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system and database connection on startup"""
    global rag_system, engine
    try:
        logger.info("Initializing FloatChat backend services...")
        
        # Initialize database engine
        engine = create_engine(os.getenv('DATABASE_URL'))
        
        # Initialize RAG system
        rag_system = AdvancedRAGEnhancedLLM()
        
        logger.info("FloatChat backend services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize backend services: {e}")
        raise

# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query about oceanographic data")
    include_sql: bool = Field(True, description="Whether to include the generated SQL in response")
    limit: Optional[int] = Field(1000, description="Maximum number of results to return")

class QueryResponse(BaseModel):
    success: bool
    query: str
    sql_query: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None
    result_count: int = 0
    columns: List[str] = []
    processing_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}

class FloatFilters(BaseModel):
    region: Optional[str] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    platform_numbers: Optional[List[str]] = None
    limit: Optional[int] = 100

class FloatInfo(BaseModel):
    platform_number: str
    cycle_number: int
    date: str
    latitude: float
    longitude: float
    project_name: str
    institution: str
    measurement_count: int

# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "FloatChat API",
        "version": "1.0.0",
        "description": "Advanced oceanographic data analysis API",
        "status": "operational"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # Test RAG system
        rag_available = rag_system is not None and rag_system.vector_store is not None
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "rag_system": "available" if rag_available else "unavailable",
            "vector_store": "loaded" if rag_available else "not_loaded"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process natural language query and return oceanographic analysis"""
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Process query using advanced RAG system
        result = rag_system.process_advanced_query(request.query)
        
        # Format response
        response_data = {
            "success": result['success'],
            "query": request.query,
            "result_count": result.get('result_count', 0),
            "columns": result.get('columns', []),
            "processing_time": result.get('processing_time', 0.0),
            "metadata": {
                "query_type": "advanced_rag",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Include SQL if requested and available
        if request.include_sql and 'sql_query' in result:
            response_data["sql_query"] = result['sql_query']
        
        # Format results for JSON response
        if result['success'] and 'results' in result:
            results_df = result['results']
            if isinstance(results_df, pd.DataFrame):
                # Convert DataFrame to list of dictionaries
                if not results_df.empty:
                    # Limit results if specified
                    if request.limit and len(results_df) > request.limit:
                        results_df = results_df.head(request.limit)
                    
                    response_data["results"] = results_df.to_dict('records')
                    response_data["result_count"] = len(results_df)
                else:
                    response_data["results"] = []
            elif isinstance(results_df, list):
                response_data["results"] = results_df
        
        # Handle errors
        if not result['success']:
            response_data["error"] = result.get('error', 'Unknown error occurred')
        
        return QueryResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResponse(
            success=False,
            query=request.query,
            error=str(e),
            processing_time=processing_time,
            metadata={"error_type": "processing_error"}
        )

@app.get("/api/floats", response_model=List[FloatInfo])
async def get_floats(filters: FloatFilters = None):
    """Get available ARGO floats with optional filtering"""
    
    try:
        # Build base query
        query = """
        SELECT 
            fm.platform_number,
            fm.cycle_number,
            fm.date,
            fm.latitude,
            fm.longitude,
            fm.project_name,
            fm.institution,
            COUNT(m.id) as measurement_count
        FROM floats_metadata fm
        LEFT JOIN measurements m ON fm.id = m.metadata_id
        """
        
        conditions = []
        params = {}
        
        # Apply filters if provided
        if filters:
            if filters.region:
                # Simple region filtering (you can enhance this with proper geographic bounds)
                if filters.region.lower() == "arabian_sea":
                    conditions.append("fm.latitude BETWEEN 8 AND 27 AND fm.longitude BETWEEN 50 AND 80")
                elif filters.region.lower() == "bay_of_bengal":
                    conditions.append("fm.latitude BETWEEN 5 AND 22 AND fm.longitude BETWEEN 80 AND 100")
            
            if filters.date_start:
                conditions.append("fm.date >= :date_start")
                params["date_start"] = filters.date_start
            
            if filters.date_end:
                conditions.append("fm.date <= :date_end")
                params["date_end"] = filters.date_end
            
            if filters.platform_numbers:
                placeholders = ", ".join([f":platform_{i}" for i in range(len(filters.platform_numbers))])
                conditions.append(f"fm.platform_number IN ({placeholders})")
                for i, platform in enumerate(filters.platform_numbers):
                    params[f"platform_{i}"] = platform
        
        # Add conditions to query
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += """
        GROUP BY fm.platform_number, fm.cycle_number, fm.date, 
                 fm.latitude, fm.longitude, fm.project_name, fm.institution
        ORDER BY fm.date DESC
        """
        
        # Add limit
        if filters and filters.limit:
            query += f" LIMIT {filters.limit}"
        else:
            query += " LIMIT 100"
        
        # Execute query
        with engine.connect() as conn:
            result = conn.execute(text(query), params)
            rows = result.fetchall()
        
        # Format response
        floats = []
        for row in rows:
            floats.append(FloatInfo(
                platform_number=row.platform_number,
                cycle_number=row.cycle_number,
                date=row.date.isoformat() if row.date else "",
                latitude=float(row.latitude) if row.latitude else 0.0,
                longitude=float(row.longitude) if row.longitude else 0.0,
                project_name=row.project_name or "",
                institution=row.institution or "",
                measurement_count=int(row.measurement_count) if row.measurement_count else 0
            ))
        
        logger.info(f"Retrieved {len(floats)} floats")
        return floats
        
    except Exception as e:
        logger.error(f"Error retrieving floats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_database_stats():
    """Get database statistics for dashboard"""
    
    try:
        stats_query = """
        SELECT 
            (SELECT COUNT(*) FROM floats_metadata) as total_floats,
            (SELECT COUNT(*) FROM measurements) as total_measurements,
            (SELECT COUNT(DISTINCT fm.project_name) FROM floats_metadata fm) as total_projects,
            (SELECT MIN(fm.date) FROM floats_metadata fm WHERE fm.date IS NOT NULL) as earliest_date,
            (SELECT MAX(fm.date) FROM floats_metadata fm WHERE fm.date IS NOT NULL) as latest_date,
            (SELECT AVG(m.temperature) FROM measurements m WHERE m.temperature IS NOT NULL) as avg_temperature,
            (SELECT AVG(m.salinity) FROM measurements m WHERE m.salinity IS NOT NULL) as avg_salinity
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(stats_query))
            row = result.fetchone()
        
        return {
            "total_floats": row.total_floats,
            "total_measurements": row.total_measurements,
            "total_projects": row.total_projects,
            "date_range": {
                "earliest": row.earliest_date.isoformat() if row.earliest_date else None,
                "latest": row.latest_date.isoformat() if row.latest_date else None
            },
            "averages": {
                "temperature": float(row.avg_temperature) if row.avg_temperature else None,
                "salinity": float(row.avg_salinity) if row.avg_salinity else None
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time chat
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_message(self, websocket: WebSocket, message: dict):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
    
    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                self.disconnect(connection)

manager = ConnectionManager()

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message from frontend
            data = await websocket.receive_text()
            message = json.loads(data)
            
            query = message.get('query', '')
            message_id = message.get('message_id', '')
            
            if not query:
                await manager.send_message(websocket, {
                    'type': 'error',
                    'message': 'Empty query received',
                    'message_id': message_id
                })
                continue
            
            # Send processing status updates
            await manager.send_message(websocket, {
                'type': 'status',
                'message': 'Processing your query...',
                'stage': 'starting',
                'message_id': message_id
            })
            
            # Simulate processing stages with real work
            await asyncio.sleep(0.5)  # Small delay for UX
            
            await manager.send_message(websocket, {
                'type': 'status',
                'message': 'Analyzing query intent...',
                'stage': 'intent_analysis',
                'message_id': message_id
            })
            
            # Classify query intent
            intent = rag_system.classify_query_intent(query)
            
            await manager.send_message(websocket, {
                'type': 'status',
                'message': f'Detected {intent["type"]} query, generating SQL...',
                'stage': 'sql_generation',
                'message_id': message_id
            })
            
            await asyncio.sleep(0.3)
            
            # Process the full query
            result = rag_system.process_advanced_query(query)
            
            # Convert DataFrame to dict if present
            if result.get('results') is not None and hasattr(result['results'], 'to_dict'):
                result['results'] = result['results'].to_dict('records')
            
            # Send final result
            await manager.send_message(websocket, {
                'type': 'result',
                'data': result,
                'message_id': message_id
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await manager.send_message(websocket, {
                'type': 'error',
                'message': f'Server error: {str(e)}',
                'message_id': message.get('message_id', '')
            })
        except:
            pass
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )