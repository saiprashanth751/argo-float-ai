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
import os

load_dotenv()

# Import the new intelligent system
from src.services.enhanced_rag_oceanographic import EnhancedOceanographicRAG
from src.services.intelligent_response_system import IntelligentResponseSystem, ResponseFormat
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FloatChat API",
    description="Advanced oceanographic data analysis using ARGO float data with intelligent RAG system",
    version="2.0.0"
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
rag_system = None
intelligent_system = None
engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize the intelligent system and database connection on startup"""
    global rag_system, intelligent_system, engine
    try:
        logger.info("Initializing FloatChat backend services...")
        
        # Initialize database engine
        engine = create_engine(
            os.getenv('DATABASE_URL', 'postgresql://argo_user:argo_password@localhost:5432/argo_data'),
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True
        )
        
        # Initialize enhanced RAG system
        rag_system = EnhancedOceanographicRAG(db_engine=engine)
        
        # Initialize intelligent response system
        intelligent_system = IntelligentResponseSystem(rag_system=rag_system)
        
        logger.info("FloatChat backend services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize backend services: {e}")
        raise

# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query about oceanographic data")
    include_sql: bool = Field(True, description="Whether to include the generated SQL in response")
    limit: Optional[int] = Field(1000, description="Maximum number of results to return")
    response_format: Optional[Dict[str, Any]] = Field(None, description="Response format preferences")

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
    
    # New intelligent response fields
    narrative_response: Optional[str] = None
    classification: Optional[Dict[str, Any]] = None
    insights: Optional[Dict[str, Any]] = None
    visualizations: Optional[List[Dict[str, Any]]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None

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

# Compatibility wrapper for existing API endpoints
class BackwardCompatibilityWrapper:
    """Wrapper to maintain compatibility with old API while using new system"""
    
    def __init__(self, intelligent_system: IntelligentResponseSystem):
        self.intelligent_system = intelligent_system
        self.rag_system = intelligent_system.rag_system
    
    def process_advanced_query(self, query: str):
        """Legacy method - uses enhanced RAG system"""
        try:
            result = self.rag_system.process_oceanographic_query(query)
            
            # Convert to legacy format
            return {
                'success': result['success'],
                'results': result.get('results'),
                'result_count': result.get('result_count', 0),
                'columns': result.get('columns', []),
                'sql_query': result.get('sql_query'),
                'processing_time': result.get('processing_time', 0.0),
                'error': result.get('error')
            }
        except Exception as e:
            logger.error(f"Legacy query processing error: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': 0.0
            }
    
    def classify_query_intent(self, query: str):
        """Legacy method - uses oceanographic intelligence"""
        try:
            classification = self.rag_system.ocean_intelligence.classify_query(query)
            return {
                'type': classification.intent.value,
                'confidence': classification.confidence,
                'complexity': classification.complexity.value
            }
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return {
                'type': 'exploration',
                'confidence': 0.5,
                'complexity': 'basic'
            }

# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "FloatChat API",
        "version": "2.0.0",
        "description": "Advanced oceanographic data analysis API with intelligent responses",
        "status": "operational",
        "features": ["Enhanced RAG", "Oceanographic Intelligence", "Multi-modal Responses"]
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # Test systems
        rag_available = rag_system is not None and rag_system.vector_store is not None
        intelligent_available = intelligent_system is not None
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "enhanced_rag_system": "available" if rag_available else "unavailable",
            "intelligent_system": "available" if intelligent_available else "unavailable",
            "vector_store": "loaded" if rag_available else "not_loaded",
            "version": "2.0.0"
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
    """Process natural language query and return intelligent oceanographic analysis"""
    
    if not intelligent_system:
        raise HTTPException(status_code=503, detail="Intelligent system not initialized")
    
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Create response format from request
        response_format = ResponseFormat()
        if request.response_format:
            if 'target_audience' in request.response_format:
                response_format.target_audience = request.response_format['target_audience']
            if 'complexity_level' in request.response_format:
                response_format.complexity_level = request.response_format['complexity_level']
            if 'include_visualizations' in request.response_format:
                response_format.include_visualizations = request.response_format['include_visualizations']
        
        # Process query using intelligent system
        result = intelligent_system.process_intelligent_query(request.query, response_format)
        
        # Format response
        response_data = {
            "success": result['success'],
            "query": request.query,
            "processing_time": result.get('processing_time', 0.0),
            "metadata": {
                "query_type": "intelligent_oceanographic",
                "timestamp": datetime.now().isoformat(),
                "version": "2.0.0"
            }
        }
        
        if result['success']:
            # Core data
            response_data.update({
                "result_count": result['results_summary']['total_records'],
                "columns": result['results_summary']['columns'],
                "classification": result['classification'],
                "narrative_response": result['narrative_response'],
                "insights": result['scientific_insights'],
                "visualizations": result['visualizations'],
                "recommendations": result['recommendations']
            })
            
            # Include SQL if requested
            if request.include_sql:
                response_data["sql_query"] = result['sql_query']
            
            # Include results data if available
            if 'results' in result and not result['results'].empty:
                results_df = result['results']
                
                # Limit results if specified
                if request.limit and len(results_df) > request.limit:
                    results_df = results_df.head(request.limit)
                
                response_data["results"] = results_df.to_dict('records')
                response_data["result_count"] = len(results_df)
        else:
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

@app.post("/api/query/legacy")
async def process_legacy_query(request: QueryRequest):
    """Legacy endpoint for backward compatibility"""
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Create compatibility wrapper
        wrapper = BackwardCompatibilityWrapper(intelligent_system)
        
        # Process using legacy method
        result = wrapper.process_advanced_query(request.query)
        
        # Format for legacy response structure
        response_data = {
            "success": result['success'],
            "query": request.query,
            "result_count": result.get('result_count', 0),
            "columns": result.get('columns', []),
            "processing_time": result.get('processing_time', 0.0),
            "metadata": {
                "query_type": "legacy_rag",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        if request.include_sql and 'sql_query' in result:
            response_data["sql_query"] = result['sql_query']
        
        if result['success'] and 'results' in result and result['results'] is not None:
            results_df = result['results']
            if isinstance(results_df, pd.DataFrame) and not results_df.empty:
                if request.limit and len(results_df) > request.limit:
                    results_df = results_df.head(request.limit)
                response_data["results"] = results_df.to_dict('records')
            elif isinstance(results_df, list):
                response_data["results"] = results_df
        
        if not result['success']:
            response_data["error"] = result.get('error', 'Unknown error occurred')
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing legacy query: {e}")
        return {
            "success": False,
            "query": request.query,
            "error": str(e),
            "processing_time": 0.0,
            "metadata": {"error_type": "legacy_processing_error"}
        }

@app.get("/api/floats", response_model=List[FloatInfo])
async def get_floats(filters: FloatFilters = None):
    """Get available ARGO floats with optional filtering"""
    
    try:
        # Build base query - updated for enhanced tables
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
        FROM enhanced_floats_metadata fm
        LEFT JOIN enhanced_measurements m ON fm.id = m.metadata_id
        """
        
        conditions = []
        params = {}
        
        # Apply filters if provided
        if filters:
            if filters.region:
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
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += """
        GROUP BY fm.platform_number, fm.cycle_number, fm.date, 
                 fm.latitude, fm.longitude, fm.project_name, fm.institution
        ORDER BY fm.date DESC
        """
        
        if filters and filters.limit:
            query += f" LIMIT {filters.limit}"
        else:
            query += " LIMIT 100"
        
        with engine.connect() as conn:
            result = conn.execute(text(query), params)
            rows = result.fetchall()
        
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
            (SELECT COUNT(*) FROM enhanced_floats_metadata) as total_floats,
            (SELECT COUNT(*) FROM enhanced_measurements) as total_measurements,
            (SELECT COUNT(DISTINCT fm.project_name) FROM enhanced_floats_metadata fm WHERE fm.project_name IS NOT NULL) as total_projects,
            (SELECT MIN(fm.date) FROM enhanced_floats_metadata fm WHERE fm.date IS NOT NULL) as earliest_date,
            (SELECT MAX(fm.date) FROM enhanced_floats_metadata fm WHERE fm.date IS NOT NULL) as latest_date,
            (SELECT AVG(m.temperature) FROM enhanced_measurements m WHERE m.temperature IS NOT NULL) as avg_temperature,
            (SELECT AVG(m.salinity) FROM enhanced_measurements m WHERE m.salinity IS NOT NULL) as avg_salinity
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
            "timestamp": datetime.now().isoformat(),
            "system_version": "2.0.0"
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
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
    
    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message, default=str))
            except:
                self.disconnect(connection)

manager = ConnectionManager()

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            query = message.get('query', '')
            message_id = message.get('message_id', '')
            response_format_data = message.get('response_format', {})
            
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
                'message': 'Processing your query with oceanographic intelligence...',
                'stage': 'starting',
                'message_id': message_id
            })
            
            await asyncio.sleep(0.3)
            
            await manager.send_message(websocket, {
                'type': 'status',
                'message': 'Analyzing query intent and oceanographic context...',
                'stage': 'intent_analysis',
                'message_id': message_id
            })
            
            # Create compatibility wrapper for WebSocket
            wrapper = BackwardCompatibilityWrapper(intelligent_system)
            intent = wrapper.classify_query_intent(query)
            
            await manager.send_message(websocket, {
                'type': 'status',
                'message': f'Detected {intent["type"]} query ({intent["confidence"]:.0%} confidence), generating response...',
                'stage': 'processing',
                'message_id': message_id
            })
            
            await asyncio.sleep(0.2)
            
            # Create response format
            response_format = ResponseFormat()
            if 'target_audience' in response_format_data:
                response_format.target_audience = response_format_data['target_audience']
            if 'complexity_level' in response_format_data:
                response_format.complexity_level = response_format_data['complexity_level']
            
            # Process with intelligent system
            result = intelligent_system.process_intelligent_query(query, response_format)
            
            # Convert DataFrame to dict if present
            if result.get('results') is not None and hasattr(result['results'], 'to_dict'):
                result['results'] = result['results'].to_dict('records')
            
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