#main.py
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

# FIXED: Import the new intelligent system with proper error handling
try:
    from src.services.enhanced_rag_oceanographic import EnhancedOceanographicRAG
    from src.services.intelligent_response_system import IntelligentResponseSystem, ResponseFormat
except ImportError:
    # Alternative import path
    import sys
    sys.path.append('src/services')
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
    allow_origins=["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances with error handling
rag_system = None
intelligent_system = None
engine = None
system_ready = False

# FIXED: Robust system initialization with fallbacks
@app.on_event("startup")
async def startup_event():
    """Initialize the intelligent system and database connection on startup"""
    global rag_system, intelligent_system, engine, system_ready
    
    try:
        logger.info("Initializing FloatChat backend services...")
        
        # Initialize database engine with connection pooling
        database_url = os.getenv('DATABASE_URL', 'postgresql://argo_user:argo_password@localhost:5432/argo_data')
        engine = create_engine(
            database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections every hour
            echo=False  # Set to True for SQL debugging
        )
        
        # Test database connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            logger.info("Database connection successful")
        
        # Initialize enhanced RAG system with error handling
        try:
            rag_system = EnhancedOceanographicRAG(db_engine=engine)
            logger.info("Enhanced RAG system initialized")
        except Exception as e:
            logger.warning(f"RAG system initialization failed: {e}")
            # Create minimal fallback system
            rag_system = None
        
        # Initialize intelligent response system
        try:
            if rag_system:
                intelligent_system = IntelligentResponseSystem(rag_system=rag_system)
                logger.info("Intelligent response system initialized")
            else:
                logger.warning("Intelligent system not available - RAG system failed")
                intelligent_system = None
        except Exception as e:
            logger.warning(f"Intelligent system initialization failed: {e}")
            intelligent_system = None
        
        system_ready = True
        logger.info("FloatChat backend services initialization completed")
        
    except Exception as e:
        logger.error(f"CRITICAL: Failed to initialize backend services: {e}")
        system_ready = False
        raise

# FIXED: Pydantic models with proper validation
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language query about oceanographic data")
    include_sql: bool = Field(True, description="Whether to include the generated SQL in response")
    limit: Optional[int] = Field(1000, ge=1, le=10000, description="Maximum number of results to return")
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
    
    # Intelligent response fields
    narrative_response: Optional[str] = None
    classification: Optional[Dict[str, Any]] = None
    insights: Optional[Dict[str, Any]] = None
    visualizations: Optional[List[Dict[str, Any]]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None

# FIXED: Robust fallback system for when AI components fail
class FallbackQueryProcessor:
    """Fallback system when AI components are unavailable"""
    
    def __init__(self, db_engine):
        self.engine = db_engine
    
    def process_basic_query(self, query: str) -> Dict[str, Any]:
        """Process basic queries using hardcoded patterns"""
        
        query_lower = query.lower()
        
        try:
            if 'count' in query_lower and 'profile' in query_lower:
                sql = "SELECT COUNT(*) as total_profiles FROM argo_profiles;"
                return self._execute_fallback_sql(sql, query)
            
            elif 'platform' in query_lower and any(char.isdigit() for char in query):
                # Extract platform number
                import re
                platform_match = re.search(r'(\d{7})', query)
                if platform_match:
                    platform_num = platform_match.group(1)
                    sql = """
                    SELECT p.platform_number, p.profile_date, p.latitude, p.longitude,
                           p.surface_temp, p.surface_salinity
                    FROM argo_profiles p
                    WHERE p.platform_number = %s
                    ORDER BY p.profile_date DESC
                    LIMIT 100;
                    """
                    return self._execute_fallback_sql(sql, query, (platform_num,))
            
            elif 'temperature' in query_lower and 'average' in query_lower:
                sql = """
                SELECT AVG(p.surface_temp) as avg_surface_temperature,
                       COUNT(*) as profile_count
                FROM argo_profiles p
                WHERE p.surface_temp IS NOT NULL;
                """
                return self._execute_fallback_sql(sql, query)
            
            elif 'salinity' in query_lower and 'average' in query_lower:
                sql = """
                SELECT AVG(p.surface_salinity) as avg_surface_salinity,
                       COUNT(*) as profile_count
                FROM argo_profiles p
                WHERE p.surface_salinity IS NOT NULL;
                """
                return self._execute_fallback_sql(sql, query)
            
            else:
                # Default query
                sql = """
                SELECT p.platform_number, p.profile_date, p.latitude, p.longitude,
                       p.surface_temp, p.surface_salinity
                FROM argo_profiles p
                WHERE p.surface_temp IS NOT NULL
                ORDER BY p.profile_date DESC
                LIMIT 50;
                """
                return self._execute_fallback_sql(sql, query)
        
        except Exception as e:
            return {
                'success': False,
                'error': f'Fallback processing failed: {str(e)}',
                'query': query,
                'processing_time': 0.0
            }
    
    def _execute_fallback_sql(self, sql: str, query: str, params=None) -> Dict[str, Any]:
        """Execute SQL with error handling"""
        
        start_time = datetime.now()
        
        try:
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(text(sql), params)
                else:
                    result = conn.execute(text(sql))
                
                rows = result.fetchall()
                columns = list(result.keys())
                
                # Convert to list of dicts
                results = [dict(zip(columns, row)) for row in rows]
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    'success': True,
                    'results': results,
                    'result_count': len(results),
                    'columns': columns,
                    'sql_query': sql,
                    'query': query,
                    'processing_time': processing_time,
                    'metadata': {'processor': 'fallback_system'}
                }
        
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'processing_time': processing_time,
                'sql_query': sql
            }

# Initialize fallback system
fallback_processor = None

@app.get("/")
async def root():
    return {
        "service": "FloatChat API",
        "version": "2.0.0", 
        "description": "Advanced oceanographic data analysis API",
        "status": "operational" if system_ready else "limited",
        "features": {
            "enhanced_rag": rag_system is not None,
            "intelligent_responses": intelligent_system is not None,
            "fallback_queries": True,
            "database": engine is not None
        },
        "timestamp": datetime.now().isoformat()
    }

# FIXED: Health check with detailed system status
@app.get("/api/health") 
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "components": {}
        }
        
        # Test database connection
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                profile_count = conn.execute(text("SELECT COUNT(*) FROM argo_profiles")).scalar()
                measurement_count = conn.execute(text("SELECT COUNT(*) FROM argo_measurements")).scalar()
                
                health_status["components"]["database"] = {
                    "status": "healthy",
                    "total_profiles": profile_count,
                    "total_measurements": measurement_count
                }
        except Exception as e:
            health_status["components"]["database"] = {"status": "unhealthy", "error": str(e)}
            health_status["status"] = "degraded"
        
        # Check RAG system
        health_status["components"]["rag_system"] = {
            "status": "available" if rag_system else "unavailable"
        }
        
        # Check intelligent system
        health_status["components"]["intelligent_system"] = {
            "status": "available" if intelligent_system else "unavailable"
        }
        
        # Check fallback system
        global fallback_processor
        if not fallback_processor and engine:
            fallback_processor = FallbackQueryProcessor(engine)
        
        health_status["components"]["fallback_system"] = {
            "status": "available" if fallback_processor else "unavailable"
        }
        
        # Overall status
        if not health_status["components"]["database"]["status"] == "healthy":
            health_status["status"] = "unhealthy"
        elif not (rag_system or fallback_processor):
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )

# FIXED: Robust query processing with multiple fallback levels
@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process natural language query with intelligent fallbacks"""
    
    if not system_ready:
        raise HTTPException(status_code=503, detail="System not ready - check health endpoint")
    
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Try intelligent system first
        if intelligent_system:
            try:
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
                
                if result['success']:
                    response_data = {
                        "success": True,
                        "query": request.query,
                        "processing_time": result.get('processing_time', 0.0),
                        "result_count": result['results_summary']['total_records'],
                        "columns": result['results_summary']['columns'],
                        "classification": result['classification'],
                        "narrative_response": result['narrative_response'],
                        "insights": result['scientific_insights'],
                        "visualizations": result['visualizations'],
                        "recommendations": result['recommendations'],
                        "metadata": {
                            "processor": "intelligent_system",
                            "timestamp": datetime.now().isoformat(),
                            "version": "2.0.0"
                        }
                    }
                    
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
                    
                    return QueryResponse(**response_data)
                
                else:
                    logger.warning(f"Intelligent system failed: {result.get('error')}, trying RAG system")
            
            except Exception as e:
                logger.warning(f"Intelligent system error: {e}, trying RAG system")
        
        # Try enhanced RAG system as fallback
        if rag_system:
            try:
                result = rag_system.process_oceanographic_query(request.query)
                
                if result['success']:
                    response_data = {
                        "success": True,
                        "query": request.query,
                        "processing_time": result.get('processing_time', 0.0),
                        "result_count": result.get('result_count', 0),
                        "columns": result.get('columns', []),
                        "classification": result.get('classification'),
                        "insights": result.get('insights'),
                        "metadata": {
                            "processor": "enhanced_rag",
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    
                    if request.include_sql:
                        response_data["sql_query"] = result.get('sql_query')
                    
                    if result.get('results') is not None and not result['results'].empty:
                        results_df = result['results']
                        if request.limit and len(results_df) > request.limit:
                            results_df = results_df.head(request.limit)
                        response_data["results"] = results_df.to_dict('records')
                        response_data["result_count"] = len(results_df)
                    
                    return QueryResponse(**response_data)
                
                else:
                    logger.warning(f"RAG system failed: {result.get('error')}, trying fallback")
            
            except Exception as e:
                logger.warning(f"RAG system error: {e}, trying fallback")
        
        # Use fallback system as last resort
        global fallback_processor
        if not fallback_processor:
            fallback_processor = FallbackQueryProcessor(engine)
        
        result = fallback_processor.process_basic_query(request.query)
        
        response_data = {
            "success": result['success'],
            "query": request.query,
            "processing_time": result.get('processing_time', 0.0),
            "result_count": result.get('result_count', 0),
            "columns": result.get('columns', []),
            "metadata": {
                "processor": "fallback_system",
                "timestamp": datetime.now().isoformat(),
                "note": "AI systems unavailable - using basic pattern matching"
            }
        }
        
        if result['success']:
            if request.include_sql:
                response_data["sql_query"] = result.get('sql_query')
            if result.get('results'):
                response_data["results"] = result['results']
        else:
            response_data["error"] = result.get('error')
        
        return QueryResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Critical error processing query: {e}")
        logger.error(traceback.format_exc())
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResponse(
            success=False,
            query=request.query,
            error=f"Server error: {str(e)}",
            processing_time=processing_time,
            metadata={"error_type": "critical_server_error"}
        )

# FIXED: Updated database queries for new schema
@app.get("/api/floats")
async def get_floats(
    region: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    platform_numbers: Optional[str] = None,  # Comma-separated
    limit: Optional[int] = 100
):
    """Get available ARGO floats with optional filtering - UPDATED FOR NEW SCHEMA"""
    
    try:
        # FIXED: Updated query for new schema
        query = """
        SELECT 
            p.platform_number,
            p.cycle_number,
            p.profile_date as date,
            p.latitude,
            p.longitude,
            'Unknown' as project_name,  -- Not available in new schema
            'Unknown' as institution,  -- Not available in new schema
            p.n_levels as measurement_count
        FROM argo_profiles p
        """
        
        conditions = []
        params = {}
        
        # Apply filters
        if region:
            if region.lower() == "arabian_sea":
                conditions.append("p.latitude BETWEEN 8 AND 27 AND p.longitude BETWEEN 50 AND 80")
            elif region.lower() == "bay_of_bengal":
                conditions.append("p.latitude BETWEEN 5 AND 22 AND p.longitude BETWEEN 80 AND 100")
        
        if date_start:
            conditions.append("p.profile_date >= :date_start")
            params["date_start"] = date_start
        
        if date_end:
            conditions.append("p.profile_date <= :date_end")
            params["date_end"] = date_end
        
        if platform_numbers:
            platform_list = [p.strip() for p in platform_numbers.split(',')]
            placeholders = ", ".join([f":platform_{i}" for i in range(len(platform_list))])
            conditions.append(f"p.platform_number IN ({placeholders})")
            for i, platform in enumerate(platform_list):
                params[f"platform_{i}"] = platform
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY p.profile_date DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        with engine.connect() as conn:
            result = conn.execute(text(query), params)
            rows = result.fetchall()
        
        floats = []
        for row in rows:
            floats.append({
                "platform_number": row.platform_number,
                "cycle_number": row.cycle_number,
                "date": row.date.isoformat() if row.date else "",
                "latitude": float(row.latitude) if row.latitude else 0.0,
                "longitude": float(row.longitude) if row.longitude else 0.0,
                "project_name": row.project_name or "Unknown",
                "institution": row.institution or "Unknown",
                "measurement_count": int(row.measurement_count) if row.measurement_count else 0
            })
        
        logger.info(f"Retrieved {len(floats)} floats")
        return floats
        
    except Exception as e:
        logger.error(f"Error retrieving floats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# FIXED: Database stats for new schema
@app.get("/api/stats")
async def get_database_stats():
    """Get database statistics for dashboard - UPDATED FOR NEW SCHEMA"""
    
    try:
        stats_query = """
        SELECT 
            (SELECT COUNT(*) FROM argo_profiles) as total_profiles,
            (SELECT COUNT(*) FROM argo_measurements) as total_measurements,
            (SELECT COUNT(DISTINCT platform_number) FROM argo_profiles) as unique_platforms,
            (SELECT MIN(profile_date) FROM argo_profiles WHERE profile_date IS NOT NULL) as earliest_date,
            (SELECT MAX(profile_date) FROM argo_profiles WHERE profile_date IS NOT NULL) as latest_date,
            (SELECT AVG(m.temperature) FROM argo_measurements m WHERE m.temperature IS NOT NULL) as avg_temperature,
            (SELECT AVG(m.salinity) FROM argo_measurements m WHERE m.salinity IS NOT NULL) as avg_salinity
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(stats_query))
            row = result.fetchone()
        
        return {
            "total_profiles": row.total_profiles,
            "total_measurements": row.total_measurements,
            "unique_platforms": row.unique_platforms,
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

# WebSocket endpoint with improved error handling
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
            self.disconnect(websocket)

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
            
            if not query:
                await manager.send_message(websocket, {
                    'type': 'error',
                    'message': 'Empty query received',
                    'message_id': message_id
                })
                continue
            
            # Send processing status
            await manager.send_message(websocket, {
                'type': 'status',
                'message': 'Processing your oceanographic query...',
                'stage': 'starting',
                'message_id': message_id
            })
            
            # Process query using the same logic as REST API
            try:
                # Create request object
                request = QueryRequest(query=query, include_sql=True)
                result = await process_query(request)
                
                # Convert to WebSocket format
                await manager.send_message(websocket, {
                    'type': 'result',
                    'data': result.dict(),
                    'message_id': message_id
                })
                
            except Exception as e:
                await manager.send_message(websocket, {
                    'type': 'error',
                    'message': f'Processing error: {str(e)}',
                    'message_id': message_id
                })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
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