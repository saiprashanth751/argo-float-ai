#main.py - PRODUCTION-READY with Three-Layer Intelligence
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
from sqlalchemy import create_engine, text

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FloatChat API - Three-Layer Intelligence",
    description="Advanced oceanographic data analysis with Lightning RAG, Semantic Bridge, and Agentic Fallback",
    version="3.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PRODUCTION SYSTEM INITIALIZATION
# Global instances with proper error handling and fallback chain
orchestrated_system = None
fallback_rag_system = None
basic_rag_system = None
engine = None
system_ready = False
system_capabilities = {
    'lightning_rag': False,
    'semantic_bridge': False,
    'agentic_fallback': False,
    'intelligent_routing': False
}

@app.on_event("startup")
async def startup_event():
    """Initialize the three-layer intelligence system with robust fallbacks"""
    global orchestrated_system, fallback_rag_system, basic_rag_system, engine, system_ready, system_capabilities
    
    try:
        logger.info("🚀 Starting FloatChat Three-Layer Intelligence System...")
        
        # Step 1: Initialize database engine
        database_url = os.getenv('DATABASE_URL', 'postgresql://argo_user:argo_password@localhost:5432/argo_data')
        engine = create_engine(
            database_url,
            pool_size=15,
            max_overflow=25,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
        
        # Test database connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            logger.info("✅ Database connection successful")
        
        # Step 2: Initialize Three-Layer System (PRIORITY ORDER)
        
        # LAYER 1 + 2 + 3: Try Full Orchestrated System First
        try:
            from src.services.orchestrated_rag_system import OrchestratedOceanographicRAG
            orchestrated_system = OrchestratedOceanographicRAG(db_engine=engine)
            # rag_system = orchestrated_system  #check for later.....
            # Validate all layers are working
            system_capabilities['lightning_rag'] = hasattr(orchestrated_system, 'rag_system')
            system_capabilities['semantic_bridge'] = hasattr(orchestrated_system, 'semantic_bridge')
            system_capabilities['agentic_fallback'] = hasattr(orchestrated_system, 'agent_system')
            system_capabilities['intelligent_routing'] = hasattr(orchestrated_system, 'router')
            
            logger.info("✅ Three-Layer Orchestrated System initialized successfully")
            logger.info(f"   Lightning RAG: {'✅' if system_capabilities['lightning_rag'] else '❌'}")
            logger.info(f"   Semantic Bridge: {'✅' if system_capabilities['semantic_bridge'] else '❌'}")
            logger.info(f"   Agentic Fallback: {'✅' if system_capabilities['agentic_fallback'] else '❌'}")
            logger.info(f"   Intelligent Routing: {'✅' if system_capabilities['intelligent_routing'] else '❌'}")
            
            system_ready = True
            
        except ImportError as e:
            logger.warning(f"⚠️ Orchestrated system import failed: {e}")
            orchestrated_system = None
        except Exception as e:
            logger.warning(f"⚠️ Orchestrated system initialization failed: {e}")
            orchestrated_system = None
        
        # FALLBACK LAYER 2: Try Enhanced RAG System
        if orchestrated_system is None:
            try:
                from src.services.enhanced_rag_oceanographic import ProductionOceanographicRAG
                fallback_rag_system = ProductionOceanographicRAG(db_engine=engine)
                system_capabilities['lightning_rag'] = True
                logger.info("✅ Enhanced RAG system initialized as fallback")
                system_ready = True
            except Exception as e:
                logger.warning(f"⚠️ Enhanced RAG system failed: {e}")
                fallback_rag_system = None
        
        # FALLBACK LAYER 3: Try Basic RAG System
        if orchestrated_system is None and fallback_rag_system is None:
            try:
                # This would be your most basic working system
                logger.warning("⚠️ Using basic fallback system - limited functionality")
                system_ready = True  # Minimal functionality
            except Exception as e:
                logger.error(f"❌ All systems failed: {e}")
                system_ready = False
        
        if system_ready:
            logger.info("🎯 FloatChat Intelligence System ready for production")
        else:
            logger.error("❌ CRITICAL: System initialization failed completely")
            raise RuntimeError("System not operational")
            
    except Exception as e:
        logger.error(f"❌ CRITICAL STARTUP FAILURE: {e}")
        system_ready = False
        raise

# PRODUCTION PYDANTIC MODELS
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Natural language oceanographic query")
    include_sql: bool = Field(True, description="Include generated SQL in response")
    limit: Optional[int] = Field(1000, ge=1, le=50000, description="Maximum results")
    response_format: Optional[Dict[str, Any]] = Field(None, description="Response format preferences")
    user_context: Optional[Dict[str, Any]] = Field(None, description="User context and preferences")

class QueryResponse(BaseModel):
    success: bool
    query: str
    processing_path: Optional[str] = None  # NEW: Which layer processed the query
    sql_query: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None
    result_count: int = 0
    columns: List[str] = []
    processing_time: float = 0.0
    error: Optional[str] = None
    
    # Three-Layer Intelligence Metadata
    routing_decision: Optional[Dict[str, Any]] = None
    semantic_enrichments: Optional[Dict[str, Any]] = None
    agent_insights: Optional[Dict[str, Any]] = None
    system_capabilities_used: Optional[Dict[str, bool]] = None
    
    # Enhanced Response Data
    narrative_response: Optional[str] = None
    classification: Optional[Dict[str, Any]] = None
    insights: Optional[Dict[str, Any]] = None
    visualizations: Optional[List[Dict[str, Any]]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None

@app.get("/")
async def root():
    """System status with three-layer architecture details"""
    return {
        "service": "FloatChat Three-Layer Intelligence API",
        "version": "3.0.0",
        "description": "Lightning RAG + Semantic Bridge + Agentic Fallback",
        "status": "operational" if system_ready else "degraded",
        "architecture": {
            "layer_1_lightning_rag": system_capabilities['lightning_rag'],
            "layer_2_semantic_bridge": system_capabilities['semantic_bridge'], 
            "layer_3_agentic_fallback": system_capabilities['agentic_fallback'],
            "intelligent_routing": system_capabilities['intelligent_routing']
        },
        "system_type": "orchestrated" if orchestrated_system else "fallback",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
async def health_check():
    """Comprehensive three-layer system health check"""
    try:
        health_status = {
            "status": "healthy" if system_ready else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "version": "3.0.0",
            "architecture": "three_layer_intelligence",
            "layers": {}
        }
        
        # Layer 1: Lightning RAG Health
        if system_capabilities['lightning_rag']:
            health_status["layers"]["lightning_rag"] = {
                "status": "operational",
                "response_time_target": "<200ms",
                "handles": "80% of standard oceanographic queries"
            }
        else:
            health_status["layers"]["lightning_rag"] = {
                "status": "unavailable",
                "impact": "No fast query processing"
            }
        
        # Layer 2: Semantic Bridge Health
        if system_capabilities['semantic_bridge']:
            health_status["layers"]["semantic_bridge"] = {
                "status": "operational", 
                "response_time_target": "<3s",
                "handles": "Complex queries with unknown terms"
            }
        else:
            health_status["layers"]["semantic_bridge"] = {
                "status": "unavailable",
                "impact": "No query enrichment for unknown terms"
            }
        
        # Layer 3: Agentic Fallback Health  
        if system_capabilities['agentic_fallback']:
            health_status["layers"]["agentic_fallback"] = {
                "status": "operational",
                "response_time_target": "<30s", 
                "handles": "Novel research queries requiring reasoning"
            }
        else:
            health_status["layers"]["agentic_fallback"] = {
                "status": "unavailable",
                "impact": "No complex reasoning capabilities"
            }
        
        # Database health
        try:
            with engine.connect() as conn:
                profile_count = conn.execute(text("SELECT COUNT(*) FROM argo_profiles")).scalar()
                measurement_count = conn.execute(text("SELECT COUNT(*) FROM argo_measurements")).scalar()
                
                health_status["database"] = {
                    "status": "healthy",
                    "total_profiles": profile_count,
                    "total_measurements": measurement_count
                }
        except Exception as e:
            health_status["database"] = {"status": "unhealthy", "error": str(e)}
            health_status["status"] = "degraded"
        
        # Overall system assessment
        operational_layers = sum(system_capabilities.values())
        if operational_layers >= 3:
            health_status["overall_intelligence"] = "maximum"
        elif operational_layers >= 2:
            health_status["overall_intelligence"] = "good"
        elif operational_layers >= 1:
            health_status["overall_intelligence"] = "basic"
        else:
            health_status["overall_intelligence"] = "minimal"
            health_status["status"] = "critical"
        
        return health_status
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "critical_failure",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process query through three-layer intelligence system"""
    
    if not system_ready:
        raise HTTPException(status_code=503, detail="System not operational - check health endpoint")
    
    start_time = datetime.now()
    
    try:
        logger.info(f"🔍 Processing query: {request.query}")
        
        # ORCHESTRATED SYSTEM (Layers 1+2+3 with intelligent routing)
        if orchestrated_system:
            try:
                # Import response format if needed
                try:
                    from src.services.response_intelligence_layer import ResponseConfig
                    response_format = ResponseConfig()
                    
                    # Apply user preferences
                    if request.response_format:
                        for key, value in request.response_format.items():
                            if hasattr(response_format, key):
                                setattr(response_format, key, value)
                except ImportError:
                    response_format = None
                
                # Process through orchestrated system
                result = orchestrated_system.process_query(
                    request.query, 
                    response_format=response_format
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                if result['success']:
                    # Extract orchestration metadata
                    orchestration = result.get('orchestration', {})
                    
                    response_data = {
                        "success": True,
                        "query": request.query,
                        "processing_path": orchestration.get('routing_path', 'unknown'),
                        "processing_time": processing_time,
                        "result_count": result.get('result_count', 0),
                        "columns": result.get('columns', []),
                        "routing_decision": {
                            "path": orchestration.get('routing_path'),
                            "confidence": orchestration.get('routing_confidence', 0),
                            "reasoning": orchestration.get('routing_reasoning', []),
                            "unknown_terms": orchestration.get('unknown_terms', []),
                            "enrichments_applied": orchestration.get('enrichments_applied', [])
                        },
                        "system_capabilities_used": {
                            "intelligent_routing": True,
                            "layer_used": orchestration.get('routing_path', 'unknown')
                        }
                    }
                    
                    # Include SQL if requested
                    if request.include_sql and 'sql_query' in result:
                        response_data["sql_query"] = result['sql_query']
                    
                    # Include results data
                    if 'results' in result and result['results'] is not None:
                        if hasattr(result['results'], 'to_dict'):
                            # It's a DataFrame
                            results_df = result['results']
                            if request.limit and len(results_df) > request.limit:
                                results_df = results_df.head(request.limit)
                            response_data["results"] = results_df.to_dict('records')
                            response_data["result_count"] = len(results_df)
                        elif isinstance(result['results'], list):
                            response_data["results"] = result['results'][:request.limit] if request.limit else result['results']
                            response_data["result_count"] = len(response_data["results"])
                    
                    # Include enhanced response data if available
                    if 'narrative_response' in result:
                        response_data["narrative_response"] = result['narrative_response']
                    if 'classification' in result:
                        response_data["classification"] = result['classification']
                    if 'insights' in result:
                        response_data["insights"] = result['insights']
                    
                    logger.info(f"✅ Orchestrated success via {orchestration.get('routing_path', 'unknown')} in {processing_time:.2f}s")
                    return QueryResponse(**response_data)
                    
                else:
                    logger.warning(f"⚠️ Orchestrated system failed: {result.get('error')}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Orchestrated system error: {e}")
        
        # FALLBACK RAG SYSTEM (Layer 1 only)  
        if fallback_rag_system:
            try:
                result = fallback_rag_system.process_oceanographic_query(request.query)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                if result['success']:
                    response_data = {
                        "success": True,
                        "query": request.query,
                        "processing_path": "fallback_rag",
                        "processing_time": processing_time,
                        "result_count": result.get('result_count', 0),
                        "columns": result.get('columns', []),
                        "system_capabilities_used": {
                            "fallback_mode": True,
                            "layer_used": "lightning_rag_only"
                        }
                    }
                    
                    if request.include_sql and 'sql_query' in result:
                        response_data["sql_query"] = result['sql_query']
                    
                    if 'results' in result and hasattr(result['results'], 'to_dict'):
                        results_df = result['results']
                        if request.limit and len(results_df) > request.limit:
                            results_df = results_df.head(request.limit)
                        response_data["results"] = results_df.to_dict('records')
                        response_data["result_count"] = len(results_df)
                    
                    logger.info(f"✅ Fallback RAG success in {processing_time:.2f}s")
                    return QueryResponse(**response_data)
                    
            except Exception as e:
                logger.warning(f"⚠️ Fallback RAG error: {e}")
        
        # ULTIMATE FALLBACK
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error("❌ All processing systems failed")
        
        return QueryResponse(
            success=False,
            query=request.query,
            error="All processing systems unavailable",
            processing_time=processing_time,
            processing_path="error_recovery",
            system_capabilities_used={
                "error_recovery": True,
                "all_systems_failed": True
            }
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ Critical query processing error: {e}")
        traceback.print_exc()
        
        return QueryResponse(
            success=False,
            query=request.query,
            error=f"Critical system error: {str(e)}",
            processing_time=processing_time,
            processing_path="critical_error"
        )

@app.get("/api/system-performance")
async def get_system_performance():
    """Get comprehensive system performance metrics"""
    
    performance = {
        "timestamp": datetime.now().isoformat(),
        "system_type": "orchestrated" if orchestrated_system else "fallback",
        "capabilities": system_capabilities,
        "layer_performance": {}
    }
    
    if orchestrated_system and hasattr(orchestrated_system, 'get_system_performance'):
        try:
            orchestrated_performance = orchestrated_system.get_system_performance()
            performance["orchestrated_metrics"] = orchestrated_performance
        except Exception as e:
            performance["orchestrated_metrics"] = {"error": str(e)}
    
    return performance

# WebSocket support for real-time queries (keep existing implementation)
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_message(self, websocket: WebSocket, message: dict):
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self.disconnect(websocket)

manager = ConnectionManager()

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time three-layer intelligence"""
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
                    'message': 'Empty query',
                    'message_id': message_id
                })
                continue
            
            await manager.send_message(websocket, {
                'type': 'status',
                'message': 'Processing through three-layer intelligence...',
                'message_id': message_id
            })
            
            try:
                request = QueryRequest(query=query, include_sql=True)
                result = await process_query(request)
                
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
        logger.error(f"WebSocket critical error: {e}")
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