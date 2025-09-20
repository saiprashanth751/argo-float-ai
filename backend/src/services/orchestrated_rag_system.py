# src/services/orchestrated_rag_system.py
"""
Orchestrated RAG System: Integration layer that uses Smart Query Router
to coordinate between Lightning RAG, Semantic Bridge, and Agentic processing.

This replaces your current process_oceanographic_query method with intelligent routing.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union, Literal
from datetime import datetime
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Your existing imports
from .oceanographic_intelligence_engine import OceanographicIntelligenceEngine
from .enhanced_rag_oceanographic import ProductionOceanographicRAG
from .smart_query_router import SmartQueryRouter, ProcessingPath, RoutingDecision
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from .response_intelligence_layer import DeepSeekResponseIntelligence, ResponseIntelligenceConfig
from utils.database_manager import get_db_engine, get_db_session

@dataclass
class ResponseFormat:
    """Response format configuration"""
    format_type: Literal["structured", "narrative", "dashboard"] = "structured"
    target_audience: Literal["researcher", "government", "maritime", "public"] = "researcher"
    complexity_level: Literal["basic", "intermediate", "advanced", "expert"] = "intermediate"
    include_visualization: bool = True
    include_metadata: bool = True

try:
    from .agent_collaboration_system import ProductionAgentCollaborationSystem
    AGENT_SYSTEM_AVAILABLE = True
except ImportError:
    AGENT_SYSTEM_AVAILABLE = False
    logger.warning("Agent collaboration system not available")


class OrchestratedOceanographicRAG:
    """
    Production RAG system with intelligent query routing.
    
    This system maintains your speed advantages while adding sophisticated
    reasoning capabilities through intelligent routing.
    """
    
    def __init__(self, persist_directory: str = None, db_engine=None):
        # Initialize core components
        self.db_engine = db_engine or get_db_engine()
        
        # Layer 1: Your existing RAG system (Lightning path)
        self.rag_system = ProductionOceanographicRAG(persist_directory, db_engine)
        
        # Layer 2: Intelligence engine for classification
        self.intelligence_engine = OceanographicIntelligenceEngine(self.db_engine)
        
        # Layer 3: Smart router (the new orchestration layer)
        self.router = SmartQueryRouter(
            intelligence_engine=self.intelligence_engine,
            vector_store=self.rag_system.vector_store
        )
        
        # Layer 4: Response system for formatting
        self.response_intelligence = DeepSeekResponseIntelligence()
        # self.response_system = self.response_intelligence   is this needed or not
        
        logger.info("Orchestrated RAG with Response Intelligence initialized")
        # Semantic bridge (will implement next)
        from .semantic_intelligence_bridge import SemanticIntelligenceBridge
        self.semantic_bridge = SemanticIntelligenceBridge(self.rag_system)
        
        # Agent system (will implement after semantic bridge)
        if AGENT_SYSTEM_AVAILABLE:
            try:
                self.agent_system = ProductionAgentCollaborationSystem(self.db_engine, max_agents=4)
                logger.info("Production agent collaboration system initialized")
            except Exception as e:
                logger.warning(f"Agent system initialization failed: {e}")
                self.agent_system = None
        else:
            self.agent_system = None
        
        # Performance monitoring
        self.processing_stats = {
            'total_queries': 0,
            'lightning_queries': 0,
            'semantic_queries': 0,
            'agent_queries': 0,
            'avg_response_times': {}
        }
        
        logger.info("Orchestrated RAG with Response Intelligence initialized")
    
    def process_query(self, natural_language_query: str, 
                 response_format=None,
                 response_config: ResponseIntelligenceConfig = None) -> Dict[str, Any]:
        """
        Enhanced single-function version that handles both old and new interfaces
        """
        logger.info(f"DIAGNOSTIC: process_query ENTRY - Query: {natural_language_query}")
        start_time = time.time()
        self.processing_stats['total_queries'] += 1
        
        # Handle backward compatibility - convert old response_format to new config
        if response_config is None:
            response_config = ResponseIntelligenceConfig()
            if response_format:
                response_config.response_format = "structured"
                if hasattr(response_format, 'target_audience'):
                    response_config.target_audience = response_format.target_audience
                if hasattr(response_format, 'complexity_level'):
                    response_config.complexity_level = response_format.complexity_level
        
        try:
            # Step 1: Routing logic
            routing_decision = self.router.route_query(natural_language_query)
            
            logger.info(f"Query routed to: {routing_decision.path.value} "
                    f"(confidence: {routing_decision.confidence:.2f})")
            
            # Step 2: Processing logic  
            logger.info(f"DIAGNOSTIC: About to call _execute_processing_path")
            result = self._execute_processing_path(
                natural_language_query, 
                routing_decision, 
                None  # response_format handled by intelligence layer
            )
            logger.info(f"DIAGNOSTIC: _execute_processing_path returned: {type(result)}")
            
            # Step 3: Apply response intelligence enhancement
            if result.get('success', False):
                logger.info(f"DIAGNOSTIC: About to call enhance_orchestrated_response")
                enhanced_result = self.response_intelligence.enhance_orchestrated_response(
                    result, response_config
                )
            else:
                enhanced_result = result
            
            # Step 4: Performance tracking
            processing_time = time.time() - start_time
            success = enhanced_result.get('success', False)
            
            self.router.record_query_result(
                query=natural_language_query,
                routing_decision=routing_decision,
                execution_time=processing_time,
                success=success,
                error_type=enhanced_result.get('error_type')
            )
            
            # Step 5: Add metadata
            enhanced_result.update({
                'orchestration': {
                    'routing_path': routing_decision.path.value,
                    'routing_confidence': routing_decision.confidence,
                    'routing_reasoning': routing_decision.reasoning,
                    'performance_budget': routing_decision.performance_budget,
                    'unknown_terms': routing_decision.unknown_terms,
                    'enrichments_applied': routing_decision.enrichments_needed,
                    'total_processing_time': processing_time,
                    'intelligence_enhanced': True
                }
            })
            
            return enhanced_result
            
        except Exception as e:
            # Error handling
            processing_time = time.time() - start_time
            logger.error(f"Orchestrated query processing failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'query': natural_language_query,
                'processing_time': processing_time,
                'orchestration': {
                    'routing_path': 'error',
                    'error_recovery_attempted': True
                }
            }
    
    async def _execute_processing_path(self, query: str, 
                                routing_decision: RoutingDecision,
                                response_format: ResponseFormat) -> Dict[str, Any]:
        """Execute the query based on routing decision"""
        
        path = routing_decision.path
        logger.info(f"DIAGNOSTIC: _execute_processing_path - path: {path.value}")
        
        if path == ProcessingPath.LIGHTNING_RAG:
            logger.info(f"DIAGNOSTIC: About to call _execute_lightning_rag")
            return await self._execute_lightning_rag(query, routing_decision, response_format)
        
        elif path == ProcessingPath.SEMANTIC_BRIDGE:
            logger.info(f"DIAGNOSTIC: About to call _execute_semantic_bridge")
            return await self._execute_semantic_bridge(query, routing_decision, response_format)
        
        elif path == ProcessingPath.AGENTIC_FALLBACK:
            logger.info(f"DIAGNOSTIC: About to call _execute_agentic_fallback")
            return await self._execute_agentic_fallback(query, routing_decision, response_format)
        
        else:  # ERROR_RECOVERY
            return await self._execute_error_recovery(query, routing_decision, response_format)
    
    def _execute_lightning_rag(self, query: str, 
                              routing_decision: RoutingDecision,
                              response_format: ResponseFormat) -> Dict[str, Any]:
        """Execute lightning-fast RAG processing (your existing system)"""
        
        logger.info("Executing Lightning RAG path")
        self.processing_stats['lightning_queries'] += 1
        
        try:
            # Use your existing RAG system directly
            result = self.rag_system.process_oceanographic_query(query)
            
            # Add path-specific metadata
            result['processing_path'] = 'lightning_rag'
            result['speed_optimized'] = True
            
            return result
            
        except Exception as e:
            logger.warning(f"Lightning RAG failed: {e}")
            
            # Fallback to semantic bridge if available
            if routing_decision.fallback_path == ProcessingPath.SEMANTIC_BRIDGE:
                logger.info("Falling back to Semantic Bridge")
                fallback_decision = RoutingDecision(
                    path=ProcessingPath.SEMANTIC_BRIDGE,
                    confidence=routing_decision.confidence * 0.8,
                    reasoning=routing_decision.reasoning + ["Fallback from Lightning RAG"],
                    performance_budget=15,
                    fallback_path=ProcessingPath.AGENTIC_FALLBACK,
                    enrichments_needed=routing_decision.enrichments_needed,
                    unknown_terms=routing_decision.unknown_terms,
                    complexity_factors=routing_decision.complexity_factors,
                    estimated_cost="medium"
                )
                return self._execute_semantic_bridge(query, fallback_decision, response_format)
            
            raise e
    
    def _execute_semantic_bridge(self, query: str,
                                routing_decision: RoutingDecision,
                                response_format: ResponseFormat) -> Dict[str, Any]:
        """Execute semantic bridge processing (to be implemented)"""
        
        logger.info("Executing Semantic Bridge path")
        self.processing_stats['semantic_queries'] += 1
        
        if self.semantic_bridge is None:
            # For now, fallback to enhanced RAG with enrichments
            logger.warning("Semantic Bridge not implemented, using enhanced RAG")
            
            # Apply query enrichments manually
            enriched_query = self._apply_basic_enrichments(query, routing_decision)
            
            try:
                result = self.rag_system.process_oceanographic_query(enriched_query)
                result['processing_path'] = 'semantic_bridge_fallback'
                result['enrichments_applied'] = routing_decision.enrichments_needed
                result['original_query'] = query
                result['enriched_query'] = enriched_query
                
                return result
                
            except Exception as e:
                # Fallback to agents if available
                if routing_decision.fallback_path == ProcessingPath.AGENTIC_FALLBACK:
                    logger.info("Falling back to Agentic processing")
                    fallback_decision = RoutingDecision(
                        path=ProcessingPath.AGENTIC_FALLBACK,
                        confidence=routing_decision.confidence * 0.7,
                        reasoning=routing_decision.reasoning + ["Fallback from Semantic Bridge"],
                        performance_budget=60,
                        fallback_path=ProcessingPath.ERROR_RECOVERY,
                        enrichments_needed=routing_decision.enrichments_needed,
                        unknown_terms=routing_decision.unknown_terms,
                        complexity_factors=routing_decision.complexity_factors,
                        estimated_cost="high"
                    )
                    return self._execute_agentic_fallback(query, fallback_decision, response_format)
                
                raise e
        
        else:
            # TODO: Implement actual semantic bridge
            return self.semantic_bridge.process_query(query, routing_decision)
    
    async def _execute_agentic_fallback(self, query: str,
                         routing_decision: RoutingDecision,
                         response_format: ResponseFormat) -> Dict[str, Any]:
        """Execute agentic processing with MCP tools (FIXED)"""
        
        logger.info(f"DIAGNOSTIC: _execute_agentic_fallback called with query: {query}")
        logger.info(f"DIAGNOSTIC: Agent collaboration system exists: {self.agent_system is not None}")
        
        logger.info("Executing Agentic Fallback path")
        self.processing_stats['agent_queries'] += 1
        
        if self.agent_system is None:
            logger.warning("Agent system not available, using intelligent response system")
            
            try:
                result = self.response_intelligence.enhance_response_with_intelligence(query, response_format)
                result['processing_path'] = 'agentic_fallback_irs'
                result['agent_reasoning'] = "Used Intelligent Response System as agent fallback"
                return result
                
            except Exception as e:
                logger.error(f"Agentic fallback failed: {e}")
                return self._execute_error_recovery(query, routing_decision, response_format)
        
        else:
            # Use actual agent collaboration system - FIXED ASYNC HANDLING
            try:
                logger.info("DIAGNOSTIC: About to call agent collaboration system")
                
                # SIMPLE DIRECT CALL - no complex async handling
                result = await self.agent_system.execute_agent_collaboration(
                    query=query,
                    routing_decision=routing_decision,
                    user_context={}
                )
                
                logger.info(f"DIAGNOSTIC: Agent collaboration returned: {result.get('success', 'unknown')}")
                
                # Add path-specific metadata
                result['processing_path'] = 'agentic_collaboration'
                result['agent_system_used'] = True
                
                return result
                
            except Exception as e:
                logger.error(f"DIAGNOSTIC: Agent collaboration failed with exception: {e}")
                logger.error(f"DIAGNOSTIC: Exception type: {type(e)}")
                import traceback
                logger.error(f"DIAGNOSTIC: Full traceback: {traceback.format_exc()}")
                
                # Fallback to intelligent response system
                try:
                    result = self.response_intelligence.enhance_response_with_intelligence(query, response_format)
                    result['processing_path'] = 'agentic_fallback_after_error'
                    result['agent_error'] = str(e)
                    return result
                except Exception as e2:
                    logger.error(f"Complete agentic fallback failed: {e2}")
                    return self._execute_error_recovery(query, routing_decision, response_format)
    
    def _execute_error_recovery(self, query: str,
                               routing_decision: RoutingDecision,
                               response_format: ResponseFormat) -> Dict[str, Any]:
        """Execute error recovery with minimal functionality"""
        
        logger.warning("Executing Error Recovery path")
        
        return {
            'success': False,
            'error': 'All processing paths failed',
            'query': query,
            'processing_path': 'error_recovery',
            'suggestions': [
                'Try simplifying your query',
                'Check if all terms are spelled correctly',
                'Consider breaking complex queries into smaller parts'
            ],
            'fallback_response': f"I encountered difficulties processing your query: '{query}'. Please try rephrasing or simplifying your request."
        }
    
    def _apply_basic_enrichments(self, query: str, routing_decision: RoutingDecision) -> str:
        """Apply basic query enrichments until semantic bridge is implemented"""
        
        enriched_query = query
        
        # Unknown term context injection
        if routing_decision.unknown_terms:
            for term in routing_decision.unknown_terms[:2]:  # Limit to avoid query bloat
                if 'chlorophyll' in term.lower():
                    enriched_query += " (considering biological productivity and ocean color)"
                elif 'flux' in term.lower():
                    enriched_query += " (considering mass transfer and biogeochemical processes)"
                elif 'gradient' in term.lower():
                    enriched_query += " (considering spatial or temporal changes)"
        
        # Regional context enhancement
        if 'arabian sea' in query.lower():
            enriched_query += " in the context of monsoon dynamics and upwelling"
        elif 'bay of bengal' in query.lower():
            enriched_query += " considering freshwater influence and cyclone effects"
        
        # Temporal context enhancement
        if 'seasonal' in query.lower():
            enriched_query += " across different months and climate patterns"
        
        return enriched_query
    
    def _update_processing_stats(self, path: ProcessingPath, 
                                processing_time: float, success: bool):
        """Update performance statistics"""
        
        path_key = path.value
        if path_key not in self.processing_stats['avg_response_times']:
            self.processing_stats['avg_response_times'][path_key] = []
        
        self.processing_stats['avg_response_times'][path_key].append(processing_time)
        
        # Keep only recent measurements
        if len(self.processing_stats['avg_response_times'][path_key]) > 100:
            self.processing_stats['avg_response_times'][path_key] = \
                self.processing_stats['avg_response_times'][path_key][-50:]
    
    def get_system_performance(self) -> Dict[str, Any]:
        """Get comprehensive system performance metrics"""
        
        # Calculate average response times
        avg_times = {}
        for path, times in self.processing_stats['avg_response_times'].items():
            if times:
                avg_times[path] = {
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'recent_queries': len(times)
                }
        
        return {
            'query_distribution': {
                'total_queries': self.processing_stats['total_queries'],
                'lightning_rag': self.processing_stats['lightning_queries'],
                'semantic_bridge': self.processing_stats['semantic_queries'],
                'agentic_fallback': self.processing_stats['agent_queries']
            },
            'performance_metrics': avg_times,
            'routing_statistics': self.router.get_routing_statistics(),
            'system_health': {
                'rag_system_ready': self.rag_system is not None,
                'semantic_bridge_ready': self.semantic_bridge is not None,
                'agent_system_ready': self.agent_system is not None,
                'intelligence_engine_ready': self.intelligence_engine is not None
            }
        }
    
    def _create_default_engine(self):
        """Create default database engine if none provided"""
        import os
        from sqlalchemy import create_engine
        
        return create_engine(
            os.getenv('DATABASE_URL', 'postgresql://argo_user:argo_password@localhost:5432/argo_production'),
            pool_size=15,
            max_overflow=25,
            pool_pre_ping=True,
            connect_args={"options": "-c timezone=UTC"}
        )


# Usage example and integration guide
def integrate_with_existing_api(user_query: str, user_response_format: ResponseFormat = None):
    """
    Example of how to integrate the orchestrated system with your existing API.
    
    Replace your current RAG system instantiation with this orchestrated version.
    """
    
    # OLD WAY (what you currently have):
    # rag_system = ProductionOceanographicRAG()
    # result = rag_system.process_oceanographic_query(query)
    
    # NEW WAY (with intelligent orchestration):
    orchestrated_system = OrchestratedOceanographicRAG()
    result = orchestrated_system.process_query(user_query, user_response_format)
    
    # The result now includes orchestration metadata:
    # result['orchestration']['routing_path'] - which path was used
    # result['orchestration']['routing_confidence'] - how confident the routing was
    # result['orchestration']['unknown_terms'] - terms that needed resolution
    # result['orchestration']['enrichments_applied'] - what enrichments were used
    
    return result


def test_orchestrated_system():
    """Test the orchestrated system with different query types"""
    
    system = OrchestratedOceanographicRAG()
    
    test_queries = [
        # Should go to Lightning RAG
        "Show temperature profiles in Arabian Sea",
        
        # Should go to Semantic Bridge  
        "What is thermocline variability during monsoon season?",
        
        # Should go to Agentic Fallback
        "How do mesoscale eddies affect biogeochemical flux patterns?",
        
        # Should trigger unknown term handling
        "Analyze chlorophyll-a concentration gradients in upwelling zones"
    ]
    
    print("Testing Orchestrated RAG System")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        
        try:
            result = system.process_query(query)
            
            if result['success']:
                orchestration = result.get('orchestration', {})
                print(f"  ✅ Success via {orchestration.get('routing_path', 'unknown')}")
                print(f"     Confidence: {orchestration.get('routing_confidence', 0):.2f}")
                print(f"     Processing time: {orchestration.get('total_processing_time', 0):.2f}s")
                
                if orchestration.get('unknown_terms'):
                    print(f"     Unknown terms handled: {orchestration['unknown_terms']}")
            else:
                print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"  ❌ Exception: {e}")
    
    # Show system performance
    print("\nSystem Performance Summary:")
    print("-" * 30)
    performance = system.get_system_performance()
    
    for path, metrics in performance['performance_metrics'].items():
        print(f"{path}: {metrics['avg_time']:.2f}s avg ({metrics['recent_queries']} queries)")


if __name__ == "__main__":
    test_orchestrated_system()