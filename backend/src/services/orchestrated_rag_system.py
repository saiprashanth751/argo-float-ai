# src/services/orchestrated_rag_system.py
"""
Orchestrated RAG System: Integration layer that uses Smart Query Router
to coordinate between Lightning RAG, Semantic Bridge, and Agentic processing.

This replaces your current process_oceanographic_query method with intelligent routing.
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Literal
from datetime import datetime
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Your existing imports
from services.oceanographic_intelligence_engine import OceanographicIntelligenceEngine
from services.enhanced_rag_oceanographic import ProductionOceanographicRAG
from services.smart_query_router import SmartQueryRouter, ProcessingPath, RoutingDecision
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from services.response_intelligence_layer import DeepSeekResponseIntelligence, ResponseIntelligenceConfig
from utils.database_manager import get_db_engine, get_db_session
from config.vector_store_config import get_vector_store_path

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
        """Initialize orchestrated system with WORKING agent system"""
        # Initialize core components
        if persist_directory is None:
            persist_directory = str(get_vector_store_path())
        
        self.db_engine = db_engine or get_db_engine()
        
        # Layer 1: Your existing RAG system (Lightning path)
        self.rag_system = ProductionOceanographicRAG(persist_directory, db_engine)
        
        # Layer 2: Intelligence engine for classification
        self.intelligence_engine = OceanographicIntelligenceEngine(self.db_engine)
        
        # Layer 3: Smart router (the orchestration layer)
        self.router = SmartQueryRouter(
            intelligence_engine=self.intelligence_engine,
            vector_store=self.rag_system.vector_store
        )
        
        # Layer 4: Semantic bridge 
        from .semantic_intelligence_bridge import SemanticIntelligenceBridge
        self.semantic_bridge = SemanticIntelligenceBridge(self.rag_system)
        
        # REMOVED: Broken IntelligentAgentCoordinator
        # Layer 5: Intelligent Agent Coordinator (NEW) - REMOVED
        # from .intelligent_agent_coordinator import IntelligentAgentCoordinator
        # self.agent_coordinator = IntelligentAgentCoordinator()
        
        # Layer 5: Response system for formatting
        self.response_intelligence = DeepSeekResponseIntelligence()
        
        # Layer 6: PRODUCTION Agent System (now PRIMARY, not "legacy")
        if AGENT_SYSTEM_AVAILABLE:
            try:
                self.agent_system = ProductionAgentCollaborationSystem(self.db_engine, max_agents=4)
                logger.info("Production agent collaboration system initialized as PRIMARY")
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
            'agent_success_rate': 0.0,  # Added
            'avg_response_times': {}
        }
        
        # Storage for inter-layer communication
        self._last_rag_result = {}
        self._last_semantic_result = {}
        
        logger.info("Orchestrated RAG with WORKING Agent System initialized")
    
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
    
    def _execute_processing_path(self, query: str, 
                            routing_decision: RoutingDecision,
                            response_format: ResponseFormat) -> Dict[str, Any]:
        """Execute the query based on routing decision - UPDATED for result storage"""
        
        path = routing_decision.path
        logger.info(f"Executing processing path: {path.value}")
        
        try:
            if path == ProcessingPath.LIGHTNING_RAG:
                result = self._execute_lightning_rag(query, routing_decision, response_format)
                # Store for potential agent use
                self._store_processing_result('rag', result)
                return result
            
            elif path == ProcessingPath.SEMANTIC_BRIDGE:
                result = self._execute_semantic_bridge(query, routing_decision, response_format)
                # Store for potential agent use
                self._store_processing_result('semantic', result)
                return result
            
            elif path == ProcessingPath.AGENTIC_FALLBACK:
                # Agent coordinator will access stored results from previous attempts
                return self._execute_agentic_fallback(query, routing_decision, response_format)
            
            else:  # ERROR_RECOVERY
                return self._execute_error_recovery(query, routing_decision, response_format)
            
        except Exception as e:
            logger.error(f"Processing path execution failed: {e}")
            return self._create_error_result(query, str(e))
    
    def _execute_lightning_rag(self, query: str, 
                          routing_decision: RoutingDecision,
                          response_format: ResponseFormat) -> Dict[str, Any]:
        """Execute lightning-fast RAG processing - UPDATED"""
        
        logger.info("Executing Lightning RAG path")
        self.processing_stats['lightning_queries'] += 1
        
        try:
            # Use your existing RAG system directly
            result = self.rag_system.process_oceanographic_query(query)
            
            # Add path-specific metadata
            result['processing_path'] = 'lightning_rag'
            result['speed_optimized'] = True
            
            # Store result immediately for potential agent coordinator use
            self._store_processing_result('rag', result)
            
            return result
            
        except Exception as e:
            logger.warning(f"Lightning RAG failed: {e}")
            
            # Store failed result
            failed_result = {
                'success': False,
                'error': str(e),
                'query': query,
                'processing_path': 'lightning_rag_failed'
            }
            self._store_processing_result('rag', failed_result)
            
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
        """Execute semantic bridge processing - UPDATED"""
        
        logger.info("Executing Semantic Bridge path")
        self.processing_stats['semantic_queries'] += 1
        
        try:
            # Use semantic bridge
            result = self.semantic_bridge.process_query(query, routing_decision)
            
            # Add path-specific metadata
            result['processing_path'] = 'semantic_bridge'
            result['enrichment_applied'] = True
            
            # Store result immediately for potential agent coordinator use
            self._store_processing_result('semantic', result)
            
            return result
            
        except Exception as e:
            logger.warning(f"Semantic Bridge failed: {e}")
            
            # Store failed result
            failed_result = {
                'success': False,
                'error': str(e),
                'query': query,
                'processing_path': 'semantic_bridge_failed'
            }
            self._store_processing_result('semantic', failed_result)
            
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
    
    async def _execute_agentic_fallback(self, query: str,
                     routing_decision: RoutingDecision,
                     response_format: ResponseFormat) -> Dict[str, Any]:
        """Execute agentic processing - FIXED to use working agent system"""
        
        logger.info("Executing Agentic Fallback path")
        self.processing_stats['agent_queries'] += 1
        
        # Check if agent system is available
        if self.agent_system is None:
            logger.warning("Agent system not available, using intelligent fallback")
            return await self._execute_intelligent_fallback_without_agents(
                query, routing_decision, response_format
            )
        
        try:
            logger.info("Using ProductionAgentCollaborationSystem")
            
            result = await self.agent_system.execute_agent_collaboration(
                query=query,
                routing_decision=routing_decision,
                user_context={}
            )
            
            # Add orchestration metadata
            result['processing_path'] = 'production_agentic'
            result['agent_system_used'] = 'ProductionAgentCollaborationSystem'
            
            # Update success tracking
            if result.get('success'):
                self._update_agent_success_metrics(True)
                logger.info(f"Agent collaboration succeeded in {result.get('processing_time', 0):.2f}s")
            else:
                self._update_agent_success_metrics(False)
                logger.warning("Agent collaboration failed")
            
            return result
            
        except Exception as e:
            logger.error(f"Agent collaboration failed with exception: {e}")
            self._update_agent_success_metrics(False)
            
            # Fallback to enhanced existing results
            return await self._execute_intelligent_fallback_without_agents(
                query, routing_decision, response_format
            )
    
    def _select_best_previous_result(self, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best result from previous processing layers"""
        
        rag_result = previous_results.get('rag_result', {})
        semantic_result = previous_results.get('semantic_result', {})
        
        # Priority: successful results with good data
        if semantic_result.get('success') and semantic_result.get('result_count', 0) > 0:
            return semantic_result
        elif rag_result.get('success') and rag_result.get('result_count', 0) > 0:
            return rag_result
        elif semantic_result.get('success'):
            return semantic_result
        elif rag_result.get('success'):
            return rag_result
        else:
            # Neither successful - return None to trigger response intelligence
            return None
    
    def _calculate_confidence_boost(self, result: Dict[str, Any], routing_decision: RoutingDecision) -> float:
        """Calculate confidence boost for enhanced results"""
        
        base_confidence = result.get('classification', {}).get('confidence', 0.5)
        
        # Boost based on routing confidence
        routing_boost = routing_decision.confidence * 0.1
        
        # Boost based on result quality
        result_count = result.get('result_count', 0)
        data_boost = min(result_count / 1000.0, 0.2)  # Up to 0.2 boost for good data
        
        total_boost = routing_boost + data_boost
        return min(total_boost, 0.3)
    
    def _execute_emergency_response(self, query: str, routing_decision: RoutingDecision, 
                              response_format: ResponseFormat) -> Dict[str, Any]:
        """Emergency response when all systems fail"""
        
        return {
            'success': False,
            'error': 'All processing systems unavailable',
            'query': query,
            'processing_path': 'emergency_response',
            'system_status': {
                'agent_system': self.agent_system is not None,
                'rag_system': self.rag_system is not None,
                'semantic_system': self.semantic_bridge is not None
            },
            'recommendations': [
                'Try simplifying your query',
                'Check system status and try again later',
                'Contact system administrator if problem persists'
            ],
            'emergency_message': f"System temporarily unable to process: '{query[:100]}...'"
        }
    
    async def _execute_intelligent_fallback_without_agents(self, 
                                       query: str, 
                                       routing_decision: RoutingDecision,
                                       response_format: ResponseFormat) -> Dict[str, Any]:
        """Intelligent fallback when agents are unavailable or fail"""
        
        logger.info("Executing intelligent fallback without agents")
        
        # Step 1: Try to enhance existing results from previous layers
        previous_results = {
            'rag_result': getattr(self, '_last_rag_result', {}),
            'semantic_result': getattr(self, '_last_semantic_result', {})
        }
        
        # Step 2: Select best existing result
        best_result = self._select_best_previous_result(previous_results)
        
        if best_result and best_result.get('success'):
            # Enhance the existing result
            enhanced_result = best_result.copy()
            enhanced_result.update({
                'processing_path': 'enhanced_existing_no_agents',
                'enhancement_applied': True,
                'fallback_reason': 'Agent system unavailable',
                'intelligence_note': 'Enhanced existing results with additional context',
                'confidence_boost': self._calculate_confidence_boost(best_result, routing_decision)
            })
            
            return enhanced_result
        
        # Step 3: If no good previous results, use response intelligence
        try:
            result = self.response_intelligence.enhance_response_with_intelligence(
                query, response_format
            )
            result.update({
                'processing_path': 'response_intelligence_fallback',
                'fallback_reason': 'No previous results available',
                'agent_system_status': 'unavailable'
            })
            return result
            
        except Exception as e:
            logger.error(f"Response intelligence fallback failed: {e}")
            
            # Final emergency fallback
            return self._execute_emergency_response(query, routing_decision, response_format)
    
    def _ensure_excellent_agent_ux(self, query: str, agent_result: Dict[str, Any], 
                               agent_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure excellent user experience for agent-processed queries"""
        
        enhanced_result = agent_result.copy()
        
        # Add user-friendly explanations
        enhanced_result['user_experience'] = {
            'processing_explanation': self._explain_agent_processing(agent_decision),
            'why_agents_used': agent_decision.get('reasoning', ['Advanced analysis required']),
            'confidence_level': self._assess_agent_result_confidence(agent_result),
            'transparency': 'high'
        }
        
        # Add metadata for monitoring
        enhanced_result['intelligent_agent_metadata'] = {
            'agent_type': agent_decision.get('agent_type', 'general'),
            'capabilities_used': agent_decision.get('expected_capabilities', []),
            'decision_confidence': agent_decision.get('confidence', 0.5),
            'fallback_available': agent_decision.get('fallback_strategy') is not None
        }
        
        return enhanced_result
    
    def _assess_agent_result_confidence(self, agent_result: Dict[str, Any]) -> str:
        """Assess and communicate confidence in agent results"""
        
        # Look for confidence indicators in result
        if 'confidence' in agent_result:
            confidence = agent_result['confidence']
            if confidence > 0.8:
                return "high"
            elif confidence > 0.6:
                return "medium"
            else:
                return "moderate"
        
        # Assess based on result structure
        if agent_result.get('success') and agent_result.get('results') is not None:
            return "high"
        elif agent_result.get('success'):
            return "medium"
        else:
            return "moderate"
    
    def _store_processing_result(self, layer: str, result: Dict[str, Any]):
        """Store processing results with enhanced metadata"""
        
        if layer == 'rag':
            self._last_rag_result = result
            # Add timestamp for freshness tracking
            self._last_rag_result['timestamp'] = time.time()
        elif layer == 'semantic':
            self._last_semantic_result = result
            self._last_semantic_result['timestamp'] = time.time()
    
    def _explain_agent_processing(self, agent_decision: Dict[str, Any]) -> str:
        """Generate user-friendly explanation of agent processing"""
        
        explanations = {
            'knowledge_synthesis': "I analyzed multiple expert sources and synthesized complex oceanographic information to provide a comprehensive answer.",
            'data_gap_handler': "I searched alternative data sources and related information since the specific data requested wasn't directly available in our primary database.",
            'calculation_specialist': "I performed advanced calculations and modeling to provide precise quantitative results for your complex query."
        }
        
        agent_type = agent_decision.get('agent_type', 'general')
        return explanations.get(agent_type, 
            "I used advanced reasoning and multiple information sources to provide the most accurate response possible.")
        
    
    
    async def _execute_intelligent_fallback(self, query: str, routing_decision: RoutingDecision,
                         response_format: ResponseFormat, error: str) -> Dict[str, Any]:
        """Execute intelligent fallback when agent coordination fails"""
        
        logger.warning(f"Executing intelligent fallback due to: {error}")
        
        # Try legacy agent system if available
        if self.agent_system is not None:
            try:
                logger.info("Attempting legacy agent system fallback")
                result = await self.agent_system.execute_agent_collaboration(
                    query=query,
                    routing_decision=routing_decision,
                    user_context={}
                )
                result['processing_path'] = 'legacy_agent_fallback'
                result['fallback_reason'] = error
                return result
            except Exception as e:
                logger.error(f"Legacy agent fallback also failed: {e}")
        
        # Final fallback to response intelligence
        try:
            result = self.response_intelligence.enhance_response_with_intelligence(query, response_format)
            result['processing_path'] = 'final_intelligence_fallback'
            result['original_error'] = error
            result['fallback_explanation'] = "Used response intelligence after agent system difficulties"
            return result
        except Exception as e:
            logger.error(f"All fallbacks failed: {e}")
            return self._execute_error_recovery(query, routing_decision, response_format)


    
    async def _enhance_existing_results(self, query: str, previous_results: Dict[str, Any], 
                     routing_decision: RoutingDecision, 
                     response_format: ResponseFormat) -> Dict[str, Any]:
        """Enhance existing results when agents aren't needed"""
        
        logger.info("Enhancing existing results instead of using agents")
        
        # Select the best result from previous layers
        best_result = self._select_best_previous_result(previous_results)
        
        if not best_result or not best_result.get('success'):
            # If no good previous results, try response intelligence fallback
            try:
                result = self.response_intelligence.enhance_response_with_intelligence(query, response_format)
                result['processing_path'] = 'enhanced_existing_fallback'
                result['enhancement_type'] = 'response_intelligence'
                return result
            except Exception as e:
                logger.error(f"Response intelligence fallback failed: {e}")
                return self._execute_error_recovery(query, routing_decision, response_format)
        
        # Enhance the best result
        enhanced_result = best_result.copy()
        enhanced_result.update({
            'processing_path': 'enhanced_existing',
            'enhancement_applied': True,
            'agent_decision': 'avoided_unnecessary_processing',
            'intelligence_boost': {
                'routing_confidence': routing_decision.confidence,
                'processing_efficiency': 'high',
                'user_experience': 'optimized'
            },
            'performance_benefit': 'Avoided unnecessary agent processing while maintaining quality'
        })
        
        # Add any final response intelligence enhancements
        try:
            if hasattr(self.response_intelligence, 'enhance_orchestrated_response'):
                from .response_intelligence_layer import ResponseIntelligenceConfig
                config = ResponseIntelligenceConfig()
                enhanced_result = self.response_intelligence.enhance_orchestrated_response(enhanced_result, config)
        except Exception as e:
            logger.warning(f"Final response enhancement failed: {e}")
        
        return enhanced_result
    
    def _select_best_previous_result(self, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best result from previous processing layers"""
        
        rag_result = previous_results.get('rag_result', {})
        semantic_result = previous_results.get('semantic_result', {})
        
        # Priority: successful results first
        if semantic_result.get('success') and rag_result.get('success'):
            # Both successful - choose semantic (more enriched)
            return semantic_result
        elif semantic_result.get('success'):
            return semantic_result
        elif rag_result.get('success'):
            return rag_result
        else:
            # Neither successful - return the one with more information
            if len(str(semantic_result)) > len(str(rag_result)):
                return semantic_result
            return rag_result
    
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
        """Get comprehensive system performance including agent metrics"""
        
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
        
        # Get agent system metrics if available
        agent_metrics = {}
        if self.agent_system:
            try:
                agent_metrics = self.agent_system.get_collaboration_metrics()
            except Exception as e:
                logger.warning(f"Could not get agent metrics: {e}")
        
        return {
            'query_distribution': {
                'total_queries': self.processing_stats['total_queries'],
                'lightning_rag': self.processing_stats['lightning_queries'],
                'semantic_bridge': self.processing_stats['semantic_queries'],
                'agentic_fallback': self.processing_stats['agent_queries']
            },
            'success_rates': {
                'agent_success_rate': self.processing_stats['agent_success_rate']
            },
            'performance_metrics': avg_times,
            'routing_statistics': self.router.get_routing_statistics(),
            'agent_system_metrics': agent_metrics,
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