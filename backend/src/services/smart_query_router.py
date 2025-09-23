# src/services/smart_query_router.py - FIXED CALIBRATION
"""
Smart Query Router: FIXED - Proper calibration for intelligent routing decisions
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import hashlib
from datetime import datetime
import json

from .oceanographic_intelligence_engine import (
    OceanographicIntelligenceEngine,
    QueryClassification,
    QueryIntent,
    ComplexityLevel,
    OceanographicContext
)

logger = logging.getLogger(__name__)

class ProcessingPath(Enum):
    """Processing paths for different query types"""
    LIGHTNING_RAG = "lightning_rag"        # Fast vector search + template SQL
    SEMANTIC_BRIDGE = "semantic_bridge"    # Query enrichment + enhanced context
    AGENTIC_FALLBACK = "agentic_fallback"  # Full agent reasoning with MCP tools
    ERROR_RECOVERY = "error_recovery"      # Fallback when other paths fail

@dataclass
class RoutingDecision:
    """Complete routing decision with reasoning and metadata"""
    path: ProcessingPath
    confidence: float
    reasoning: List[str]
    performance_budget: int  # seconds
    fallback_path: Optional[ProcessingPath]
    enrichments_needed: List[str]
    unknown_terms: List[str]
    complexity_factors: Dict[str, Any]
    estimated_cost: str  # computational cost estimate

@dataclass
class QueryMetrics:
    """Performance and success metrics for queries"""
    query_hash: str
    processing_path: ProcessingPath
    execution_time: float
    success: bool
    confidence_score: float
    error_type: Optional[str] = None
    enrichments_used: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class SmartQueryRouter:
    """
    FIXED: Smart query router with PROPER calibration for production use
    
    Key Fixes:
    1. Balanced scoring that actually allows agent selection
    2. Realistic complexity thresholds
    3. Proper unknown term handling
    4. Intelligence-based routing decisions
    """
    
    def __init__(self, intelligence_engine: OceanographicIntelligenceEngine,
                 vector_store=None):
        self.intelligence_engine = intelligence_engine
        self.vector_store = vector_store
        
        # Known oceanographic terms and concepts
        self.known_terms = self._initialize_known_terms()
        
        # FIXED: Realistic performance thresholds
        self.performance_thresholds = self._initialize_realistic_thresholds()
        
        # Learning system for query patterns
        self.query_history: List[QueryMetrics] = []
        self.routing_patterns = {}
        
        # FIXED: Balanced system health monitoring
        self.system_health = {
            'rag_response_time': 0.5,
            'semantic_response_time': 8.0,
            'agent_response_time': 45.0,
            'error_rates': {
                ProcessingPath.LIGHTNING_RAG: 0.05,
                ProcessingPath.SEMANTIC_BRIDGE: 0.12,
                ProcessingPath.AGENTIC_FALLBACK: 0.20
            }
        }
        
        logger.info("Smart Query Router initialized with FIXED calibration")
    
    def route_query(self, query: str, user_context: Dict[str, Any] = None) -> RoutingDecision:
        """
        FIXED: Main routing method with proper intelligence-based decisions
        """
        
        start_time = time.time()
        
        # Step 1: Basic classification
        classification = self.intelligence_engine.classify_query(query)
        
        # Step 2: Detect unknown terms (improved logic)
        unknown_terms = self._detect_unknown_terms_intelligent(query)
        
        # Step 3: Assess complexity factors (enhanced)
        complexity_factors = self._analyze_complexity_factors_enhanced(query, classification)
        
        # Step 4: Check historical patterns
        historical_pattern = self._check_historical_patterns(query)
        
        # Step 5: FIXED routing decision with balanced scoring
        routing_decision = self._make_balanced_routing_decision(
            query=query,
            classification=classification,
            unknown_terms=unknown_terms,
            complexity_factors=complexity_factors,
            historical_pattern=historical_pattern,
            user_context=user_context
        )
        
        decision_time = time.time() - start_time
        logger.info(f"FIXED Router: {routing_decision.path.value} "
                   f"(confidence: {routing_decision.confidence:.2f}, "
                   f"decision_time: {decision_time:.3f}s)")
        
        return routing_decision
    
    def _detect_unknown_terms_intelligent(self, query: str) -> List[str]:
        """
        IMPROVED: More intelligent unknown term detection
        """
        
        query_lower = query.lower()
        
        # Extract potential scientific terms (improved patterns)
        scientific_patterns = [
            r'\b[a-zA-Z]{6,}\b',  # Long scientific terms
            r'\b[a-zA-Z]+(?:tion|ity|ism|phy|logy)\b',  # Scientific suffixes
            r'\b(?:bio|geo|hydro|thermo)[a-zA-Z]+\b'  # Scientific prefixes
        ]
        
        potential_terms = set()
        for pattern in scientific_patterns:
            matches = re.findall(pattern, query_lower)
            potential_terms.update(matches)
        
        # Enhanced filtering
        common_words = {
            'show', 'find', 'get', 'data', 'from', 'with', 'what', 'where',
            'when', 'how', 'why', 'the', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'by', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off',
            'over', 'under', 'again', 'further', 'then', 'once', 'analysis',
            'measurement', 'value', 'level', 'time', 'year', 'month', 'day',
            'temperature', 'salinity', 'pressure', 'density', 'depth'  # Known oceanographic terms
        }
        
        # Identify truly unknown terms
        unknown_terms = []
        for term in potential_terms:
            if (term not in self.known_terms and 
                term not in common_words and 
                len(term) > 4):
                
                # Score the term for scientific importance
                importance_score = self._score_term_importance(term, query_lower)
                if importance_score > 0.4:
                    unknown_terms.append(term)
        
        # Sort by importance and return top 5
        scored_terms = [(term, self._score_term_importance(term, query_lower)) 
                       for term in unknown_terms]
        scored_terms.sort(key=lambda x: x[1], reverse=True)
        
        return [term for term, score in scored_terms[:5]]
    
    def _score_term_importance(self, term: str, query_context: str) -> float:
        """
        IMPROVED: Better term importance scoring
        """
        
        score = 0.0
        
        # Length bonus (longer terms often more specific/scientific)
        if len(term) > 10:
            score += 0.4
        elif len(term) > 7:
            score += 0.2
        
        # Scientific pattern recognition
        scientific_indicators = [
            ('biochem', 0.5), ('geophys', 0.5), ('thermo', 0.4),
            ('flux', 0.3), ('gradient', 0.3), ('anomaly', 0.3),
            ('coefficient', 0.4), ('ratio', 0.2)
        ]
        
        for indicator, bonus in scientific_indicators:
            if indicator in term.lower():
                score += bonus
                break
        
        # Context importance (appears multiple times or in key positions)
        term_frequency = query_context.count(term)
        if term_frequency > 1:
            score += 0.2
        
        # Position importance (terms at beginning often more important)
        if query_context.find(term) < len(query_context) * 0.3:
            score += 0.1
        
        return min(score, 1.0)
    
    def _analyze_complexity_factors_enhanced(self, query: str, 
                                          classification: QueryClassification) -> Dict[str, Any]:
        """
        ENHANCED: Better complexity factor analysis
        """
        
        query_lower = query.lower()
        
        factors = {
            'base_complexity': classification.complexity.value,
            'confidence_score': classification.confidence,
            'parameter_count': len(classification.context.parameters),
            'has_spatial_bounds': classification.context.spatial_bounds is not None,
            'has_temporal_bounds': classification.context.temporal_range is not None,
            'requires_joins': False,
            'requires_aggregation': False,
            'requires_calculation': len(classification.required_calculations) > 0,
            'query_length': len(query.split()),
            'question_words': 0,
            'comparative_terms': 0,
            'scientific_terms': 0,
            'calculation_complexity': 'none'
        }
        
        # Enhanced detection patterns
        join_indicators = ['profile', 'measurement', 'depth', 'pressure', 'vertical', 'cast']
        factors['requires_joins'] = any(indicator in query_lower for indicator in join_indicators)
        
        agg_indicators = ['average', 'mean', 'sum', 'count', 'maximum', 'minimum', 'total', 'statistics']
        factors['requires_aggregation'] = any(indicator in query_lower for indicator in agg_indicators)
        
        question_words = ['what', 'where', 'when', 'how', 'why', 'which', 'who']
        factors['question_words'] = sum(1 for word in question_words if word in query_lower)
        
        comparative_terms = ['compare', 'versus', 'difference', 'between', 'correlation', 'against']
        factors['comparative_terms'] = sum(1 for term in comparative_terms if term in query_lower)
        
        # Scientific term detection
        scientific_terms = ['flux', 'gradient', 'anomaly', 'biogeochemical', 'coefficient', 'ratio']
        factors['scientific_terms'] = sum(1 for term in scientific_terms if term in query_lower)
        
        # Calculation complexity assessment
        if any(term in query_lower for term in ['calculate', 'compute', 'derive', 'model']):
            if any(term in query_lower for term in ['complex', 'advanced', 'research']):
                factors['calculation_complexity'] = 'expert'
            elif any(term in query_lower for term in ['correlation', 'relationship', 'pattern']):
                factors['calculation_complexity'] = 'advanced'
            else:
                factors['calculation_complexity'] = 'intermediate'
        
        return factors
    
    def _make_balanced_routing_decision(self, query: str, 
                                      classification: QueryClassification,
                                      unknown_terms: List[str],
                                      complexity_factors: Dict[str, Any],
                                      historical_pattern: Optional[Dict[str, Any]],
                                      user_context: Optional[Dict[str, Any]]) -> RoutingDecision:
        """
        FIXED: Balanced routing decision with fair scoring for all paths
        """
        
        reasoning = []
        base_confidence = classification.confidence
        
        # FIXED: Balanced base scores (no artificial bias)
        rag_score = 1.0
        semantic_score = 1.0  
        agent_score = 1.0
        
        # === RAG PATH SCORING ===
        
        # RAG strengths: simple queries, known patterns, fast retrieval
        if base_confidence > 0.7:
            rag_score += 0.3
            reasoning.append(f"High classification confidence ({base_confidence:.2f}) favors RAG")
        
        if not unknown_terms:
            rag_score += 0.2
            reasoning.append("All terms recognized - RAG suitable")
        
        if complexity_factors['base_complexity'] == 'basic':
            rag_score += 0.3
            reasoning.append("Basic complexity well-suited for RAG")
        
        # Simple statistical operations that RAG templates handle well
        if (complexity_factors['requires_aggregation'] and 
            not complexity_factors['requires_calculation'] and
            complexity_factors['scientific_terms'] == 0):
            rag_score += 0.2
            reasoning.append("Simple aggregation suitable for RAG templates")
        
        # === SEMANTIC BRIDGE SCORING ===
        
        # Semantic bridge strengths: moderate complexity, some unknowns, enrichment needs
        if 0.4 < base_confidence <= 0.7:
            semantic_score += 0.3
            reasoning.append(f"Moderate confidence ({base_confidence:.2f}) suggests semantic enrichment needed")
        
        if 1 <= len(unknown_terms) <= 3:
            semantic_score += 0.4
            reasoning.append(f"Moderate unknown terms ({len(unknown_terms)}) suitable for semantic bridge")
        
        if complexity_factors['base_complexity'] in ['intermediate', 'advanced']:
            semantic_score += 0.3
            reasoning.append(f"Intermediate/advanced complexity benefits from semantic enrichment")
        
        if complexity_factors['requires_calculation'] and complexity_factors['calculation_complexity'] != 'expert':
            semantic_score += 0.2
            reasoning.append("Moderate calculations suitable for enhanced processing")
        
        # === AGENT SYSTEM SCORING ===
        
        # Agent strengths: high complexity, many unknowns, novel scenarios, research-level queries
        if len(unknown_terms) > 3:
            agent_score += 0.5
            reasoning.append(f"Many unknown terms ({len(unknown_terms)}) require agent reasoning")
        
        if complexity_factors['base_complexity'] in ['advanced', 'expert']:
            agent_score += 0.4
            reasoning.append(f"Advanced/expert complexity requires agent reasoning")
        
        if complexity_factors['calculation_complexity'] == 'expert':
            agent_score += 0.5
            reasoning.append("Expert-level calculations require agent collaboration")
        
        if complexity_factors['scientific_terms'] > 2:
            agent_score += 0.3
            reasoning.append(f"Multiple scientific terms ({complexity_factors['scientific_terms']}) suggest agent handling")
        
        if complexity_factors['comparative_terms'] > 0:
            agent_score += 0.2
            reasoning.append("Comparative analysis benefits from agent reasoning")
        
        # Novel query patterns (low confidence but high complexity)
        if base_confidence < 0.4 and complexity_factors['base_complexity'] != 'basic':
            agent_score += 0.4
            reasoning.append("Low confidence with complexity suggests novel scenario requiring agents")
        
        # === APPLY SYSTEM HEALTH ADJUSTMENTS ===
        current_health = self._assess_current_system_health()
        
        # Slight preference for faster systems if they're performing well
        if current_health['rag_performance'] < 2.0:  # Good performance
            rag_score *= 1.1
            reasoning.append("RAG system performing well - slight preference boost")
        
        if current_health.get('agent_availability', True):
            agent_score *= 1.0  # No penalty for agent availability
        else:
            agent_score *= 0.5  # Significant penalty if agents unavailable
            reasoning.append("Agent system availability issues - reduced score")
        
        # === HISTORICAL PATTERN INFLUENCE ===
        if historical_pattern:
            recommended_path = historical_pattern['recommended_path']
            boost = min(historical_pattern.get('confidence', 0.5) * 0.3, 0.2)
            
            if recommended_path == ProcessingPath.LIGHTNING_RAG:
                rag_score += boost
            elif recommended_path == ProcessingPath.SEMANTIC_BRIDGE:
                semantic_score += boost
            elif recommended_path == ProcessingPath.AGENTIC_FALLBACK:
                agent_score += boost
            
            reasoning.append(f"Historical success with {recommended_path.value} (+{boost:.2f})")
        
        # === MAKE FINAL DECISION ===
        scores = {
            ProcessingPath.LIGHTNING_RAG: rag_score,
            ProcessingPath.SEMANTIC_BRIDGE: semantic_score,
            ProcessingPath.AGENTIC_FALLBACK: agent_score
        }
        
        selected_path = max(scores.items(), key=lambda x: x[1])[0]
        final_confidence = min(scores[selected_path] / 2.0, 1.0)  # Normalize to 0-1 range
        
        # Debug logging
        logger.debug(f"ROUTING SCORES - RAG: {rag_score:.2f}, Semantic: {semantic_score:.2f}, Agents: {agent_score:.2f}")
        logger.debug(f"SELECTED: {selected_path.value} with confidence {final_confidence:.2f}")
        
        # Build final decision
        return RoutingDecision(
            path=selected_path,
            confidence=final_confidence,
            reasoning=reasoning,
            performance_budget=self._get_performance_budget(selected_path),
            fallback_path=self._get_fallback_path(selected_path),
            enrichments_needed=self._determine_enrichments(query, classification),
            unknown_terms=unknown_terms,
            complexity_factors=complexity_factors,
            estimated_cost=self._estimate_cost(selected_path)
        )
    
    def _initialize_realistic_thresholds(self) -> Dict[str, Dict]:
        """
        FIXED: Realistic performance thresholds that allow all paths
        """
        return {
            'response_time_targets': {
                ProcessingPath.LIGHTNING_RAG: 2.0,     # Realistic: 2 seconds
                ProcessingPath.SEMANTIC_BRIDGE: 10.0,   # Realistic: 10 seconds  
                ProcessingPath.AGENTIC_FALLBACK: 60.0   # Realistic: 60 seconds
            },
            'confidence_thresholds': {
                'high': 0.7,      # Lowered from 0.8
                'medium': 0.5,    # Lowered from 0.6  
                'low': 0.3        # Lowered from 0.4
            },
            'complexity_routing': {
                ComplexityLevel.BASIC: ProcessingPath.LIGHTNING_RAG,
                ComplexityLevel.INTERMEDIATE: ProcessingPath.SEMANTIC_BRIDGE,  # Changed
                ComplexityLevel.ADVANCED: ProcessingPath.AGENTIC_FALLBACK,     # Changed
                ComplexityLevel.EXPERT: ProcessingPath.AGENTIC_FALLBACK
            }
        }
    
    # Keep existing helper methods (they're working fine)
    def _initialize_known_terms(self) -> Set[str]:
        """Initialize known oceanographic terms"""
        
        known_terms = set()
        
        # Core parameters
        core_params = {
            'temperature', 'salinity', 'pressure', 'density', 'depth',
            'latitude', 'longitude', 'platform_number', 'cycle_number',
            'mixed_layer_depth', 'thermocline', 'pycnocline', 'halocline'
        }
        known_terms.update(core_params)
        
        # Analysis terms
        analysis_terms = {
            'variability', 'anomaly', 'gradient', 'flux', 'circulation',
            'stratification', 'biogeochemical', 'mesoscale', 'seasonal',
            'temporal', 'spatial', 'correlation', 'regression', 'trend',
            'climatology', 'interannual', 'diurnal', 'vertical', 'horizontal'
        }
        known_terms.update(analysis_terms)
        
        # Oceanographic processes
        processes = {
            'upwelling', 'downwelling', 'mixing', 'convection', 'advection',
            'diffusion', 'eddy', 'current', 'front', 'gyre', 'meandering',
            'instability', 'turbulence', 'entrainment', 'detrainment'
        }
        known_terms.update(processes)
        
        # Regional terms
        regions = {
            'indian_ocean', 'arabian_sea', 'bay_of_bengal', 'equatorial',
            'tropical', 'subtropical', 'northern', 'southern', 'western', 'eastern',
            'coastal', 'offshore', 'continental_shelf', 'abyssal', 'pelagic'
        }
        known_terms.update(regions)
        
        # Statistical terms
        stats_terms = {
            'average', 'mean', 'median', 'standard_deviation', 'variance',
            'percentile', 'quartile', 'maximum', 'minimum', 'range',
            'distribution', 'histogram', 'correlation', 'covariance'
        }
        known_terms.update(stats_terms)
        
        return known_terms
    
    def _check_historical_patterns(self, query: str) -> Optional[Dict[str, Any]]:
        """Check historical routing patterns"""
        
        if not self.query_history:
            return None
        
        # Simple pattern matching for now
        query_signature = self._create_query_signature(query)
        
        similar_queries = []
        for metric in self.query_history[-50:]:  # Check last 50 queries
            if metric.query_hash == query_signature:
                similar_queries.append(metric)
        
        if not similar_queries:
            return None
        
        # Find most successful path
        successful_queries = [q for q in similar_queries if q.success]
        if not successful_queries:
            return None
        
        # Get most common successful path
        path_counts = {}
        for query in successful_queries:
            path = query.processing_path
            path_counts[path] = path_counts.get(path, 0) + 1
        
        if path_counts:
            best_path = max(path_counts.items(), key=lambda x: x[1])[0]
            success_rate = len(successful_queries) / len(similar_queries)
            
            return {
                'recommended_path': best_path,
                'confidence': success_rate,
                'sample_size': len(similar_queries)
            }
        
        return None
    
    def _assess_current_system_health(self) -> Dict[str, Any]:
        """Assess current system health"""
        return {
            'rag_available': True,
            'rag_performance': self.system_health['rag_response_time'],
            'semantic_available': True,
            'agent_availability': True,
            'timestamp': datetime.now()
        }
    
    def _get_performance_budget(self, path: ProcessingPath) -> int:
        """Get performance budget for path"""
        budgets = {
            ProcessingPath.LIGHTNING_RAG: 5,
            ProcessingPath.SEMANTIC_BRIDGE: 15,
            ProcessingPath.AGENTIC_FALLBACK: 60,
            ProcessingPath.ERROR_RECOVERY: 5
        }
        return budgets.get(path, 30)
    
    def _get_fallback_path(self, path: ProcessingPath) -> Optional[ProcessingPath]:
        """Get fallback path"""
        fallbacks = {
            ProcessingPath.LIGHTNING_RAG: ProcessingPath.SEMANTIC_BRIDGE,
            ProcessingPath.SEMANTIC_BRIDGE: ProcessingPath.AGENTIC_FALLBACK,
            ProcessingPath.AGENTIC_FALLBACK: ProcessingPath.ERROR_RECOVERY,
            ProcessingPath.ERROR_RECOVERY: None
        }
        return fallbacks.get(path)
    
    def _determine_enrichments(self, query: str, classification: QueryClassification) -> List[str]:
        """Determine needed enrichments"""
        enrichments = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['seasonal', 'monthly', 'annual']):
            enrichments.append('temporal_context')
        
        if any(word in query_lower for word in ['regional', 'spatial', 'geographic']):
            enrichments.append('spatial_context')
        
        if any(word in query_lower for word in ['biogeochemical', 'ecosystem']):
            enrichments.append('domain_knowledge')
        
        return enrichments
    
    def _estimate_cost(self, path: ProcessingPath) -> str:
        """Estimate computational cost"""
        costs = {
            ProcessingPath.LIGHTNING_RAG: "low",
            ProcessingPath.SEMANTIC_BRIDGE: "medium",
            ProcessingPath.AGENTIC_FALLBACK: "high",
            ProcessingPath.ERROR_RECOVERY: "low"
        }
        return costs.get(path, "medium")
    
    def _create_query_signature(self, query: str) -> str:
        """Create query signature for similarity matching"""
        normalized = re.sub(r'[^\w\s]', '', query.lower())
        words = normalized.split()
        important_words = [w for w in words if len(w) > 3 and w in self.known_terms]
        signature = ' '.join(sorted(important_words))
        return hashlib.md5(signature.encode()).hexdigest()
    
    def record_query_result(self, query: str, routing_decision: RoutingDecision,
                           execution_time: float, success: bool, 
                           error_type: Optional[str] = None):
        """Record query results for learning"""
        
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        metrics = QueryMetrics(
            query_hash=query_hash,
            processing_path=routing_decision.path,
            execution_time=execution_time,
            success=success,
            confidence_score=routing_decision.confidence,
            error_type=error_type,
            enrichments_used=routing_decision.enrichments_needed
        )
        
        self.query_history.append(metrics)
        
        # Keep only recent history
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-500:]
        
        logger.info(f"Recorded result: {routing_decision.path.value} "
                   f"({'success' if success else 'failure'}, {execution_time:.2f}s)")
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics for monitoring"""
        
        if not self.query_history:
            return {"message": "No routing history available"}
        
        total_queries = len(self.query_history)
        path_stats = {}
        
        for path in ProcessingPath:
            path_queries = [q for q in self.query_history if q.processing_path == path]
            if path_queries:
                path_stats[path.value] = {
                    'count': len(path_queries),
                    'percentage': len(path_queries) / total_queries * 100,
                    'avg_execution_time': sum(q.execution_time for q in path_queries) / len(path_queries),
                    'success_rate': sum(1 for q in path_queries if q.success) / len(path_queries) * 100,
                    'avg_confidence': sum(q.confidence_score for q in path_queries) / len(path_queries)
                }
        
        return {
            'total_queries': total_queries,
            'path_statistics': path_stats,
            'system_health': self.system_health
        }