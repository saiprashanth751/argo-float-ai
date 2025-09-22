# src/services/smart_query_router.py - FIXED VERSION
"""
Smart Query Router: FIXED - Added missing methods that were causing crashes
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
    FIXED: Smart query router with all missing methods implemented
    """
    
    def __init__(self, intelligence_engine: OceanographicIntelligenceEngine,
                 vector_store=None):
        self.intelligence_engine = intelligence_engine
        self.vector_store = vector_store
        
        # Known oceanographic terms and concepts
        self.known_terms = self._initialize_known_terms()
        
        # Performance thresholds for routing decisions
        self.performance_thresholds = self._initialize_performance_thresholds()
        
        # Learning system for query patterns
        self.query_history: List[QueryMetrics] = []
        self.routing_patterns = {}
        
        # System health monitoring
        self.system_health = {
            'rag_response_time': 0.2,  # seconds
            'semantic_response_time': 3.0,
            'agent_response_time': 20.0,
            'error_rates': {
                ProcessingPath.LIGHTNING_RAG: 0.05,
                ProcessingPath.SEMANTIC_BRIDGE: 0.10,
                ProcessingPath.AGENTIC_FALLBACK: 0.15
            }
        }
        
        logger.info("Smart Query Router initialized with all methods")
    
    def route_query(self, query: str, user_context: Dict[str, Any] = None) -> RoutingDecision:
        """
        Main routing method - determines optimal processing path for query
        """
        
        start_time = time.time()
        
        # Step 1: Basic classification using existing intelligence engine
        logger.info("ROUTING : self.intelligence_engine.classify_query is going to execute...")
        classification = self.intelligence_engine.classify_query(query)
        logger.info("ROUTING : self.intelligence_engine.classify_query has been executed...")
        # Step 2: Detect unknown terms and concepts
        unknown_terms = self._detect_unknown_terms(query)
        logger.info("ROUTING : self._detect_unknown_terms has been executed...")
        
        # Step 3: Assess query characteristics
        complexity_factors = self._analyze_complexity_factors(query, classification)
        logger.info("ROUTING : self._analyze_complexity_factors has been executed...")
        
        # Step 4: Check historical patterns
        historical_pattern = self._check_historical_patterns(query)
        logger.info("ROUTING : self._check_historical_patterns has been executed...")
        
        # Step 5: Make routing decision
        routing_decision = self._make_routing_decision(
            query=query,
            classification=classification,
            unknown_terms=unknown_terms,
            complexity_factors=complexity_factors,
            historical_pattern=historical_pattern,
            user_context=user_context
        )
        logger.info("ROUTING : self._make_routing_decision has been executed...")
        
        # Step 6: Log decision for learning
        decision_time = time.time() - start_time
        logger.info("ROUTING : SUCCESSFULL ROUTING MESSAGE")
        logger.info(f"Routed query to {routing_decision.path.value} "
                   f"(confidence: {routing_decision.confidence:.2f}, "
                   f"decision_time: {decision_time:.3f}s)")
        
        return routing_decision
    
    def _initialize_known_terms(self) -> Set[str]:
        """EXPANDED oceanographic terms dictionary"""
        
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
        
        # Quality terms
        quality_terms = {
            'quality', 'flag', 'qc', 'validation', 'calibration', 'drift',
            'accuracy', 'precision', 'uncertainty', 'error', 'bias'
        }
        known_terms.update(quality_terms)
        
        logger.info(f"Expanded known terms to {len(known_terms)} oceanographic terms")
        return known_terms
    
    def _detect_unknown_terms(self, query: str) -> List[str]:
        """Detect terms in query that are not in known oceanographic vocabulary"""
        
        # Extract potential scientific terms
        scientific_pattern = r'\b[a-zA-Z]{4,}\b'
        potential_terms = re.findall(scientific_pattern, query.lower())
        
        # Common English words to ignore
        common_words = {
            'show', 'find', 'get', 'data', 'from', 'with', 'what', 'where',
            'when', 'how', 'why', 'the', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'by', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off',
            'over', 'under', 'again', 'further', 'then', 'once', 'analysis',
            'measurement', 'value', 'level', 'time', 'year', 'month', 'day'
        }
        
        # Identify unknown terms
        unknown_terms = []
        for term in potential_terms:
            if (term not in self.known_terms and 
                term not in common_words and 
                len(term) > 3):
                unknown_terms.append(term)
        
        # Remove duplicates and score by importance
        unknown_terms = list(set(unknown_terms))
        scored_unknowns = []
        for term in unknown_terms:
            score = self._score_unknown_term(term, query)
            if score > 0.3:
                scored_unknowns.append((term, score))
        
        scored_unknowns.sort(key=lambda x: x[1], reverse=True)
        return [term for term, score in scored_unknowns]
    
    def _score_unknown_term(self, term: str, query: str) -> float:
        """Score how likely an unknown term is to be scientifically important"""
        
        score = 0.0
        
        # Length bonus
        if len(term) > 8:
            score += 0.3
        elif len(term) > 6:
            score += 0.2
        
        # Context clues
        context_indicators = [
            'coefficient', 'index', 'ratio', 'gradient', 'flux', 'concentration',
            'velocity', 'frequency', 'amplitude', 'phase', 'spectrum'
        ]
        
        for indicator in context_indicators:
            if indicator in term.lower():
                score += 0.4
                break
        
        # Chemical/biological patterns
        if (term.endswith('ate') or term.endswith('ide') or 
            term.endswith('ine') or term.startswith('bio') or
            term.startswith('geo') or term.startswith('hydro')):
            score += 0.3
        
        # Units or measurements
        if any(unit in term.lower() for unit in ['gram', 'meter', 'liter', 'mol', 'bar']):
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_complexity_factors(self, query: str, 
                                   classification: QueryClassification) -> Dict[str, Any]:
        """Analyze factors that contribute to query complexity"""
        
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
            'comparative_terms': 0
        }
        
        query_lower = query.lower()
        
        # Detect joins needed
        join_indicators = ['profile', 'measurement', 'depth', 'pressure', 'vertical']
        if any(indicator in query_lower for indicator in join_indicators):
            factors['requires_joins'] = True
        
        # Detect aggregation needs
        agg_indicators = ['average', 'mean', 'sum', 'count', 'maximum', 'minimum', 'total']
        if any(indicator in query_lower for indicator in agg_indicators):
            factors['requires_aggregation'] = True
        
        # Count question complexity
        question_words = ['what', 'where', 'when', 'how', 'why', 'which', 'who']
        factors['question_words'] = sum(1 for word in question_words if word in query_lower)
        
        # Detect comparative analysis
        comparative_terms = ['compare', 'versus', 'difference', 'between', 'correlation']
        factors['comparative_terms'] = sum(1 for term in comparative_terms if term in query_lower)
        
        return factors
    
    def _check_historical_patterns(self, query: str) -> Optional[Dict[str, Any]]:
        """Check if similar queries have been processed before"""
        
        if not self.query_history:
            return None
        
        # Create query signature for similarity matching
        query_signature = self._create_query_signature(query)
        
        # Look for similar queries in history
        similar_queries = []
        for metric in self.query_history[-100:]:  # Check last 100 queries
            similarity = self._calculate_query_similarity(query_signature, metric.query_hash)
            if similarity > 0.7:  # High similarity threshold
                similar_queries.append((metric, similarity))
        
        if not similar_queries:
            return None
        
        # Analyze historical performance
        successful_paths = {}
        for metric, similarity in similar_queries:
            if metric.success:
                path = metric.processing_path
                if path not in successful_paths:
                    successful_paths[path] = []
                successful_paths[path].append({
                    'execution_time': metric.execution_time,
                    'confidence': metric.confidence_score,
                    'similarity': similarity
                })
        
        if not successful_paths:
            return None
        
        # Find best performing path
        best_path = None
        best_score = 0
        
        for path, metrics in successful_paths.items():
            avg_time = sum(m['execution_time'] for m in metrics) / len(metrics)
            avg_confidence = sum(m['confidence'] for m in metrics) / len(metrics)
            success_rate = len(metrics) / len([m for m, s in similar_queries])
            
            # Composite score
            score = (avg_confidence * success_rate) / (avg_time + 1)
            
            if score > best_score:
                best_score = score
                best_path = path
        
        return {
            'recommended_path': best_path,
            'historical_performance': successful_paths[best_path],
            'confidence': best_score,
            'similar_query_count': len(similar_queries)
        }
    
    def _make_routing_decision(self, query: str, 
                          classification: QueryClassification,
                          unknown_terms: List[str],
                          complexity_factors: Dict[str, Any],
                          historical_pattern: Optional[Dict[str, Any]],
                          user_context: Optional[Dict[str, Any]]) -> RoutingDecision:
        """FIXED: Calibrated routing decision - favor Lightning RAG more aggressively"""
        
        
        try:
            reasoning = []
            confidence = classification.confidence
            
            # Much more realistic confidence thresholds
            if confidence > 0.6:
                reasoning.append(f"Good classification confidence ({confidence:.2f})")
                path_score_rag = 1.2  # BOOSTED
            elif confidence > 0.4:
                reasoning.append(f"Acceptable classification confidence ({confidence:.2f})")
                path_score_rag = 1.0  # Still good
            else:
                reasoning.append(f"Low classification confidence ({confidence:.2f})")
                path_score_rag = 0.7  # Less penalty
            
            # Smarter unknown terms handling
            critical_unknown_terms = [term for term in unknown_terms 
                                    if not self._is_likely_simple_variant(term)]
            
            if critical_unknown_terms:
                reasoning.append(f"Critical unknown terms: {', '.join(critical_unknown_terms[:3])}")
                path_score_rag *= 0.9  # MINIMAL penalty
                path_score_semantic = 1.1  # Slight boost
                path_score_agents = 0.5   # REDUCED
            else:
                if unknown_terms:
                    reasoning.append(f"Minor unknown terms: {', '.join(unknown_terms[:3])}")
                    path_score_rag *= 0.98  # TINY penalty
                else:
                    reasoning.append("All terms recognized")
                path_score_semantic = 0.4  # REDUCED
                path_score_agents = 0.2   # MUCH REDUCED
            
            # Complexity handling - HEAVILY favor Lightning RAG
            complexity = classification.complexity
            if complexity == ComplexityLevel.BASIC:
                reasoning.append("Basic complexity - Lightning RAG optimal")
                complexity_boost_rag = 1.5     # MASSIVE boost
                complexity_boost_semantic = 0.5
                complexity_boost_agents = 0.2
            elif complexity == ComplexityLevel.INTERMEDIATE:
                reasoning.append("Intermediate complexity - Lightning RAG preferred")
                complexity_boost_rag = 1.3     # BIG boost
                complexity_boost_semantic = 0.8
                complexity_boost_agents = 0.4
            elif complexity == ComplexityLevel.ADVANCED:
                if self._is_analytical_not_complex(query, classification):
                    reasoning.append("Advanced analysis - Lightning RAG with semantic support")
                    complexity_boost_rag = 1.1      # STILL boost RAG
                    complexity_boost_semantic = 1.2
                    complexity_boost_agents = 0.6
                else:
                    reasoning.append("Complex analysis - Semantic bridge preferred")
                    complexity_boost_rag = 0.8      # Slight penalty
                    complexity_boost_semantic = 1.3
                    complexity_boost_agents = 0.9
            else:  # EXPERT
                reasoning.append("Expert-level complexity - Agent system may be needed")
                complexity_boost_rag = 0.5
                complexity_boost_semantic = 1.1
                complexity_boost_agents = 1.2
            
            # Calculate path scores
            rag_score = path_score_rag * complexity_boost_rag
            semantic_score = (path_score_semantic * complexity_boost_semantic 
                            if critical_unknown_terms or complexity in [ComplexityLevel.ADVANCED, ComplexityLevel.EXPERT] 
                            else 0.2)  # VERY LOW fallback
            agent_score = path_score_agents * complexity_boost_agents
            
            # System health boost for RAG
            current_health = self._assess_current_system_health()
            if current_health['rag_performance'] < 2.0:
                rag_score *= 1.4  # BIG boost for good RAG performance
                reasoning.append("RAG system performing excellently")
            
            # Historical pattern boost
            if historical_pattern:
                recommended_path = historical_pattern['recommended_path']
                reasoning.append(f"Historical success with {recommended_path.value}")
                
                if recommended_path == ProcessingPath.LIGHTNING_RAG:
                    rag_score *= 1.3  # Big boost
                elif recommended_path == ProcessingPath.SEMANTIC_BRIDGE:
                    semantic_score *= 1.1
                elif recommended_path == ProcessingPath.AGENTIC_FALLBACK:
                    agent_score *= 1.05  # Small boost
            
            # FINAL DECISION
            scores = {
                ProcessingPath.LIGHTNING_RAG: rag_score,
                ProcessingPath.SEMANTIC_BRIDGE: semantic_score,
                ProcessingPath.AGENTIC_FALLBACK: agent_score
            }
            
            selected_path = max(scores.items(), key=lambda x: x[1])[0]
            final_confidence = min(scores[selected_path], 1.0)
            
            # Debug logging
            logger.debug(f"Routing scores - RAG: {rag_score:.2f}, Semantic: {semantic_score:.2f}, Agents: {agent_score:.2f}")
            logger.debug(f"Selected: {selected_path.value} with confidence {final_confidence:.2f}")
            
            try:
                routing_decision = RoutingDecision(
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
                
                # Final validation
                if not isinstance(routing_decision.path, ProcessingPath):
                    logger.error(f"CRITICAL: routing_decision.path corrupted after creation: {type(routing_decision.path)}")
                    raise ValueError("RoutingDecision.path was corrupted during object creation")
                
                logger.debug(f"RoutingDecision created successfully with path: {routing_decision.path}")
                return routing_decision
                
            except Exception as e:
                logger.error(f"Failed to create RoutingDecision: {e}")
                raise
        except Exception as e:
            logger.error(f"Routing decision failed: {str(e)}")
            # Return a safe fallback decision
            return RoutingDecision(
                path=ProcessingPath.AGENTIC_FALLBACK,
                confidence=0.1,
                reasoning=["Error in routing decision"],
                unknown_terms=unknown_terms
            )
    
    # FIXED: Add all missing methods    
    def _is_likely_simple_variant(self, term: str) -> bool:
        """Check if unknown term is likely just a variant of known terms"""
        simple_variants = [
            (r'(.+)s$', r'\1'),      # plurals
            (r'(.+)ing$', r'\1'),    # gerunds
            (r'(.+)ed$', r'\1'),     # past tense
            (r'(.+)tion$', r'\1'),   # -tion endings
            (r'(.+)ity$', r'\1'),    # -ity endings
        ]
        
        for pattern, replacement in simple_variants:
            base_term = re.sub(pattern, replacement, term.lower())
            if base_term in self.known_terms:
                return True
        return False
    
    def _is_analytical_not_complex(self, query: str, classification: QueryClassification) -> bool:
        """Distinguish between analytical and truly complex queries"""
        query_lower = query.lower()
        
        analytical_patterns = [
            'average', 'mean', 'distribution', 'correlation', 'comparison',
            'seasonal', 'temporal', 'spatial', 'regional', 'profile',
            'statistics', 'summary', 'count', 'maximum', 'minimum'
        ]
        
        complex_patterns = [
            'predict', 'forecast', 'model', 'simulate', 'calculate complex',
            'biogeochemical cycle', 'ecosystem interaction', 'climate impact',
            'mass balance', 'heat budget', 'carbon cycle'
        ]
        
        analytical_score = sum(2 if pattern in query_lower else 0 for pattern in analytical_patterns)
        complex_score = sum(3 if pattern in query_lower else 0 for pattern in complex_patterns)
        
        if any(word in query_lower for word in ['show', 'display', 'get', 'find']):
            analytical_score += 1
        
        return analytical_score >= complex_score
    
    def _get_performance_budget(self, path: ProcessingPath) -> int:
        """Get performance budget in seconds for the given path"""
        budgets = {
            ProcessingPath.LIGHTNING_RAG: 5,        # 5 seconds
            ProcessingPath.SEMANTIC_BRIDGE: 15,     # 15 seconds  
            ProcessingPath.AGENTIC_FALLBACK: 60,    # 60 seconds
            ProcessingPath.ERROR_RECOVERY: 5        # 5 seconds
        }
        return budgets.get(path, 30)
    
    def _get_fallback_path(self, path: ProcessingPath) -> Optional[ProcessingPath]:
        """Get fallback path for the given primary path"""
        fallbacks = {
            ProcessingPath.LIGHTNING_RAG: ProcessingPath.SEMANTIC_BRIDGE,
            ProcessingPath.SEMANTIC_BRIDGE: ProcessingPath.AGENTIC_FALLBACK,
            ProcessingPath.AGENTIC_FALLBACK: ProcessingPath.ERROR_RECOVERY,
            ProcessingPath.ERROR_RECOVERY: None
        }
        return fallbacks.get(path)
    
    def _determine_enrichments(self, query: str, classification: QueryClassification) -> List[str]:
        """Determine what enrichments are needed for the query"""
        enrichments = []
        
        query_lower = query.lower()
        
        # Temporal enrichments
        if any(word in query_lower for word in ['seasonal', 'monthly', 'annual', 'trend']):
            enrichments.append('temporal_context')
        
        # Spatial enrichments
        if any(word in query_lower for word in ['regional', 'spatial', 'geographic']):
            enrichments.append('spatial_context')
        
        # Domain enrichments
        if any(word in query_lower for word in ['biogeochemical', 'ecosystem', 'climate']):
            enrichments.append('domain_knowledge')
        
        # Statistical enrichments
        if any(word in query_lower for word in ['correlation', 'regression', 'statistics']):
            enrichments.append('statistical_analysis')
        
        return enrichments
    
    def _estimate_cost(self, path: ProcessingPath) -> str:
        """Estimate computational cost for the given path"""
        costs = {
            ProcessingPath.LIGHTNING_RAG: "low",
            ProcessingPath.SEMANTIC_BRIDGE: "medium", 
            ProcessingPath.AGENTIC_FALLBACK: "high",
            ProcessingPath.ERROR_RECOVERY: "low"
        }
        return costs.get(path, "medium")
    
    def _assess_current_system_health(self) -> Dict[str, Any]:
        """Assess current system health for routing decisions"""
        return {
            'rag_available': True,
            'rag_performance': self.system_health['rag_response_time'],
            'semantic_available': True,
            'agents_available': True,
            'timestamp': datetime.now()
        }
    
    def record_query_result(self, query: str, routing_decision: RoutingDecision,
                           execution_time: float, success: bool, 
                           error_type: Optional[str] = None):
        """Record query result for learning and optimization"""
        
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
        
        # Update routing patterns
        self._update_routing_patterns(query, routing_decision, success, execution_time)
        
        logger.info(f"Recorded query result: {routing_decision.path.value} "
                   f"({'success' if success else 'failure'}, {execution_time:.2f}s)")
    
    # Helper methods
    def _initialize_performance_thresholds(self) -> Dict[str, Dict]:
        """Initialize performance thresholds for routing decisions"""
        return {
            'response_time_targets': {
                ProcessingPath.LIGHTNING_RAG: 1.0,
                ProcessingPath.SEMANTIC_BRIDGE: 10.0,
                ProcessingPath.AGENTIC_FALLBACK: 60.0
            },
            'confidence_thresholds': {
                'high': 0.8,
                'medium': 0.6,
                'low': 0.4
            },
            'complexity_routing': {
                ComplexityLevel.BASIC: ProcessingPath.LIGHTNING_RAG,
                ComplexityLevel.INTERMEDIATE: ProcessingPath.SEMANTIC_BRIDGE,
                ComplexityLevel.ADVANCED: ProcessingPath.SEMANTIC_BRIDGE,
                ComplexityLevel.EXPERT: ProcessingPath.AGENTIC_FALLBACK
            }
        }
    
    def _create_query_signature(self, query: str) -> str:
        """Create normalized signature for query similarity matching"""
        normalized = re.sub(r'[^\w\s]', '', query.lower())
        words = normalized.split()
        important_words = [w for w in words if len(w) > 3 and w in self.known_terms]
        signature = ' '.join(sorted(important_words))
        return hashlib.md5(signature.encode()).hexdigest()
    
    def _calculate_query_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between query signatures"""
        return 1.0 if sig1 == sig2 else 0.0
    
    def _update_routing_patterns(self, query: str, decision: RoutingDecision,
                                success: bool, execution_time: float):
        """Update learned routing patterns based on results"""
        
        pattern_key = f"{decision.path.value}_{len(decision.unknown_terms)}_{decision.complexity_factors['base_complexity']}"
        
        if pattern_key not in self.routing_patterns:
            self.routing_patterns[pattern_key] = {
                'total_queries': 0,
                'successful_queries': 0,
                'avg_execution_time': 0.0,
                'confidence_sum': 0.0
            }
        
        pattern = self.routing_patterns[pattern_key]
        pattern['total_queries'] += 1
        
        if success:
            pattern['successful_queries'] += 1
        
        # Update running averages
        old_count = pattern['total_queries'] - 1
        if old_count > 0:
            pattern['avg_execution_time'] = (
                (pattern['avg_execution_time'] * old_count + execution_time) / 
                pattern['total_queries']
            )
            pattern['confidence_sum'] = (
                (pattern['confidence_sum'] * old_count + decision.confidence) /
                pattern['total_queries']
            )
        else:
            pattern['avg_execution_time'] = execution_time
            pattern['confidence_sum'] = decision.confidence
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics for monitoring and optimization"""
        
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
            'routing_patterns': self.routing_patterns,
            'system_health': self.system_health
        }