#src/services/smart_query_router.py
"""
Smart Query Router: The intelligence layer that determines optimal processing path
for oceanographic queries based on confidence, complexity, and system capabilities.

This is the critical component that preserves speed advantages while enabling
sophisticated reasoning for complex queries.
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
    Intelligent query router that determines optimal processing path based on:
    1. Query confidence and complexity
    2. Unknown term detection
    3. Historical performance patterns
    4. System resource availability
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
        
        logger.info("Smart Query Router initialized")
    
    def route_query(self, query: str, user_context: Dict[str, Any] = None) -> RoutingDecision:
        """
        Main routing method - determines optimal processing path for query
        
        Args:
            query: Natural language query
            user_context: Optional user preferences and history
            
        Returns:
            RoutingDecision with complete routing strategy
        """
        
        start_time = time.time()
        
        # Step 1: Basic classification using existing intelligence engine
        classification = self.intelligence_engine.classify_query(query)
        
        # Step 2: Detect unknown terms and concepts
        unknown_terms = self._detect_unknown_terms(query)
        
        # Step 3: Assess query characteristics
        complexity_factors = self._analyze_complexity_factors(query, classification)
        
        # Step 4: Check historical patterns
        historical_pattern = self._check_historical_patterns(query)
        
        # Step 5: Make routing decision
        routing_decision = self._make_routing_decision(
            query=query,
            classification=classification,
            unknown_terms=unknown_terms,
            complexity_factors=complexity_factors,
            historical_pattern=historical_pattern,
            user_context=user_context
        )
        
        # Step 6: Log decision for learning
        decision_time = time.time() - start_time
        logger.info(f"Routed query to {routing_decision.path.value} "
                   f"(confidence: {routing_decision.confidence:.2f}, "
                   f"decision_time: {decision_time:.3f}s)")
        
        return routing_decision
    
    def _initialize_known_terms(self) -> Set[str]:
        """Initialize set of known oceanographic terms from vector DB and domain knowledge"""
        
        known_terms = set()
        
        # Core oceanographic parameters
        core_params = {
            'temperature', 'salinity', 'pressure', 'density', 'depth',
            'latitude', 'longitude', 'platform_number', 'cycle_number',
            'mixed_layer_depth', 'thermocline', 'pycnocline', 'halocline'
        }
        known_terms.update(core_params)
        
        # Regional terms
        regions = {
            'indian_ocean', 'arabian_sea', 'bay_of_bengal', 'equatorial',
            'tropical', 'subtropical', 'northern', 'southern', 'western', 'eastern'
        }
        known_terms.update(regions)
        
        # Temporal terms
        temporal = {
            'seasonal', 'annual', 'monthly', 'daily', 'climatology',
            'anomaly', 'trend', 'variability', 'monsoon', 'winter', 'summer'
        }
        known_terms.update(temporal)
        
        # Physical processes
        processes = {
            'upwelling', 'downwelling', 'mixing', 'stratification', 'convection',
            'advection', 'diffusion', 'circulation', 'current', 'eddy', 'front'
        }
        known_terms.update(processes)
        
        # Analysis types
        analysis = {
            'profile', 'distribution', 'correlation', 'regression', 'statistics',
            'average', 'mean', 'median', 'standard_deviation', 'percentile'
        }
        known_terms.update(analysis)
        
        # Quality and instrumentation
        quality = {
            'quality', 'flag', 'qc', 'validation', 'calibration', 'drift',
            'argo', 'float', 'ctd', 'sensor', 'measurement'
        }
        known_terms.update(quality)
        
        # If vector store available, extract terms from documents
        if self.vector_store:
            try:
                # This would extract terms from your vector DB metadata
                vector_terms = self._extract_terms_from_vector_db()
                known_terms.update(vector_terms)
            except Exception as e:
                logger.warning(f"Could not extract terms from vector DB: {e}")
        
        logger.info(f"Initialized {len(known_terms)} known oceanographic terms")
        return known_terms
    
    def _detect_unknown_terms(self, query: str) -> List[str]:
        """Detect terms in query that are not in known oceanographic vocabulary"""
        
        # Extract potential scientific terms (not common words)
        scientific_pattern = r'\b[a-zA-Z]{4,}\b'  # Words 4+ chars
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
        
        # Remove duplicates and sort by potential importance
        unknown_terms = list(set(unknown_terms))
        
        # Score unknown terms by context clues
        scored_unknowns = []
        for term in unknown_terms:
            score = self._score_unknown_term(term, query)
            if score > 0.3:  # Only include terms likely to be scientific
                scored_unknowns.append((term, score))
        
        # Return sorted by importance
        scored_unknowns.sort(key=lambda x: x[1], reverse=True)
        return [term for term, score in scored_unknowns]
    
    def _score_unknown_term(self, term: str, query: str) -> float:
        """Score how likely an unknown term is to be scientifically important"""
        
        score = 0.0
        
        # Length bonus (longer scientific terms more likely important)
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
            
            # Composite score (lower time, higher confidence, higher success rate)
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
        """Make the final routing decision based on all analysis"""
        
        reasoning = []
        confidence = classification.confidence
        
        # Factor 1: Classification confidence
        if confidence > 0.8:
            reasoning.append(f"High classification confidence ({confidence:.2f})")
            path_score_rag = 1.0
        elif confidence > 0.6:
            reasoning.append(f"Medium classification confidence ({confidence:.2f})")
            path_score_rag = 0.7
        else:
            reasoning.append(f"Low classification confidence ({confidence:.2f})")
            path_score_rag = 0.3
        
        # Factor 2: Unknown terms
        if unknown_terms:
            reasoning.append(f"Unknown terms detected: {', '.join(unknown_terms[:3])}")
            path_score_rag *= 0.5  # Penalize RAG for unknown terms
            path_score_semantic = 0.8
            path_score_agents = 1.0
        else:
            reasoning.append("All terms recognized")
            path_score_semantic = 0.6
            path_score_agents = 0.4
        
        # Factor 3: Query complexity
        complexity = classification.complexity
        if complexity == ComplexityLevel.BASIC:
            reasoning.append("Basic complexity query")
            complexity_boost_rag = 1.2
            complexity_boost_semantic = 0.8
            complexity_boost_agents = 0.6
        elif complexity == ComplexityLevel.INTERMEDIATE:
            reasoning.append("Intermediate complexity query")
            complexity_boost_rag = 1.0
            complexity_boost_semantic = 1.1
            complexity_boost_agents = 0.9
        elif complexity == ComplexityLevel.ADVANCED:
            reasoning.append("Advanced complexity query")
            complexity_boost_rag = 0.7
            complexity_boost_semantic = 1.2
            complexity_boost_agents = 1.1
        else:  # EXPERT
            reasoning.append("Expert-level complexity query")
            complexity_boost_rag = 0.4
            complexity_boost_semantic = 0.9
            complexity_boost_agents = 1.3
        
        # Calculate path scores
        rag_score = path_score_rag * complexity_boost_rag
        semantic_score = path_score_semantic * complexity_boost_semantic if unknown_terms else 0.3
        agent_score = path_score_agents * complexity_boost_agents
        
        # Factor 4: Historical patterns
        if historical_pattern:
            recommended_path = historical_pattern['recommended_path']
            reasoning.append(f"Historical data suggests {recommended_path.value}")
            
            # Boost recommended path
            if recommended_path == ProcessingPath.LIGHTNING_RAG:
                rag_score *= 1.3
            elif recommended_path == ProcessingPath.SEMANTIC_BRIDGE:
                semantic_score *= 1.3
            elif recommended_path == ProcessingPath.AGENTIC_FALLBACK:
                agent_score *= 1.3
        
        # Factor 5: System health and performance
        current_health = self._assess_current_system_health()
        if current_health['rag_available'] and current_health['rag_performance'] < 0.5:
            rag_score *= 1.2
            reasoning.append("RAG system performing well")
        
        if not current_health['agents_available']:
            agent_score = 0
            reasoning.append("Agent system unavailable")
        
        # Make final decision
        scores = {
            ProcessingPath.LIGHTNING_RAG: rag_score,
            ProcessingPath.SEMANTIC_BRIDGE: semantic_score,
            ProcessingPath.AGENTIC_FALLBACK: agent_score
        }
        
        selected_path = max(scores.items(), key=lambda x: x[1])[0]
        final_confidence = min(scores[selected_path], 1.0)
        
        # Determine enrichments needed
        enrichments = []
        if unknown_terms and selected_path in [ProcessingPath.SEMANTIC_BRIDGE, ProcessingPath.AGENTIC_FALLBACK]:
            enrichments.append("unknown_term_resolution")
        
        if complexity_factors['requires_calculation']:
            enrichments.append("parameter_calculation")
        
        if complexity_factors['comparative_terms'] > 0:
            enrichments.append("comparative_context")
        
        # Set performance budget
        performance_budgets = {
            ProcessingPath.LIGHTNING_RAG: 5,
            ProcessingPath.SEMANTIC_BRIDGE: 15,
            ProcessingPath.AGENTIC_FALLBACK: 60
        }
        
        # Determine fallback path
        fallback_map = {
            ProcessingPath.LIGHTNING_RAG: ProcessingPath.SEMANTIC_BRIDGE,
            ProcessingPath.SEMANTIC_BRIDGE: ProcessingPath.AGENTIC_FALLBACK,
            ProcessingPath.AGENTIC_FALLBACK: ProcessingPath.ERROR_RECOVERY
        }
        
        # Estimate computational cost
        cost_estimates = {
            ProcessingPath.LIGHTNING_RAG: "low",
            ProcessingPath.SEMANTIC_BRIDGE: "medium", 
            ProcessingPath.AGENTIC_FALLBACK: "high"
        }
        
        return RoutingDecision(
            path=selected_path,
            confidence=final_confidence,
            reasoning=reasoning,
            performance_budget=performance_budgets[selected_path],
            fallback_path=fallback_map[selected_path],
            enrichments_needed=enrichments,
            unknown_terms=unknown_terms,
            complexity_factors=complexity_factors,
            estimated_cost=cost_estimates[selected_path]
        )
    
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
        
        # Keep only recent history for performance
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
                ProcessingPath.LIGHTNING_RAG: 1.0,      # 1 second
                ProcessingPath.SEMANTIC_BRIDGE: 10.0,   # 10 seconds
                ProcessingPath.AGENTIC_FALLBACK: 60.0   # 1 minute
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
    
    def _extract_terms_from_vector_db(self) -> Set[str]:
        """Extract known terms from vector database documents"""
        terms = set()
        
        try:
            if hasattr(self.vector_store, '_collection'):
                # This would depend on your vector store implementation
                # For now, return empty set
                pass
        except Exception as e:
            logger.warning(f"Could not extract terms from vector DB: {e}")
        
        return terms
    
    def _create_query_signature(self, query: str) -> str:
        """Create normalized signature for query similarity matching"""
        
        # Normalize query
        normalized = re.sub(r'[^\w\s]', '', query.lower())
        words = normalized.split()
        
        # Remove common words and sort
        important_words = [w for w in words if len(w) > 3 and w in self.known_terms]
        signature = ' '.join(sorted(important_words))
        
        return hashlib.md5(signature.encode()).hexdigest()
    
    def _calculate_query_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between query signatures"""
        # Simple implementation - could be enhanced with ML similarity
        return 1.0 if sig1 == sig2 else 0.0
    
    def _assess_current_system_health(self) -> Dict[str, Any]:
        """Assess current system health for routing decisions"""
        return {
            'rag_available': True,
            'rag_performance': self.system_health['rag_response_time'],
            'semantic_available': True,
            'agents_available': True,
            'timestamp': datetime.now()
        }
    
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


# Test function for the router
def test_smart_router():
    """Test the smart query router with various oceanographic queries"""
    
    from unittest.mock import Mock
    
    # Create mock intelligence engine
    mock_engine = Mock()
    mock_engine.classify_query.return_value = Mock(
        confidence=0.85,
        complexity=ComplexityLevel.INTERMEDIATE,
        context=Mock(
            parameters=['temperature', 'salinity'],
            spatial_bounds={'lat_min': 10, 'lat_max': 25, 'lon_min': 50, 'lon_max': 78},
            temporal_range=None
        ),
        required_calculations=['mixed_layer_depth']
    )
    
    # Initialize router
    router = SmartQueryRouter(mock_engine)
    
    # Test queries
    test_queries = [
        "Show temperature profile for platform 1900121",
        "What is the average thermocline depth in Arabian Sea during monsoon?",
        "Compare biogeochemical flux patterns between different ocean basins",
        "Find chlorophyll concentration gradients in upwelling zones",
        "Surface temperature distribution in Indian Ocean"
    ]
    
    print("Testing Smart Query Router")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        decision = router.route_query(query)
        
        print(f"  → Path: {decision.path.value}")
        print(f"  → Confidence: {decision.confidence:.2f}")
        print(f"  → Budget: {decision.performance_budget}s")
        print(f"  → Unknown terms: {decision.unknown_terms}")
        print(f"  → Enrichments: {decision.enrichments_needed}")
        print(f"  → Reasoning: {decision.reasoning[0] if decision.reasoning else 'None'}")
        
        # Simulate recording result
        router.record_query_result(query, decision, 2.5, True)
    
    # Show statistics
    print("\nRouting Statistics:")
    print("-" * 30)
    stats = router.get_routing_statistics()
    for path, data in stats['path_statistics'].items():
        print(f"{path}: {data['count']} queries ({data['percentage']:.1f}%), "
              f"{data['success_rate']:.1f}% success")


if __name__ == "__main__":
    test_smart_router()