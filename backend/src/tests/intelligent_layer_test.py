#backend\src\tests\intelligent_layer_test.py
"""
Complete Integration Test Suite - Production Readiness Validation
"""
import sys
import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import os
from pathlib import Path

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Core system imports
from services.orchestrated_rag_system import OrchestratedOceanographicRAG
from services.smart_query_router import ProcessingPath, SmartQueryRouter
from services.response_intelligence_layer import ResponseConfig
from services.agent_collaboration_system import ProductionAgentCollaborationSystem
from services.oceanographic_intelligence_engine import (
    QueryClassification, 
    QueryIntent,
    ComplexityLevel
)
from services.types_core import RoutingDecision, ProcessingPath, ComplexityLevel, QueryIntent

logger = logging.getLogger(__name__)

# @dataclass
# class RoutingDecision:
#     """Test routing decision class that mimics production RoutingDecision"""
#     processing_path: ProcessingPath
#     confidence: float
#     complexity_level: ComplexityLevel
#     query_intent: QueryIntent
#     fallback_paths: List[ProcessingPath]
#     metadata: Dict[str, Any]
#     enrichments_needed: List[str]
#     unknown_terms: List[str]
#     performance_budget: int
#     reasoning: List[str]
#     estimated_cost: str

class CompleteSystemTest:
    """
    Production-ready test suite that validates all system layers and integrations
    """
    
    def __init__(self):
        self.test_results = {
            'overall_status': 'pending',
            'test_suites': {},
            'performance_metrics': {},
            'system_health': {},
            'start_time': datetime.now().isoformat()
        }
        
        # Initialize test system
    async def initialize_system(self):
        self.rag_system = OrchestratedOceanographicRAG()

    async def run_complete_test_suite(self) -> Dict[str, Any]:
        """Run all test suites with comprehensive validation"""
        
        test_suites = [
            self._test_layer1_lightning_rag(),
            self._test_layer2_semantic_bridge(),
            self._test_layer3_agent_system(),
            self._test_cross_layer_integration(),
            self._test_error_handling(),
            self._test_performance_metrics()
        ]
        
        for suite in test_suites:
            await suite
            
        self._calculate_final_status()
        return self.test_results

    async def _test_layer1_lightning_rag(self):
        """Test Lightning RAG capabilities"""
        
        logger.info("Testing Layer 1: Lightning RAG")
        results = {'status': 'running', 'tests': []}
        
        test_queries = [
            "Show temperature at 1000m depth",
            "Count profiles in Arabian Sea",
            "List all platforms with salinity data"
        ]
        
        for query in test_queries:
            try:
                start_time = time.time()
                response = self.rag_system._execute_lightning_rag(
                    query=query,
                    routing_decision=self._create_test_routing_decision(),  # This now returns TestRoutingDecision
                    response_format=ResponseConfig()
                )
                execution_time = time.time() - start_time
                
                results['tests'].append({
                    'query': query,
                    'success': response['success'],
                    'execution_time': execution_time,
                    'meets_sla': execution_time < 0.2  # 200ms target
                })
                
            except Exception as e:
                results['tests'].append({
                    'query': query,
                    'success': False,
                    'error': str(e)
                })
        
        self.test_results['test_suites']['lightning_rag'] = results

    async def _test_layer2_semantic_bridge(self):
        """Test Semantic Bridge capabilities"""
        
        logger.info("Testing Layer 2: Semantic Bridge")
        results = {'status': 'running', 'tests': []}
        
        test_cases = [
            {
                'query': "What is the thermocline depth variation?",
                'expected_enrichments': ['mixed_layer_depth', 'density_gradient']
            },
            {
                'query': "Show upwelling patterns in Arabian Sea during monsoon",
                'expected_enrichments': ['seasonal_effects', 'wind_driven_circulation']
            }
        ]
        
        for case in test_cases:
            try:
                start_time = time.time()
                response = self.rag_system._execute_semantic_bridge(
                    query=case['query'],
                    routing_decision=self._create_test_routing_decision(),  # This now returns TestRoutingDecision
                    response_format=ResponseConfig()
                )
                execution_time = time.time() - start_time
                
                # Validate enrichments
                enrichments_found = set(response.get('enrichments', []))
                expected_enrichments = set(case['expected_enrichments'])
                
                results['tests'].append({
                    'query': case['query'],
                    'success': response['success'],
                    'execution_time': execution_time,
                    'meets_sla': execution_time < 3.0,  # 3s target
                    'enrichment_coverage': len(enrichments_found & expected_enrichments) / len(expected_enrichments)
                })
                
            except Exception as e:
                results['tests'].append({
                    'query': case['query'],
                    'success': False,
                    'error': str(e)
                })
        
        self.test_results['test_suites']['semantic_bridge'] = results

    async def _test_layer3_agent_system(self):
        """Test Agent Collaboration System - HONEST VERSION"""
        
        logger.info("Testing Layer 3: Agent System")
        results = {'status': 'running', 'tests': []}
        
        test_scenarios = [
            {
                'query': "Compare biogeochemical processes between Bay of Bengal and Arabian Sea",
                'expected_agents': ['domain_researcher', 'sql_specialist', 'result_validator']
            },
            {
                'query': "Analyze deep water formation patterns with quality validation",
                'expected_agents': ['domain_researcher', 'quality_assessor', 'integration_coordinator']
            }
        ]
        
        for scenario in test_scenarios:
            try:
                start_time = time.time()
                response = await self.rag_system._execute_agentic_fallback(
                    query=scenario['query'],
                    routing_decision=self._create_test_routing_decision(),
                    response_format=ResponseConfig()
                )
                execution_time = time.time() - start_time
                
                # HONEST VALIDATION: Check if this is real agent collaboration or fallback
                is_real_collaboration = (
                    response.get('success', False) and 
                    not response.get('fallback_mode', False) and
                    not response.get('agent_error') and
                    len(response.get('agent_results', {})) > 0 and
                    execution_time > 0.5  # Real collaboration takes time
                )
                
                if is_real_collaboration:
                    # Real agent system worked
                    participating_agents = set(response.get('agent_results', {}).keys())
                    expected_agents = set(scenario['expected_agents'])
                    agent_coverage = len(participating_agents & expected_agents) / len(expected_agents)
                    result_confidence = response.get('confidence_score', 0.0)
                    test_success = True
                    test_type = "real_collaboration"
                else:
                    # This is fallback/fake - mark as failed
                    agent_coverage = 0.0
                    result_confidence = 0.0
                    test_success = False
                    test_type = "fallback_detected"
                
                results['tests'].append({
                    'query': scenario['query'],
                    'success': test_success,  # HONEST: Only true if real collaboration
                    'execution_time': execution_time,
                    'meets_sla': execution_time < 30.0,
                    'agent_coverage': agent_coverage,
                    'result_confidence': result_confidence,
                    'test_type': test_type,
                    'fallback_mode': response.get('fallback_mode', False),
                    'has_agent_error': bool(response.get('agent_error')),
                    'agent_results_count': len(response.get('agent_results', {}))
                })
                
            except Exception as e:
                results['tests'].append({
                    'query': scenario['query'],
                    'success': False,
                    'error': str(e),
                    'test_type': "exception"
                })
        
        self.test_results['test_suites']['agent_system'] = results

    async def _test_cross_layer_integration(self):
        """Test integration between layers"""
        # Implementation of cross-layer integration tests
        ...

    async def _test_error_handling(self):
        """Test error handling and recovery"""
        # Implementation of error handling tests
        ...

    async def _test_performance_metrics(self):
        """Test performance and resource usage"""
        # Implementation of performance tests
        ...

    def _calculate_final_status(self):
        """Calculate final system status - HONEST VERSION"""
        
        all_tests = []
        for suite in self.test_results['test_suites'].values():
            all_tests.extend(suite['tests'])
        
        successful_tests = len([t for t in all_tests if t.get('success', False)])
        total_tests = len(all_tests)
        
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # HONEST: No partial credit for fallbacks
        critical_issues = [
            test for test in all_tests 
            if not test.get('success') or test.get('test_type') == 'fallback_detected'
        ]
        
        # HONEST criteria
        if success_rate >= 95 and not critical_issues:
            status = "PRODUCTION_READY"
        elif success_rate >= 70:
            status = "READY_WITH_WARNINGS"
        else:
            status = "NOT_PRODUCTION_READY"
            
        self.test_results.update({
            'overall_status': status,
            'success_rate': success_rate,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'critical_issues': len(critical_issues),
            'completion_time': datetime.now().isoformat()
        })
        
        
    def _create_test_routing_decision(self) -> RoutingDecision:
        """Create test routing decision with default test values"""
        
        return RoutingDecision(
            path=ProcessingPath.AGENTIC_FALLBACK,  # Changed from LIGHTNING_RAG
            confidence=0.95,
            reasoning=["Test routing decision"],
            performance_budget=300,  # Increased timeout
            fallback_path=ProcessingPath.SEMANTIC_BRIDGE,
            enrichments_needed=[],
            unknown_terms=[],
            complexity_factors={'base_complexity': 'intermediate'},
            estimated_cost='medium'
        )

async def main():
    """Run the complete integration test suite"""
    
    logger.info("Starting Complete System Integration Test")
    logger.info("=" * 60)
    
    test_suite = CompleteSystemTest()
    await test_suite.initialize_system()
    
    try:
        results = await test_suite.run_complete_test_suite()
        
        logger.info("\nTest Suite Results:")
        logger.info(f"Overall Status: {results['overall_status']}")
        logger.info(f"Success Rate: {results['success_rate']:.1f}%")
        logger.info(f"Total Tests: {results['total_tests']}")
        logger.info(f"Critical Issues: {results['critical_issues']}")
        
        return 0 if results['overall_status'] == "PRODUCTION_READY" else 1
        
    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))