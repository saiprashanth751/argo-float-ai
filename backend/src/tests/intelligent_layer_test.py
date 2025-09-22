#backend\src\tests\intelligent_layer_test.py
"""
Complete Integration Test Suite - Production Readiness Validation
Enhanced to test Phase 1, 2, and 3 improvements
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
import json

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Core system imports
try:
    from services.orchestrated_rag_system import OrchestratedOceanographicRAG
    from services.smart_query_router import ProcessingPath, SmartQueryRouter
    from services.agent_collaboration_system import ProductionAgentCollaborationSystem, CollaborationPattern
    from services.oceanographic_intelligence_engine import (
        QueryClassification, 
        QueryIntent,
        ComplexityLevel,
        OceanographicIntelligenceEngine
    )
    from services.types_core import RoutingDecision, ProcessingPath, ComplexityLevel, QueryIntent
    from utils.database_manager import get_db_engine
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available")
    sys.exit(1)

logger = logging.getLogger(__name__)

class EnhancedSystemTest:
    """
    Production-ready test suite that validates all Phase 1, 2, 3 enhancements
    """
    
    def __init__(self):
        self.test_results = {
            'overall_status': 'pending',
            'phase_results': {},
            'integration_results': {},
            'performance_metrics': {},
            'system_health': {},
            'start_time': datetime.now().isoformat()
        }
        
        # Initialize systems
        self.db_engine = None
        self.intelligence_engine = None
        self.router = None
        self.orchestrated_system = None
        self.agent_system = None

    async def initialize_systems(self):
        """Initialize all system components for testing"""
        try:
            print("Initializing database engine...")
            self.db_engine = get_db_engine()
            
            print("Initializing intelligence engine...")
            self.intelligence_engine = OceanographicIntelligenceEngine(self.db_engine)
            
            print("Initializing smart router...")
            self.router = SmartQueryRouter(self.intelligence_engine)
            
            print("Initializing orchestrated system...")
            self.orchestrated_system = OrchestratedOceanographicRAG(db_engine=self.db_engine)
            
            # Try to initialize agent system
            print("Attempting to initialize agent system...")
            try:
                self.agent_system = ProductionAgentCollaborationSystem(self.db_engine, max_agents=4)
                logger.info("Agent collaboration system initialized successfully")
            except Exception as e:
                logger.warning(f"Agent system initialization failed: {e}")
                self.agent_system = None
            
            logger.info("All systems initialized for testing")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise

    async def run_complete_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive test suite covering all phases"""
        
        logger.info("Starting Enhanced Production Test Suite")
        logger.info("=" * 60)
        
        # Phase 1: Test Smart Routing Improvements
        await self._test_phase1_smart_routing()
        
        # Phase 2: Test Orchestrated System Fixes
        await self._test_phase2_orchestration_fixes()
        
        # Phase 3: Test Enhanced Agent System
        await self._test_phase3_agent_enhancements()
        
        # Integration Tests: End-to-End Scenarios
        await self._test_cross_system_integration()
        
        # Performance and Production Readiness
        await self._test_production_readiness()
        
        self._calculate_final_assessment()
        return self.test_results

    async def _test_phase1_smart_routing(self):
        """Test Phase 1: Smart routing calibration and improvements"""
        
        logger.info("Phase 1: Testing Smart Routing Calibration")
        results = {'status': 'running', 'routing_tests': [], 'calibration_analysis': {}}
        
        # Test routing decisions for different query types
        test_cases = [
            {
                'query': "Show temperature profiles in Arabian Sea",
                'expected_path': ProcessingPath.LIGHTNING_RAG,
                'description': "Simple query - should go to Lightning RAG"
            },
            {
                'query': "What is thermocline depth variability during monsoon?",
                'expected_path': ProcessingPath.LIGHTNING_RAG,  # Should now route to RAG (Phase 1 fix)
                'description': "Analytical query - should go to Lightning RAG (improved routing)"
            },
            {
                'query': "Average salinity measurements in Bay of Bengal",
                'expected_path': ProcessingPath.LIGHTNING_RAG,
                'description': "Statistical query - should go to Lightning RAG"
            },
            {
                'query': "What is biogeochemical flux patterns?",
                'expected_path': ProcessingPath.SEMANTIC_BRIDGE,  # Unknown term should trigger semantic
                'description': "Query with unknown terms - should go to Semantic Bridge"
            },
            {
                'query': "Calculate complex biogeochemical carbon cycle interactions with ecosystem modeling",
                'expected_path': ProcessingPath.AGENTIC_FALLBACK,
                'description': "Complex query requiring external knowledge - should go to Agents"
            }
        ]
        
        routing_distribution = {
            ProcessingPath.LIGHTNING_RAG: 0,
            ProcessingPath.SEMANTIC_BRIDGE: 0,
            ProcessingPath.AGENTIC_FALLBACK: 0
        }
        
        for case in test_cases:
            try:
                # Step 1: Route the query
                logger.debug(f"Testing query: {case['query']}")
                decision = self.router.route_query(case['query'])
                logger.debug(f"Routing completed successfully")
                
                # Step 2: Validate the decision object
                if not hasattr(decision, 'path') or not hasattr(decision, 'confidence'):
                    raise ValueError(f"Invalid routing decision object: {type(decision)}")
                
                if not isinstance(decision.path, ProcessingPath):
                    raise ValueError(f"decision.path is not ProcessingPath enum: {type(decision.path)} = {decision.path}")
                
                logger.debug(f"Decision validation passed")
                
                # Step 3: Track routing distribution
                routing_distribution[decision.path] += 1
                logger.debug(f"Distribution tracking completed")
                
                # Step 4: Evaluate routing accuracy
                correct_routing = decision.path == case['expected_path']
                logger.debug(f"Routing accuracy evaluation completed: {correct_routing}")
                
                # Step 5: Create test result
                test_result = {
                    'query': case['query'][:50] + '...' if len(case['query']) > 50 else case['query'],
                    'expected_path': case['expected_path'].value, 
                    'actual_path': decision.path.value,
                    'correct_routing': correct_routing,
                    'confidence': decision.confidence,
                    'reasoning': decision.reasoning[:2] if decision.reasoning else [],
                    'unknown_terms': decision.unknown_terms,
                    'description': case['description']
                }
                logger.debug(f"Test result creation completed")
                
                # Step 6: Append to results
                results['routing_tests'].append(test_result)
                logger.debug(f"Test result appended to results")

                logger.info(f"  Query: {case['query'][:40]}...")
                logger.info(f"  -> Expected: {case['expected_path'].value}")
                logger.info(f"  -> Actual: {decision.path.value} (confidence: {decision.confidence:.2f})")
                logger.info(f"  -> Correct: {correct_routing}")

            except Exception as e:
                logger.error(f"Routing test failed for query '{case['query'][:40]}...'")
                logger.error(f"Exception type: {type(e)}")
                logger.error(f"Exception message: {str(e)}")
                logger.error(f"Exception repr: {repr(e)}")
                
                # Add stack trace for debugging
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                
                results['routing_tests'].append({
                    'query': case['query'],
                    'error': str(e),
                    'error_type': str(type(e)),
                    'correct_routing': False
                })
        
        # Analyze routing distribution (Phase 1 goal: ~65-70% to Lightning RAG)
        total_queries = len(test_cases)
        rag_percentage = (routing_distribution[ProcessingPath.LIGHTNING_RAG] / total_queries) * 100
        semantic_percentage = (routing_distribution[ProcessingPath.SEMANTIC_BRIDGE] / total_queries) * 100
        agent_percentage = (routing_distribution[ProcessingPath.AGENTIC_FALLBACK] / total_queries) * 100
        
        results['calibration_analysis'] = {
            'rag_percentage': rag_percentage,
            'semantic_percentage': semantic_percentage,
            'agent_percentage': agent_percentage,
            'meets_target_distribution': 60 <= rag_percentage <= 75,  # Target range
            'routing_accuracy': len([t for t in results['routing_tests'] if t.get('correct_routing', False)]) / total_queries * 100
        }
        
        logger.info(f"  Routing Distribution: RAG={rag_percentage:.1f}%, Semantic={semantic_percentage:.1f}%, Agents={agent_percentage:.1f}%")
        logger.info(f"  Target Met: {results['calibration_analysis']['meets_target_distribution']}")
        
        self.test_results['phase_results']['phase1_routing'] = results

    async def _test_phase2_orchestration_fixes(self):
        """Test Phase 2: Orchestrated system fixes and working agent connection"""
        
        logger.info("Phase 2: Testing Orchestrated System Fixes")
        results = {'status': 'running', 'orchestration_tests': [], 'agent_connection_test': {}}
        
        # Test basic orchestration
        basic_queries = [
            "Show temperature at 100m depth in Arabian Sea",
            "Average salinity measurements in Indian Ocean",
            "What is mixed layer depth variation?"
        ]
        
        for query in basic_queries:
            try:
                start_time = time.time()
                response = self.orchestrated_system.process_query(query)
                processing_time = time.time() - start_time
                
                test_result = {
                    'query': query,
                    'success': response.get('success', False),
                    'processing_time': processing_time,
                    'routing_path': response.get('orchestration', {}).get('routing_path', 'unknown'),
                    'has_orchestration_metadata': 'orchestration' in response,
                    'intelligence_enhanced': response.get('orchestration', {}).get('intelligence_enhanced', False)
                }
                
                results['orchestration_tests'].append(test_result)
                
                logger.info(f"  Query: {query[:40]}...")
                logger.info(f"  -> Success: {test_result['success']}")
                logger.info(f"  -> Path: {test_result['routing_path']}")
                logger.info(f"  -> Time: {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Orchestration test failed for query '{query[:40]}...': {e}")
                results['orchestration_tests'].append({
                    'query': query,
                    'success': False,
                    'error': str(e)
                })
        
        # Test agent system connection (Phase 2 fix)
        try:
            logger.info("  Testing agent system connection...")
            
            # Test if agent system is properly connected (not broken coordinator)
            agent_available = self.orchestrated_system.agent_system is not None
            
            if agent_available:
                # Test actual agent execution with a complex query
                complex_query = "Analyze biogeochemical processes with validation and quality assessment"
                
                start_time = time.time()
                agent_response = self.orchestrated_system.process_query(complex_query)
                execution_time = time.time() - start_time
                
                # Check if this is real agent collaboration or fallback
                routing_path = agent_response.get('orchestration', {}).get('routing_path', '')
                is_agent_path = 'agent' in routing_path.lower()
                has_agent_results = len(agent_response.get('agent_results', {})) > 0
                not_fallback_only = not agent_response.get('fallback_mode', False)
                
                agent_connection_working = is_agent_path and (has_agent_results or not_fallback_only)
                
                results['agent_connection_test'] = {
                    'agent_system_available': True,
                    'agent_connection_working': agent_connection_working,
                    'execution_time': execution_time,
                    'routing_path': routing_path,
                    'has_agent_results': has_agent_results,
                    'fallback_mode': agent_response.get('fallback_mode', False),
                    'response_success': agent_response.get('success', False)
                }
                
                logger.info(f"  -> Agent system connected: {agent_connection_working}")
                logger.info(f"  -> Execution time: {execution_time:.2f}s")
                logger.info(f"  -> Has agent results: {has_agent_results}")
                
            else:
                results['agent_connection_test'] = {
                    'agent_system_available': False,
                    'agent_connection_working': False,
                    'reason': 'Agent system not initialized'
                }
                
                logger.info("  -> Agent system not available")
                
        except Exception as e:
            logger.error(f"Agent connection test failed: {e}")
            results['agent_connection_test'] = {
                'agent_system_available': False,
                'agent_connection_working': False,
                'error': str(e)
            }
        
        self.test_results['phase_results']['phase2_orchestration'] = results

    async def _test_phase3_agent_enhancements(self):
        """Test Phase 3: Enhanced agent system with new collaboration patterns"""
        
        logger.info("Phase 3: Testing Enhanced Agent System")
        results = {'status': 'running', 'pattern_tests': [], 'production_use_cases': []}
        
        if not self.agent_system:
            logger.warning("  Agent system not available - skipping Phase 3 tests")
            results['status'] = 'skipped'
            results['reason'] = 'Agent system not initialized'
            self.test_results['phase_results']['phase3_agents'] = results
            return
        
        # Test new collaboration patterns
        pattern_test_cases = [
            {
                'pattern': CollaborationPattern.EXTERNAL_KNOWLEDGE_SYNTHESIS,
                'query': "Calculate primary productivity using biogeochemical formulas in Arabian Sea",
                'description': "External knowledge synthesis pattern",
                'expected_agents': ['domain_researcher', 'integration_coordinator', 'result_validator']
            },
            {
                'pattern': CollaborationPattern.DATA_GAP_INTELLIGENT_RESPONSE,
                'query': "Show dissolved oxygen at 3000m depth from 1990-1995",
                'description': "Data gap intelligent response pattern",
                'expected_agents': ['schema_explorer', 'integration_coordinator', 'quality_assessor']
            },
            {
                'pattern': CollaborationPattern.INTELLIGENT_APPROXIMATION,
                'query': "Estimate heat transport approximately at 15N using available temperature data",
                'description': "Intelligent approximation pattern", 
                'expected_agents': ['domain_researcher', 'integration_coordinator', 'quality_assessor', 'result_validator']
            }
        ]
        
        for case in pattern_test_cases:
            try:
                logger.info(f"  Testing {case['pattern'].value}...")
                
                # Create routing decision for agent system
                routing_decision = RoutingDecision(
                    path=ProcessingPath.AGENTIC_FALLBACK,
                    confidence=0.7,
                    reasoning=[f"Testing {case['pattern'].value}"],
                    performance_budget=240,
                    fallback_path=ProcessingPath.ERROR_RECOVERY,
                    enrichments_needed=[],
                    unknown_terms=[],
                    complexity_factors={'base_complexity': 'advanced'},
                    estimated_cost='high'
                )
                
                start_time = time.time()
                response = await self.agent_system.execute_agent_collaboration(
                    query=case['query'],
                    routing_decision=routing_decision,
                    user_context={}
                )
                execution_time = time.time() - start_time
                
                # Analyze response quality
                is_successful = response.get('success', False)
                has_pattern_info = 'collaboration_pattern' in response
                pattern_matches = (response.get('collaboration_pattern') == case['pattern'].value 
                                 if has_pattern_info else False)
                
                # Check for real agent execution vs fallback
                agent_results = response.get('agent_results', {})
                participating_agents = set(agent_results.keys()) if agent_results else set()
                expected_agents = set(case['expected_agents'])
                agent_coverage = len(participating_agents & expected_agents) / len(expected_agents) if expected_agents else 0
                
                is_real_collaboration = (
                    is_successful and 
                    not response.get('fallback_mode', False) and
                    len(participating_agents) > 0 and
                    execution_time > 1.0  # Real collaboration takes time
                )
                
                test_result = {
                    'pattern': case['pattern'].value,
                    'query': case['query'][:60] + '...' if len(case['query']) > 60 else case['query'],
                    'success': is_successful,
                    'execution_time': execution_time,
                    'pattern_correctly_selected': pattern_matches,
                    'agent_coverage': agent_coverage,
                    'participating_agents': list(participating_agents),
                    'expected_agents': case['expected_agents'],
                    'is_real_collaboration': is_real_collaboration,
                    'fallback_mode': response.get('fallback_mode', False),
                    'has_integration_result': 'final_result' in response
                }
                
                results['pattern_tests'].append(test_result)
                
                logger.info(f"    -> Success: {is_successful}")
                logger.info(f"    -> Real collaboration: {is_real_collaboration}")
                logger.info(f"    -> Agent coverage: {agent_coverage:.1%}")
                logger.info(f"    -> Time: {execution_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Pattern test failed for {case['pattern'].value}: {e}")
                results['pattern_tests'].append({
                    'pattern': case['pattern'].value,
                    'query': case['query'],
                    'success': False,
                    'error': str(e)
                })
        
        # Test production use case scenarios
        production_scenarios = [
            {
                'scenario': 'complex_calculation',
                'query': 'Calculate net primary productivity with carbon flux analysis',
                'expected_outcome': 'external_knowledge_integration'
            },
            {
                'scenario': 'missing_data_handling',
                'query': 'Show trace metal concentrations in deep Arabian Sea waters',
                'expected_outcome': 'transparent_data_gap_communication'
            }
        ]
        
        for scenario in production_scenarios:
            try:
                # Use orchestrated system to test full integration
                response = self.orchestrated_system.process_query(scenario['query'])
                
                scenario_result = {
                    'scenario': scenario['scenario'],
                    'success': response.get('success', False),
                    'used_agents': 'agent' in response.get('orchestration', {}).get('routing_path', ''),
                    'has_intelligence_enhancement': response.get('orchestration', {}).get('intelligence_enhanced', False),
                    'transparent_communication': 'limitation' in str(response).lower() or 'uncertainty' in str(response).lower()
                }
                
                results['production_use_cases'].append(scenario_result)
                
            except Exception as e:
                logger.error(f"Production scenario test failed for {scenario['scenario']}: {e}")
                results['production_use_cases'].append({
                    'scenario': scenario['scenario'],
                    'success': False,
                    'error': str(e)
                })
        
        self.test_results['phase_results']['phase3_agents'] = results

    async def _test_cross_system_integration(self):
        """Test integration between all system components"""
        
        logger.info("Integration: Testing Cross-System Integration")
        results = {'status': 'running', 'integration_scenarios': []}
        
        # End-to-end integration scenarios
        integration_scenarios = [
            {
                'name': 'simple_to_complex_escalation',
                'queries': [
                    "Show temperature data",  # Should go to Lightning RAG
                    "What is temperature variability with seasonal patterns?",  # Should go to Semantic Bridge
                    "Analyze temperature impact on biogeochemical processes with validation"  # Should go to Agents
                ],
                'expected_escalation': True
            },
            {
                'name': 'consistent_high_performance',
                'queries': [
                    "Count profiles in Arabian Sea",
                    "Average salinity at 500m depth", 
                    "Maximum temperature in Bay of Bengal"
                ],
                'expected_path': ProcessingPath.LIGHTNING_RAG,
                'target_time': 2.0  # All should be fast
            }
        ]
        
        for scenario in integration_scenarios:
            try:
                scenario_result = {
                    'name': scenario['name'],
                    'queries_tested': len(scenario['queries']),
                    'query_results': []
                }
                
                for query in scenario['queries']:
                    start_time = time.time()
                    response = self.orchestrated_system.process_query(query)
                    processing_time = time.time() - start_time
                    
                    query_result = {
                        'query': query[:40] + '...' if len(query) > 40 else query,
                        'success': response.get('success', False),
                        'processing_time': processing_time,
                        'routing_path': response.get('orchestration', {}).get('routing_path', 'unknown')
                    }
                    
                    scenario_result['query_results'].append(query_result)
                
                # Analyze scenario success
                all_successful = all(qr['success'] for qr in scenario_result['query_results'])
                
                if scenario['name'] == 'simple_to_complex_escalation':
                    paths_used = [qr['routing_path'] for qr in scenario_result['query_results']]
                    shows_escalation = len(set(paths_used)) > 1  # Multiple paths used
                    scenario_result['shows_escalation'] = shows_escalation
                    scenario_result['success'] = all_successful and shows_escalation
                    
                elif scenario['name'] == 'consistent_high_performance':
                    avg_time = sum(qr['processing_time'] for qr in scenario_result['query_results']) / len(scenario_result['query_results'])
                    meets_performance = avg_time <= scenario['target_time']
                    scenario_result['average_time'] = avg_time
                    scenario_result['meets_performance'] = meets_performance
                    scenario_result['success'] = all_successful and meets_performance
                
                results['integration_scenarios'].append(scenario_result)
                
                logger.info(f"  Scenario: {scenario['name']}")
                logger.info(f"  -> Success: {scenario_result['success']}")
                
            except Exception as e:
                logger.error(f"Integration scenario failed for {scenario['name']}: {e}")
                results['integration_scenarios'].append({
                    'name': scenario['name'],
                    'success': False,
                    'error': str(e)
                })
        
        self.test_results['integration_results'] = results

    async def _test_production_readiness(self):
        """Test production readiness and system health"""
        
        logger.info("Production: Testing System Readiness")
        results = {'system_health': {}, 'performance_metrics': {}, 'readiness_score': 0}
        
        # System health checks
        try:
            # Check orchestrated system health
            orchestrated_performance = self.orchestrated_system.get_system_performance()
            
            results['system_health'] = {
                'orchestrated_system_ready': True,
                'query_distribution': orchestrated_performance.get('query_distribution', {}),
                'success_rates': orchestrated_performance.get('success_rates', {}),
                'system_health_components': orchestrated_performance.get('system_health', {})
            }
            
            # Check agent system health if available
            if self.agent_system:
                try:
                    agent_status = self.agent_system.get_production_system_status()
                    results['system_health']['agent_system_status'] = agent_status['status']
                    results['system_health']['agent_health_score'] = agent_status['overall_health_score']
                except Exception as e:
                    results['system_health']['agent_system_error'] = str(e)
            
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            results['system_health']['error'] = str(e)
        
        # Performance stress test
        try:
            stress_queries = [
                "Show temperature profiles",
                "Average salinity measurements", 
                "Count argo profiles",
                "Maximum depth measurements",
                "Minimum temperature values"
            ]
            
            start_time = time.time()
            stress_results = []
            
            for query in stress_queries:
                query_start = time.time()
                response = self.orchestrated_system.process_query(query)
                query_time = time.time() - query_start
                
                stress_results.append({
                    'success': response.get('success', False),
                    'time': query_time
                })
            
            total_time = time.time() - start_time
            successful_queries = [r for r in stress_results if r['success']]
            
            results['performance_metrics'] = {
                'total_queries': len(stress_queries),
                'successful_queries': len(successful_queries),
                'success_rate': len(successful_queries) / len(stress_queries) * 100,
                'total_time': total_time,
                'average_query_time': sum(r['time'] for r in successful_queries) / len(successful_queries) if successful_queries else 0,
                'queries_per_second': len(stress_queries) / total_time
            }
            
        except Exception as e:
            logger.error(f"Performance stress test failed: {e}")
            results['performance_metrics']['error'] = str(e)
        
        # Calculate readiness score
        readiness_factors = []
        
        # System health factor
        if results['system_health'].get('orchestrated_system_ready', False):
            readiness_factors.append(1.0)
        else:
            readiness_factors.append(0.0)
        
        # Performance factor
        if results['performance_metrics'].get('success_rate', 0) >= 80:
            readiness_factors.append(1.0)
        elif results['performance_metrics'].get('success_rate', 0) >= 60:
            readiness_factors.append(0.7)
        else:
            readiness_factors.append(0.0)
        
        # Agent system factor
        if results['system_health'].get('agent_system_status') == 'healthy':
            readiness_factors.append(1.0)
        elif self.agent_system is not None:
            readiness_factors.append(0.5)
        else:
            readiness_factors.append(0.0)
        
        results['readiness_score'] = sum(readiness_factors) / len(readiness_factors)
        
        self.test_results['system_health'] = results

    def _calculate_final_assessment(self):
        """Calculate comprehensive final assessment"""
        
        phase_scores = []
        critical_issues = []
        
        # Phase 1 Assessment
        phase1 = self.test_results['phase_results'].get('phase1_routing', {})
        if phase1.get('calibration_analysis', {}).get('meets_target_distribution', False):
            phase1_score = phase1['calibration_analysis']['routing_accuracy'] / 100
        else:
            phase1_score = 0.0
            critical_issues.append("Phase 1: Routing calibration failed")
        phase_scores.append(phase1_score)
        
        # Phase 2 Assessment  
        phase2 = self.test_results['phase_results'].get('phase2_orchestration', {})
        orchestration_success_rate = len([t for t in phase2.get('orchestration_tests', []) if t.get('success', False)]) / max(len(phase2.get('orchestration_tests', [])), 1)
        agent_connection_working = phase2.get('agent_connection_test', {}).get('agent_connection_working', False)
        
        phase2_score = (orchestration_success_rate + (0.5 if agent_connection_working else 0.0)) / 1.5
        if not agent_connection_working:
            critical_issues.append("Phase 2: Agent system connection issues")
        phase_scores.append(phase2_score)
        
        # Phase 3 Assessment
        phase3 = self.test_results['phase_results'].get('phase3_agents', {})
        if phase3.get('status') == 'skipped':
            phase3_score = 0.5  # Partial credit for system without agents
        else:
            pattern_success_rate = len([t for t in phase3.get('pattern_tests', []) if t.get('is_real_collaboration', False)]) / max(len(phase3.get('pattern_tests', [])), 1)
            phase3_score = pattern_success_rate
            if pattern_success_rate < 0.5:
                critical_issues.append("Phase 3: Agent collaboration patterns failing")
        phase_scores.append(phase3_score)
        
        # Overall assessment
        overall_score = sum(phase_scores) / len(phase_scores)
        readiness_score = self.test_results['system_health'].get('readiness_score', 0)
        
        final_score = (overall_score * 0.7 + readiness_score * 0.3)
        
        # Determine final status
        if final_score >= 0.9 and len(critical_issues) == 0:
            status = "PRODUCTION_READY"
        elif final_score >= 0.7 and len(critical_issues) <= 1:
            status = "READY_WITH_MONITORING"
        elif final_score >= 0.5:
            status = "NEEDS_IMPROVEMENTS"
        else:
            status = "NOT_PRODUCTION_READY"
        
        self.test_results.update({
            'overall_status': status,
            'final_score': final_score,
            'phase_scores': {
                'phase1_routing': phase1_score,
                'phase2_orchestration': phase2_score, 
                'phase3_agents': phase3_score
            },
            'critical_issues': critical_issues,
            'completion_time': datetime.now().isoformat(),
            'recommendations': self._generate_recommendations(status, critical_issues, final_score)
        })

    def _generate_recommendations(self, status: str, critical_issues: List[str], final_score: float) -> List[str]:
        """Generate deployment recommendations based on test results"""
        
        recommendations = []
        
        if status == "PRODUCTION_READY":
            recommendations.extend([
                "System is ready for production deployment",
                "Set up monitoring and alerting",
                "Configure production logging",
                "Implement backup and recovery procedures"
            ])
        elif status == "READY_WITH_MONITORING":
            recommendations.extend([
                "System can be deployed with close monitoring",
                "Address critical issues as high priority",
                "Implement comprehensive monitoring", 
                "Have rollback procedures ready"
            ])
        else:
            recommendations.extend([
                "System requires improvements before production",
                "Address all critical issues",
                "Re-run tests after fixes",
                "Consider phased deployment approach"
            ])
        
        # Add specific recommendations based on issues
        for issue in critical_issues:
            if "Phase 1" in issue:
                recommendations.append("- Review and adjust smart routing thresholds")
            elif "Phase 2" in issue:
                recommendations.append("- Fix orchestrated system and agent connections")
            elif "Phase 3" in issue:
                recommendations.append("- Debug agent collaboration system initialization")
        
        return recommendations


async def run_enhanced_tests():
    """Run the complete enhanced integration test suite"""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🚀 Starting Enhanced Production Test Suite")
    print("=" * 60)
    
    test_suite = EnhancedSystemTest()
    
    try:
        # Initialize all systems
        print("📋 Initializing test systems...")
        await test_suite.initialize_systems()
        
        # Run comprehensive test suite
        results = await test_suite.run_complete_test_suite()
        
        # Display comprehensive results
        print("\n" + "=" * 60)
        print("🎯 ENHANCED PRODUCTION TEST RESULTS")
        print("=" * 60)
        
        # Overall Status
        status = results['overall_status']
        final_score = results.get('final_score', 0)
        
        status_emoji = {
            'PRODUCTION_READY': '✅',
            'READY_WITH_MONITORING': '⚠️',
            'NEEDS_IMPROVEMENTS': '🔧',
            'NOT_PRODUCTION_READY': '❌'
        }.get(status, '❓')
        
        print(f"{status_emoji} Overall Status: {status}")
        print(f"📊 Final Score: {final_score:.2f}/1.0")
        
        # Phase Results
        print("\n📈 Phase Results:")
        phase_scores = results.get('phase_scores', {})
        for phase, score in phase_scores.items():
            score_emoji = "✅" if score > 0.8 else "⚠️" if score > 0.6 else "❌"
            print(f"  {score_emoji} {phase}: {score:.2f}")
        
        # Critical Issues
        critical_issues = results.get('critical_issues', [])
        if critical_issues:
            print("\n🚨 Critical Issues:")
            for issue in critical_issues:
                print(f"  ❌ {issue}")
        else:
            print("\n✅ No critical issues detected")
        
        # Performance Metrics
        system_health = results.get('system_health', {})
        perf_metrics = system_health.get('performance_metrics', {})
        
        if perf_metrics.get('success_rate'):
            print(f"\n⚡ Performance Summary:")
            print(f"  Success Rate: {perf_metrics['success_rate']:.1f}%")
            print(f"  Average Query Time: {perf_metrics.get('average_query_time', 0):.2f}s")
            print(f"  Queries Per Second: {perf_metrics.get('queries_per_second', 0):.1f}")
        
        # Component Health
        comp_health = system_health.get('system_health', {})
        if comp_health:
            print(f"\n🏥 System Health:")
            for component, ready in comp_health.items():
                health_emoji = "✅" if ready else "❌"
                print(f"  {health_emoji} {component}: {'Ready' if ready else 'Not Ready'}")
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations[:5]:  # Show top 5
                print(f"  • {rec}")
        
        # Detailed Phase Analysis
        print(f"\n📊 DETAILED PHASE ANALYSIS")
        print("=" * 40)
        
        # Phase 1 Details
        phase1 = results['phase_results'].get('phase1_routing', {})
        if phase1.get('calibration_analysis'):
            cal = phase1['calibration_analysis']
            print(f"Phase 1 - Smart Routing:")
            print(f"  Distribution: RAG={cal['rag_percentage']:.1f}%, Semantic={cal['semantic_percentage']:.1f}%, Agent={cal['agent_percentage']:.1f}%")
            print(f"  Target Met: {'✅' if cal['meets_target_distribution'] else '❌'}")
            print(f"  Routing Accuracy: {cal['routing_accuracy']:.1f}%")
        
        # Phase 2 Details
        phase2 = results['phase_results'].get('phase2_orchestration', {})
        if phase2.get('agent_connection_test'):
            agent_test = phase2['agent_connection_test']
            print(f"Phase 2 - Orchestration:")
            print(f"  Agent System Available: {'✅' if agent_test['agent_system_available'] else '❌'}")
            print(f"  Agent Connection Working: {'✅' if agent_test.get('agent_connection_working', False) else '❌'}")
            if 'execution_time' in agent_test:
                print(f"  Agent Execution Time: {agent_test['execution_time']:.2f}s")
        
        # Phase 3 Details
        phase3 = results['phase_results'].get('phase3_agents', {})
        if phase3.get('pattern_tests'):
            successful_patterns = len([t for t in phase3['pattern_tests'] if t.get('is_real_collaboration', False)])
            total_patterns = len(phase3['pattern_tests'])
            print(f"Phase 3 - Agent Enhancements:")
            print(f"  Pattern Tests: {successful_patterns}/{total_patterns} successful")
            print(f"  Real Collaboration Rate: {successful_patterns/total_patterns*100:.1f}%" if total_patterns > 0 else "  No pattern tests")
        elif phase3.get('status') == 'skipped':
            print(f"Phase 3 - Agent Enhancements: ⏭️ Skipped ({phase3.get('reason', 'Unknown reason')})")
        
        # Integration Test Details
        integration = results.get('integration_results', {})
        if integration.get('integration_scenarios'):
            successful_scenarios = len([s for s in integration['integration_scenarios'] if s.get('success', False)])
            total_scenarios = len(integration['integration_scenarios'])
            print(f"Integration Tests:")
            print(f"  Scenarios: {successful_scenarios}/{total_scenarios} successful")
        
        # Final Verdict
        print(f"\n🎯 FINAL VERDICT")
        print("=" * 20)
        
        verdict_messages = {
            'PRODUCTION_READY': "🎉 SYSTEM IS PRODUCTION READY! Deploy with confidence.",
            'READY_WITH_MONITORING': "⚠️ SYSTEM CAN BE DEPLOYED with close monitoring required.",
            'NEEDS_IMPROVEMENTS': "🔧 SYSTEM NEEDS IMPROVEMENTS before production deployment.",
            'NOT_PRODUCTION_READY': "❌ SYSTEM NOT READY for production - address critical issues first."
        }
        
        print(verdict_messages.get(status, "❓ Status unclear - review results manually."))
        
        # Save detailed results to file
        results_file = Path("test_results_enhanced.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Test Suite Failed: {e}")
        import traceback
        traceback.print_exc()
        return {'overall_status': 'FAILED', 'error': str(e)}


def main():
    """Main entry point for the enhanced test suite"""
    print("Enhanced Production Test Suite")
    print("Testing Phases 1, 2, and 3 improvements")
    print("=" * 50)
    
    try:
        # Run the async test suite
        results = asyncio.run(run_enhanced_tests())
        
        # Return exit code based on results
        if results['overall_status'] in ['PRODUCTION_READY', 'READY_WITH_MONITORING']:
            return 0  # Success
        else:
            return 1  # Failure
            
    except KeyboardInterrupt:
        print("\n⏹️ Test suite interrupted by user")
        return 2
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 3


if __name__ == "__main__":
    exit(main())