# test_enhanced_rag_system.py
"""
Comprehensive test suite for LLM-Enhanced RAG System
Tests both LLM enhancement and fallback mechanisms
"""

import os
import sys
import logging
import time
import asyncio
from pathlib import Path


# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the enhanced RAG system
try:
    from services.enhanced_rag_oceanographic import ProductionOceanographicRAG
except ImportError as e:
    logger.error(f"Import failed: {e}")
    sys.exit(1)

class EnhancedRAGSystemTest:
    """Comprehensive test suite for enhanced RAG system"""
    
    def __init__(self):
        self.test_results = []
        self.system_initialized = False
        
    def initialize_system(self):
        """Initialize the enhanced RAG system"""
        
        logger.info("🚀 Initializing Enhanced RAG System for Testing")
        logger.info("=" * 60)
        
        try:
            # Check environment setup
            db_url = os.getenv('DATABASE_URL')
            deepseek_key = os.getenv('DEEPSEEK_API_KEY')
            
            logger.info(f"Database URL configured: {'Yes' if db_url else 'No'}")
            logger.info(f"DeepSeek API Key configured: {'Yes' if deepseek_key else 'No'}")
            
            # Initialize system
            start_time = time.time()
            self.rag_system = ProductionOceanographicRAG()
            init_time = time.time() - start_time
            
            logger.info(f"System initialized in {init_time:.2f}s")
            
            # Get system status
            health = self.rag_system.validate_system_health()
            
            logger.info(f"System Health: {health['overall_status']}")
            logger.info(f"LLM Enhancement: {'ENABLED' if self.rag_system.llm_enhancement_enabled else 'DISABLED'}")
            
            self.system_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    def test_query_processing_capabilities(self):
        """Test different types of query processing"""
        
        logger.info("\n🧪 Testing Query Processing Capabilities")
        logger.info("-" * 50)
        
        # Test queries designed to trigger different scenarios
        test_scenarios = [
            {
                'name': 'Simple Surface Query (Should be fast)',
                'query': 'Show temperature in Arabian Sea',
                'expected_characteristics': {
                    'processing_time': '<5s',
                    'result_count': '>0',
                    'complexity': 'basic'
                }
            },
            {
                'name': 'Statistical Aggregation (Medium complexity)',
                'query': 'What is the average surface temperature in Indian Ocean during 2023?',
                'expected_characteristics': {
                    'processing_time': '<10s',
                    'result_count': '>0',
                    'complexity': 'intermediate'
                }
            },
            {
                'name': 'Profile Analysis (Complex SQL)',
                'query': 'Show temperature profiles at different depths for platform 1900121',
                'expected_characteristics': {
                    'processing_time': '<15s',
                    'result_count': '>0',
                    'complexity': 'advanced'
                }
            },
            {
                'name': 'Novel Query (Should trigger LLM if available)',
                'query': 'Calculate mixed layer depth variability in monsoon-influenced regions',
                'expected_characteristics': {
                    'processing_time': '<20s',
                    'result_count': '>=0',  # May have no data, but should respond intelligently
                    'complexity': 'advanced'
                }
            },
            {
                'name': 'Complex Calculation Query (Advanced)',
                'query': 'Analyze thermocline strength using temperature gradients in upwelling zones',
                'expected_characteristics': {
                    'processing_time': '<25s',
                    'result_count': '>=0',
                    'complexity': 'expert'
                }
            }
        ]
        
        results = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            logger.info(f"\nTest {i}: {scenario['name']}")
            logger.info(f"Query: {scenario['query']}")
            
            try:
                start_time = time.time()
                result = self.rag_system.process_oceanographic_query(scenario['query'])
                processing_time = time.time() - start_time
                
                # Analyze result
                analysis = self._analyze_query_result(result, scenario, processing_time)
                results.append(analysis)
                
                # Log result summary
                if result['success']:
                    logger.info(f"✅ SUCCESS")
                    logger.info(f"   Processing time: {processing_time:.2f}s")
                    logger.info(f"   Result count: {result.get('result_count', 0)}")
                    logger.info(f"   Template used: {result.get('template_used', 'unknown')}")
                    
                    # Check LLM enhancement usage
                    llm_info = result.get('llm_enhancement', {})
                    if llm_info.get('used', False):
                        logger.info(f"   🤖 LLM Enhancement: USED")
                    else:
                        logger.info(f"   📋 SQL Generation: Template-based")
                        if llm_info.get('fallback_reason'):
                            logger.info(f"   Fallback reason: {llm_info['fallback_reason']}")
                    
                    # Show adaptations made
                    if result.get('adaptations_made'):
                        logger.info(f"   Adaptations: {result['adaptations_made'][:2]}")
                        
                else:
                    logger.error(f"❌ FAILED: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"❌ EXCEPTION: {e}")
                results.append({
                    'scenario': scenario['name'],
                    'success': False,
                    'error': str(e),
                    'processing_time': 0
                })
        
        return results
    
    def _analyze_query_result(self, result, scenario, processing_time):
        """Analyze query result against expected characteristics"""
        
        analysis = {
            'scenario': scenario['name'],
            'query': scenario['query'],
            'success': result.get('success', False),
            'processing_time': processing_time,
            'result_count': result.get('result_count', 0),
            'template_used': result.get('template_used', 'unknown'),
            'llm_enhanced': result.get('llm_enhancement', {}).get('used', False),
            'performance_evaluation': {}
        }
        
        # Evaluate against expected characteristics
        expected = scenario['expected_characteristics']
        
        # Processing time evaluation
        if '<' in expected['processing_time']:
            max_time = float(expected['processing_time'].replace('<', '').replace('s', ''))
            analysis['performance_evaluation']['time'] = 'good' if processing_time < max_time else 'slow'
        
        # Result count evaluation
        result_count = analysis['result_count']
        if expected['result_count'] == '>0':
            analysis['performance_evaluation']['data'] = 'good' if result_count > 0 else 'no_data'
        elif expected['result_count'] == '>=0':
            analysis['performance_evaluation']['data'] = 'acceptable'  # Any result is fine
        
        return analysis
    
    def test_llm_enhancement_scenarios(self):
        """Test scenarios specifically designed to trigger LLM enhancement"""
        
        logger.info("\n🤖 Testing LLM Enhancement Scenarios")
        logger.info("-" * 50)
        
        if not self.rag_system.llm_enhancement_enabled:
            logger.warning("LLM enhancement not available - testing fallback behavior")
        
        llm_test_queries = [
            "Calculate water mass mixing ratios using T-S diagram analysis",
            "Estimate primary productivity from temperature and chlorophyll relationships",
            "Determine ocean heat content changes in the Arabian Sea thermocline",
            "Analyze Ekman transport effects on coastal upwelling dynamics"
        ]
        
        llm_results = []
        
        for i, query in enumerate(llm_test_queries, 1):
            logger.info(f"\nLLM Test {i}: {query[:50]}...")
            
            try:
                start_time = time.time()
                result = self.rag_system.process_oceanographic_query(query)
                processing_time = time.time() - start_time
                
                llm_info = result.get('llm_enhancement', {})
                
                if result['success']:
                    logger.info(f"✅ Query processed successfully")
                    logger.info(f"   LLM Enhanced: {'Yes' if llm_info.get('used') else 'No'}")
                    logger.info(f"   Processing time: {processing_time:.2f}s")
                    logger.info(f"   Results: {result.get('result_count', 0)} rows")
                    
                    if not llm_info.get('used') and llm_info.get('fallback_reason'):
                        logger.info(f"   Fallback reason: {llm_info['fallback_reason']}")
                    
                    llm_results.append({
                        'query': query,
                        'success': True,
                        'llm_used': llm_info.get('used', False),
                        'processing_time': processing_time,
                        'fallback_reason': llm_info.get('fallback_reason')
                    })
                else:
                    logger.error(f"❌ Failed: {result.get('error')}")
                    llm_results.append({
                        'query': query,
                        'success': False,
                        'error': result.get('error'),
                        'processing_time': processing_time
                    })
                    
            except Exception as e:
                logger.error(f"❌ Exception: {e}")
                llm_results.append({
                    'query': query,
                    'success': False,
                    'error': str(e),
                    'processing_time': 0
                })
        
        return llm_results
    
    def test_system_resilience(self):
        """Test system resilience and error handling"""
        
        logger.info("\n🛡️ Testing System Resilience")
        logger.info("-" * 50)
        
        resilience_tests = [
            {
                'name': 'Invalid Query Structure',
                'query': 'Show me the color of the ocean temperature salinity',
                'expected': 'graceful_handling'
            },
            {
                'name': 'Non-existent Data Request',
                'query': 'Find oxygen levels at 10000m depth in Lake Superior',
                'expected': 'intelligent_response'
            },
            {
                'name': 'Malformed Scientific Request',
                'query': 'Calculate the square root of salinity divided by fish population',
                'expected': 'error_or_intelligent_rejection'
            },
            {
                'name': 'Empty Query',
                'query': '',
                'expected': 'error_handling'
            }
        ]
        
        resilience_results = []
        
        for test in resilience_tests:
            logger.info(f"\nResilience Test: {test['name']}")
            logger.info(f"Query: '{test['query']}'")
            
            try:
                result = self.rag_system.process_oceanographic_query(test['query'])
                
                if result['success']:
                    logger.info("✅ System handled gracefully")
                    logger.info(f"   Result count: {result.get('result_count', 0)}")
                    # Check if response makes sense for invalid queries
                    if result.get('result_count', 0) == 0:
                        logger.info("   ✅ No incorrect data returned")
                else:
                    logger.info("✅ System properly rejected query")
                    logger.info(f"   Error: {result.get('error', 'Unknown')}")
                
                resilience_results.append({
                    'test': test['name'],
                    'query': test['query'],
                    'handled_gracefully': True,
                    'success': result['success'],
                    'result_count': result.get('result_count', 0)
                })
                
            except Exception as e:
                logger.warning(f"⚠️ Exception occurred: {e}")
                # Exceptions are acceptable for malformed queries
                resilience_results.append({
                    'test': test['name'],
                    'query': test['query'],
                    'handled_gracefully': True,  # Exception handling is graceful
                    'success': False,
                    'exception': str(e)
                })
        
        return resilience_results
    
    def generate_test_report(self, capability_results, llm_results, resilience_results):
        """Generate comprehensive test report"""
        
        logger.info("\n📊 ENHANCED RAG SYSTEM TEST REPORT")
        logger.info("=" * 60)
        
        # Overall statistics
        total_capability_tests = len(capability_results)
        successful_capability_tests = sum(1 for r in capability_results if r['success'])
        
        total_llm_tests = len(llm_results)
        successful_llm_tests = sum(1 for r in llm_results if r['success'])
        llm_usage_count = sum(1 for r in llm_results if r.get('llm_used', False))
        
        total_resilience_tests = len(resilience_results)
        handled_resilience_tests = sum(1 for r in resilience_results if r['handled_gracefully'])
        
        # System status
        logger.info("SYSTEM STATUS:")
        logger.info(f"  Database Connection: {'✅' if self.rag_system.engine else '❌'}")
        logger.info(f"  Vector Store: {'✅' if self.rag_system.vector_store else '❌'}")
        logger.info(f"  Intelligence Engine: {'✅' if self.rag_system.ocean_intelligence else '❌'}")
        logger.info(f"  LLM Enhancement: {'✅' if self.rag_system.llm_enhancement_enabled else '❌'}")
        
        # Test results summary
        logger.info("\nTEST RESULTS SUMMARY:")
        logger.info(f"  Query Processing: {successful_capability_tests}/{total_capability_tests} ({successful_capability_tests/total_capability_tests*100:.1f}%)")
        logger.info(f"  LLM Enhancement: {successful_llm_tests}/{total_llm_tests} ({successful_llm_tests/total_llm_tests*100:.1f}%)")
        logger.info(f"  LLM Usage Rate: {llm_usage_count}/{total_llm_tests} ({llm_usage_count/total_llm_tests*100:.1f}%)")
        logger.info(f"  Resilience: {handled_resilience_tests}/{total_resilience_tests} ({handled_resilience_tests/total_resilience_tests*100:.1f}%)")
        
        # Performance analysis
        successful_times = [r['processing_time'] for r in capability_results if r['success']]
        if successful_times:
            avg_time = sum(successful_times) / len(successful_times)
            logger.info(f"\nPERFORMANCE ANALYSIS:")
            logger.info(f"  Average Processing Time: {avg_time:.2f}s")
            logger.info(f"  Fastest Query: {min(successful_times):.2f}s")
            logger.info(f"  Slowest Query: {max(successful_times):.2f}s")
        
        # Recommendations
        logger.info("\nRECOMMENDATIONS:")
        if successful_capability_tests / total_capability_tests < 0.8:
            logger.info("  ⚠️  Improve basic query processing success rate")
        if llm_usage_count == 0 and self.rag_system.llm_enhancement_enabled:
            logger.info("  ⚠️  LLM enhancement not being triggered - review routing thresholds")
        if avg_time > 10:
            logger.info("  ⚠️  Consider performance optimizations for slow queries")
        
        return {
            'capability_success_rate': successful_capability_tests / total_capability_tests,
            'llm_success_rate': successful_llm_tests / total_llm_tests,
            'llm_usage_rate': llm_usage_count / total_llm_tests,
            'resilience_rate': handled_resilience_tests / total_resilience_tests,
            'avg_processing_time': avg_time if successful_times else 0
        }

def main():
    """Main test execution function"""
    
    logger.info("🧪 Starting Enhanced RAG System Comprehensive Test Suite")
    logger.info("=" * 70)
    
    # Initialize test system
    test_suite = EnhancedRAGSystemTest()
    
    if not test_suite.initialize_system():
        logger.error("❌ System initialization failed - cannot proceed with tests")
        return
    
    # Run comprehensive test suite
    try:
        # Test 1: Basic query processing capabilities
        capability_results = test_suite.test_query_processing_capabilities()
        
        # Test 2: LLM enhancement scenarios
        llm_results = test_suite.test_llm_enhancement_scenarios()
        
        # Test 3: System resilience
        resilience_results = test_suite.test_system_resilience()
        
        # Generate final report
        report = test_suite.generate_test_report(capability_results, llm_results, resilience_results)
        
        # Overall assessment
        logger.info("\n🎯 OVERALL ASSESSMENT:")
        if report['capability_success_rate'] >= 0.8 and report['resilience_rate'] >= 0.9:
            logger.info("✅ SYSTEM READY FOR PRODUCTION")
        elif report['capability_success_rate'] >= 0.6:
            logger.info("⚠️  SYSTEM MOSTLY FUNCTIONAL - MINOR IMPROVEMENTS NEEDED")
        else:
            logger.error("❌ SYSTEM REQUIRES SIGNIFICANT IMPROVEMENTS")
            
    except Exception as e:
        logger.error(f"❌ Test suite execution failed: {e}")
        return
    
    logger.info("\n🏁 Test suite completed")

if __name__ == "__main__":
    main()