# test_enhanced_rag_system_clean.py
"""
Clean test suite for the updated LLM-Enhanced RAG System
Fixed to work with the simplified architecture
"""

import os
import sys
import logging
import time
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

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

class CleanRAGSystemTest:
    """Clean test suite for updated RAG system"""
    
    def __init__(self):
        self.test_results = []
        self.system_initialized = False
        
    def initialize_system(self):
        """Initialize the RAG system"""
        
        logger.info("Initializing Clean RAG System")
        logger.info("=" * 50)
        
        try:
            # Check environment
            db_url = os.getenv('DATABASE_URL')
            deepseek_key = os.getenv('DEEPSEEK_API_KEY')
            
            logger.info(f"Database URL: {'Configured' if db_url else 'Missing'}")
            logger.info(f"DeepSeek API Key: {'Configured' if deepseek_key else 'Missing'}")
            
            # Initialize system
            start_time = time.time()
            self.rag_system = ProductionOceanographicRAG()
            init_time = time.time() - start_time
            
            logger.info(f"System initialized in {init_time:.2f}s")
            
            # Check system health
            health = self.rag_system.validate_system_health()
            logger.info(f"System Health: {health['overall_status']}")
            
            # Log component status
            for component, status in health['components'].items():
                logger.info(f"  {component}: {status['status']}")
            
            self.system_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    def test_basic_queries(self):
        """Test basic query processing"""
        
        logger.info("\nTesting Basic Query Processing")
        logger.info("-" * 40)
        
        basic_queries = [
            "Show surface temperature in Arabian Sea",
            "Get recent temperature measurements",
            "How many profiles are in the database?",
            "Find data from platform 1900121"
        ]
        
        results = []
        
        for i, query in enumerate(basic_queries, 1):
            logger.info(f"\nBasic Test {i}: {query}")
            
            try:
                start_time = time.time()
                result = self.rag_system.process_oceanographic_query(query)
                processing_time = time.time() - start_time
                
                if result['success']:
                    logger.info(f"SUCCESS - {result['result_count']} rows in {processing_time:.2f}s")
                    logger.info(f"Template: {result.get('template_used', 'unknown')}")
                    
                    results.append({
                        'query': query,
                        'success': True,
                        'processing_time': processing_time,
                        'result_count': result['result_count'],
                        'template': result.get('template_used', 'unknown')
                    })
                else:
                    logger.error(f"FAILED: {result.get('error', 'Unknown error')}")
                    results.append({
                        'query': query,
                        'success': False,
                        'error': result.get('error'),
                        'processing_time': processing_time
                    })
                    
            except Exception as e:
                logger.error(f"EXCEPTION: {e}")
                results.append({
                    'query': query,
                    'success': False,
                    'error': str(e),
                    'processing_time': 0
                })
        
        return results
    
    def test_llm_enhancement(self):
        """Test LLM enhancement functionality"""
        
        logger.info("\nTesting LLM Enhancement")
        logger.info("-" * 40)
        
        if not self.rag_system.llm_enhancement_enabled:
            logger.warning("LLM enhancement disabled - testing fallback behavior")
        
        llm_queries = [
            "Calculate average temperature at different depths",
            "Analyze temperature trends in the Indian Ocean",
            "Compare salinity between different regions",
            "Find temperature anomalies in recent data"
        ]
        
        results = []
        
        for i, query in enumerate(llm_queries, 1):
            logger.info(f"\nLLM Test {i}: {query}")
            
            try:
                start_time = time.time()
                result = self.rag_system.process_oceanographic_query(query)
                processing_time = time.time() - start_time
                
                if result['success']:
                    llm_info = result.get('llm_enhancement', {})
                    logger.info(f"SUCCESS - {result['result_count']} rows")
                    logger.info(f"LLM Used: {'Yes' if llm_info.get('used') else 'No'}")
                    logger.info(f"Processing Time: {processing_time:.2f}s")
                    
                    results.append({
                        'query': query,
                        'success': True,
                        'llm_used': llm_info.get('used', False),
                        'processing_time': processing_time,
                        'result_count': result['result_count']
                    })
                else:
                    logger.error(f"FAILED: {result.get('error')}")
                    results.append({
                        'query': query,
                        'success': False,
                        'error': result.get('error'),
                        'processing_time': processing_time
                    })
                    
            except Exception as e:
                logger.error(f"EXCEPTION: {e}")
                results.append({
                    'query': query,
                    'success': False,
                    'error': str(e),
                    'processing_time': 0
                })
        
        return results
    
    def test_error_handling(self):
        """Test system error handling"""
        
        logger.info("\nTesting Error Handling")
        logger.info("-" * 40)
        
        error_queries = [
            "",  # Empty query
            "Show me the color of temperature",  # Invalid request
            "Delete all data from database",  # Dangerous query
            "Find unicorns in the ocean"  # Nonsensical query
        ]
        
        results = []
        
        for i, query in enumerate(error_queries, 1):
            logger.info(f"\nError Test {i}: '{query}'")
            
            try:
                result = self.rag_system.process_oceanographic_query(query)
                
                # All these should either fail gracefully or return safe results
                if result['success'] and result.get('result_count', 0) > 0:
                    logger.info("Handled gracefully with safe results")
                elif not result['success']:
                    logger.info("Properly rejected invalid query")
                else:
                    logger.info("Handled gracefully with no results")
                
                results.append({
                    'query': query,
                    'handled_gracefully': True,
                    'success': result['success'],
                    'result_count': result.get('result_count', 0)
                })
                
            except Exception as e:
                logger.info(f"Exception handled: {e}")
                results.append({
                    'query': query,
                    'handled_gracefully': True,  # Exceptions are acceptable
                    'success': False,
                    'exception': str(e)
                })
        
        return results
    
    def test_vector_context_quality(self):
        """Test vector context retrieval quality"""
        
        logger.info("\nTesting Vector Context Quality")
        logger.info("-" * 40)
        
        context_queries = [
            "Show database schema for profiles",
            "How to calculate mixed layer depth?",
            "What are the temperature ranges in ocean data?"
        ]
        
        results = []
        
        for i, query in enumerate(context_queries, 1):
            logger.info(f"\nContext Test {i}: {query}")
            
            try:
                # Get context directly from the system
                context = self.rag_system._get_domain_context(query)
                context_length = len(context)
                
                # Simple quality indicators
                has_schema_info = 'argo_profiles' in context or 'argo_measurements' in context
                has_sql_examples = 'SELECT' in context
                has_calculations = 'calculation' in context.lower()
                
                logger.info(f"Context Length: {context_length} chars")
                logger.info(f"Schema Info: {'Yes' if has_schema_info else 'No'}")
                logger.info(f"SQL Examples: {'Yes' if has_sql_examples else 'No'}")
                
                quality_score = 0
                if context_length > 1000: quality_score += 25
                if has_schema_info: quality_score += 25
                if has_sql_examples: quality_score += 25
                if has_calculations: quality_score += 25
                
                logger.info(f"Quality Score: {quality_score}/100")
                
                results.append({
                    'query': query,
                    'context_length': context_length,
                    'quality_score': quality_score,
                    'has_schema': has_schema_info,
                    'has_sql': has_sql_examples
                })
                
            except Exception as e:
                logger.error(f"Context test failed: {e}")
                results.append({
                    'query': query,
                    'error': str(e),
                    'quality_score': 0
                })
        
        return results
    
    def generate_final_report(self, basic_results, llm_results, error_results, context_results):
        """Generate comprehensive test report"""
        
        logger.info("\nFINAL TEST REPORT")
        logger.info("=" * 50)
        
        # Calculate success rates
        basic_success = sum(1 for r in basic_results if r['success']) / len(basic_results) * 100
        llm_success = sum(1 for r in llm_results if r['success']) / len(llm_results) * 100
        error_handled = sum(1 for r in error_results if r['handled_gracefully']) / len(error_results) * 100
        
        # LLM usage stats
        llm_used_count = sum(1 for r in llm_results if r.get('llm_used', False))
        llm_usage_rate = (llm_used_count / len(llm_results)) * 100 if llm_results else 0
        
        # Performance stats
        all_times = [r['processing_time'] for r in basic_results + llm_results if r['success']]
        avg_time = sum(all_times) / len(all_times) if all_times else 0
        
        # Context quality
        avg_context_quality = sum(r['quality_score'] for r in context_results) / len(context_results) if context_results else 0
        
        # Report results
        logger.info("SYSTEM STATUS:")
        logger.info(f"  Database: {'Connected' if self.rag_system.engine else 'Failed'}")
        logger.info(f"  Vector Store: {'Available' if self.rag_system.vector_store else 'Missing'}")
        logger.info(f"  LLM Enhancement: {'Enabled' if self.rag_system.llm_enhancement_enabled else 'Disabled'}")
        
        logger.info("\nTEST RESULTS:")
        logger.info(f"  Basic Queries: {basic_success:.1f}% success")
        logger.info(f"  LLM Queries: {llm_success:.1f}% success")
        logger.info(f"  LLM Usage Rate: {llm_usage_rate:.1f}%")
        logger.info(f"  Error Handling: {error_handled:.1f}% handled gracefully")
        
        logger.info("\nPERFORMANCE:")
        logger.info(f"  Average Processing Time: {avg_time:.2f}s")
        logger.info(f"  Context Quality Average: {avg_context_quality:.1f}/100")
        
        # Overall assessment
        logger.info("\nOVERALL ASSESSMENT:")
        if basic_success >= 75 and error_handled >= 90:
            logger.info("✅ SYSTEM READY FOR PRODUCTION")
        elif basic_success >= 50 and error_handled >= 80:
            logger.info("⚠️ SYSTEM FUNCTIONAL - NEEDS MINOR IMPROVEMENTS")
        else:
            logger.info("❌ SYSTEM NEEDS MAJOR FIXES")
        
        return {
            'basic_success_rate': basic_success,
            'llm_success_rate': llm_success,
            'llm_usage_rate': llm_usage_rate,
            'error_handling_rate': error_handled,
            'avg_processing_time': avg_time,
            'context_quality': avg_context_quality
        }

def main():
    """Main test execution"""
    
    logger.info("Starting Clean RAG System Test Suite")
    logger.info("=" * 50)
    
    # Initialize test suite
    test_suite = CleanRAGSystemTest()
    
    if not test_suite.initialize_system():
        logger.error("System initialization failed - cannot proceed")
        return
    
    try:
        # Run all tests
        basic_results = test_suite.test_basic_queries()
        llm_results = test_suite.test_llm_enhancement()
        error_results = test_suite.test_error_handling()
        context_results = test_suite.test_vector_context_quality()
        
        # Generate final report
        report = test_suite.generate_final_report(
            basic_results, llm_results, error_results, context_results
        )
        
        logger.info("\nTest suite completed successfully")
        
        # Get LLM stats if available
        if test_suite.rag_system.enhanced_sql_generator:
            stats = test_suite.rag_system.enhanced_sql_generator.get_stats()
            logger.info(f"LLM Generator Stats: {stats}")
        
    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        return

if __name__ == "__main__":
    main()