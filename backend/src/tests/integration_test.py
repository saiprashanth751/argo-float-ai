# integration_test.py - COMPLETE VERSION
# Location: src/tests/integration_test.py

import os
import sys
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))

from services.enhanced_rag_oceanographic import EnhancedOceanographicRAG
from services.intelligent_response_system import IntelligentResponseSystem, ResponseFormat
from services.oceanographic_intelligence_engine import QueryIntent, ComplexityLevel
from utils.vector_db_initializer import initialize_production_vector_db

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemIntegrationTest:
    """Comprehensive integration test for the oceanographic system"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.rag_system = None
        self.response_system = None
    
    async def run_full_integration_test(self) -> Dict[str, Any]:
        """Run complete integration test suite"""
        
        logger.info("🧪 Starting System Integration Test Suite")
        logger.info("=" * 60)
        self.start_time = time.time()
        
        # Test phases
        test_phases = [
            ("Database Connection", self.test_database_connection),
            ("Vector Database", self.test_vector_database),
            ("RAG System", self.test_rag_system),
            ("Intelligence Engine", self.test_intelligence_engine),
            ("Response System", self.test_response_system),
            ("End-to-End Queries", self.test_end_to_end_queries),
            ("Performance Benchmarks", self.test_performance_benchmarks),
            ("Error Handling", self.test_error_handling)
        ]
        
        for phase_name, test_func in test_phases:
            logger.info(f"\n🔍 Testing: {phase_name}")
            logger.info("-" * 40)
            
            try:
                result = await test_func() if asyncio.iscoroutinefunction(test_func) else test_func()
                self.test_results[phase_name] = {
                    'status': 'PASSED' if result else 'FAILED',
                    'details': result if isinstance(result, dict) else {'success': result}
                }
                
                if result:
                    logger.info(f"✅ {phase_name} - PASSED")
                else:
                    logger.error(f"❌ {phase_name} - FAILED")
                    
            except Exception as e:
                logger.error(f"❌ {phase_name} - ERROR: {e}")
                self.test_results[phase_name] = {
                    'status': 'ERROR',
                    'details': {'error': str(e)}
                }
        
        # Generate final report
        total_time = time.time() - self.start_time
        return self.generate_test_report(total_time)
    
    def test_database_connection(self) -> bool:
        """Test database connectivity and schema validation"""
        
        try:
            from sqlalchemy import create_engine, text
            
            engine = create_engine(os.getenv('DATABASE_URL'), pool_pre_ping=True)
            
            with engine.connect() as conn:
                # Test basic connectivity
                result = conn.execute(text("SELECT 1")).scalar()
                if result != 1:
                    return False
                
                # Test schema tables exist
                required_tables = ['argo_profiles', 'argo_measurements', 'data_processing_log']
                for table in required_tables:
                    result = conn.execute(text(f"""
                        SELECT COUNT(*) FROM information_schema.tables 
                        WHERE table_name = '{table}'
                    """)).scalar()
                    
                    if result == 0:
                        logger.error(f"Required table missing: {table}")
                        return False
                
                # Test data availability
                profile_count = conn.execute(text("SELECT COUNT(*) FROM argo_profiles")).scalar()
                measurement_count = conn.execute(text("SELECT COUNT(*) FROM argo_measurements")).scalar()
                
                logger.info(f"Database contains {profile_count:,} profiles and {measurement_count:,} measurements")
                
                if profile_count == 0:
                    logger.warning("No profiles found in database")
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def test_vector_database(self) -> bool:
        """Test vector database initialization and functionality"""
        
        try:
            # Initialize vector database if needed
            vector_db_success = initialize_production_vector_db()
            
            if not vector_db_success:
                logger.error("Vector database initialization failed")
                return False
            
            # Test embeddings functionality
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from langchain_community.vectorstores import Chroma
            
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv('GEMINI_API_KEY')
            )
            
            vector_store = Chroma(
                persist_directory="storage/chroma_db_oceanographic",
                embedding_function=embeddings
            )
            
            # Test similarity search
            test_query = "How to query ARGO temperature profiles?"
            results = vector_store.similarity_search_with_score(test_query, k=3)
            
            if len(results) == 0:
                logger.error("Vector database returns no results for test query")
                return False
            
            logger.info(f"Vector search returned {len(results)} relevant documents")
            return True
            
        except Exception as e:
            logger.error(f"Vector database test failed: {e}")
            return False
    
    def test_rag_system(self) -> Dict[str, Any]:
        """Test RAG system functionality"""
        
        try:
            self.rag_system = EnhancedOceanographicRAG()
            
            # Test simple query
            test_query = "Count total profiles in database"
            result = self.rag_system.process_oceanographic_query(test_query)
            
            if not result['success']:
                logger.error(f"RAG system failed on simple query: {result.get('error')}")
                return {'success': False, 'error': result.get('error')}
            
            # Test query classification
            classification = result['classification']
            expected_intent = 'statistical_summary'
            
            if classification['intent'] != expected_intent:
                logger.warning(f"Query classification mismatch: expected {expected_intent}, got {classification['intent']}")
            
            # Test SQL generation
            sql_query = result['sql_query']
            if 'SELECT' not in sql_query.upper():
                logger.error("Generated SQL does not contain SELECT statement")
                return {'success': False, 'error': 'Invalid SQL generated'}
            
            logger.info(f"RAG system processed query successfully: {result['result_count']} records")
            
            return {
                'success': True,
                'result_count': result['result_count'],
                'processing_time': result['processing_time'],
                'classification': classification
            }
            
        except Exception as e:
            logger.error(f"RAG system test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_intelligence_engine(self) -> Dict[str, Any]:
        """Test oceanographic intelligence engine"""
        
        try:
            from services.oceanographic_intelligence_engine import OceanographicIntelligenceEngine
            from sqlalchemy import create_engine
            
            engine = create_engine(os.getenv('DATABASE_URL'))
            intel_engine = OceanographicIntelligenceEngine(engine)
            
            # Test query classification with various query types
            test_queries = [
                ("Show temperature profiles", QueryIntent.PROFILE_ANALYSIS),
                ("What is the average salinity?", QueryIntent.STATISTICAL_SUMMARY),
                ("Map temperature distribution", QueryIntent.SPATIAL_MAPPING),
                ("Show temperature trends over time", QueryIntent.TEMPORAL_TRENDS)
            ]
            
            classification_results = []
            
            for query_text, expected_intent in test_queries:
                classification = intel_engine.classify_query(query_text)
                
                result = {
                    'query': query_text,
                    'expected_intent': expected_intent.value,
                    'actual_intent': classification.intent.value,
                    'confidence': classification.confidence,
                    'complexity': classification.complexity.value,
                    'parameters': classification.context.parameters
                }
                
                classification_results.append(result)
                
                # Check if classification is reasonable
                if classification.confidence < 0.3:
                    logger.warning(f"Low confidence classification for: {query_text}")
            
            logger.info(f"Tested {len(test_queries)} query classifications")
            
            return {
                'success': True,
                'classifications': classification_results,
                'average_confidence': sum(r['confidence'] for r in classification_results) / len(classification_results)
            }
            
        except Exception as e:
            logger.error(f"Intelligence engine test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_response_system(self) -> Dict[str, Any]:
        """Test intelligent response system"""
        
        try:
            if not self.rag_system:
                self.rag_system = EnhancedOceanographicRAG()
            
            self.response_system = IntelligentResponseSystem(self.rag_system)
            
            # Test different response formats
            test_query = "Show surface temperature statistics"
            
            response_formats = [
                ResponseFormat(target_audience="researcher", complexity_level="advanced"),
                ResponseFormat(target_audience="general_public", complexity_level="basic"),
                ResponseFormat(target_audience="government_official", complexity_level="intermediate")
            ]
            
            response_results = []
            
            for response_format in response_formats:
                result = self.response_system.process_intelligent_query(test_query, response_format)
                
                if result['success']:
                    response_results.append({
                        'audience': response_format.target_audience,
                        'complexity': response_format.complexity_level,
                        'processing_time': result['processing_time'],
                        'visualizations_count': len(result['visualizations']),
                        'recommendations_count': len(result['recommendations']),
                        'narrative_length': len(result['narrative_response'])
                    })
                else:
                    logger.error(f"Response system failed for {response_format.target_audience}: {result.get('error')}")
                    return {'success': False, 'error': result.get('error')}
            
            logger.info(f"Generated {len(response_results)} different response formats")
            
            return {
                'success': True,
                'response_formats_tested': response_results,
                'average_processing_time': sum(r['processing_time'] for r in response_results) / len(response_results)
            }
            
        except Exception as e:
            logger.error(f"Response system test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def test_end_to_end_queries(self) -> Dict[str, Any]:
        """Test complete end-to-end query processing"""
        
        try:
            if not self.response_system:
                if not self.rag_system:
                    self.rag_system = EnhancedOceanographicRAG()
                self.response_system = IntelligentResponseSystem(self.rag_system)
            
            # Test queries representing real user scenarios
            end_to_end_queries = [
                {
                    'query': "What is the temperature at 1000 meters depth?",
                    'expected_complexity': 'intermediate',
                    'expected_intent': 'profile_analysis'
                },
                {
                    'query': "Show me salinity distribution in the Arabian Sea",
                    'expected_complexity': 'intermediate', 
                    'expected_intent': 'spatial_mapping'
                },
                {
                    'query': "Count how many ARGO floats we have data for",
                    'expected_complexity': 'basic',
                    'expected_intent': 'statistical_summary'
                },
                {
                    'query': "Compare water temperatures between 2020 and 2023",
                    'expected_complexity': 'advanced',
                    'expected_intent': 'temporal_trends'
                }
            ]
            
            query_results = []
            
            for test_case in end_to_end_queries:
                start_time = time.time()
                result = self.response_system.process_intelligent_query(
                    test_case['query'],
                    ResponseFormat(complexity_level="intermediate", target_audience="researcher")
                )
                end_time = time.time()
                
                if result['success']:
                    query_results.append({
                        'query': test_case['query'],
                        'success': True,
                        'processing_time': end_time - start_time,
                        'records_returned': result['results_summary']['total_records'],
                        'classification_confidence': result['classification']['confidence'],
                        'actual_intent': result['classification']['intent'],
                        'expected_intent': test_case['expected_intent'],
                        'visualizations_generated': len(result['visualizations']),
                        'recommendations_generated': len(result['recommendations'])
                    })
                    
                    logger.info(f"✅ E2E Query: '{test_case['query']}' -> {result['results_summary']['total_records']} records")
                else:
                    query_results.append({
                        'query': test_case['query'],
                        'success': False,
                        'error': result.get('error'),
                        'processing_time': end_time - start_time
                    })
                    
                    logger.error(f"❌ E2E Query failed: '{test_case['query']}' -> {result.get('error')}")
            
            successful_queries = sum(1 for r in query_results if r['success'])
            average_processing_time = sum(r['processing_time'] for r in query_results if r['success']) / max(successful_queries, 1)
            
            return {
                'success': successful_queries > 0,
                'total_queries': len(end_to_end_queries),
                'successful_queries': successful_queries,
                'average_processing_time': average_processing_time,
                'query_results': query_results
            }
            
        except Exception as e:
            logger.error(f"End-to-end test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test system performance under different loads"""
        
        try:
            if not self.response_system:
                if not self.rag_system:
                    self.rag_system = EnhancedOceanographicRAG()
                self.response_system = IntelligentResponseSystem(self.rag_system)
            
            # Performance benchmarks
            benchmark_queries = [
                "Count total profiles",  # Simple query
                "Show temperature profiles for platform 1900121",  # Medium complexity
                "Analyze temperature variability across all platforms"  # Complex query
            ]
            
            performance_results = []
            
            for query in benchmark_queries:
                # Run query multiple times to get average
                times = []
                
                for i in range(3):  # 3 runs for average
                    start_time = time.time()
                    result = self.response_system.process_intelligent_query(
                        query, 
                        ResponseFormat(complexity_level="basic", target_audience="researcher")
                    )
                    end_time = time.time()
                    
                    if result['success']:
                        times.append(end_time - start_time)
                    else:
                        logger.warning(f"Performance test query failed: {query}")
                
                if times:
                    performance_results.append({
                        'query': query,
                        'average_time': sum(times) / len(times),
                        'min_time': min(times),
                        'max_time': max(times),
                        'runs': len(times)
                    })
            
            # Performance thresholds
            acceptable_time = 10.0  # seconds
            slow_queries = [r for r in performance_results if r['average_time'] > acceptable_time]
            
            if slow_queries:
                logger.warning(f"{len(slow_queries)} queries exceed {acceptable_time}s threshold")
            
            return {
                'success': len(performance_results) > 0,
                'benchmark_results': performance_results,
                'slow_queries': len(slow_queries),
                'average_query_time': sum(r['average_time'] for r in performance_results) / max(len(performance_results), 1)
            }
            
        except Exception as e:
            logger.error(f"Performance benchmark failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test system error handling and recovery"""
        
        try:
            if not self.response_system:
                if not self.rag_system:
                    self.rag_system = EnhancedOceanographicRAG()
                self.response_system = IntelligentResponseSystem(self.rag_system)
            
            # Test various error conditions
            error_test_cases = [
                {
                    'query': "",  # Empty query
                    'expected_behavior': 'graceful_failure'
                },
                {
                    'query': "SELECT * FROM nonexistent_table",  # Invalid SQL-like query
                    'expected_behavior': 'graceful_failure'
                },
                {
                    'query': "Show data for platform INVALID123456789",  # Nonexistent platform
                    'expected_behavior': 'empty_results'
                },
                {
                    'query': "a" * 1000,  # Very long query
                    'expected_behavior': 'processing_attempt'
                },
                {
                    'query': "🌊🌡️📊 ocean temperature data",  # Query with emojis
                    'expected_behavior': 'processing_attempt'
                }
            ]
            
            error_results = []
            
            for test_case in error_test_cases:
                try:
                    result = self.response_system.process_intelligent_query(
                        test_case['query'],
                        ResponseFormat(complexity_level="basic", target_audience="researcher")
                    )
                    
                    error_results.append({
                        'query': test_case['query'][:50] + "..." if len(test_case['query']) > 50 else test_case['query'],
                        'expected': test_case['expected_behavior'],
                        'success': result['success'],
                        'handled_gracefully': True,  # If we get here, no exception was thrown
                        'error_message': result.get('error') if not result['success'] else None
                    })
                    
                except Exception as e:
                    error_results.append({
                        'query': test_case['query'][:50] + "..." if len(test_case['query']) > 50 else test_case['query'],
                        'expected': test_case['expected_behavior'],
                        'success': False,
                        'handled_gracefully': False,  # Exception was thrown
                        'error_message': str(e)
                    })
            
            graceful_failures = sum(1 for r in error_results if r['handled_gracefully'])
            
            logger.info(f"Error handling: {graceful_failures}/{len(error_results)} cases handled gracefully")
            
            return {
                'success': graceful_failures == len(error_results),
                'total_error_cases': len(error_results),
                'gracefully_handled': graceful_failures,
                'error_test_results': error_results
            }
            
        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_test_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 INTEGRATION TEST REPORT")
        logger.info("=" * 60)
        
        # Count test results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r['status'] == 'PASSED')
        failed_tests = sum(1 for r in self.test_results.values() if r['status'] == 'FAILED')
        error_tests = sum(1 for r in self.test_results.values() if r['status'] == 'ERROR')
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info(f"Test Summary:")
        logger.info(f"  Total Tests: {total_tests}")
        logger.info(f"  Passed: {passed_tests} ✅")
        logger.info(f"  Failed: {failed_tests} ❌")
        logger.info(f"  Errors: {error_tests} ⚠️")
        logger.info(f"  Success Rate: {success_rate:.1f}%")
        logger.info(f"  Total Time: {total_time:.2f}s")
        
        # Detailed results
        logger.info(f"\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result['status'] == 'PASSED' else "❌" if result['status'] == 'FAILED' else "⚠️"
            logger.info(f"  {status_emoji} {test_name}: {result['status']}")
            
            if result['status'] in ['FAILED', 'ERROR'] and 'error' in result['details']:
                logger.info(f"    Error: {result['details']['error']}")
        
        # System readiness assessment
        critical_tests = ['Database Connection', 'RAG System', 'Response System']
        critical_passed = all(
            self.test_results.get(test, {}).get('status') == 'PASSED' 
            for test in critical_tests
        )
        
        if critical_passed and success_rate >= 75:
            readiness = "READY FOR PRODUCTION"
            logger.info(f"\n🟢 System Status: {readiness}")
        elif critical_passed:
            readiness = "READY WITH MINOR ISSUES"
            logger.info(f"\n🟡 System Status: {readiness}")
        else:
            readiness = "NOT READY - CRITICAL ISSUES"
            logger.info(f"\n🔴 System Status: {readiness}")
        
        # Recommendations
        recommendations = []
        
        if failed_tests > 0:
            recommendations.append("Investigate and fix failing test cases")
        
        if error_tests > 0:
            recommendations.append("Resolve system errors before deployment")
        
        if success_rate < 90:
            recommendations.append("Improve system reliability before production use")
        
        # Performance insights
        performance_data = self.test_results.get('Performance Benchmarks', {}).get('details', {})
        if performance_data.get('slow_queries', 0) > 0:
            recommendations.append("Optimize slow query performance")
        
        if recommendations:
            logger.info(f"\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"  {i}. {rec}")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_time': total_time,
            'test_summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'errors': error_tests,
                'success_rate': success_rate
            },
            'system_status': readiness,
            'test_results': self.test_results,
            'recommendations': recommendations,
            'critical_systems_ready': critical_passed
        }


async def main():
    """Run the complete integration test suite"""
    
    test_suite = SystemIntegrationTest()
    report = await test_suite.run_full_integration_test()
    
    # Save report to file
    report_path = Path("test_reports") / f"integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        import json
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"\n📁 Test report saved to: {report_path}")
    
    return report


if __name__ == "__main__":
    asyncio.run(main())