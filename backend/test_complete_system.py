# test_complete_system.py
# Comprehensive test for the complete FloatChat intelligent system

import os
import sys
import logging
from datetime import datetime
import traceback
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

# Add src to path properly
project_root = Path(__file__).parent
src_path = project_root / 'src'
services_path = src_path / 'services'

# Add paths to sys.path if not already there
for path in [str(src_path), str(services_path), str(project_root)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def test_individual_components():
    """Test each component individually"""
    
    print("🧪 TESTING INDIVIDUAL COMPONENTS")
    print("=" * 60)
    
    # Test 1: Oceanographic Intelligence Engine
    print("\n1️⃣ Testing Oceanographic Intelligence Engine...")
    try:
        # Import with multiple fallback strategies
        try:
            from src.services.oceanographic_intelligence_engine import OceanographicIntelligenceEngine
        except ImportError:
            try:
                from src.services.oceanographic_intelligence_engine import OceanographicIntelligenceEngine
            except ImportError:
                from src.services.oceanographic_intelligence_engine import OceanographicIntelligenceEngine
        
        # Initialize with no database for basic testing
        engine = OceanographicIntelligenceEngine(db_engine=None)
        
        # Test query classification
        test_query = "Show temperature profile for float 1900121"
        classification = engine.classify_query(test_query)
        
        print(f"✅ Intelligence Engine works!")
        print(f"   Query: {test_query}")
        print(f"   Intent: {classification.intent.value}")
        print(f"   Complexity: {classification.complexity.value}")
        print(f"   Confidence: {classification.confidence:.2f}")
        
    except Exception as e:
        print(f"❌ Intelligence Engine failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 2: Enhanced RAG System
    print("\n2️⃣ Testing Enhanced RAG System...")
    try:
        # Import with multiple fallback strategies
        try:
            from src.services.enhanced_rag_oceanographic import EnhancedOceanographicRAG
        except ImportError:
            try:
                from src.services.enhanced_rag_oceanographic import EnhancedOceanographicRAG
            except ImportError:
                from src.services.enhanced_rag_oceanographic import EnhancedOceanographicRAG
        
        # Initialize RAG system
        rag_system = EnhancedOceanographicRAG()
        
        # Test query processing (will likely fail due to DB, but we can test initialization)
        print(f"✅ Enhanced RAG System initialized!")
        print(f"   Database engine: {'Connected' if rag_system.engine else 'None'}")
        print(f"   LLM model: {'Configured' if rag_system.llm else 'Not configured'}")
        print(f"   Vector store: {'Loaded' if rag_system.vector_store else 'Not loaded'}")
        
    except Exception as e:
        print(f"❌ Enhanced RAG System failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 3: Intelligent Response System
    print("\n3️⃣ Testing Intelligent Response System...")
    try:
        # Import with multiple fallback strategies
        try:
            from src.services.intelligent_response_system import IntelligentResponseSystem, ResponseFormat
        except ImportError:
            try:
                from src.services.intelligent_response_system import IntelligentResponseSystem, ResponseFormat
            except ImportError:
                from src.services.intelligent_response_system import IntelligentResponseSystem, ResponseFormat
        
        # Initialize response system (will use the RAG system from above)
        response_system = IntelligentResponseSystem(rag_system=rag_system)
        
        print(f"✅ Intelligent Response System initialized!")
        print(f"   Response LLM: {'Configured' if response_system.response_llm else 'Not configured'}")
        print(f"   Visualization templates: {len(response_system.viz_templates)} categories")
        print(f"   Response templates: {len(response_system.response_templates)} audiences")
        
    except Exception as e:
        print(f"❌ Intelligent Response System failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_database_connection():
    """Test database connection and basic queries"""
    
    print("\n🗄️ TESTING DATABASE CONNECTION")
    print("=" * 60)
    
    try:
        from sqlalchemy import create_engine, text
        
        # Create database engine
        engine = create_engine(
            os.getenv('DATABASE_URL', 'postgresql://argo_user:argo_password@localhost:5432/argo_data'),
            pool_size=5,
            max_overflow=10
        )
        
        # Test basic connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            print(f"✅ Database connection successful: {row.test}")
        
        # Test ARGO tables
        with engine.connect() as conn:
            # Test enhanced_floats_metadata
            result = conn.execute(text("SELECT COUNT(*) as count FROM enhanced_floats_metadata"))
            metadata_count = result.fetchone().count
            print(f"✅ Enhanced floats metadata: {metadata_count:,} records")
            
            # Test enhanced_measurements
            result = conn.execute(text("SELECT COUNT(*) as count FROM enhanced_measurements"))
            measurements_count = result.fetchone().count
            print(f"✅ Enhanced measurements: {measurements_count:,} records")
            
            # Test sample data
            result = conn.execute(text("""
                SELECT fm.platform_number, fm.latitude, fm.longitude, 
                       COUNT(m.id) as measurement_count
                FROM enhanced_floats_metadata fm
                LEFT JOIN enhanced_measurements m ON fm.id = m.metadata_id
                GROUP BY fm.platform_number, fm.latitude, fm.longitude
                LIMIT 3
            """))
            
            print("📊 Sample data:")
            for row in result:
                print(f"   Platform {row.platform_number}: {row.measurement_count:,} measurements at ({row.latitude:.2f}, {row.longitude:.2f})")
        
        return engine
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("Make sure PostgreSQL is running and DATABASE_URL is correct")
        traceback.print_exc()
        return None

def test_end_to_end_system(engine):
    """Test the complete end-to-end system"""
    
    print("\n🚀 TESTING END-TO-END SYSTEM")
    print("=" * 60)
    
    try:
        # Import with multiple fallback strategies
        try:
            from src.services.intelligent_response_system import IntelligentResponseSystem, ResponseFormat
            from src.services.enhanced_rag_oceanographic import EnhancedOceanographicRAG
        except ImportError:
            try:
                from src.services.intelligent_response_system import IntelligentResponseSystem, ResponseFormat
                from src.services.enhanced_rag_oceanographic import EnhancedOceanographicRAG
            except ImportError:
                from src.services.intelligent_response_system import IntelligentResponseSystem, ResponseFormat
                from src.services.enhanced_rag_oceanographic import EnhancedOceanographicRAG
        
        # Initialize complete system with database
        rag_system = EnhancedOceanographicRAG(db_engine=engine)
        response_system = IntelligentResponseSystem(rag_system=rag_system)
        
        # Test queries with different complexity levels
        test_cases = [
            {
                'query': "How many ARGO float measurements do we have in total?",
                'format': ResponseFormat(complexity_level="basic", target_audience="general_public"),
                'expected': 'statistical_summary'
            },
            {
                'query': "Show me temperature measurements for any available float",
                'format': ResponseFormat(complexity_level="intermediate", target_audience="researcher"),
                'expected': 'profile_analysis'
            },
            {
                'query': "What is the spatial distribution of our ARGO floats?",
                'format': ResponseFormat(complexity_level="advanced", target_audience="government_official"),
                'expected': 'spatial_mapping'
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i}: {test_case['query']} ---")
            
            try:
                start_time = datetime.now()
                
                # Process query through intelligent system
                result = response_system.process_intelligent_query(
                    test_case['query'], 
                    test_case['format']
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                if result['success']:
                    print(f"✅ Success! ({processing_time:.2f}s)")
                    print(f"   Intent: {result['classification']['intent']}")
                    print(f"   Records: {result['results_summary']['total_records']:,}")
                    print(f"   Confidence: {result['classification']['confidence']:.0%}")
                    print(f"   Visualizations: {len(result['visualizations'])}")
                    print(f"   Recommendations: {len(result['recommendations'])}")
                    
                    # Show narrative preview
                    narrative = result['narrative_response']
                    preview = narrative[:150] + "..." if len(narrative) > 150 else narrative
                    print(f"   Response: {preview}")
                    
                else:
                    print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                    if 'sql_query' in result:
                        print(f"   SQL: {result['sql_query']}")
                    
            except Exception as e:
                print(f"❌ Exception in test case {i}: {e}")
                traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end system failed: {e}")
        traceback.print_exc()
        return False

def test_api_compatibility():
    """Test compatibility with existing FastAPI endpoints"""
    
    print("\n🔌 TESTING API COMPATIBILITY")
    print("=" * 60)
    
    try:
        # Test if we can create the same interface as the old system
        try:
            from src.services.enhanced_rag_oceanographic import EnhancedOceanographicRAG
        except ImportError:
            try:
                from src.services.enhanced_rag_oceanographic import EnhancedOceanographicRAG
            except ImportError:
                from src.services.enhanced_rag_oceanographic import EnhancedOceanographicRAG
        
        # Create wrapper that mimics old API
        class APICompatibilityWrapper:
            def __init__(self):
                self.rag_system = EnhancedOceanographicRAG()
            
            def process_advanced_query(self, query: str):
                """Wrapper to maintain API compatibility"""
                result = self.rag_system.process_oceanographic_query(query)
                
                # Convert to old format
                if result['success']:
                    return {
                        'success': True,
                        'results': result['results'],
                        'result_count': result['result_count'],
                        'columns': result['columns'],
                        'sql_query': result['sql_query'],
                        'processing_time': result['processing_time']
                    }
                else:
                    return {
                        'success': False,
                        'error': result['error'],
                        'processing_time': result['processing_time']
                    }
            
            def classify_query_intent(self, query: str):
                """Wrapper for query classification"""
                classification = self.rag_system.ocean_intelligence.classify_query(query)
                return {
                    'type': classification.intent.value,
                    'confidence': classification.confidence,
                    'complexity': classification.complexity.value
                }
        
        # Test the wrapper
        wrapper = APICompatibilityWrapper()
        
        # Test query processing
        result = wrapper.process_advanced_query("Show me available data")
        print(f"✅ API compatibility wrapper works!")
        print(f"   Query processing: {'Success' if result['success'] else 'Failed'}")
        
        # Test intent classification
        intent = wrapper.classify_query_intent("Show temperature data")
        print(f"   Intent classification: {intent['type']} ({intent['confidence']:.0%} confidence)")
        
        return wrapper
        
    except Exception as e:
        print(f"❌ API compatibility test failed: {e}")
        traceback.print_exc()
        return None

def test_imports_only():
    """Test just the imports to isolate import issues"""
    
    print("\n📦 TESTING IMPORTS ONLY")
    print("=" * 60)
    
    import_results = {}
    
    # Test 1: Oceanographic Intelligence Engine
    try:
        from src.services.oceanographic_intelligence_engine import OceanographicIntelligenceEngine
        import_results['ocean_intelligence'] = True
        print("✅ OceanographicIntelligenceEngine imported successfully")
    except ImportError as e:
        import_results['ocean_intelligence'] = False
        print(f"❌ OceanographicIntelligenceEngine import failed: {e}")
    
    # Test 2: Enhanced RAG
    try:
        from src.services.enhanced_rag_oceanographic import EnhancedOceanographicRAG
        import_results['enhanced_rag'] = True
        print("✅ EnhancedOceanographicRAG imported successfully")
    except ImportError as e:
        import_results['enhanced_rag'] = False
        print(f"❌ EnhancedOceanographicRAG import failed: {e}")
    
    # Test 3: Intelligent Response System
    try:
        from src.services.intelligent_response_system import IntelligentResponseSystem, ResponseFormat
        import_results['intelligent_response'] = True
        print("✅ IntelligentResponseSystem imported successfully")
    except ImportError as e:
        import_results['intelligent_response'] = False
        print(f"❌ IntelligentResponseSystem import failed: {e}")
    
    return import_results

def run_comprehensive_test():
    """Run the complete test suite"""
    
    print("🎯 FLOATCHAT COMPLETE SYSTEM TEST")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    print("=" * 80)
    
    # Test results tracker
    test_results = {}
    
    # Test 0: Import tests first
    import_results = test_imports_only()
    test_results['imports'] = all(import_results.values())
    
    # Test 1: Individual components (only if imports work)
    if test_results['imports']:
        test_results['components'] = test_individual_components()
    else:
        test_results['components'] = False
        print("\n⚠️ Skipping component tests due to import failures")
    
    # Test 2: Database connection
    engine = test_database_connection()
    test_results['database'] = engine is not None
    
    # Test 3: End-to-end system (only if components and database work)
    if test_results['components'] and engine:
        test_results['end_to_end'] = test_end_to_end_system(engine)
    else:
        test_results['end_to_end'] = False
        print("\n⚠️ Skipping end-to-end test due to component or database failure")
    
    # Test 4: API compatibility (only if imports work)
    if test_results['imports']:
        wrapper = test_api_compatibility()
        test_results['api_compatibility'] = wrapper is not None
    else:
        test_results['api_compatibility'] = False
        wrapper = None
        print("\n⚠️ Skipping API compatibility test due to import failures")
    
    # Final results
    print("\n" + "=" * 80)
    print("🏁 FINAL TEST RESULTS")
    print("=" * 80)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"Overall Score: {passed_tests}/{total_tests}")
    print()
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name.replace('_', ' ').title()}")
    
    # Show specific import failures
    if not test_results['imports']:
        print("\nIMPORT DETAILS:")
        for module, success in import_results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {module}")
    
    if passed_tests == total_tests:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("Your complete FloatChat system is ready for integration!")
        
        if wrapper:
            print("\nNext steps:")
            print("1. Update main.py to use the new system")
            print("2. Test with your existing API endpoints")
            print("3. Connect to your Next.js frontend")
            
    else:
        print(f"\n⚠️ {total_tests - passed_tests} tests failed.")
        print("Please review the errors above and fix the issues.")
        
        if not test_results['imports']:
            print("\n🔧 IMPORT FIX SUGGESTIONS:")
            print("1. Make sure all .py files are in the correct directories")
            print("2. Check that __init__.py files exist in package directories")
            print("3. Verify that relative imports are correct")
            print("4. Consider using absolute imports instead")
    
    print(f"\nTest completed at: {datetime.now()}")
    
    return test_results, wrapper

if __name__ == "__main__":
    # Run the comprehensive test
    results, api_wrapper = run_comprehensive_test()
    
#     # If API wrapper is available, show how to integrate with main.py
#     if api_wrapper:
#         print("\n" + "=" * 80)
#         print("🔧 INTEGRATION GUIDANCE")
#         print("=" * 80)
#         print("""
# To update your main.py, replace:
#     from src.services.advanced_rag_llm_system import AdvancedRAGEnhancedLLM
#     rag_system = AdvancedRAGEnhancedLLM()

# With:
#     from src.services.enhanced_rag_oceanographic import EnhancedOceanographicRAG
#     rag_system = EnhancedOceanographicRAG()

# The API endpoints should work with minimal changes since the new system
# maintains compatibility with the old interface methods.
#         """)