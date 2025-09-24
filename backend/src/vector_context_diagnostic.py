# vector_context_diagnostic.py
"""
Vector Store Context Diagnostic Tool
Analyzes what context is actually being retrieved for different queries
and assesses whether it's sufficient for intelligent SQL generation.
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorContextDiagnostic:
    """Diagnostic tool to analyze vector store context quality"""
    
    def __init__(self):
        self.diagnostic_results = []
        self.system_initialized = False
        
    def initialize_system(self):
        """Initialize just the vector store components for testing"""
        
        logger.info("🔍 Initializing Vector Store Diagnostic")
        logger.info("=" * 50)
        
        try:
            # Import the RAG system
            from services.enhanced_rag_oceanographic import ProductionOceanographicRAG
            
            # Initialize minimal system (just vector store)
            self.rag_system = ProductionOceanographicRAG()
            
            # Check system components
            logger.info(f"Vector Store Available: {'Yes' if self.rag_system.vector_store else 'No'}")
            logger.info(f"Embeddings Model: {'Available' if hasattr(self.rag_system, 'embeddings') else 'Not Available'}")
            
            if self.rag_system.vector_store:
                try:
                    # Test basic vector store functionality
                    test_docs = self.rag_system.vector_store.similarity_search("test", k=1)
                    logger.info(f"Vector Store Test: {'Passed' if test_docs else 'Failed - No documents'}")
                    if test_docs:
                        logger.info(f"Document Count Available: Yes")
                except Exception as e:
                    logger.warning(f"Vector Store Test Failed: {e}")
            
            self.system_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    def analyze_vector_context_for_queries(self):
        """Analyze vector context retrieval for different query types"""
        
        logger.info("\n🔍 VECTOR CONTEXT ANALYSIS")
        logger.info("=" * 50)
        
        # Test queries of different types and complexities
        test_queries = [
            {
                'name': 'Simple Surface Query',
                'query': 'Show temperature in Arabian Sea',
                'category': 'basic',
                'expected_needs': ['argo_profiles table', 'surface_temp column', 'spatial filtering']
            },
            {
                'name': 'Statistical Aggregation',
                'query': 'What is the average surface temperature in Indian Ocean during 2023?',
                'category': 'statistical',
                'expected_needs': ['argo_profiles table', 'AVG function', 'temporal filtering', 'spatial filtering']
            },
            {
                'name': 'Profile Analysis',
                'query': 'Show temperature profiles at different depths for platform 1900121',
                'category': 'profile',
                'expected_needs': ['argo_profiles table', 'argo_measurements table', 'JOIN relationship', 'depth/pressure']
            },
            {
                'name': 'Complex Scientific Query',
                'query': 'Calculate mixed layer depth variability in monsoon-influenced regions',
                'category': 'complex',
                'expected_needs': ['both tables', 'mixed_layer_depth calculation', 'advanced filtering']
            },
            {
                'name': 'Schema Information Query',
                'query': 'Show me the database schema structure for oceanographic data',
                'category': 'metadata',
                'expected_needs': ['table descriptions', 'column definitions', 'relationships']
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_queries, 1):
            logger.info(f"\n--- Query {i}: {test_case['name']} ---")
            logger.info(f"Query: {test_case['query']}")
            logger.info(f"Category: {test_case['category']}")
            
            # Perform context analysis
            context_analysis = self._analyze_single_query_context(test_case)
            results.append(context_analysis)
            
            # Log immediate results
            self._log_context_analysis(context_analysis)
        
        return results
    
    def _analyze_single_query_context(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context retrieval for a single query"""
        
        query = test_case['query']
        
        try:
            # Step 1: Get vector context using the same method as the RAG system
            start_time = time.time()
            
            if hasattr(self.rag_system, '_get_domain_context'):
                vector_context = self.rag_system._get_domain_context(query, k=5)
            else:
                # Fallback direct retrieval
                if self.rag_system.vector_store:
                    docs = self.rag_system.vector_store.similarity_search(query, k=5)
                    vector_context = "\n\n".join([doc.page_content for doc in docs])
                else:
                    vector_context = ""
            
            retrieval_time = time.time() - start_time
            
            # Step 2: Analyze context quality
            context_analysis = {
                'test_case': test_case['name'],
                'query': query,
                'category': test_case['category'],
                'retrieval_time': retrieval_time,
                'context_length': len(vector_context),
                'context_content': vector_context,
                'quality_assessment': self._assess_context_quality(vector_context, test_case),
                'schema_coverage': self._assess_schema_coverage(vector_context),
                'sql_example_coverage': self._assess_sql_examples(vector_context),
                'expected_needs_met': self._check_expected_needs(vector_context, test_case['expected_needs']),
                'llm_usability_score': 0,  # Will calculate based on other factors
                'recommendations': []
            }
            
            # Calculate overall LLM usability score
            context_analysis['llm_usability_score'] = self._calculate_llm_usability_score(context_analysis)
            
            # Generate recommendations
            context_analysis['recommendations'] = self._generate_recommendations(context_analysis)
            
            return context_analysis
            
        except Exception as e:
            logger.error(f"Context analysis failed for query: {query}, Error: {e}")
            return {
                'test_case': test_case['name'],
                'query': query,
                'category': test_case['category'],
                'error': str(e),
                'retrieval_time': 0,
                'context_length': 0,
                'context_content': '',
                'quality_assessment': 'failed',
                'llm_usability_score': 0,
                'recommendations': ['Fix vector store retrieval error']
            }
    
    def _assess_context_quality(self, context: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of retrieved context"""
        
        if not context or len(context.strip()) < 50:
            return {
                'overall': 'insufficient',
                'reason': 'Context too short or empty',
                'detailed_issues': ['No meaningful content retrieved']
            }
        
        context_lower = context.lower()
        issues = []
        strengths = []
        
        # Check for essential database information
        if 'argo_profiles' in context_lower:
            strengths.append('Contains argo_profiles table info')
        else:
            issues.append('Missing argo_profiles table information')
        
        if 'argo_measurements' in context_lower:
            strengths.append('Contains argo_measurements table info')
        elif test_case['category'] in ['profile', 'complex']:
            issues.append('Missing argo_measurements table info (needed for this query type)')
        
        # Check for JOIN information
        if 'join' in context_lower:
            strengths.append('Contains JOIN relationship information')
        elif test_case['category'] in ['profile', 'complex']:
            issues.append('Missing JOIN relationship information')
        
        # Check for column information
        essential_columns = ['platform_number', 'latitude', 'longitude', 'temperature', 'salinity']
        found_columns = [col for col in essential_columns if col in context_lower]
        if found_columns:
            strengths.append(f'Contains column info: {found_columns}')
        else:
            issues.append('Missing essential column information')
        
        # Overall assessment
        if len(issues) == 0:
            overall = 'excellent'
        elif len(strengths) > len(issues):
            overall = 'good'
        elif len(strengths) == len(issues):
            overall = 'moderate'
        else:
            overall = 'poor'
        
        return {
            'overall': overall,
            'strengths': strengths,
            'issues': issues,
            'detailed_analysis': f'{len(strengths)} strengths, {len(issues)} issues'
        }
    
    def _assess_schema_coverage(self, context: str) -> Dict[str, Any]:
        """Assess how well the context covers database schema"""
        
        context_lower = context.lower()
        
        # Essential schema elements
        schema_elements = {
            'tables': ['argo_profiles', 'argo_measurements'],
            'key_columns': ['platform_number', 'profile_date', 'latitude', 'longitude', 
                           'surface_temp', 'surface_salinity', 'temperature', 'salinity', 'pressure'],
            'relationships': ['join', 'foreign key', 'profile_id'],
            'data_types': ['varchar', 'integer', 'timestamp', 'double precision'],
            'constraints': ['primary key', 'not null', 'index']
        }
        
        coverage = {}
        for category, elements in schema_elements.items():
            found = [elem for elem in elements if elem in context_lower]
            coverage[category] = {
                'found': found,
                'missing': [elem for elem in elements if elem not in found],
                'coverage_percent': (len(found) / len(elements)) * 100
            }
        
        # Overall coverage score
        total_found = sum(len(info['found']) for info in coverage.values())
        total_possible = sum(len(elements) for elements in schema_elements.values())
        overall_coverage = (total_found / total_possible) * 100
        
        return {
            'overall_coverage_percent': overall_coverage,
            'by_category': coverage,
            'assessment': 'excellent' if overall_coverage > 80 else 
                         'good' if overall_coverage > 60 else
                         'moderate' if overall_coverage > 40 else 'poor'
        }
    
    def _assess_sql_examples(self, context: str) -> Dict[str, Any]:
        """Assess SQL examples in the context"""
        
        context_lower = context.lower()
        
        # Look for SQL patterns
        sql_indicators = ['select', 'from', 'where', 'join', 'group by', 'order by']
        sql_patterns_found = [pattern for pattern in sql_indicators if pattern in context_lower]
        
        # Count apparent SQL queries
        select_count = context_lower.count('select')
        
        # Look for specific query types
        query_types = {
            'basic_select': 'select.*from.*argo_profiles' in context_lower,
            'join_query': 'join.*argo_measurements' in context_lower,
            'aggregation': any(agg in context_lower for agg in ['avg', 'count', 'sum', 'max', 'min']),
            'spatial_filter': any(spatial in context_lower for spatial in ['latitude', 'longitude', 'between']),
            'temporal_filter': any(temporal in context_lower for temporal in ['profile_date', 'date_trunc'])
        }
        
        examples_quality = 'excellent' if select_count >= 3 and sum(query_types.values()) >= 3 else \
                          'good' if select_count >= 2 and sum(query_types.values()) >= 2 else \
                          'moderate' if select_count >= 1 else 'poor'
        
        return {
            'sql_patterns_found': sql_patterns_found,
            'apparent_query_count': select_count,
            'query_types_covered': query_types,
            'examples_quality': examples_quality,
            'has_working_examples': select_count > 0 and 'from argo_profiles' in context_lower
        }
    
    def _check_expected_needs(self, context: str, expected_needs: List[str]) -> Dict[str, Any]:
        """Check if context meets expected informational needs"""
        
        context_lower = context.lower()
        
        needs_met = {}
        for need in expected_needs:
            need_lower = need.lower()
            # Simple keyword matching for now
            met = any(keyword in context_lower for keyword in need_lower.split())
            needs_met[need] = met
        
        total_needs = len(expected_needs)
        met_count = sum(needs_met.values())
        coverage_percent = (met_count / total_needs * 100) if total_needs > 0 else 0
        
        return {
            'needs_coverage': needs_met,
            'coverage_percent': coverage_percent,
            'met_count': met_count,
            'total_needs': total_needs,
            'assessment': 'excellent' if coverage_percent >= 90 else
                         'good' if coverage_percent >= 70 else
                         'moderate' if coverage_percent >= 50 else 'poor'
        }
    
    def _calculate_llm_usability_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall LLM usability score (0-100)"""
        
        if 'error' in analysis:
            return 0.0
        
        # Weight different factors
        weights = {
            'context_length': 0.15,      # Having sufficient content
            'schema_coverage': 0.30,     # Database schema information
            'sql_examples': 0.25,        # Working SQL examples
            'expected_needs': 0.30       # Meeting query-specific needs
        }
        
        scores = {}
        
        # Context length score (0-100 based on reasonable thresholds)
        length = analysis['context_length']
        if length >= 2000:
            scores['context_length'] = 100
        elif length >= 1000:
            scores['context_length'] = 80
        elif length >= 500:
            scores['context_length'] = 60
        elif length >= 200:
            scores['context_length'] = 40
        else:
            scores['context_length'] = 20
        
        # Schema coverage score
        schema_percent = analysis['schema_coverage']['overall_coverage_percent']
        scores['schema_coverage'] = schema_percent
        
        # SQL examples score
        sql_quality = analysis['sql_example_coverage']['examples_quality']
        sql_scores = {'excellent': 100, 'good': 80, 'moderate': 60, 'poor': 20}
        scores['sql_examples'] = sql_scores.get(sql_quality, 0)
        
        # Expected needs score
        needs_percent = analysis['expected_needs_met']['coverage_percent']
        scores['expected_needs'] = needs_percent
        
        # Calculate weighted average
        total_score = sum(scores[factor] * weights[factor] for factor in weights)
        
        return round(total_score, 2)
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on context analysis"""
        
        recommendations = []
        
        if 'error' in analysis:
            recommendations.append("Fix vector store retrieval functionality")
            return recommendations
        
        # Context length recommendations
        if analysis['context_length'] < 500:
            recommendations.append("Increase vector context retrieval (k parameter) or improve document chunking")
        
        # Schema coverage recommendations
        schema_coverage = analysis['schema_coverage']['overall_coverage_percent']
        if schema_coverage < 60:
            recommendations.append("Add more comprehensive database schema documentation to vector store")
        
        # SQL examples recommendations
        if analysis['sql_example_coverage']['apparent_query_count'] < 2:
            recommendations.append("Include more working SQL query examples in vector store documents")
        
        # Query type specific recommendations
        if analysis['category'] in ['profile', 'complex']:
            if not analysis['sql_example_coverage']['query_types_covered']['join_query']:
                recommendations.append("Add JOIN query examples for profile analysis capabilities")
        
        # Expected needs recommendations
        if analysis['expected_needs_met']['coverage_percent'] < 70:
            missing_needs = [need for need, met in analysis['expected_needs_met']['needs_coverage'].items() if not met]
            recommendations.append(f"Address missing information needs: {', '.join(missing_needs)}")
        
        # Overall usability recommendations
        if analysis['llm_usability_score'] < 50:
            recommendations.append("CRITICAL: Context quality too low for reliable LLM SQL generation")
        elif analysis['llm_usability_score'] < 70:
            recommendations.append("Context quality moderate - expect inconsistent LLM performance")
        
        return recommendations
    
    def _log_context_analysis(self, analysis: Dict[str, Any]):
        """Log the analysis results in a readable format"""
        
        logger.info(f"Context Length: {analysis['context_length']} characters")
        logger.info(f"Retrieval Time: {analysis['retrieval_time']:.3f}s")
        
        # Quality assessment
        quality = analysis['quality_assessment']
        logger.info(f"Quality Assessment: {quality['overall'].upper()}")
        if quality.get('strengths'):
            logger.info(f"  Strengths: {len(quality['strengths'])} found")
        if quality.get('issues'):
            logger.info(f"  Issues: {len(quality['issues'])} found")
        
        # Schema coverage
        schema = analysis['schema_coverage']
        logger.info(f"Schema Coverage: {schema['overall_coverage_percent']:.1f}% ({schema['assessment']})")
        
        # SQL examples
        sql = analysis['sql_example_coverage']
        logger.info(f"SQL Examples: {sql['apparent_query_count']} queries, {sql['examples_quality']} quality")
        
        # Expected needs
        needs = analysis['expected_needs_met']
        logger.info(f"Expected Needs Met: {needs['met_count']}/{needs['total_needs']} ({needs['coverage_percent']:.1f}%)")
        
        # Overall score
        logger.info(f"LLM Usability Score: {analysis['llm_usability_score']}/100")
        
        # Recommendations
        if analysis['recommendations']:
            logger.info("Recommendations:")
            for i, rec in enumerate(analysis['recommendations'], 1):
                logger.info(f"  {i}. {rec}")
        
        # Show actual context (first 200 chars)
        if analysis['context_content']:
            preview = analysis['context_content'][:200] + "..." if len(analysis['context_content']) > 200 else analysis['context_content']
            logger.info(f"Context Preview: {preview}")
        
        logger.info("-" * 50)
    
    def generate_diagnostic_report(self, results: List[Dict[str, Any]]):
        """Generate comprehensive diagnostic report"""
        
        logger.info("\n📊 VECTOR STORE DIAGNOSTIC REPORT")
        logger.info("=" * 60)
        
        if not results:
            logger.error("No results to analyze")
            return
        
        # Overall statistics
        total_queries = len(results)
        successful_retrievals = len([r for r in results if 'error' not in r])
        avg_context_length = sum(r.get('context_length', 0) for r in results) / total_queries
        avg_usability_score = sum(r.get('llm_usability_score', 0) for r in results) / total_queries
        
        logger.info(f"RETRIEVAL STATISTICS:")
        logger.info(f"  Total Queries Tested: {total_queries}")
        logger.info(f"  Successful Retrievals: {successful_retrievals}/{total_queries} ({successful_retrievals/total_queries*100:.1f}%)")
        logger.info(f"  Average Context Length: {avg_context_length:.0f} characters")
        logger.info(f"  Average LLM Usability Score: {avg_usability_score:.1f}/100")
        
        # Quality distribution
        quality_distribution = {}
        for result in results:
            if 'quality_assessment' in result:
                quality = result['quality_assessment']['overall']
                quality_distribution[quality] = quality_distribution.get(quality, 0) + 1
        
        logger.info(f"\nQUALITY DISTRIBUTION:")
        for quality, count in quality_distribution.items():
            logger.info(f"  {quality.title()}: {count} queries")
        
        # Score distribution
        score_ranges = {
            'Excellent (80-100)': len([r for r in results if r.get('llm_usability_score', 0) >= 80]),
            'Good (60-79)': len([r for r in results if 60 <= r.get('llm_usability_score', 0) < 80]),
            'Moderate (40-59)': len([r for r in results if 40 <= r.get('llm_usability_score', 0) < 60]),
            'Poor (0-39)': len([r for r in results if r.get('llm_usability_score', 0) < 40])
        }
        
        logger.info(f"\nUSABILITY SCORE DISTRIBUTION:")
        for range_name, count in score_ranges.items():
            logger.info(f"  {range_name}: {count} queries")
        
        # Common issues
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.get('recommendations', []))
        
        from collections import Counter
        common_issues = Counter(all_recommendations).most_common(5)
        
        logger.info(f"\nMOST COMMON ISSUES:")
        for issue, count in common_issues:
            logger.info(f"  {count}x: {issue}")
        
        # Overall assessment
        logger.info(f"\nOVERALL ASSESSMENT:")
        if avg_usability_score >= 70:
            logger.info("✅ VECTOR STORE READY FOR INTELLIGENT LLM USAGE")
            logger.info("   Context quality is sufficient for reliable SQL generation")
        elif avg_usability_score >= 50:
            logger.info("⚠️  VECTOR STORE PARTIALLY FUNCTIONAL")
            logger.info("   Context quality moderate - expect inconsistent results")
        else:
            logger.error("❌ VECTOR STORE NOT READY FOR LLM USAGE")
            logger.info("   Context quality too low for reliable SQL generation")
        
        # Next steps
        logger.info(f"\nNEXT STEPS:")
        if avg_usability_score < 50:
            logger.info("1. Fix vector store content and retrieval")
            logger.info("2. Add comprehensive schema documentation")
            logger.info("3. Include working SQL examples")
        elif avg_usability_score < 70:
            logger.info("1. Improve context retrieval quality")
            logger.info("2. Add missing schema elements")
            logger.info("3. Test with actual LLM integration")
        else:
            logger.info("1. Proceed with LLM integration testing")
            logger.info("2. Fine-tune retrieval parameters if needed")
        
        # Save detailed results
        self._save_detailed_results(results)
    
    def _save_detailed_results(self, results: List[Dict[str, Any]]):
        """Save detailed results to file for further analysis"""
        
        try:
            output_file = f"vector_diagnostic_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Prepare results for JSON serialization
            json_results = []
            for result in results:
                json_result = {}
                for key, value in result.items():
                    if key == 'context_content' and len(str(value)) > 1000:
                        # Truncate very long context for readability
                        json_result[key] = str(value)[:1000] + "... [truncated]"
                    else:
                        json_result[key] = value
                json_results.append(json_result)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'total_queries': len(results),
                    'results': json_results
                }, f, indent=2)
            
            logger.info(f"Detailed results saved to: {output_file}")
            
        except Exception as e:
            logger.warning(f"Could not save detailed results: {e}")

def main():
    """Main diagnostic execution"""
    
    logger.info("🔍 Vector Store Context Diagnostic Tool")
    logger.info("=" * 60)
    
    # Initialize diagnostic tool
    diagnostic = VectorContextDiagnostic()
    
    if not diagnostic.initialize_system():
        logger.error("❌ Failed to initialize system - cannot run diagnostics")
        return
    
    try:
        # Run context analysis
        results = diagnostic.analyze_vector_context_for_queries()
        
        # Generate comprehensive report
        diagnostic.generate_diagnostic_report(results)
        
        logger.info("\n🏁 Diagnostic completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Diagnostic failed: {e}")
        raise

if __name__ == "__main__":
    main()
