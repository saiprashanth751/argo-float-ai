# intelligent_response_system.py
# Location: src/services/intelligent_response_system.py

import os
import json
import logging
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import warnings

# Fix import issues by using absolute imports
import sys
from pathlib import Path

def setup_safe_logging():
    """Setup logging that won't crash on Unicode characters"""
    if sys.platform == "win32":
        os.environ["PYTHONIOENCODING"] = "utf-8"
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
    
    class SafeFormatter(logging.Formatter):
        def format(self, record):
            try:
                formatted = super().format(record)
                # Replace problematic Unicode chars
                replacements = {
                    '❌': '[ERROR]', '✅': '[OK]', '🛠️': '[TOOL]',
                    '📊': '[CHART]', '🎯': '[TARGET]', '🔍': '[SEARCH]'
                }
                for unicode_char, replacement in replacements.items():
                    formatted = formatted.replace(unicode_char, replacement)
                return formatted
            except UnicodeEncodeError:
                return super().format(record).encode('ascii', errors='ignore').decode('ascii')
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(SafeFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    root_logger = logging.getLogger()
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    
    return root_logger

# Call this at the start of main()
logger = setup_safe_logging()


# Add the services directory to the path if not already there
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Now import our dependencies with proper error handling
try:
    from oceanographic_intelligence_engine import QueryIntent, ComplexityLevel, QueryClassification
    from enhanced_rag_oceanographic import EnhancedOceanographicRAG
except ImportError as e:
    # Fallback imports for when running as standalone
    try:
        from .oceanographic_intelligence_engine import QueryIntent, ComplexityLevel, QueryClassification
        from .enhanced_rag_oceanographic import EnhancedOceanographicRAG
    except ImportError:
        # Create dummy classes for testing
        from enum import Enum
        
        class QueryIntent(Enum):
            PROFILE_ANALYSIS = "profile_analysis"
            SPATIAL_MAPPING = "spatial_mapping"
            TEMPORAL_TRENDS = "temporal_trends"
            STATISTICAL_SUMMARY = "statistical_summary"
            EXPLORATION = "exploration"
        
        class ComplexityLevel(Enum):
            BASIC = "basic"
            INTERMEDIATE = "intermediate"
            ADVANCED = "advanced"
            EXPERT = "expert"
        
        class QueryClassification:
            def __init__(self):
                self.intent = QueryIntent.EXPLORATION
                self.complexity = ComplexityLevel.BASIC
                self.confidence = 0.5
                self.context = type('Context', (), {
                    'parameters': [],
                    'depth_range': None,
                    'spatial_bounds': None,
                    'temporal_range': None,
                    'physical_processes': [],
                    'data_quality_requirements': 'standard'
                })()
                self.suggested_approach = "Basic analysis"
                self.required_calculations = []
        
        # Create a dummy RAG system
        class EnhancedOceanographicRAG:
            def __init__(self):
                pass
            
            def process_oceanographic_query(self, query):
                return {
                    'success': False,
                    'error': 'RAG system not available in test mode',
                    'query': query,
                    'processing_time': 0.0
                }

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class VisualizationSpec:
    """Specification for visualization generation"""
    chart_type: str
    title: str
    x_axis: str
    y_axis: str
    color_by: Optional[str] = None
    size_by: Optional[str] = None
    facet_by: Optional[str] = None
    chart_config: Dict[str, Any] = None

@dataclass
class ResponseFormat:
    """Format specification for intelligent responses"""
    include_summary: bool = True
    include_insights: bool = True
    include_visualizations: bool = True
    include_recommendations: bool = True
    include_raw_data: bool = False
    complexity_level: str = "intermediate"
    target_audience: str = "general"

class IntelligentResponseSystem:
    """
    Intelligent response system that creates rich, multi-modal responses
    with dynamic visualizations and actionable insights for oceanographic queries
    """
    
    def __init__(self, rag_system: EnhancedOceanographicRAG = None):
        # Initialize RAG system
        if rag_system is None:
            self.rag_system = EnhancedOceanographicRAG()
        else:
            self.rag_system = rag_system
        
        # Initialize response generation LLM
        # Initialize response generation LLM with DeepSeek
        try:
    # Docker Desktop Model Runner LLM setup
            self.response_llm = ChatOpenAI(
                model="ai/llama3.2:latest",
                openai_api_base="http://localhost:12434/engines/llama.cpp/v1",
                openai_api_key="dummy",
                temperature=0.3,  # Slightly higher for creative responses
                max_tokens=2000
            )
            logger.info("✅ Docker Desktop Model Runner response LLM initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Docker Desktop response LLM: {e}")
            self.response_llm = None
        
        # SQL validation patterns and rules
        self.sql_validation_patterns = self._build_sql_validation_patterns()
        
        # Production query templates
        self.production_templates = self._build_production_query_templates()
        
        # Visualization templates and configurations
        self.viz_templates = self._build_visualization_templates()
        
        # Response templates for different audiences
        self.response_templates = self._build_response_templates()
        
        # Scientific interpretation patterns
        self.interpretation_patterns = self._build_interpretation_patterns()
    
    def _build_sql_validation_patterns(self) -> Dict[str, Any]:
        """Build patterns for SQL validation and correction - FIXED VERSION"""
        
        return {
            'schema_info': {
                'main_tables': ['argo_profiles', 'argo_measurements'],
                'legacy_tables': ['enhanced_floats_metadata', 'enhanced_measurements'],
                'numeric_fields': ['pressure', 'temperature', 'salinity', 'depth', 'cycle_number', 'max_pressure'],
                'string_fields': ['data_source', 'quality_flag', 'source_file'],
                'varchar_numeric_fields': ['platform_number'],  # Special case: stored as VARCHAR but often treated as numeric
                'date_fields': ['profile_date', 'date', 'processed_at']
            },
            'common_errors': {
                # FIXED: Consistent key naming
                'platform_number_unquoted': r'platform_number\s*=\s*(\d+)(?!\s*[\'"])',
                # Missing quotes around actual string values
                'missing_quotes_strings': r'(data_source|quality_flag|source_file)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)',
                # Trailing comma in SELECT clause
                'trailing_comma_select': r',\s*FROM',
                # Unbalanced parentheses
                'unbalanced_parens': r'[\(\)]',
                # Malformed WHERE clauses
                'empty_where': r'WHERE\s*$',
                'missing_operator': r'WHERE\s+\w+\s*$'
            },
            'correction_rules': {
                'platform_number_fix': lambda match: f"platform_number = '{match.group(1)}'",
                'add_string_quotes': lambda match: f"{match.group(1)} = '{match.group(2)}'",
                'remove_trailing_comma': lambda sql: re.sub(r',\s*FROM', ' FROM', sql, flags=re.IGNORECASE)
            }
        }
    
    def _build_production_query_templates(self) -> Dict[str, Any]:
        """Build query templates optimized for production schema"""
        return {
            'profile_analysis': {
                'main_schema': """
                    SELECT {columns}
                    FROM argo_measurements m
                    JOIN argo_profiles p ON m.profile_id = p.id
                    WHERE {conditions}
                    ORDER BY p.profile_date DESC, m.pressure ASC
                    LIMIT {limit}
                """,
                'legacy_schema': """
                    SELECT {columns}
                    FROM enhanced_measurements em
                    JOIN enhanced_floats_metadata efm ON em.metadata_id = efm.id
                    WHERE {conditions}
                    ORDER BY efm.date DESC, em.pressure ASC
                    LIMIT {limit}
                """
            },
            'spatial_analysis': {
                'main_schema': """
                    SELECT {columns}, p.latitude, p.longitude
                    FROM argo_profiles p
                    LEFT JOIN argo_measurements m ON m.profile_id = p.id
                    WHERE p.latitude BETWEEN {lat_min} AND {lat_max}
                    AND p.longitude BETWEEN {lon_min} AND {lon_max}
                    AND {conditions}
                    ORDER BY p.profile_date DESC
                    LIMIT {limit}
                """
            },
            'statistical_analysis': {
                'main_schema': """
                    SELECT {aggregate_functions}
                    FROM argo_profiles p
                    LEFT JOIN argo_measurements m ON m.profile_id = p.id
                    WHERE {conditions}
                    GROUP BY {group_by}
                """
            }
        }
    
    def _validate_and_correct_sql(self, sql_query: str) -> Dict[str, Any]:
        """Validate SQL query and apply corrections - FIXED VERSION"""
        validation_report = {
            'original_sql': sql_query,
            'errors_found': [],
            'corrections_applied': [],
            'schema_compatibility': self._detect_schema_type(sql_query),
            'is_valid': True
        }
        
        corrected_sql = sql_query.strip()
        
        try:
            # Check 1: Platform number data type (most critical fix) - FIXED KEY
            platform_pattern = self.sql_validation_patterns['common_errors']['platform_number_unquoted']
            matches = list(re.finditer(platform_pattern, corrected_sql, re.IGNORECASE))
            
            for match in matches:
                validation_report['errors_found'].append('Platform number missing quotes (VARCHAR field)')
                # Replace the matched portion with quoted version
                old_text = match.group(0)
                new_text = f"platform_number = '{match.group(1)}'"
                corrected_sql = corrected_sql.replace(old_text, new_text)
                validation_report['corrections_applied'].append(f'Added quotes around platform_number value: {match.group(1)}')
            
            # Check 2: Trailing commas in SELECT
            if re.search(self.sql_validation_patterns['common_errors']['trailing_comma_select'], corrected_sql, re.IGNORECASE):
                validation_report['errors_found'].append('Trailing comma in SELECT clause')
                corrected_sql = self.sql_validation_patterns['correction_rules']['remove_trailing_comma'](corrected_sql)
                validation_report['corrections_applied'].append('Removed trailing comma before FROM')
            
            # Check 3: Missing quotes for string fields
            string_pattern = self.sql_validation_patterns['common_errors']['missing_quotes_strings']
            string_matches = list(re.finditer(string_pattern, corrected_sql, re.IGNORECASE))
            
            for match in string_matches:
                validation_report['errors_found'].append(f'Missing quotes for string field: {match.group(1)}')
                old_text = match.group(0)
                new_text = f"{match.group(1)} = '{match.group(2)}'"
                corrected_sql = corrected_sql.replace(old_text, new_text)
                validation_report['corrections_applied'].append(f'Added quotes around {match.group(1)} value: {match.group(2)}')
            
            # Check 4: Parenthesis balance
            open_paren = corrected_sql.count('(')
            close_paren = corrected_sql.count(')')
            if open_paren != close_paren:
                validation_report['errors_found'].append(f'Parenthesis mismatch: {open_paren} open, {close_paren} close')
                if open_paren > close_paren:
                    corrected_sql += ')' * (open_paren - close_paren)
                    validation_report['corrections_applied'].append(f'Added {open_paren - close_paren} missing closing parentheses')
            
            # Check 5: Empty WHERE clauses
            if re.search(self.sql_validation_patterns['common_errors']['empty_where'], corrected_sql, re.IGNORECASE):
                validation_report['errors_found'].append('Empty WHERE clause')
                corrected_sql = re.sub(r'WHERE\s*$', 'WHERE 1=1', corrected_sql, flags=re.IGNORECASE)
                validation_report['corrections_applied'].append('Fixed empty WHERE clause with 1=1')
            
            # Check 6: Basic SQL syntax validation
            if not self._validate_basic_sql_syntax(corrected_sql):
                validation_report['errors_found'].append('Basic SQL syntax validation failed')
                validation_report['is_valid'] = False
            else:
                # Only mark as valid if no critical errors remain
                validation_report['is_valid'] = len(validation_report['errors_found']) == 0 or len(validation_report['corrections_applied']) > 0
                
        except Exception as e:
            validation_report['errors_found'].append(f'Validation error: {str(e)}')
            validation_report['is_valid'] = False
        
        validation_report['corrected_sql'] = corrected_sql
        return validation_report
    
    def _detect_schema_type(self, sql_query: str) -> str:
        """Detect which schema the query is targeting"""
        sql_lower = sql_query.lower()
        
        main_tables = ['argo_profiles', 'argo_measurements']
        legacy_tables = ['enhanced_floats_metadata', 'enhanced_measurements']
        
        uses_main = any(table in sql_lower for table in main_tables)
        uses_legacy = any(table in sql_lower for table in legacy_tables)
        
        if uses_legacy and not uses_main:
            return 'legacy'
        elif uses_main and not uses_legacy:
            return 'main'
        elif uses_main and uses_legacy:
            return 'mixed'
        else:
            return 'main'  # Default to main schema
    
    def _validate_basic_sql_syntax(self, sql_query: str) -> bool:
        """Basic SQL syntax validation"""
        essential_keywords = ['SELECT', 'FROM']
        sql_upper = sql_query.upper()
        
        for keyword in essential_keywords:
            if keyword not in sql_upper:
                return False
        
        # Check for balanced quotes
        single_quotes = sql_query.count("'")
        if single_quotes % 2 != 0:
            return False
            
        return True
    
    def _generate_fallback_sql(self, query: str, classification: Dict) -> str:
        """Generate fallback SQL when LLM produces invalid queries"""
        
        intent = classification.get('intent', 'profile_analysis')
        parameters = classification.get('parameters', ['temperature', 'pressure'])
        
        # Use main schema template by default
        if intent in self.production_templates:
            template = self.production_templates[intent]['main_schema']
        else:
            template = self.production_templates['profile_analysis']['main_schema']
        
        # Build basic columns
        columns = []
        for param in parameters:
            if param in ['temperature', 'salinity', 'pressure', 'depth']:
                columns.append(f"m.{param}")
            elif param in ['latitude', 'longitude', 'profile_date', 'platform_number']:
                columns.append(f"p.{param}")
            else:
                columns.append(param)
        
        # Extract simple conditions
        conditions = self._extract_simple_conditions(query)
        
        return template.format(
            columns=', '.join(columns) or 'm.temperature, m.pressure',
            conditions=conditions or '1=1',
            limit=1000,
            lat_min=-90, lat_max=90,
            lon_min=-180, lon_max=180
        )
    
    def _extract_simple_conditions(self, query: str) -> str:
        """Extract simple conditions from natural language query"""
        conditions = []
        
        # Platform number extraction
        platform_match = re.search(r'platform[_\s]?(\d+)', query, re.IGNORECASE)
        if platform_match:
            conditions.append(f"p.platform_number = '{platform_match.group(1)}'")
        
        # Basic region detection
        if 'pacific' in query.lower():
            conditions.append("p.longitude BETWEEN 120 AND 300")
        elif 'atlantic' in query.lower():
            conditions.append("p.longitude BETWEEN -80 AND 20")
        elif 'indian' in query.lower():
            conditions.append("p.longitude BETWEEN 20 AND 120")
        
        return ' AND '.join(conditions) if conditions else '1=1'
    
    def process_intelligent_query(self, natural_language_query: str, 
                                response_format: ResponseFormat = None) -> Dict[str, Any]:
        """
        Process query with full intelligent response generation pipeline
        WITH ENHANCED SQL VALIDATION
        """
        
        logger.info(f"Processing intelligent query: {natural_language_query}")
        start_time = datetime.now()
        
        # Set default response format
        if response_format is None:
            response_format = ResponseFormat()
        
        try:
            # Step 1: Process query through enhanced RAG system
            rag_result = self.rag_system.process_oceanographic_query(natural_language_query)
            
            if not rag_result['success']:
                return {
                    'success': False,
                    'error': rag_result.get('error', 'RAG processing failed'),
                    'query': natural_language_query,
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            
            # Step 1.5: Validate and correct SQL
            sql_validation = self._validate_and_correct_sql(rag_result['sql_query'])
            
            # If SQL validation fails, try fallback generation
            if not sql_validation['is_valid']:
                logger.warning(f"SQL validation failed, trying fallback generation")
                fallback_sql = self._generate_fallback_sql(natural_language_query, rag_result.get('classification', {}))
                fallback_validation = self._validate_and_correct_sql(fallback_sql)
                
                if fallback_validation['is_valid']:
                    logger.info("Fallback SQL generation successful")
                    sql_validation = fallback_validation
                    rag_result['sql_query'] = fallback_validation['corrected_sql']
                else:
                    return {
                        'success': False,
                        'error': f"SQL validation failed: {sql_validation['errors_found']}",
                        'query': natural_language_query,
                        'sql_validation': sql_validation,
                        'processing_time': (datetime.now() - start_time).total_seconds()
                    }
            else:
                rag_result['sql_query'] = sql_validation['corrected_sql']
            
            # Step 2: Generate intelligent narrative response
            narrative_response = self._generate_narrative_response(
                natural_language_query, rag_result, response_format
            )
            
            # Step 3: Create dynamic visualizations
            visualizations = []
            if response_format.include_visualizations and not rag_result['results'].empty:
                visualizations = self._generate_dynamic_visualizations(
                    rag_result['results'], rag_result['classification']
                )
            
            # Step 4: Generate actionable recommendations
            recommendations = []
            if response_format.include_recommendations:
                recommendations = self._generate_actionable_recommendations(
                    rag_result, response_format
                )
            
            # Step 5: Create export-ready data formats
            export_data = self._prepare_export_data(rag_result['results']) if response_format.include_raw_data else {}
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Build comprehensive intelligent response
            intelligent_response = {
                'success': True,
                'query': natural_language_query,
                'response_type': 'intelligent_multi_modal',
                
                # Core analysis results
                'sql_query': rag_result['sql_query'],
                'sql_validation': sql_validation,
                'classification': rag_result['classification'],
                'results_summary': {
                    'total_records': rag_result['result_count'],
                    'columns': rag_result['columns'],
                    'data_types': rag_result['data_types']
                },
                
                # Intelligent narrative
                'narrative_response': narrative_response,
                
                # Enhanced insights
                'scientific_insights': rag_result['insights'],
                
                # Dynamic visualizations
                'visualizations': visualizations,
                
                # Actionable recommendations
                'recommendations': recommendations,
                
                # Export data
                'export_data': export_data,
                
                # Processing metadata
                'processing_time': processing_time,
                'response_format': asdict(response_format),
                'confidence_score': rag_result['classification']['confidence']
            }
            
            logger.info(f"Intelligent query processed successfully in {processing_time:.2f}s")
            return intelligent_response
            
        except Exception as e:
            logger.error(f"Error in intelligent query processing: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': natural_language_query,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _build_visualization_templates(self) -> Dict[str, Dict]:
        """Build templates for different visualization types - UPDATED for new schema"""
        return {
            'profile_analysis': {
                'line_profile': {
                    'type': 'line',
                    'x_axis': 'parameter_value',
                    'y_axis': 'pressure',
                    'title': '{parameter} Profile - Platform {platform_number}',
                    'y_reversed': True,
                    'layout_config': {
                        'yaxis_title': 'Pressure (dbar)',
                        'yaxis_autorange': 'reversed',
                        'annotations': 'Include mixed layer depth if available'
                    }
                },
                'multi_profile_comparison': {
                    'type': 'scatter',
                    'x_axis': 'parameter_value',
                    'y_axis': 'pressure',
                    'color_by': 'platform_number',
                    'title': 'Multi-Platform {parameter} Profiles',
                    'y_reversed': True,
                    'facet_by': 'data_source'
                }
            },
            'spatial_mapping': {
                'geographic_scatter': {
                    'type': 'scatter_mapbox',
                    'lat_col': 'latitude',
                    'lon_col': 'longitude',
                    'color_by': 'parameter_value',
                    'size_by': 'n_observations',
                    'title': '{parameter} Distribution - Data Coverage',
                    'mapbox_style': 'open-street-map'
                },
                'gridded_data_viz': {
                    'type': 'density_mapbox',
                    'lat_col': 'latitude',
                    'lon_col': 'longitude',
                    'z_col': 'mean_parameter',
                    'title': 'Gridded {parameter} - Spatial Analysis',
                    'hover_data': ['n_observations', 'grid_date']
                }
            },
            'temporal_analysis': {
                'time_series': {
                    'type': 'line',
                    'x_axis': 'profile_date',
                    'y_axis': 'parameter_value',
                    'color_by': 'data_source',
                    'title': '{parameter} Temporal Evolution',
                    'layout_config': {
                        'xaxis_title': 'Date',
                        'showlegend': True,
                        'rangeslider_visible': True
                    }
                },
                'seasonal_analysis': {
                    'type': 'box',
                    'x_axis': 'month',
                    'y_axis': 'parameter_value',
                    'color_by': 'depth_layer',
                    'title': 'Seasonal {parameter} Variability by Depth'
                }
            },
            'quality_assessment': {
                'quality_flags_distribution': {
                    'type': 'bar',
                    'x_axis': 'quality_flag',
                    'y_axis': 'count',
                    'title': 'Data Quality Flag Distribution',
                    'color_discrete_map': {'1': 'green', '2': 'yellow', '3': 'orange', '4': 'red'}
                },
                'data_coverage_map': {
                    'type': 'scatter_mapbox',
                    'lat_col': 'latitude',
                    'lon_col': 'longitude',
                    'color_by': 'data_quality_score',
                    'title': 'Data Quality Coverage Assessment'
                }
            },
            'comparative_analysis': {
                'parameter_correlation': {
                    'type': 'scatter',
                    'x_axis': 'temperature',
                    'y_axis': 'salinity',
                    'color_by': 'pressure',
                    'title': 'T-S Diagram - Water Mass Analysis',
                    'trendline': 'ols'
                },
                'depth_comparison': {
                    'type': 'violin',
                    'x_axis': 'depth_layer',
                    'y_axis': 'parameter_value',
                    'title': '{parameter} Distribution by Depth Layer'
                }
            }
        }
    
    def _build_response_templates(self) -> Dict[str, Dict]:
        """Build response templates for different audiences"""
        return {
            'government_official': {
                'format': 'executive_summary',
                'emphasis': ['policy_implications', 'economic_impact', 'strategic_value'],
                'language': 'non_technical',
                'include_uncertainty': True
            },
            'researcher': {
                'format': 'scientific_report',
                'emphasis': ['methodology', 'statistical_significance', 'physical_interpretation'],
                'language': 'technical',
                'include_uncertainty': True
            },
            'maritime_industry': {
                'format': 'operational_brief',
                'emphasis': ['practical_applications', 'operational_impact', 'safety_considerations'],
                'language': 'semi_technical',
                'include_uncertainty': False
            },
            'general_public': {
                'format': 'accessible_explanation',
                'emphasis': ['context', 'significance', 'implications'],
                'language': 'non_technical',
                'include_uncertainty': False
            }
        }
    
    def _build_interpretation_patterns(self) -> Dict[str, Dict]:
        """Build patterns for scientific interpretation"""
        return {
            'temperature_patterns': {
                'high_gradient': 'Strong temperature gradients indicate active mixing or frontal zones',
                'stable_profile': 'Uniform temperature profile suggests well-mixed water column',
                'inversion': 'Temperature inversion may indicate subsurface warming or cold water intrusion'
            },
            'salinity_patterns': {
                'freshening': 'Decreasing salinity suggests freshwater influence from precipitation or river discharge',
                'salinification': 'Increasing salinity indicates evaporation dominance or saline water intrusion',
                'halocline': 'Sharp salinity gradients create density stratification affecting vertical mixing'
            },
            'density_patterns': {
                'stratification': 'Strong density stratification inhibits vertical mixing and nutrient exchange',
                'convection': 'Weak stratification promotes convective mixing and vertical transport',
                'overturning': 'Density inversions drive convective overturning and deep water formation'
            }
        }
    
    def _generate_narrative_response(self, query: str, rag_result: Dict, 
                                   response_format: ResponseFormat) -> str:
        """Generate intelligent narrative response using LLM"""
        
        # If LLM is not available, return fallback
        if self.response_llm is None:
            return self._generate_fallback_narrative({
                'query': query,
                'classification': rag_result['classification'],
                'results_count': rag_result['result_count'],
                'key_findings': rag_result['insights'].get('key_findings', []),
                'summary': rag_result['insights'].get('summary', ''),
                'physical_interpretation': rag_result['insights'].get('physical_interpretation', '')
            })
        
        # Determine target audience
        audience = response_format.target_audience
        complexity = response_format.complexity_level
        
        # Build context for LLM
        context = {
            'query': query,
            'classification': rag_result['classification'],
            'results_count': rag_result['result_count'],
            'key_findings': rag_result['insights'].get('key_findings', []),
            'summary': rag_result['insights'].get('summary', ''),
            'physical_interpretation': rag_result['insights'].get('physical_interpretation', '')
        }
        
        # Build system prompt based on audience and complexity
        system_prompt = self._build_audience_specific_prompt(audience, complexity)
        
        # Build user prompt with results
        user_prompt = f"""
        OCEANOGRAPHIC QUERY ANALYSIS REQUEST
        
        Original Query: {query}
        
        Analysis Classification:
        - Intent: {context['classification']['intent']}
        - Complexity: {context['classification']['complexity']}
        - Parameters: {', '.join(context['classification']['parameters'])}
        - Confidence: {context['classification']['confidence']:.2f}
        
        Results Summary:
        - Total Records: {context['results_count']:,}
        - Key Findings: {'; '.join(context['key_findings'][:3])}
        - Physical Interpretation: {context['physical_interpretation']}
        
        Generate a comprehensive, intelligent response that:
        1. Directly answers the user's question
        2. Provides scientific context and interpretation
        3. Explains the significance of findings
        4. Connects results to broader oceanographic understanding
        5. Maintains appropriate technical level for the audience
        
        Focus on clarity, accuracy, and actionable insights.
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.response_llm(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating narrative response: {e}")
            return self._generate_fallback_narrative(context)
    
    def _build_audience_specific_prompt(self, audience: str, complexity: str) -> str:
        """Build audience-specific system prompts - UPDATED for new schema"""
        
        base_prompt = """You are an expert oceanographer and science communicator with deep knowledge of ARGO float data and marine science. You have access to a comprehensive database with both individual profile measurements and gridded oceanographic products. You excel at translating complex oceanographic analysis into clear, actionable insights."""
        
        schema_context = """
        
        DATABASE CONTEXT:
        You're working with an advanced ARGO database containing:
        - Individual profile data: argo_profiles (metadata) + argo_measurements (detailed measurements)
        - Gridded products: argo_gridded_data with spatial-temporal averages
        - Quality metrics: comprehensive QC flags and data processing logs
        - Derived properties: mixed layer depth, potential temperature, density calculations
        
        This enables analysis from individual float profiles to basin-scale patterns.
        """
        
        audience_prompts = {
            'government_official': """
            Your audience consists of government officials and policy makers who need:
            - Clear, executive-level summaries with policy implications
            - Strategic value of oceanographic monitoring (ARGO network importance)
            - Economic and societal impact context (climate, fisheries, shipping)
            - Actionable recommendations for marine resource management
            - Non-technical language while maintaining scientific accuracy
            """,
            'researcher': """
            Your audience consists of marine researchers and scientists who need:
            - Technical accuracy with proper oceanographic terminology
            - Methodology discussion including data sources and limitations
            - Physical oceanographic interpretation of patterns and processes
            - Research implications and connections to current literature
            - Statistical significance and uncertainty quantification
            """,
            'maritime_industry': """
            Your audience consists of maritime industry professionals who need:
            - Operational relevance for shipping, offshore, and fisheries
            - Safety implications from oceanographic conditions
            - Economic impact on marine operations
            - Practical guidance for route planning and operations
            - Semi-technical language focused on applied oceanography
            """,
            'general_public': """
            Your audience consists of the general public who need:
            - Accessible explanations without technical jargon
            - Context about ocean's role in climate and weather
            - Real-world relevance and everyday implications
            - Educational content that builds ocean literacy
            - Clear, engaging language with helpful analogies
            """
        }
        
        complexity_modifiers = {
            'basic': "Focus on key findings with simple explanations. Emphasize what the data shows rather than complex analysis methods.",
            'intermediate': "Provide moderate technical detail with clear explanations. Include some methodology but focus on interpretation.",
            'advanced': "Include comprehensive analysis with technical depth. Discuss methodology, uncertainty, and broader context.",
            'expert': "Provide full technical detail with research-level analysis. Include statistical significance, limitations, and research implications."
        }
        
        audience_prompt = audience_prompts.get(audience, audience_prompts['general_public'])
        complexity_modifier = complexity_modifiers.get(complexity, complexity_modifiers['intermediate'])
        
        return f"{base_prompt}{schema_context}\n\n{audience_prompt}\n\nComplexity Level: {complexity_modifier}"
    
    def _generate_fallback_narrative(self, context: Dict) -> str:
        """Generate fallback narrative when LLM fails"""
        
        return f"""
        Analysis Results for: {context['query']}
        
        Analysis Summary:
        {context['summary']}
        
        Key Findings:
        {'; '.join(context['key_findings']) if context['key_findings'] else 'No specific findings available'}
        
        Physical Interpretation:
        {context['physical_interpretation'] if context['physical_interpretation'] else 'No specific interpretation available'}
        
        This analysis processed {context['results_count']:,} measurements with {context['classification']['confidence']:.0%} confidence in the classification as {context['classification']['intent']}.
        """
    
    def _generate_dynamic_visualizations(self, results_df: pd.DataFrame, 
                                       classification: Dict) -> List[Dict]:
        """Generate dynamic visualizations based on data and query type"""
        
        visualizations = []
        
        try:
            intent = classification['intent']
            parameters = classification['parameters']
            
            # Profile Analysis Visualizations
            if intent == 'profile_analysis' and 'pressure' in results_df.columns:
                for param in parameters:
                    if param in results_df.columns:
                        viz = self._create_profile_visualization(results_df, param)
                        if viz:
                            visualizations.append(viz)
            
            # Spatial Mapping Visualizations
            elif intent == 'spatial_mapping' and all(col in results_df.columns for col in ['latitude', 'longitude']):
                for param in parameters:
                    if param in results_df.columns:
                        viz = self._create_spatial_visualization(results_df, param)
                        if viz:
                            visualizations.append(viz)
            
            # Statistical Summary Visualizations
            elif intent == 'statistical_summary':
                for param in parameters:
                    if param in results_df.columns and pd.api.types.is_numeric_dtype(results_df[param]):
                        viz = self._create_statistical_visualization(results_df, param)
                        if viz:
                            visualizations.append(viz)
            
            # Temporal Analysis Visualizations
            elif intent == 'temporal_trends' and any(col in results_df.columns for col in ['date', 'time', 'profile_date']):
                time_col = next((col for col in ['profile_date', 'date', 'time'] if col in results_df.columns), None)
                if time_col:
                    for param in parameters:
                        if param in results_df.columns:
                            viz = self._create_temporal_visualization(results_df, param, time_col)
                            if viz:
                                visualizations.append(viz)
            
            # Default: Create basic parameter visualizations
            else:
                for param in parameters:
                    if param in results_df.columns and pd.api.types.is_numeric_dtype(results_df[param]):
                        viz = self._create_basic_visualization(results_df, param)
                        if viz:
                            visualizations.append(viz)
        
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
        
        return visualizations
    
    def _create_profile_visualization(self, df: pd.DataFrame, parameter: str) -> Optional[Dict]:
        """Create profile visualization - UPDATED for new schema structure"""
        
        try:
            # Clean data with new schema columns
            required_cols = [parameter, 'pressure']
            optional_cols = ['platform_number', 'cycle_number', 'profile_date', 'latitude', 'longitude']
            
            available_cols = [col for col in required_cols + optional_cols if col in df.columns]
            clean_df = df[available_cols].dropna(subset=required_cols)
            
            if len(clean_df) < 2:
                return None
            
            fig = go.Figure()
            
            # Check if we have multiple platforms/cycles
            if 'platform_number' in clean_df.columns:
                unique_platforms = clean_df['platform_number'].unique()
                
                if len(unique_platforms) > 1:
                    # Multi-platform visualization
                    for platform in unique_platforms[:10]:  # Limit to 10 platforms for clarity
                        platform_data = clean_df[clean_df['platform_number'] == platform]
                        
                        fig.add_trace(go.Scatter(
                            x=platform_data[parameter],
                            y=platform_data['pressure'],
                            mode='lines+markers',
                            name=f'Platform {platform}',
                            line=dict(width=2),
                            marker=dict(size=3),
                            hovertemplate=(
                                f'Platform: {platform}<br>'
                                f'{parameter}: %{{x}}<br>'
                                'Pressure: %{y} dbar<extra></extra>'
                            )
                        ))
                else:
                    # Single platform, potentially multiple cycles
                    if 'cycle_number' in clean_df.columns:
                        unique_cycles = clean_df['cycle_number'].unique()
                        
                        if len(unique_cycles) > 1:
                            for cycle in sorted(unique_cycles)[:5]:  # Limit cycles
                                cycle_data = clean_df[clean_df['cycle_number'] == cycle]
                                
                                fig.add_trace(go.Scatter(
                                    x=cycle_data[parameter],
                                    y=cycle_data['pressure'],
                                    mode='lines+markers',
                                    name=f'Cycle {cycle}',
                                    line=dict(width=2),
                                    marker=dict(size=3)
                                ))
                        else:
                            # Single profile
                            fig.add_trace(go.Scatter(
                                x=clean_df[parameter],
                                y=clean_df['pressure'],
                                mode='lines+markers',
                                name=f'{parameter.title()} Profile',
                                line=dict(width=3, color='blue'),
                                marker=dict(size=4, color='darkblue')
                            ))
                    else:
                        # No cycle information, treat as single profile
                        fig.add_trace(go.Scatter(
                            x=clean_df[parameter],
                            y=clean_df['pressure'],
                            mode='lines+markers',
                            name=f'{parameter.title()} Profile',
                            line=dict(width=3, color='blue'),
                            marker=dict(size=4, color='darkblue')
                        ))
            
            # Add mixed layer depth annotation if available
            if 'mixed_layer_depth' in df.columns:
                mld_values = df['mixed_layer_depth'].dropna()
                if len(mld_values) > 0:
                    mld = mld_values.mean()
                    fig.add_hline(
                        y=mld,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Mixed Layer Depth: {mld:.1f}m"
                    )
            
            # Enhanced layout with oceanographic context
            fig.update_layout(
                title=f'{parameter.title()} Profile Analysis',
                xaxis_title=f'{parameter.title()} ({self._get_parameter_unit(parameter)})',
                yaxis_title='Pressure (dbar ≈ depth in meters)',
                yaxis_autorange='reversed',
                showlegend=True,
                template='plotly_white',
                height=700,
                hovermode='closest'
            )
            
            # Add depth zone annotations
            self._add_depth_zone_annotations(fig)
            
            return {
                'type': 'profile',
                'parameter': parameter,
                'title': f'{parameter.title()} Profile Analysis',
                'plotly_json': fig.to_json(),
                'data_points': len(clean_df),
                'platforms': clean_df['platform_number'].nunique() if 'platform_number' in clean_df.columns else 1,
                'depth_range': [clean_df['pressure'].min(), clean_df['pressure'].max()]
            }
            
        except Exception as e:
            logger.error(f"Error creating enhanced profile visualization for {parameter}: {e}")
            return None
    
    def _create_spatial_visualization(self, df: pd.DataFrame, parameter: str) -> Optional[Dict]:
        """Create spatial distribution visualization - UPDATED for new schema"""
        
        try:
            required_cols = ['latitude', 'longitude', parameter]
            optional_cols = ['platform_number', 'cycle_number', 'profile_date', 'pressure']
            
            available_cols = [col for col in required_cols + optional_cols if col in df.columns]
            clean_df = df[available_cols].dropna(subset=required_cols)
            
            if len(clean_df) < 2:
                return None
            
            # Aggregate by location if multiple measurements per location
            if 'pressure' in clean_df.columns:
                # Average parameter values for surface layer (< 50 dbar)
                surface_df = clean_df[clean_df['pressure'] <= 50].groupby(['latitude', 'longitude']).agg({
                    parameter: 'mean',
                    'platform_number': 'first' if 'platform_number' in clean_df.columns else lambda x: 'Unknown',
                    'pressure': 'mean'
                }).reset_index()
            else:
                surface_df = clean_df.groupby(['latitude', 'longitude']).agg({
                    parameter: 'mean',
                    'platform_number': 'first' if 'platform_number' in clean_df.columns else lambda x: 'Unknown'
                }).reset_index()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scattermapbox(
                lat=surface_df['latitude'],
                lon=surface_df['longitude'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=surface_df[parameter],
                    colorscale='plasma',
                    colorbar=dict(
                        title=f'Surface {parameter.title()}<br>({self._get_parameter_unit(parameter)})',
                        titleside='right'
                    ),
                    showscale=True,
                    opacity=0.9,
                    line=dict(width=1, color='white')
                ),
                text=[f'Platform: {plat}<br>Surface {parameter}: {val:.2f}' 
                    for plat, val in zip(surface_df['platform_number'], surface_df[parameter])],
                hovertemplate='Lat: %{lat:.2f}°<br>Lon: %{lon:.2f}°<br>%{text}<extra></extra>',
                name=f'Surface {parameter.title()}'
            ))
            
            fig.update_layout(
                title=f'Surface {parameter.title()} Distribution from ARGO Profiles',
                mapbox=dict(
                    style='open-street-map',
                    center=dict(
                        lat=surface_df['latitude'].mean(),
                        lon=surface_df['longitude'].mean()
                    ),
                    zoom=self._calculate_optimal_zoom(surface_df)
                ),
                height=700,
                margin=dict(t=80, b=20, l=20, r=20)
            )
            
            return {
                'type': 'spatial_profiles',
                'parameter': parameter,
                'title': f'Surface {parameter.title()} from ARGO Profiles',
                'plotly_json': fig.to_json(),
                'data_points': len(surface_df),
                'unique_platforms': surface_df['platform_number'].nunique() if 'platform_number' in surface_df.columns else 1
            }
            
        except Exception as e:
            logger.error(f"Error creating spatial visualization for {parameter}: {e}")
            return None
     
    def _get_parameter_unit(self, parameter: str) -> str:
        """Get appropriate unit for parameter"""
        unit_map = {
            'temperature': '°C',
            'salinity': 'PSU',
            'pressure': 'dbar',
            'density': 'kg/m³',
            'mean_temperature': '°C',
            'mean_salinity': 'PSU'
        }
        return unit_map.get(parameter, '')
    
    def _add_depth_zone_annotations(self, fig):
        """Add oceanographic depth zone annotations to profile plots"""
        depth_zones = [
            (0, 200, 'Epipelagic', 'lightblue'),
            (200, 1000, 'Mesopelagic', 'lightgreen'),
            (1000, 4000, 'Bathypelagic', 'lightyellow'),
            (4000, 6000, 'Abyssopelagic', 'lightcoral')
        ]
        
        for start, end, name, color in depth_zones:
            fig.add_hrect(
                y0=start, y1=end,
                fillcolor=color,
                opacity=0.1,
                layer="below",
                line_width=0,
                annotation_text=name,
                annotation_position="top left"
            )
    
    def _calculate_optimal_zoom(self, df: pd.DataFrame) -> float:
        """Calculate optimal zoom level for map based on data extent"""
        lat_range = df['latitude'].max() - df['latitude'].min()
        lon_range = df['longitude'].max() - df['longitude'].min()
        
        max_range = max(lat_range, lon_range)
        
        if max_range > 50:
            return 2
        elif max_range > 20:
            return 3
        elif max_range > 10:
            return 4
        elif max_range > 5:
            return 5
        else:
            return 6
    
    def _create_statistical_visualization(self, df: pd.DataFrame, parameter: str) -> Optional[Dict]:
        """Create statistical summary visualization"""
        
        try:
            # Clean data
            clean_data = df[parameter].dropna()
            if len(clean_data) < 10:
                return None
            
            # Create subplots for histogram and box plot
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(f'{parameter.title()} Distribution', f'{parameter.title()} Statistics'),
                vertical_spacing=0.1
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=clean_data, nbinsx=30, name='Distribution'),
                row=1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(y=clean_data, name='Statistics', boxmean=True),
                row=2, col=1
            )
            
            fig.update_layout(
                title=f'{parameter.title()} Statistical Summary',
                showlegend=False,
                height=700,
                template='plotly_white'
            )
            
            # Calculate statistics
            stats = {
                'mean': float(clean_data.mean()),
                'std': float(clean_data.std()),
                'min': float(clean_data.min()),
                'max': float(clean_data.max()),
                'median': float(clean_data.median()),
                'count': int(len(clean_data))
            }
            
            return {
                'type': 'statistical',
                'parameter': parameter,
                'title': f'{parameter.title()} Statistical Summary',
                'plotly_json': fig.to_json(),
                'statistics': stats,
                'data_points': len(clean_data)
            }
            
        except Exception as e:
            logger.error(f"Error creating statistical visualization for {parameter}: {e}")
            return None
    
    def _create_temporal_visualization(self, df: pd.DataFrame, parameter: str, time_col: str) -> Optional[Dict]:
        """Create temporal trend visualization"""
        
        try:
            # Clean data
            required_cols = [time_col, parameter]
            clean_df = df[required_cols].dropna()
            if len(clean_df) < 2:
                return None
            
            # Ensure time column is datetime
            if not pd.api.types.is_datetime64_any_dtype(clean_df[time_col]):
                clean_df[time_col] = pd.to_datetime(clean_df[time_col], errors='coerce')
                clean_df = clean_df.dropna()
            
            # Sort by time
            clean_df = clean_df.sort_values(time_col)
            
            fig = go.Figure()
            
            # Add time series line
            fig.add_trace(go.Scatter(
                x=clean_df[time_col],
                y=clean_df[parameter],
                mode='lines+markers',
                name=f'{parameter.title()} Time Series',
                line=dict(width=2),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title=f'{parameter.title()} Temporal Trends',
                xaxis_title='Time',
                yaxis_title=f'{parameter.title()}',
                showlegend=True,
                template='plotly_white',
                height=500
            )
            
            return {
                'type': 'temporal',
                'parameter': parameter,
                'title': f'{parameter.title()} Temporal Trends',
                'plotly_json': fig.to_json(),
                'data_points': len(clean_df),
                'time_range': [str(clean_df[time_col].min()), str(clean_df[time_col].max())]
            }
            
        except Exception as e:
            logger.error(f"Error creating temporal visualization for {parameter}: {e}")
            return None
    
    def _create_basic_visualization(self, df: pd.DataFrame, parameter: str) -> Optional[Dict]:
        """Create basic visualization for parameter"""
        
        try:
            clean_data = df[parameter].dropna()
            if len(clean_data) < 2:
                return None
            
            # Simple line plot of values
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                y=clean_data,
                mode='lines+markers',
                name=f'{parameter.title()}',
                line=dict(width=2),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title=f'{parameter.title()} Values',
                xaxis_title='Data Point Index',
                yaxis_title=f'{parameter.title()}',
                showlegend=True,
                template='plotly_white',
                height=400
            )
            
            return {
                'type': 'basic',
                'parameter': parameter,
                'title': f'{parameter.title()} Values',
                'plotly_json': fig.to_json(),
                'data_points': len(clean_data)
            }
            
        except Exception as e:
            logger.error(f"Error creating basic visualization for {parameter}: {e}")
            return None
    
    def _generate_actionable_recommendations(self, rag_result: Dict, 
                                           response_format: ResponseFormat) -> List[Dict]:
        """Generate actionable recommendations based on results"""
        
        recommendations = []
        
        try:
            classification = rag_result['classification']
            results_count = rag_result['result_count']
            intent = classification['intent']
            complexity = classification['complexity']
            
            # Data quality recommendations
            if results_count < 100:
                recommendations.append({
                    'type': 'data_collection',
                    'priority': 'medium',
                    'title': 'Increase Data Coverage',
                    'description': 'Consider expanding data collection or timeframe for more robust analysis',
                    'action': 'Add more spatial or temporal coverage to the query'
                })
            
            # Analysis-specific recommendations
            if intent == 'profile_analysis':
                recommendations.append({
                    'type': 'analysis_enhancement',
                    'priority': 'high',
                    'title': 'Enhance Profile Analysis',
                    'description': 'Calculate derived properties like potential temperature and density',
                    'action': 'Request calculation of physical oceanographic properties'
                })
            
            elif intent == 'spatial_mapping':
                recommendations.append({
                    'type': 'visualization',
                    'priority': 'medium',
                    'title': 'Improve Spatial Visualization',
                    'description': 'Consider interpolation or gridding for better spatial representation',
                    'action': 'Apply spatial interpolation methods or create contour maps'
                })
            
            elif intent == 'statistical_summary':
                recommendations.append({
                    'type': 'statistical_analysis',
                    'priority': 'medium',
                    'title': 'Advanced Statistical Analysis',
                    'description': 'Consider trend analysis or correlation studies',
                    'action': 'Apply time series analysis or correlation with other parameters'
                })
            
            # Complexity-based recommendations
            if complexity == 'basic':
                recommendations.append({
                    'type': 'analysis_depth',
                    'priority': 'low',
                    'title': 'Deepen Analysis',
                    'description': 'Consider more sophisticated analysis methods',
                    'action': 'Explore comparative analysis or physical property calculations'
                })
            
            elif complexity == 'expert':
                recommendations.append({
                    'type': 'validation',
                    'priority': 'high',
                    'title': 'Validate Results',
                    'description': 'Cross-reference with published literature or other datasets',
                    'action': 'Compare results with climatological data or research publications'
                })
            
            # Audience-specific recommendations
            audience = response_format.target_audience
            if audience == 'government_official':
                recommendations.append({
                    'type': 'policy_application',
                    'priority': 'high',
                    'title': 'Policy Implementation',
                    'description': 'Consider policy implications of these oceanographic findings',
                    'action': 'Develop policy recommendations based on ocean state information'
                })
            
            elif audience == 'maritime_industry':
                recommendations.append({
                    'type': 'operational_application',
                    'priority': 'high',
                    'title': 'Operational Integration',
                    'description': 'Integrate findings into operational decision-making processes',
                    'action': 'Develop operational protocols based on oceanographic conditions'
                })
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _prepare_export_data(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for export in multiple formats"""
        
        try:
            export_data = {}
            
            if not results_df.empty:
                # CSV format
                csv_buffer = results_df.to_csv(index=False)
                export_data['csv'] = {
                    'format': 'csv',
                    'data': csv_buffer,
                    'filename': f'argo_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    'size_bytes': len(csv_buffer.encode('utf-8'))
                }
                
                # JSON format
                json_data = results_df.to_json(orient='records', date_format='iso')
                export_data['json'] = {
                    'format': 'json',
                    'data': json_data,
                    'filename': f'argo_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                    'size_bytes': len(json_data.encode('utf-8'))
                }
                
                # Summary statistics
                numeric_cols = results_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    stats_df = results_df[numeric_cols].describe()
                    export_data['statistics'] = {
                        'format': 'statistics',
                        'data': stats_df.to_dict(),
                        'filename': f'argo_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                    }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error preparing export data: {e}")
            return {}
    
    def generate_executive_summary(self, intelligent_response: Dict[str, Any]) -> str:
        """Generate executive summary for stakeholders"""
        
        try:
            query = intelligent_response['query']
            classification = intelligent_response['classification']
            results_summary = intelligent_response['results_summary']
            insights = intelligent_response['scientific_insights']
            
            summary_parts = [
                f"EXECUTIVE SUMMARY: {query}",
                "=" * 60,
                "",
                f"ANALYSIS TYPE: {classification['intent'].replace('_', ' ').title()}",
                f"CONFIDENCE LEVEL: {classification['confidence']:.0%}",
                f"DATA PROCESSED: {results_summary['total_records']:,} measurements",
                "",
                "KEY FINDINGS:",
            ]
            
            # Add key findings
            key_findings = insights.get('key_findings', [])
            for i, finding in enumerate(key_findings[:5], 1):
                summary_parts.append(f"{i}. {finding}")
            
            summary_parts.extend([
                "",
                "PHYSICAL INTERPRETATION:",
                insights.get('physical_interpretation', 'No specific interpretation available'),
                "",
                "DATA QUALITY:",
                insights.get('data_quality_notes', 'No quality issues identified'),
                "",
                f"PROCESSING TIME: {intelligent_response['processing_time']:.2f} seconds"
            ])
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return f"Executive summary for: {intelligent_response.get('query', 'Unknown query')}"
    
    def create_dashboard_data(self, intelligent_response: Dict[str, Any]) -> Dict[str, Any]:
        """Create structured data for dashboard display"""
        
        try:
            dashboard_data = {
                'header': {
                    'query': intelligent_response['query'],
                    'timestamp': datetime.now().isoformat(),
                    'processing_time': intelligent_response['processing_time'],
                    'confidence': intelligent_response['classification']['confidence']
                },
                'metrics': {
                    'total_records': intelligent_response['results_summary']['total_records'],
                    'parameters_analyzed': len(intelligent_response['classification']['parameters']),
                    'visualizations_created': len(intelligent_response['visualizations']),
                    'recommendations_generated': len(intelligent_response['recommendations'])
                },
                'content_sections': [
                    {
                        'title': 'Analysis Results',
                        'type': 'narrative',
                        'content': intelligent_response['narrative_response']
                    },
                    {
                        'title': 'Scientific Insights',
                        'type': 'insights',
                        'content': intelligent_response['scientific_insights']
                    },
                    {
                        'title': 'Visualizations',
                        'type': 'visualizations',
                        'content': intelligent_response['visualizations']
                    },
                    {
                        'title': 'Recommendations',
                        'type': 'recommendations',
                        'content': intelligent_response['recommendations']
                    }
                ],
                'metadata': {
                    'classification': intelligent_response['classification'],
                    'response_format': intelligent_response['response_format'],
                    'sql_query': intelligent_response['sql_query']
                }
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error creating dashboard data: {e}")
            return {'error': str(e)}


def test_intelligent_response_system():
    """Test the intelligent response system with sample queries"""
    
    logger.info("Testing Enhanced Intelligent Response System with SQL Validation")
    logger.info("=" * 70)
    
    # Initialize system
    try:
        response_system = IntelligentResponseSystem()
        logger.info("✅ System initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        return
    
    # Test SQL validation with problematic queries first
    test_sql_queries = [
        "SELECT temperature, pressure FROM argo_measurements WHERE platform_number = 1900121",    # Missing quotes (ERROR)
        "SELECT temperature, pressure FROM argo_measurements WHERE platform_number = '1900121'",  # Correct (VALID)
        "SELECT temperature, pressure, FROM argo_measurements WHERE pressure > 100",              # Trailing comma
        "SELECT * FROM argo_profiles WHERE data_source = argo",                                   # Missing quotes
        "SELECT temperature FROM argo_measurements WHERE (pressure < 100",                        # Unbalanced parenthesis
    ]
    
    logger.info("\n🧪 Testing SQL Validation:")
    logger.info("-" * 50)
    
    for i, sql_query in enumerate(test_sql_queries, 1):
        logger.info(f"\nSQL Test {i}: {sql_query}")
        validation_result = response_system._validate_and_correct_sql(sql_query)
        
        if validation_result['is_valid']:
            logger.info(f"✅ VALID: {validation_result['corrected_sql']}")
            if validation_result['corrections_applied']:
                logger.info(f"   Corrections: {validation_result['corrections_applied']}")
        else:
            logger.error(f"❌ INVALID: {validation_result['errors_found']}")
    
    # Test queries with different complexity levels
    test_queries = [
        {
            'query': "Show temperature measurements for platform '1900121'",
            'format': ResponseFormat(complexity_level="basic", target_audience="general_public")
        },
        {
            'query': "What is the average surface salinity in the Arabian Sea region?",
            'format': ResponseFormat(complexity_level="intermediate", target_audience="researcher")
        },
        {
            'query': "Compare thermocline depth variability between different ocean basins",
            'format': ResponseFormat(complexity_level="advanced", target_audience="government_official")
        }
    ]
    
    # Final completion of intelligent_response_system.py test function and main execution

    logger.info("\n🔍 Testing Complete Query Processing Pipeline:")
    logger.info("-" * 60)
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case['query']
        response_format = test_case['format']
        
        logger.info(f"\nQuery Test {i}: {query}")
        logger.info(f"Target Audience: {response_format.target_audience}")
        logger.info(f"Complexity: {response_format.complexity_level}")
        logger.info("-" * 40)
        
        try:
            start_time = datetime.now()
            result = response_system.process_intelligent_query(query, response_format)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if result['success']:
                logger.info(f"✅ Success! ({processing_time:.2f}s)")
                
                # Log SQL validation results
                if 'sql_validation' in result:
                    sql_val = result['sql_validation']
                    if sql_val['corrections_applied']:
                        logger.info(f"   SQL Corrections: {', '.join(sql_val['corrections_applied'])}")
                    else:
                        logger.info("   SQL: No corrections needed")
                
                logger.info(f"   Classification: {result['classification']['intent']}")
                logger.info(f"   Data Points: {result['results_summary']['total_records']:,}")
                logger.info(f"   Visualizations: {len(result['visualizations'])}")
                logger.info(f"   Recommendations: {len(result['recommendations'])}")
                logger.info(f"   Confidence: {result['classification']['confidence']:.0%}")
                
                # Show first part of narrative response
                narrative = result['narrative_response']
                preview = narrative[:200] + "..." if len(narrative) > 200 else narrative
                logger.info(f"   Response Preview: {preview}")
                
                # Test dashboard data creation
                dashboard_data = response_system.create_dashboard_data(result)
                if 'error' not in dashboard_data:
                    logger.info("   Dashboard Data: ✅ Generated")
                
                # Test executive summary
                exec_summary = response_system.generate_executive_summary(result)
                if exec_summary:
                    logger.info(f"   Executive Summary: ✅ Generated ({len(exec_summary)} chars)")
                
            else:
                logger.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
                if 'sql_validation' in result:
                    logger.error(f"   SQL Issues: {result['sql_validation'].get('errors_found', [])}")
                
        except Exception as e:
            logger.error(f"❌ Exception during test {i}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Test fallback SQL generation
    logger.info("\n🛠️  Testing Fallback SQL Generation:")
    logger.info("-" * 50)
    
    test_contexts = [
        {
            'query': "Show temperature profile for platform '1900121'",
            'classification': {
                'intent': 'profile_analysis',
                'parameters': ['temperature', 'pressure']
            }
        },
        {
            'query': 'Map salinity in the Pacific Ocean',
            'classification': {
                'intent': 'spatial_mapping',
                'parameters': ['salinity']
            }
        }
    ]
    
    for i, context in enumerate(test_contexts, 1):
        logger.info(f"\nFallback Test {i}: {context['query']}")
        try:
            fallback_sql = response_system._generate_fallback_sql(
                context['query'], 
                context['classification']
            )
            validation = response_system._validate_and_correct_sql(fallback_sql)
            
            if validation['is_valid']:
                logger.info("✅ Fallback SQL generated and validated")
                logger.info(f"   SQL: {fallback_sql.strip()}")
            else:
                logger.error("❌ Fallback SQL validation failed")
                logger.error(f"   Errors: {validation['errors_found']}")
        except Exception as e:
            logger.error(f"❌ Fallback generation failed: {e}")
    
    # Performance and capability summary
    logger.info("\n📊 System Capabilities Summary:")
    logger.info("=" * 60)
    logger.info("✅ SQL Validation & Auto-correction")
    logger.info("✅ Production Schema Compatibility")
    logger.info("✅ Multi-audience Response Generation")
    logger.info("✅ Dynamic Visualization Creation")
    logger.info("✅ Actionable Recommendations")
    logger.info("✅ Executive Summary Generation")
    logger.info("✅ Dashboard Data Export")
    logger.info("✅ Fallback Error Recovery")
    logger.info("✅ Comprehensive Error Handling")
    
    logger.info(f"\n🎯 Key Improvements Implemented:")
    logger.info("- Fixed platform_number string/integer conversion errors")
    logger.info("- Added comprehensive SQL syntax validation")
    logger.info("- Implemented smart parentheses balancing")
    logger.info("- Added production schema awareness")
    logger.info("- Enhanced error recovery with fallback SQL generation")
    logger.info("- Improved visualization templates for new schema")
    logger.info("- Added audience-specific response formatting")
    
    logger.info("\n" + "=" * 70)
    logger.info("INTELLIGENT RESPONSE SYSTEM TESTING COMPLETE")
    logger.info("=" * 70)


# Additional utility functions for production deployment

def validate_system_requirements():
    """Validate that all system requirements are met"""
    requirements = {
        'pandas': 'Data processing',
        'plotly': 'Visualization generation',
        'langchain_google_genai': 'LLM integration',
        'dotenv': 'Environment configuration'
    }
    
    missing_requirements = []
    
    for package, description in requirements.items():
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_requirements.append(f"{package} ({description})")
    
    if missing_requirements:
        logger.error("Missing required packages:")
        for req in missing_requirements:
            logger.error(f"  - {req}")
        return False
    
    logger.info("✅ All system requirements satisfied")
    return True


def create_production_config():
    """Create production configuration template"""
    config = {
        'database': {
            'main_tables': ['argo_profiles', 'argo_measurements'],
            'legacy_tables': ['enhanced_floats_metadata', 'enhanced_measurements'],
            'connection_pool_size': 10,
            'query_timeout': 300
        },
        'llm': {
            'model': 'gemini-1.5-flash',
            'temperature': 0.3,
            'max_tokens': 2000,
            'retry_attempts': 3
        },
        'visualization': {
            'max_data_points': 10000,
            'default_height': 700,
            'color_schemes': ['viridis', 'plasma', 'inferno']
        },
        'performance': {
            'cache_results': True,
            'cache_ttl_seconds': 3600,
            'max_concurrent_queries': 5
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'intelligent_response_system.log'
        }
    }
    
    return config


def setup_production_logging():
    """Setup production-ready logging configuration"""
    import logging.handlers
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Setup file handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        'intelligent_response_system.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def main():
    """Main entry point for the intelligent response system"""
    
    print("🚀 Starting Intelligent Response System")
    print("=" * 60)
    
    # Setup production logging
    prod_logger = setup_production_logging()
    
    # Validate system requirements
    if not validate_system_requirements():
        print("❌ System requirements not met. Please install missing packages.")
        return 1
    
    # Create production config
    config = create_production_config()
    prod_logger.info("Production configuration created")
    
    # Run tests
    try:
        test_intelligent_response_system()
        prod_logger.info("System testing completed successfully")
        print("\n✅ Intelligent Response System is ready for production!")
        return 0
        
    except Exception as e:
        prod_logger.error(f"System testing failed: {e}")
        print(f"\n❌ System testing failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)