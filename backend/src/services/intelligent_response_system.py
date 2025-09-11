# intelligent_response_system.py
# Location: src/services/intelligent_response_system.py

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import warnings

# Fix import issues by using absolute imports
import sys
from pathlib import Path

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
        try:
            self.response_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.3,  # Slightly higher for creative responses
                max_tokens=2000,
                google_api_key=os.getenv('GEMINI_API_KEY')
            )
        except Exception as e:
            logger.warning(f"Failed to initialize response LLM: {e}")
            self.response_llm = None
        
        # Visualization templates and configurations
        self.viz_templates = self._build_visualization_templates()
        
        # Response templates for different audiences
        self.response_templates = self._build_response_templates()
        
        # Scientific interpretation patterns
        self.interpretation_patterns = self._build_interpretation_patterns()
    
    def _build_visualization_templates(self) -> Dict[str, Dict]:
        """Build templates for different visualization types"""
        return {
            'profile_analysis': {
                'line_profile': {
                    'type': 'line',
                    'x_axis': 'parameter_value',
                    'y_axis': 'pressure',
                    'title': '{parameter} Profile',
                    'y_reversed': True,
                    'layout_config': {
                        'yaxis_title': 'Pressure (dbar)',
                        'yaxis_autorange': 'reversed'
                    }
                },
                'scatter_profile': {
                    'type': 'scatter',
                    'x_axis': 'parameter_value',
                    'y_axis': 'pressure',
                    'color_by': 'float_id',
                    'title': 'Multi-Float {parameter} Profiles',
                    'y_reversed': True
                }
            },
            'spatial_mapping': {
                'geographic_scatter': {
                    'type': 'scatter_mapbox',
                    'lat_col': 'latitude',
                    'lon_col': 'longitude',
                    'color_by': 'parameter_value',
                    'title': '{parameter} Distribution',
                    'mapbox_style': 'open-street-map'
                },
                'contour_map': {
                    'type': 'density_mapbox',
                    'lat_col': 'latitude',
                    'lon_col': 'longitude',
                    'z_col': 'parameter_value',
                    'title': '{parameter} Density Map'
                }
            },
            'temporal_analysis': {
                'time_series': {
                    'type': 'line',
                    'x_axis': 'time',
                    'y_axis': 'parameter_value',
                    'title': '{parameter} Time Series',
                    'layout_config': {
                        'xaxis_title': 'Time',
                        'showlegend': True
                    }
                },
                'seasonal_box': {
                    'type': 'box',
                    'x_axis': 'month',
                    'y_axis': 'parameter_value',
                    'title': 'Seasonal {parameter} Variability'
                }
            },
            'statistical_analysis': {
                'histogram': {
                    'type': 'histogram',
                    'x_axis': 'parameter_value',
                    'title': '{parameter} Distribution',
                    'bins': 50
                },
                'violin_plot': {
                    'type': 'violin',
                    'x_axis': 'category',
                    'y_axis': 'parameter_value',
                    'title': '{parameter} Distribution by {category}'
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
    
    def process_intelligent_query(self, natural_language_query: str, 
                                response_format: ResponseFormat = None) -> Dict[str, Any]:
        """
        Process query with full intelligent response generation pipeline
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
        """Build audience-specific system prompts"""
        
        base_prompt = """You are an expert oceanographer and science communicator with deep knowledge of ARGO float data and marine science. You excel at translating complex oceanographic analysis into clear, actionable insights."""
        
        audience_prompts = {
            'government_official': """
            Your audience consists of government officials and policy makers who need:
            - Clear, executive-level summaries
            - Policy implications and strategic value
            - Economic and societal impact context
            - Actionable recommendations for decision-making
            - Non-technical language with essential scientific context
            """,
            'researcher': """
            Your audience consists of marine researchers and scientists who need:
            - Technical accuracy and scientific rigor
            - Methodology and statistical significance discussion
            - Physical oceanographic interpretation
            - Research implications and future directions
            - Technical terminology and detailed analysis
            """,
            'maritime_industry': """
            Your audience consists of maritime industry professionals who need:
            - Operational relevance and practical applications
            - Safety and efficiency implications
            - Economic impact on operations
            - Clear, actionable guidance
            - Semi-technical language focused on applications
            """,
            'general_public': """
            Your audience consists of the general public who need:
            - Accessible explanations without jargon
            - Context and broader significance
            - Real-world implications and relevance
            - Engaging and educational content
            - Clear, simple language with analogies when helpful
            """
        }
        
        complexity_modifiers = {
            'basic': "Keep explanations simple and focus on key takeaways.",
            'intermediate': "Provide moderate technical detail with clear explanations.",
            'advanced': "Include comprehensive analysis with technical depth.",
            'expert': "Provide full technical detail and research-level analysis."
        }
        
        audience_prompt = audience_prompts.get(audience, audience_prompts['general_public'])
        complexity_modifier = complexity_modifiers.get(complexity, complexity_modifiers['intermediate'])
        
        return f"{base_prompt}\n\n{audience_prompt}\n\nComplexity Level: {complexity_modifier}"
    
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
            elif intent == 'temporal_trends' and any(col in results_df.columns for col in ['date', 'time']):
                time_col = 'date' if 'date' in results_df.columns else 'time'
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
        """Create profile visualization (depth vs parameter)"""
        
        try:
            # Clean data
            clean_df = df[[parameter, 'pressure']].dropna()
            if len(clean_df) < 2:
                return None
            
            fig = go.Figure()
            
            # Add profile line
            fig.add_trace(go.Scatter(
                x=clean_df[parameter],
                y=clean_df['pressure'],
                mode='lines+markers',
                name=f'{parameter.title()} Profile',
                line=dict(width=2),
                marker=dict(size=4)
            ))
            
            # Update layout for profile (depth on y-axis, reversed)
            fig.update_layout(
                title=f'{parameter.title()} Profile',
                xaxis_title=f'{parameter.title()}',
                yaxis_title='Pressure (dbar)',
                yaxis_autorange='reversed',
                showlegend=True,
                template='plotly_white',
                height=600
            )
            
            return {
                'type': 'profile',
                'parameter': parameter,
                'title': f'{parameter.title()} Profile',
                'plotly_json': fig.to_json(),
                'data_points': len(clean_df)
            }
            
        except Exception as e:
            logger.error(f"Error creating profile visualization for {parameter}: {e}")
            return None
    
    def _create_spatial_visualization(self, df: pd.DataFrame, parameter: str) -> Optional[Dict]:
        """Create spatial distribution visualization"""
        
        try:
            # Clean data
            required_cols = ['latitude', 'longitude', parameter]
            clean_df = df[required_cols].dropna()
            if len(clean_df) < 2:
                return None
            
            fig = go.Figure()
            
            # Add scatter points
            fig.add_trace(go.Scattermapbox(
                lat=clean_df['latitude'],
                lon=clean_df['longitude'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=clean_df[parameter],
                    colorscale='viridis',
                    colorbar=dict(title=f'{parameter.title()}'),
                    showscale=True
                ),
                text=[f'Lat: {lat:.2f}, Lon: {lon:.2f}<br>{parameter}: {val:.2f}' 
                      for lat, lon, val in zip(clean_df['latitude'], clean_df['longitude'], clean_df[parameter])],
                name=f'{parameter.title()} Distribution'
            ))
            
            # Update layout for map
            fig.update_layout(
                title=f'{parameter.title()} Spatial Distribution',
                mapbox=dict(
                    style='open-street-map',
                    center=dict(
                        lat=clean_df['latitude'].mean(),
                        lon=clean_df['longitude'].mean()
                    ),
                    zoom=4
                ),
                height=600,
                margin=dict(t=50, b=0, l=0, r=0)
            )
            
            return {
                'type': 'spatial',
                'parameter': parameter,
                'title': f'{parameter.title()} Spatial Distribution',
                'plotly_json': fig.to_json(),
                'data_points': len(clean_df)
            }
            
        except Exception as e:
            logger.error(f"Error creating spatial visualization for {parameter}: {e}")
            return None
    
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
    
    logger.info("Testing Intelligent Response System")
    logger.info("=" * 60)
    
    # Initialize system
    try:
        response_system = IntelligentResponseSystem()
        logger.info("✅ System initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        return
    
    # Test queries with different complexity levels
    test_queries = [
        {
            'query': "Show temperature measurements for platform 1900121",
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
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case['query']
        response_format = test_case['format']
        
        logger.info(f"\n🔍 Test {i}: {query}")
        logger.info(f"Target Audience: {response_format.target_audience}")
        logger.info(f"Complexity: {response_format.complexity_level}")
        logger.info("-" * 50)
        
        try:
            start_time = datetime.now()
            result = response_system.process_intelligent_query(query, response_format)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if result['success']:
                logger.info(f"✅ Success! ({processing_time:.2f}s)")
                logger.info(f"   Classification: {result['classification']['intent']}")
                logger.info(f"   Data Points: {result['results_summary']['total_records']:,}")
                logger.info(f"   Visualizations: {len(result['visualizations'])}")
                logger.info(f"   Recommendations: {len(result['recommendations'])}")
                
                # Show first part of narrative response
                narrative = result['narrative_response']
                preview = narrative[:200] + "..." if len(narrative) > 200 else narrative
                logger.info(f"   Response Preview: {preview}")
                
                # Test dashboard data creation
                dashboard_data = response_system.create_dashboard_data(result)
                if 'error' not in dashboard_data:
                    logger.info(f"   Dashboard Data: ✅ Generated")
                
                # Test executive summary
                exec_summary = response_system.generate_executive_summary(result)
                if exec_summary:
                    logger.info(f"   Executive Summary: ✅ Generated ({len(exec_summary)} chars)")
                
            else:
                logger.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"❌ Exception during test {i}: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Intelligent Response System Testing Complete")


if __name__ == "__main__":
    test_intelligent_response_system()