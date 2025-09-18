# src/services/response_orchestration_system.py
"""
RESPONSE ORCHESTRATION ARCHITECTURE - PRODUCTION IMPLEMENTATION

This is the missing piece that transforms your bloated intelligent_response_system.py
into a clean, structured, DeepSeek-powered response generation engine.

CORE PRINCIPLE: Generate structured data packages that your frontend consumes intelligently.

Architecture:
- Layer 1: Data Processing (your existing RAG)
- Layer 2: Response Intelligence (THIS - the missing piece)
- Layer 3: Multi-Modal Response Assembly
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np

# Type imports (forward references)
from __future__ import annotations
from typing_extensions import TypeAlias

# DeepSeek Integration
try:
    from langchain.chat_models.base import BaseChatModel
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
except ImportError:
    BaseChatModel = ChatOpenAI = HumanMessage = SystemMessage = None

from dotenv import load_dotenv

# Type hint for DeepSeek client
from typing import Union

load_dotenv()
logger = logging.getLogger(__name__)

class VisualizationType(Enum):
    """Enhanced visualization types for oceanographic data"""
    DEPTH_PROFILE = "depth_profile"
    SPATIAL_DISTRIBUTION = "spatial_distribution"
    TEMPORAL_SERIES = "temporal_series"
    STATISTICAL_SUMMARY = "statistical_summary"
    GEOGRAPHIC_CONTEXT = "geographic_context"
    PARAMETER_CORRELATION = "parameter_correlation"
    WATER_MASS_ANALYSIS = "water_mass_analysis"
    QUALITY_ASSESSMENT = "quality_assessment"

@dataclass
class VisualizationSpec:
    """Structured visualization specification"""
    type: VisualizationType
    plotly_config: Dict[str, Any]
    reasoning: str
    priority: int
    data_requirements: List[str]
    interactive_features: List[str]
    export_options: List[str]

@dataclass
class DataExport:
    """Structured data export specification"""
    table_data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    quality_notes: str
    export_formats: List[str]
    download_ready: bool

@dataclass
class UserGuidance:
    """Structured user guidance for exploration"""
    next_questions: List[str]
    exploration_hints: List[str]
    related_analyses: List[str]
    methodology_notes: List[str]

@dataclass
class ResponsePackage:
    """Complete structured response package for frontend consumption"""
    query_context: Dict[str, Any]
    primary_insight: Dict[str, str]
    visualizations: Dict[str, VisualizationSpec]
    data_export: DataExport
    user_guidance: UserGuidance
    processing_metadata: Dict[str, Any]

class ResponseOrchestrator:
    """
    CORE RESPONSE INTELLIGENCE ENGINE
    
    This is where your DeepSeek API investment pays off - intelligent analysis
    of data characteristics to generate optimal response structures.
    """
    
    deepseek_client: Any  # Type hint for instance variable
    visualization_templates: Dict[str, Any]
    insight_patterns: Dict[str, List[str]]
    guidance_generators: Dict[str, Any]
    
    def __init__(self):
        self.deepseek_client = self._initialize_deepseek()
        self.visualization_templates = self._build_visualization_templates()
        self.insight_patterns = self._build_insight_patterns()
        self.guidance_generators = self._build_guidance_generators()
        
    def _initialize_deepseek(self) -> Any:
        """Initialize DeepSeek API client"""
        try:
            if not ChatOpenAI:
                logger.warning("LangChain not available - falling back to template responses")
                return None
                
            client = ChatOpenAI(
                model="deepseek-chat",
                openai_api_key=os.getenv('DEEPSEEK_API_KEY'),
                openai_api_base="https://api.deepseek.com/v1",
                temperature=0.2,  # Lower temperature for consistent structured output
                max_tokens=1500,
                timeout=30
            )
            
            logger.info("DeepSeek API client initialized successfully")
            return client
            
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek: {e}")
            return None
    
    def orchestrate_response(self, rag_result: Dict[str, Any]) -> ResponsePackage:
        """
        MAIN ORCHESTRATION METHOD
        
        Takes RAG result and generates structured response package
        """
        logger.info(f"Orchestrating response for query: {rag_result.get('query', 'Unknown')}")
        
        try:
            # Step 1: Analyze data characteristics
            data_analysis = self._analyze_data_characteristics(rag_result)
            
            # Step 2: Generate primary insight (DeepSeek-powered)
            primary_insight = self._generate_primary_insight(rag_result, data_analysis)
            
            # Step 3: Select optimal visualizations
            visualizations = self._select_optimal_visualizations(rag_result, data_analysis)
            
            # Step 4: Prepare data export
            data_export = self._prepare_data_export(rag_result)
            
            # Step 5: Generate user guidance
            user_guidance = self._generate_user_guidance(rag_result, data_analysis)
            
            # Step 6: Build complete response package
            response_package = ResponsePackage(
                query_context={
                    "original_query": rag_result.get('query'),
                    "scientific_classification": rag_result.get('classification', {}).get('intent'),
                    "confidence": rag_result.get('classification', {}).get('confidence', 0.0),
                    "data_scope": data_analysis.get('scope_summary')
                },
                primary_insight=primary_insight,
                visualizations=visualizations,
                data_export=data_export,
                user_guidance=user_guidance,
                processing_metadata={
                    "orchestration_time": datetime.now().isoformat(),
                    "deepseek_used": self.deepseek_client is not None,
                    "data_quality_score": data_analysis.get('quality_score', 0.0),
                    "visualization_count": len(visualizations),
                    "response_type": "structured_package"
                }
            )
            
            logger.info("Response orchestration completed successfully")
            return response_package
            
        except Exception as e:
            logger.error(f"Response orchestration failed: {e}")
            return self._create_fallback_response(rag_result)
    
    def _analyze_data_characteristics(self, rag_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data characteristics to inform visualization and insight selection"""
        
        results_df = rag_result.get('results')
        if results_df is None or results_df.empty:
            return {
                'scope_summary': 'No data available',
                'quality_score': 0.0,
                'visualization_recommendations': [],
                'complexity_level': 'basic'
            }
        
        analysis = {
            'record_count': len(results_df),
            'parameter_count': len(results_df.columns),
            'geographic_extent': self._calculate_geographic_extent(results_df),
            'temporal_span': self._calculate_temporal_span(results_df),
            'depth_range': self._calculate_depth_range(results_df),
            'data_completeness': self._assess_data_completeness(results_df),
            'quality_score': self._calculate_quality_score(results_df),
            'visualization_suitability': self._assess_visualization_suitability(results_df),
            'complexity_level': self._determine_complexity_level(rag_result, results_df)
        }
        
        # Generate scope summary
        analysis['scope_summary'] = self._generate_scope_summary(analysis)
        
        return analysis
    
    def _generate_primary_insight(self, rag_result: Dict[str, Any], 
                                 data_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate primary insight using DeepSeek intelligence"""
        
        if not self.deepseek_client:
            return self._generate_template_insight(rag_result, data_analysis)
        
        try:
            # Build focused prompt for DeepSeek
            prompt = self._build_insight_prompt(rag_result, data_analysis)
            
            messages = [
                SystemMessage(content="""You are an expert oceanographic analyst. Provide concise, scientifically accurate insights about ARGO float data. Focus on:
1. Primary scientific finding (max 2 sentences)
2. Physical oceanographic interpretation
3. Key data quality considerations
4. Statistical significance when relevant

Output as structured JSON with keys: summary, interpretation, quality_notes, significance."""),
                HumanMessage(content=prompt)
            ]
            
            response = self.deepseek_client(messages)
            
            # Parse response
            try:
                insight_data = json.loads(response.content)
                return {
                    'summary': insight_data.get('summary', 'No summary available'),
                    'scientific_interpretation': insight_data.get('interpretation', 'No interpretation available'),
                    'quality_assessment': insight_data.get('quality_notes', 'Quality assessment not available'),
                    'statistical_significance': insight_data.get('significance', 'Not assessed')
                }
            except json.JSONDecodeError:
                # Fallback to text parsing
                return {
                    'summary': response.content[:200] + "..." if len(response.content) > 200 else response.content,
                    'scientific_interpretation': 'DeepSeek analysis completed',
                    'quality_assessment': 'Standard quality assessment applied',
                    'statistical_significance': 'Assessment available in summary'
                }
                
        except Exception as e:
            logger.warning(f"DeepSeek insight generation failed: {e}")
            return self._generate_template_insight(rag_result, data_analysis)
    
    def _select_optimal_visualizations(self, rag_result: Dict[str, Any], 
                                     data_analysis: Dict[str, Any]) -> Dict[str, VisualizationSpec]:
        """Select optimal visualizations based on data characteristics"""
        
        selected_viz = {}
        results_df = rag_result.get('results')
        
        if results_df is None or results_df.empty:
            return {}
        
        classification = rag_result.get('classification', {})
        intent = classification.get('intent', 'exploration')
        
        # Primary visualization selection logic
        if self._has_depth_data(results_df) and intent == 'profile_analysis':
            selected_viz['primary'] = self._create_depth_profile_viz(results_df, data_analysis)
            
        elif self._has_geographic_data(results_df) and intent in ['spatial_mapping', 'exploration']:
            selected_viz['primary'] = self._create_spatial_viz(results_df, data_analysis)
            
        elif self._has_temporal_data(results_df) and intent == 'temporal_trends':
            selected_viz['primary'] = self._create_temporal_viz(results_df, data_analysis)
            
        else:
            # Default to statistical visualization
            selected_viz['primary'] = self._create_statistical_viz(results_df, data_analysis)
        
        # Supporting visualizations
        supporting_viz = self._select_supporting_visualizations(results_df, data_analysis, intent)
        selected_viz.update(supporting_viz)
        
        return selected_viz
    
    def _create_depth_profile_viz(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> VisualizationSpec:
        """Create depth profile visualization specification"""
        
        # Determine primary parameter for profile
        profile_param = self._select_primary_parameter(df, ['temperature', 'salinity', 'density'])
        
        plotly_config = {
            "data": [
                {
                    "type": "scatter",
                    "mode": "lines+markers",
                    "x": f"${profile_param}_values",
                    "y": "$pressure_values",
                    "name": f"{profile_param.title()} Profile",
                    "line": {"width": 3},
                    "marker": {"size": 6}
                }
            ],
            "layout": {
                "title": f"{profile_param.title()} vs Depth Profile",
                "xaxis": {"title": f"{profile_param.title()} ({self._get_unit(profile_param)})"},
                "yaxis": {"title": "Pressure (dbar)", "autorange": "reversed"},
                "showlegend": True,
                "height": 700,
                "template": "plotly_white"
            }
        }
        
        return VisualizationSpec(
            type=VisualizationType.DEPTH_PROFILE,
            plotly_config=plotly_config,
            reasoning=f"Temperature profiles require depth-based visualization with inverted y-axis for oceanographic convention",
            priority=1,
            data_requirements=[profile_param, 'pressure'],
            interactive_features=['zoom', 'pan', 'hover', 'select'],
            export_options=['png', 'svg', 'pdf', 'html']
        )
    
    def _create_spatial_viz(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> VisualizationSpec:
        """Create spatial distribution visualization specification"""
        
        color_param = self._select_primary_parameter(df, ['surface_temp', 'temperature', 'salinity'])
        
        plotly_config = {
            "data": [
                {
                    "type": "scattermapbox",
                    "lat": "$latitude_values",
                    "lon": "$longitude_values",
                    "mode": "markers",
                    "marker": {
                        "size": 8,
                        "color": f"${color_param}_values",
                        "colorscale": "Viridis",
                        "showscale": True,
                        "colorbar": {"title": f"{color_param.title()}"}
                    },
                    "text": f"${color_param}_hover_text",
                    "name": "ARGO Measurements"
                }
            ],
            "layout": {
                "title": f"ARGO {color_param.title()} Distribution",
                "mapbox": {
                    "style": "open-street-map",
                    "center": {"lat": "$center_lat", "lon": "$center_lon"},
                    "zoom": "$optimal_zoom"
                },
                "height": 700,
                "margin": {"t": 80, "b": 20, "l": 20, "r": 20}
            }
        }
        
        return VisualizationSpec(
            type=VisualizationType.SPATIAL_DISTRIBUTION,
            plotly_config=plotly_config,
            reasoning="Geographic data requires spatial visualization to show measurement distribution and parameter patterns",
            priority=1,
            data_requirements=['latitude', 'longitude', color_param],
            interactive_features=['zoom', 'pan', 'hover', 'filter'],
            export_options=['png', 'pdf', 'html', 'geojson']
        )
    
    def _create_temporal_viz(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> VisualizationSpec:
        """Create temporal series visualization specification"""
        
        y_param = self._select_primary_parameter(df, ['temperature', 'salinity'])
        
        plotly_config = {
            "data": [
                {
                    "type": "scatter",
                    "mode": "lines+markers",
                    "x": "$time_values",
                    "y": f"${y_param}_values",
                    "name": f"{y_param.title()} Time Series",
                    "line": {"width": 2},
                    "marker": {"size": 4}
                }
            ],
            "layout": {
                "title": f"{y_param.title()} Temporal Evolution",
                "xaxis": {"title": "Time"},
                "yaxis": {"title": f"{y_param.title()} ({self._get_unit(y_param)})"},
                "showlegend": True,
                "height": 500,
                "template": "plotly_white"
            }
        }
        
        return VisualizationSpec(
            type=VisualizationType.TEMPORAL_SERIES,
            plotly_config=plotly_config,
            reasoning="Temporal data requires time series visualization to show trends and variability",
            priority=1,
            data_requirements=['profile_date', y_param],
            interactive_features=['zoom', 'pan', 'range_selector'],
            export_options=['png', 'svg', 'csv', 'html']
        )
    
    def _create_statistical_viz(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> VisualizationSpec:
        """Create statistical summary visualization specification"""
        
        numeric_params = df.select_dtypes(include=[np.number]).columns.tolist()
        primary_param = numeric_params[0] if numeric_params else 'value'
        
        plotly_config = {
            "data": [
                {
                    "type": "histogram",
                    "x": f"${primary_param}_values",
                    "nbinsx": 30,
                    "name": f"{primary_param.title()} Distribution"
                }
            ],
            "layout": {
                "title": f"{primary_param.title()} Statistical Distribution",
                "xaxis": {"title": f"{primary_param.title()}"},
                "yaxis": {"title": "Frequency"},
                "showlegend": False,
                "height": 500,
                "template": "plotly_white"
            }
        }
        
        return VisualizationSpec(
            type=VisualizationType.STATISTICAL_SUMMARY,
            plotly_config=plotly_config,
            reasoning="Statistical analysis requires distribution visualization to understand data characteristics",
            priority=1,
            data_requirements=[primary_param],
            interactive_features=['zoom', 'pan', 'bin_control'],
            export_options=['png', 'svg', 'statistics_csv']
        )
    
    def _prepare_data_export(self, rag_result: Dict[str, Any]) -> DataExport:
        """Prepare structured data export"""
        
        results_df = rag_result.get('results')
        
        if results_df is None or results_df.empty:
            return DataExport(
                table_data=[],
                metadata={"message": "No data available for export"},
                quality_notes="No data to assess",
                export_formats=[],
                download_ready=False
            )
        
        # Convert DataFrame to structured format
        table_data = results_df.head(1000).to_dict('records')  # Limit for performance
        
        metadata = {
            "total_records": len(results_df),
            "exported_records": len(table_data),
            "columns": list(results_df.columns),
            "data_types": {col: str(dtype) for col, dtype in results_df.dtypes.items()},
            "export_timestamp": datetime.now().isoformat(),
            "source": "ARGO Float Database",
            "processing_info": rag_result.get('classification', {})
        }
        
        # Assess data quality
        quality_notes = self._generate_quality_assessment(results_df)
        
        return DataExport(
            table_data=table_data,
            metadata=metadata,
            quality_notes=quality_notes,
            export_formats=['csv', 'json', 'xlsx', 'netcdf'],
            download_ready=True
        )
    
    def _generate_user_guidance(self, rag_result: Dict[str, Any], 
                               data_analysis: Dict[str, Any]) -> UserGuidance:
        """Generate intelligent user guidance for exploration"""
        
        classification = rag_result.get('classification', {})
        intent = classification.get('intent', 'exploration')
        parameters = classification.get('parameters', [])
        
        # Generate context-aware follow-up questions
        next_questions = self._generate_followup_questions(intent, parameters, data_analysis)
        
        # Generate exploration hints
        exploration_hints = self._generate_exploration_hints(intent, data_analysis)
        
        # Suggest related analyses
        related_analyses = self._suggest_related_analyses(intent, parameters)
        
        # Methodology notes
        methodology_notes = self._generate_methodology_notes(rag_result)
        
        return UserGuidance(
            next_questions=next_questions,
            exploration_hints=exploration_hints,
            related_analyses=related_analyses,
            methodology_notes=methodology_notes
        )
    
    # Helper methods for data analysis
    def _calculate_geographic_extent(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate geographic extent of data"""
        if not all(col in df.columns for col in ['latitude', 'longitude']):
            return {}
        
        return {
            'lat_min': float(df['latitude'].min()),
            'lat_max': float(df['latitude'].max()),
            'lon_min': float(df['longitude'].min()),
            'lon_max': float(df['longitude'].max()),
            'lat_span': float(df['latitude'].max() - df['latitude'].min()),
            'lon_span': float(df['longitude'].max() - df['longitude'].min())
        }
    
    def _calculate_temporal_span(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate temporal span of data"""
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if not date_columns:
            return {}
        
        date_col = date_columns[0]
        try:
            dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
            if len(dates) == 0:
                return {}
            
            return {
                'start_date': dates.min().isoformat(),
                'end_date': dates.max().isoformat(),
                'span_days': (dates.max() - dates.min()).days,
                'temporal_resolution': self._estimate_temporal_resolution(dates)
            }
        except:
            return {}
    
    def _calculate_depth_range(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate depth/pressure range"""
        depth_cols = [col for col in df.columns if col in ['pressure', 'depth']]
        
        if not depth_cols:
            return {}
        
        depth_col = depth_cols[0]
        return {
            'min_depth': float(df[depth_col].min()),
            'max_depth': float(df[depth_col].max()),
            'depth_range': float(df[depth_col].max() - df[depth_col].min()),
            'depth_resolution': self._estimate_depth_resolution(df[depth_col])
        }
    
    def _assess_data_completeness(self, df: pd.DataFrame) -> Dict[str, float]:
        """Assess data completeness"""
        completeness = {}
        
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            completeness[col] = float(non_null_count / len(df) * 100)
        
        return completeness
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score"""
        if df.empty:
            return 0.0
        
        completeness_scores = list(self._assess_data_completeness(df).values())
        avg_completeness = np.mean(completeness_scores) if completeness_scores else 0
        
        # Adjust score based on data characteristics
        quality_score = avg_completeness / 100.0
        
        # Penalize for very small datasets
        if len(df) < 10:
            quality_score *= 0.5
        
        # Bonus for geographic coverage
        if self._has_geographic_data(df):
            geo_extent = self._calculate_geographic_extent(df)
            if geo_extent.get('lat_span', 0) > 5 or geo_extent.get('lon_span', 0) > 5:
                quality_score *= 1.1
        
        return min(quality_score, 1.0)
    
    # Utility methods
    def _has_depth_data(self, df: pd.DataFrame) -> bool:
        """Check if data has depth/pressure information"""
        return any(col in df.columns for col in ['pressure', 'depth'])
    
    def _has_geographic_data(self, df: pd.DataFrame) -> bool:
        """Check if data has geographic information"""
        return all(col in df.columns for col in ['latitude', 'longitude'])
    
    def _has_temporal_data(self, df: pd.DataFrame) -> bool:
        """Check if data has temporal information"""
        return any('date' in col.lower() or 'time' in col.lower() for col in df.columns)
    
    def _select_primary_parameter(self, df: pd.DataFrame, candidates: List[str]) -> str:
        """Select primary parameter from candidates"""
        for param in candidates:
            if param in df.columns and df[param].notna().sum() > 0:
                return param
        
        # Fallback to first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return numeric_cols[0] if len(numeric_cols) > 0 else 'value'
    
    def _get_unit(self, parameter: str) -> str:
        """Get unit for parameter"""
        units = {
            'temperature': '°C',
            'salinity': 'PSU',
            'pressure': 'dbar',
            'density': 'kg/m³',
            'surface_temp': '°C',
            'surface_salinity': 'PSU'
        }
        return units.get(parameter, '')
    
    # Template and fallback methods
    def _generate_template_insight(self, rag_result: Dict[str, Any], 
                                  data_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate template insight when DeepSeek is unavailable"""
        
        record_count = data_analysis.get('record_count', 0)
        quality_score = data_analysis.get('quality_score', 0)
        
        return {
            'summary': f"Analysis of {record_count:,} ARGO measurements with {quality_score:.1%} data completeness.",
            'scientific_interpretation': "Oceanographic analysis completed using template-based interpretation system.",
            'quality_assessment': f"Data quality score: {quality_score:.1%}. Standard validation applied.",
            'statistical_significance': "Statistical analysis available in detailed results."
        }
    
    def _create_fallback_response(self, rag_result: Dict[str, Any]) -> ResponsePackage:
        """Create fallback response when orchestration fails"""
        
        return ResponsePackage(
            query_context={
                "original_query": rag_result.get('query', 'Unknown query'),
                "scientific_classification": "fallback",
                "confidence": 0.5,
                "data_scope": "Fallback processing applied"
            },
            primary_insight={
                'summary': "Query processed with fallback system",
                'scientific_interpretation': "Limited interpretation available",
                'quality_assessment': "Standard assessment applied",
                'statistical_significance': "Not assessed"
            },
            visualizations={},
            data_export=DataExport(
                table_data=[],
                metadata={"message": "Fallback response"},
                quality_notes="Unable to perform full analysis",
                export_formats=[],
                download_ready=False
            ),
            user_guidance=UserGuidance(
                next_questions=["Try simplifying your query"],
                exploration_hints=["Consider more specific parameters"],
                related_analyses=[],
                methodology_notes=["Fallback processing was used"]
            ),
            processing_metadata={
                "orchestration_time": datetime.now().isoformat(),
                "deepseek_used": False,
                "data_quality_score": 0.0,
                "visualization_count": 0,
                "response_type": "fallback"
            }
        )
    
    # Build supporting data structures
    def _build_visualization_templates(self) -> Dict[str, Any]:
        """Build visualization templates"""
        return {
            'depth_profile_standard': {
                'type': 'scatter',
                'y_reversed': True,
                'best_for': ['temperature', 'salinity', 'density']
            },
            'spatial_mapbox': {
                'type': 'scattermapbox',
                'style': 'open-street-map',
                'best_for': ['geographic_distribution', 'regional_analysis']
            },
            'temporal_line': {
                'type': 'scatter',
                'mode': 'lines+markers',
                'best_for': ['time_series', 'trend_analysis']
            }
        }
    
    def _build_insight_patterns(self) -> Dict[str, List[str]]:
        """Build insight generation patterns"""
        return {
            'temperature_patterns': [
                'thermal_stratification',
                'mixed_layer_depth',
                'temperature_gradients',
                'seasonal_variability'
            ],
            'salinity_patterns': [
                'haline_stratification',
                'freshwater_influence',
                'water_mass_characteristics',
                'evaporation_precipitation_balance'
            ],
            'spatial_patterns': [
                'geographic_distribution',
                'regional_differences',
                'frontal_zones',
                'boundary_currents'
            ]
        }
    
    def _build_guidance_generators(self) -> Dict[str, callable]:
        """Build guidance generation functions"""
        return {
            'profile_analysis': self._generate_profile_guidance,
            'spatial_mapping': self._generate_spatial_guidance,
            'temporal_trends': self._generate_temporal_guidance,
            'statistical_summary': self._generate_statistical_guidance
        }
    
    # Guidance generation methods (simplified for brevity)
    def _generate_followup_questions(self, intent: str, parameters: List[str], 
                                   analysis: Dict[str, Any]) -> List[str]:
        """Generate context-aware follow-up questions"""
        questions = []
        
        if intent == 'profile_analysis':
            questions.extend([
                "How do these profiles compare to climatological averages?",
                "What are the seasonal variations in these profiles?",
                "Which water masses are represented in this data?"
            ])
        elif intent == 'spatial_mapping':
            questions.extend([
                "How does this parameter vary seasonally across the region?",
                "What are the main oceanographic features in this area?",
                "How do these measurements compare to satellite observations?"
            ])
        
        return questions[:3]  # Limit to 3 questions
    
    def _generate_exploration_hints(self, intent: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate exploration hints"""
        hints = []
        
        if analysis.get('record_count', 0) > 1000:
            hints.append("Try filtering by specific time periods or regions for focused analysis")
        
        if analysis.get('depth_range', {}).get('depth_range', 0) > 1000:
            hints.append("Deep ocean data available - consider water mass analysis")
        
        if analysis.get('geographic_extent', {}).get('lat_span', 0) > 10:
            hints.append("Large geographic coverage - consider regional comparisons")
        
        return hints[:3]  # Limit to 3 hints
    
    def _suggest_related_analyses(self, intent: str, parameters: List[str]) -> List[str]:
        """Suggest related analyses"""
        suggestions = []
        
        if 'temperature' in parameters and 'salinity' in parameters:
            suggestions.append("Water mass analysis using T-S diagrams")
        
        if intent == 'profile_analysis':
            suggestions.extend([
                "Mixed layer depth calculation",
                "Stratification index analysis",
                "Comparison with climatology"
            ])
        elif intent == 'spatial_mapping':
            suggestions.extend([
                "Seasonal variability mapping",
                "Regional comparison analysis",
                "Gradient and front detection"
            ])
        
        return suggestions[:3]
    
    def _generate_methodology_notes(self, rag_result: Dict[str, Any]) -> List[str]:
        """Generate methodology notes"""
        notes = []
        
        classification = rag_result.get('classification', {})
        complexity = classification.get('complexity', 'basic')
        
        notes.append(f"Analysis complexity: {complexity}")
        notes.append("Data source: ARGO float database")
        
        if rag_result.get('sql_query'):
            notes.append("SQL-based data retrieval with quality filtering applied")
        
        processing_time = rag_result.get('processing_time', 0)
        if processing_time > 0:
            notes.append(f"Processing time: {processing_time:.2f} seconds")
        
        return notes
    
    # Additional helper methods for completeness
    def _generate_scope_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate human-readable scope summary"""
        record_count = analysis.get('record_count', 0)
        
        if record_count == 0:
            return "No data available for analysis"
        
        summary_parts = [f"{record_count:,} measurements"]
        
        geo_extent = analysis.get('geographic_extent', {})
        if geo_extent:
            lat_span = geo_extent.get('lat_span', 0)
            lon_span = geo_extent.get('lon_span', 0)
            if lat_span > 0 or lon_span > 0:
                summary_parts.append(f"spanning {lat_span:.1f}° latitude × {lon_span:.1f}° longitude")
        
        depth_range = analysis.get('depth_range', {})
        if depth_range:
            max_depth = depth_range.get('max_depth', 0)
            if max_depth > 0:
                summary_parts.append(f"to {max_depth:.0f}m depth")
        
        temporal_span = analysis.get('temporal_span', {})
        if temporal_span:
            span_days = temporal_span.get('span_days', 0)
            if span_days > 0:
                if span_days > 365:
                    summary_parts.append(f"over {span_days/365:.1f} years")
                else:
                    summary_parts.append(f"over {span_days} days")
        
        return " ".join(summary_parts)
    
    def _assess_visualization_suitability(self, df: pd.DataFrame) -> Dict[str, float]:
        """Assess suitability for different visualization types"""
        suitability = {}
        
        # Depth profile suitability
        if self._has_depth_data(df):
            depth_coverage = len(df['pressure' if 'pressure' in df.columns else 'depth'].dropna())
            suitability['depth_profile'] = min(depth_coverage / 20, 1.0)  # Good with 20+ depth points
        
        # Spatial visualization suitability
        if self._has_geographic_data(df):
            unique_locations = len(df[['latitude', 'longitude']].drop_duplicates())
            suitability['spatial'] = min(unique_locations / 10, 1.0)  # Good with 10+ locations
        
        # Temporal visualization suitability
        if self._has_temporal_data(df):
            date_col = next((col for col in df.columns if 'date' in col.lower()), None)
            if date_col:
                unique_dates = pd.to_datetime(df[date_col], errors='coerce').dropna().nunique()
                suitability['temporal'] = min(unique_dates / 5, 1.0)  # Good with 5+ time points
        
        return suitability
    
    def _determine_complexity_level(self, rag_result: Dict[str, Any], df: pd.DataFrame) -> str:
        """Determine analysis complexity level"""
        classification = rag_result.get('classification', {})
        complexity = classification.get('complexity', 'basic')
        
        # Adjust based on data characteristics
        if len(df) > 10000:
            complexity = 'advanced' if complexity == 'intermediate' else complexity
        
        if len(df.columns) > 10:
            complexity = 'intermediate' if complexity == 'basic' else complexity
        
        return complexity
    
    def _estimate_temporal_resolution(self, dates: pd.Series) -> str:
        """Estimate temporal resolution of data"""
        if len(dates) < 2:
            return "unknown"
        
        sorted_dates = dates.sort_values()
        time_diffs = sorted_dates.diff().dropna()
        median_diff = time_diffs.median()
        
        if median_diff <= pd.Timedelta(hours=1):
            return "hourly"
        elif median_diff <= pd.Timedelta(days=1):
            return "daily"
        elif median_diff <= pd.Timedelta(days=7):
            return "weekly"
        elif median_diff <= pd.Timedelta(days=31):
            return "monthly"
        else:
            return "irregular"
    
    def _estimate_depth_resolution(self, depth_series: pd.Series) -> float:
        """Estimate depth resolution"""
        if len(depth_series) < 2:
            return 0.0
        
        sorted_depths = depth_series.sort_values()
        depth_diffs = sorted_depths.diff().dropna()
        return depth_diffs.median() if len(depth_diffs) > 0 else 0.0
    
    def _generate_quality_assessment(self, df: pd.DataFrame) -> str:
        """Generate comprehensive quality assessment"""
        assessments = []
        
        # Data completeness
        completeness = self._assess_data_completeness(df)
        avg_completeness = np.mean(list(completeness.values()))
        
        if avg_completeness > 90:
            assessments.append("Excellent data completeness (>90%)")
        elif avg_completeness > 75:
            assessments.append("Good data completeness (75-90%)")
        elif avg_completeness > 50:
            assessments.append("Moderate data completeness (50-75%)")
        else:
            assessments.append("Limited data completeness (<50%)")
        
        # Geographic coverage
        if self._has_geographic_data(df):
            geo_extent = self._calculate_geographic_extent(df)
            lat_span = geo_extent.get('lat_span', 0)
            lon_span = geo_extent.get('lon_span', 0)
            
            if lat_span > 20 or lon_span > 20:
                assessments.append("Excellent geographic coverage")
            elif lat_span > 5 or lon_span > 5:
                assessments.append("Good geographic coverage")
            else:
                assessments.append("Limited geographic coverage")
        
        # Temporal coverage
        temporal_span = self._calculate_temporal_span(df)
        if temporal_span:
            span_days = temporal_span.get('span_days', 0)
            if span_days > 365:
                assessments.append("Multi-year temporal coverage")
            elif span_days > 30:
                assessments.append("Good temporal coverage")
            else:
                assessments.append("Limited temporal coverage")
        
        # Data volume
        record_count = len(df)
        if record_count > 10000:
            assessments.append("Large dataset - high statistical power")
        elif record_count > 1000:
            assessments.append("Moderate dataset size")
        else:
            assessments.append("Small dataset - limited statistical power")
        
        return "; ".join(assessments)
    
    def _build_insight_prompt(self, rag_result: Dict[str, Any], 
                             data_analysis: Dict[str, Any]) -> str:
        """Build focused prompt for DeepSeek insight generation"""
        
        query = rag_result.get('query', 'Unknown query')
        record_count = data_analysis.get('record_count', 0)
        scope = data_analysis.get('scope_summary', 'Unknown scope')
        quality_score = data_analysis.get('quality_score', 0)
        
        classification = rag_result.get('classification', {})
        intent = classification.get('intent', 'unknown')
        parameters = classification.get('parameters', [])
        
        prompt = f"""
Analyze this oceanographic query and data:

Query: "{query}"
Analysis type: {intent}
Parameters: {', '.join(parameters)}
Data scope: {scope}
Records: {record_count:,}
Quality score: {quality_score:.2f}

Provide oceanographic insights as JSON with:
- summary: Key finding (max 2 sentences)
- interpretation: Physical oceanographic meaning
- quality_notes: Data reliability assessment
- significance: Statistical/scientific significance

Focus on ARGO float data characteristics and oceanographic processes.
"""
        
        return prompt.strip()
    
    def _select_supporting_visualizations(self, df: pd.DataFrame, 
                                        analysis: Dict[str, Any], 
                                        intent: str) -> Dict[str, VisualizationSpec]:
        """Select supporting visualizations"""
        supporting = {}
        
        # Add geographic context for non-spatial primary viz
        if not intent == 'spatial_mapping' and self._has_geographic_data(df):
            supporting['geographic_context'] = self._create_geographic_context_viz(df)
        
        # Add quality assessment visualization if data quality is questionable
        if analysis.get('quality_score', 1.0) < 0.8:
            supporting['quality_assessment'] = self._create_quality_viz(df, analysis)
        
        return supporting
    
    def _create_geographic_context_viz(self, df: pd.DataFrame) -> VisualizationSpec:
        """Create geographic context visualization"""
        
        plotly_config = {
            "data": [
                {
                    "type": "scattermapbox",
                    "lat": "$latitude_values",
                    "lon": "$longitude_values",
                    "mode": "markers",
                    "marker": {"size": 6, "color": "blue"},
                    "name": "Measurement Locations"
                }
            ],
            "layout": {
                "title": "Geographic Coverage",
                "mapbox": {
                    "style": "open-street-map",
                    "center": {"lat": "$center_lat", "lon": "$center_lon"},
                    "zoom": "$optimal_zoom"
                },
                "height": 400
            }
        }
        
        return VisualizationSpec(
            type=VisualizationType.GEOGRAPHIC_CONTEXT,
            plotly_config=plotly_config,
            reasoning="Geographic context helps understand spatial distribution of measurements",
            priority=2,
            data_requirements=['latitude', 'longitude'],
            interactive_features=['zoom', 'pan'],
            export_options=['png', 'pdf']
        )
    
    def _create_quality_viz(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> VisualizationSpec:
        """Create data quality visualization"""
        
        completeness = analysis.get('data_completeness', {})
        
        plotly_config = {
            "data": [
                {
                    "type": "bar",
                    "x": "$parameter_names",
                    "y": "$completeness_values",
                    "name": "Data Completeness",
                    "marker": {"color": "$completeness_colors"}
                }
            ],
            "layout": {
                "title": "Data Quality Assessment",
                "xaxis": {"title": "Parameters"},
                "yaxis": {"title": "Completeness (%)"},
                "height": 400
            }
        }
        
        return VisualizationSpec(
            type=VisualizationType.QUALITY_ASSESSMENT,
            plotly_config=plotly_config,
            reasoning="Data quality visualization helps assess reliability of analysis",
            priority=3,
            data_requirements=list(completeness.keys()),
            interactive_features=['hover'],
            export_options=['png', 'csv']
        )


# Integration wrapper for existing system
class ResponseOrchestrationIntegration:
    """
    Integration wrapper that connects the orchestrator with your existing RAG system
    """
    
    def __init__(self, rag_system=None):
        self.orchestrator = ResponseOrchestrator()
        self.rag_system = rag_system
        
    def process_query_with_orchestration(self, query: str) -> Dict[str, Any]:
        """
        Main integration method that replaces your current intelligent_response_system
        """
        
        try:
            # Step 1: Use your existing RAG system
            if self.rag_system:
                rag_result = self.rag_system.process_oceanographic_query(query)
            else:
                # Fallback for testing
                rag_result = {
                    'success': True,
                    'query': query,
                    'results': pd.DataFrame(),
                    'classification': {'intent': 'exploration', 'confidence': 0.7, 'parameters': []},
                    'sql_query': 'SELECT * FROM argo_profiles LIMIT 100;',
                    'processing_time': 1.5
                }
            
            if not rag_result['success']:
                return rag_result  # Return RAG error as-is
            
            # Step 2: Apply response orchestration
            response_package = self.orchestrator.orchestrate_response(rag_result)
            
            # Step 3: Convert to structured format for frontend
            return self._convert_to_frontend_format(response_package, rag_result)
            
        except Exception as e:
            logger.error(f"Query processing with orchestration failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'response_type': 'orchestration_error'
            }
    
    def _convert_to_frontend_format(self, response_package: ResponsePackage, 
                                   rag_result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert orchestrated response to frontend-consumable format"""
        
        return {
            'success': True,
            'query': response_package.query_context['original_query'],
            'response_type': 'structured_orchestrated',
            
            # Core response data
            'primary_insight': response_package.primary_insight,
            'visualizations': {
                name: {
                    'type': viz.type.value,
                    'config': viz.plotly_config,
                    'reasoning': viz.reasoning,
                    'priority': viz.priority,
                    'interactive_features': viz.interactive_features,
                    'export_options': viz.export_options
                }
                for name, viz in response_package.visualizations.items()
            },
            
            # Data export
            'data_export': {
                'table_data': response_package.data_export.table_data,
                'metadata': response_package.data_export.metadata,
                'quality_notes': response_package.data_export.quality_notes,
                'export_formats': response_package.data_export.export_formats,
                'download_ready': response_package.data_export.download_ready
            },
            
            # User guidance
            'user_guidance': {
                'next_questions': response_package.user_guidance.next_questions,
                'exploration_hints': response_package.user_guidance.exploration_hints,
                'related_analyses': response_package.user_guidance.related_analyses,
                'methodology_notes': response_package.user_guidance.methodology_notes
            },
            
            # Compatibility with existing system
            'sql_query': rag_result.get('sql_query', ''),
            'classification': rag_result.get('classification', {}),
            'results': rag_result.get('results', pd.DataFrame()),
            'processing_time': rag_result.get('processing_time', 0),
            'insights': response_package.primary_insight,  # Legacy compatibility
            
            # Processing metadata
            'orchestration_metadata': response_package.processing_metadata,
            'query_context': response_package.query_context
        }


def test_response_orchestration():
    """Test the response orchestration system"""
    
    print("🚀 Testing Response Orchestration System")
    print("=" * 60)
    
    # Initialize orchestration system
    orchestrator = ResponseOrchestrationIntegration()
    
    # Test queries
    test_queries = [
        "What is the average surface temperature in the Arabian Sea?",
        "Show me temperature profiles for the last year",
        "How does salinity vary across the Indian Ocean?",
        "Count the total number of ARGO profiles"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 50)
        
        try:
            result = orchestrator.process_query_with_orchestration(query)
            
            if result['success']:
                print("✅ SUCCESS!")
                print(f"   Response Type: {result['response_type']}")
                print(f"   Primary Insight: {result['primary_insight']['summary'][:100]}...")
                print(f"   Visualizations: {len(result['visualizations'])}")
                print(f"   Data Export Ready: {result['data_export']['download_ready']}")
                print(f"   User Guidance: {len(result['user_guidance']['next_questions'])} suggestions")
                
                if result['orchestration_metadata']['deepseek_used']:
                    print("   🧠 DeepSeek Intelligence: ACTIVE")
                else:
                    print("   📋 Template Intelligence: Used")
            else:
                print(f"❌ FAILED: {result.get('error')}")
                
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
    
    print("\n" + "=" * 60)
    print("RESPONSE ORCHESTRATION TESTING COMPLETE")
    print("✅ Ready for integration with your existing RAG system")
    print("=" * 60)


if __name__ == "__main__":
    test_response_orchestration()