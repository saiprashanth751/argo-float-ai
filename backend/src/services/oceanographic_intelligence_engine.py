# src/services/oceanographic_intelligence_engine.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import re
import logging
from datetime import datetime, timedelta
import json
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Sophisticated query intent classification for oceanographic data"""
    PROFILE_ANALYSIS = "profile_analysis"
    SPATIAL_MAPPING = "spatial_mapping"
    TEMPORAL_TRENDS = "temporal_trends"
    STATISTICAL_SUMMARY = "statistical_summary"
    ANOMALY_DETECTION = "anomaly_detection"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    PHYSICAL_PROPERTIES = "physical_properties"
    QUALITY_ASSESSMENT = "quality_assessment"
    PREDICTIVE_ANALYSIS = "predictive_analysis"
    EXPLORATION = "exploration"

class ComplexityLevel(Enum):
    """Query complexity levels for intelligent routing"""
    BASIC = "basic"          # Simple data retrieval
    INTERMEDIATE = "intermediate"  # Calculations and aggregations
    ADVANCED = "advanced"    # Complex analytics and insights
    EXPERT = "expert"        # Research-level analysis

@dataclass
class OceanographicContext:
    """Rich context for oceanographic queries"""
    parameters: List[str]
    depth_range: Optional[Tuple[float, float]]
    spatial_bounds: Optional[Dict[str, float]]
    temporal_range: Optional[Tuple[datetime, datetime]]
    analysis_type: str
    physical_processes: List[str]
    data_quality_requirements: str

@dataclass
class QueryClassification:
    """Complete query classification result"""
    intent: QueryIntent
    complexity: ComplexityLevel
    context: OceanographicContext
    confidence: float
    suggested_approach: str
    required_calculations: List[str]

class OceanographicIntelligenceEngine:
    """
    Advanced oceanographic intelligence system that understands
    the scientific context behind queries and provides intelligent responses
    """
    
    def __init__(self, db_engine=None):
        self.engine = db_engine
        self.parameter_relationships = self._build_parameter_relationships()
        self.physical_process_keywords = self._build_physical_process_keywords()
        self.regional_boundaries = self._build_regional_boundaries()
        
        # Cache for calculated properties
        self.calculation_cache = {}
        
        # Initialize oceanographic constants
        self.SEAWATER_CONSTANTS = {
            'reference_pressure': 0.0,  # dbar
            'reference_temperature': 15.0,  # Celsius
            'reference_salinity': 35.0,  # PSU
            'gravity': 9.80665,  # m/s^2
            'earth_rotation': 7.2921e-5  # rad/s
        }
    
    def _build_parameter_relationships(self) -> Dict[str, Dict]:
        """Build parameter relationships for intelligent analysis"""
        return {
            'temperature': {
                'related_params': ['salinity', 'pressure', 'density'],
                'derived_properties': ['potential_temperature', 'conservative_temperature'],
                'physical_processes': ['mixing', 'advection', 'convection', 'upwelling'],
                'typical_ranges': {'surface': (20, 30), 'deep': (1, 4)},
                'units': 'degrees_celsius'
            },
            'salinity': {
                'related_params': ['temperature', 'density'],
                'derived_properties': ['absolute_salinity'],
                'physical_processes': ['evaporation', 'precipitation', 'mixing', 'river_input'],
                'typical_ranges': {'surface': (34, 37), 'deep': (34.6, 34.8)},
                'units': 'psu'
            },
            'pressure': {
                'related_params': ['depth', 'density'],
                'derived_properties': ['depth', 'potential_density'],
                'physical_processes': ['hydrostatic_balance'],
                'conversion_factor': 1.0194,  # dbar to meters (approximate)
                'units': 'dbar'
            },
            'density': {
                'related_params': ['temperature', 'salinity', 'pressure'],
                'derived_properties': ['potential_density', 'buoyancy_frequency'],
                'physical_processes': ['stratification', 'mixing', 'convection'],
                'typical_ranges': {'surface': (1020, 1027), 'deep': (1027, 1028)},
                'units': 'kg_m3'
            }
        }
    
    def _build_physical_process_keywords(self) -> Dict[str, List[str]]:
        """Build keywords for physical process identification"""
        return {
            'mixing': ['mix', 'turbul', 'vertical', 'convection', 'stirring'],
            'stratification': ['stratif', 'layer', 'thermocline', 'pycnocline', 'halocline'],
            'upwelling': ['upwell', 'divergence', 'coastal', 'equatorial'],
            'currents': ['current', 'flow', 'circulation', 'transport', 'advection'],
            'fronts': ['front', 'boundary', 'transition', 'gradient'],
            'eddies': ['eddy', 'vortex', 'mesoscale', 'swirl', 'circulation'],
            'waves': ['wave', 'internal', 'tidal', 'oscillation'],
            'air_sea': ['surface', 'atmosphere', 'heat flux', 'gas exchange']
        }
    
    def _build_regional_boundaries(self) -> Dict[str, Dict]:
        """Define regional boundaries for spatial context"""
        return {
            'arabian_sea': {
                'lat_range': (10, 25), 'lon_range': (50, 78),
                'characteristics': ['low_oxygen', 'monsoon_influence', 'upwelling']
            },
            'bay_of_bengal': {
                'lat_range': (5, 22), 'lon_range': (78, 100),
                'characteristics': ['freshwater_influence', 'cyclones', 'river_discharge']
            },
            'indian_ocean_central': {
                'lat_range': (-20, 10), 'lon_range': (60, 100),
                'characteristics': ['deep_water', 'equatorial_currents', 'dipole']
            },
            'southern_ocean': {
                'lat_range': (-60, -30), 'lon_range': (20, 150),
                'characteristics': ['circumpolar_current', 'deep_water_formation', 'stormy']
            }
        }
    
    def classify_query(self, query: str) -> QueryClassification:
        """
        Perform sophisticated classification of oceanographic queries
        """
        query_lower = query.lower()
        
        # Initialize classification components
        intent = self._determine_intent(query_lower)
        complexity = self._assess_complexity(query_lower, intent)
        context = self._extract_oceanographic_context(query_lower)
        confidence = self._calculate_confidence(query_lower, intent, context)
        
        # Determine suggested approach
        suggested_approach = self._suggest_analysis_approach(intent, complexity, context)
        
        # Identify required calculations
        required_calculations = self._identify_required_calculations(query_lower, intent, context)
        
        return QueryClassification(
            intent=intent,
            complexity=complexity,
            context=context,
            confidence=confidence,
            suggested_approach=suggested_approach,
            required_calculations=required_calculations
        )
    
    def _determine_intent(self, query_lower: str) -> QueryIntent:
        """Determine the primary intent of the query"""
        
        intent_patterns = {
            QueryIntent.PROFILE_ANALYSIS: [
                'profile', 'vertical', 'depth', 'cast', 'sounding',
                'thermocline', 'pycnocline', 'mixed layer'
            ],
            QueryIntent.SPATIAL_MAPPING: [
                'map', 'spatial', 'distribution', 'geographic', 'region',
                'latitude', 'longitude', 'area', 'basin'
            ],
            QueryIntent.TEMPORAL_TRENDS: [
                'trend', 'time', 'seasonal', 'monthly', 'yearly',
                'change', 'evolution', 'history', 'series'
            ],
            QueryIntent.STATISTICAL_SUMMARY: [
                'average', 'mean', 'median', 'statistics', 'summary',
                'count', 'total', 'maximum', 'minimum', 'range'
            ],
            QueryIntent.ANOMALY_DETECTION: [
                'anomaly', 'unusual', 'extreme', 'outlier', 'abnormal',
                'deviation', 'strange', 'rare', 'exceptional'
            ],
            QueryIntent.COMPARATIVE_ANALYSIS: [
                'compare', 'comparison', 'versus', 'difference', 'between',
                'contrast', 'relative', 'against'
            ],
            QueryIntent.PHYSICAL_PROPERTIES: [
                'density', 'buoyancy', 'stability', 'mixing', 'stratification',
                'potential', 'conservative', 'derived', 'calculated'
            ],
            QueryIntent.QUALITY_ASSESSMENT: [
                'quality', 'valid', 'missing', 'error', 'flag',
                'reliable', 'accurate', 'precision', 'uncertainty'
            ],
            QueryIntent.PREDICTIVE_ANALYSIS: [
                'predict', 'forecast', 'future', 'projection', 'model',
                'estimate', 'expect', 'anticipate'
            ]
        }
        
        intent_scores = {}
        for intent, keywords in intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        if not intent_scores:
            return QueryIntent.EXPLORATION
        
        return max(intent_scores.items(), key=lambda x: x[1])[0]
    
    def _assess_complexity(self, query_lower: str, intent: QueryIntent) -> ComplexityLevel:
        """Assess the complexity level of the query"""
        
        complexity_indicators = {
            ComplexityLevel.BASIC: ['show', 'get', 'find', 'list', 'display'],
            ComplexityLevel.INTERMEDIATE: ['average', 'calculate', 'group', 'aggregate', 'summarize'],
            ComplexityLevel.ADVANCED: ['analyze', 'correlate', 'relationship', 'pattern', 'trend'],
            ComplexityLevel.EXPERT: ['predict', 'model', 'complex', 'research', 'investigate']
        }
        
        # Base complexity from intent
        intent_complexity = {
            QueryIntent.EXPLORATION: ComplexityLevel.BASIC,
            QueryIntent.STATISTICAL_SUMMARY: ComplexityLevel.INTERMEDIATE,
            QueryIntent.SPATIAL_MAPPING: ComplexityLevel.INTERMEDIATE,
            QueryIntent.PROFILE_ANALYSIS: ComplexityLevel.INTERMEDIATE,
            QueryIntent.TEMPORAL_TRENDS: ComplexityLevel.ADVANCED,
            QueryIntent.COMPARATIVE_ANALYSIS: ComplexityLevel.ADVANCED,
            QueryIntent.ANOMALY_DETECTION: ComplexityLevel.ADVANCED,
            QueryIntent.PHYSICAL_PROPERTIES: ComplexityLevel.ADVANCED,
            QueryIntent.QUALITY_ASSESSMENT: ComplexityLevel.INTERMEDIATE,
            QueryIntent.PREDICTIVE_ANALYSIS: ComplexityLevel.EXPERT
        }
        
        base_complexity = intent_complexity.get(intent, ComplexityLevel.BASIC)
        
        # Adjust based on keywords
        keyword_scores = {}
        for complexity, keywords in complexity_indicators.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                keyword_scores[complexity] = score
        
        if keyword_scores:
            keyword_complexity = max(keyword_scores.items(), key=lambda x: x[1])[0]
            # Take the higher complexity level
            complexity_order = [ComplexityLevel.BASIC, ComplexityLevel.INTERMEDIATE, 
                              ComplexityLevel.ADVANCED, ComplexityLevel.EXPERT]
            base_idx = complexity_order.index(base_complexity)
            keyword_idx = complexity_order.index(keyword_complexity)
            return complexity_order[max(base_idx, keyword_idx)]
        
        return base_complexity
    
    def _extract_oceanographic_context(self, query_lower: str) -> OceanographicContext:
        """Extract rich oceanographic context from the query"""
        
        # Extract parameters
        parameters = []
        for param, info in self.parameter_relationships.items():
            if any(keyword in query_lower for keyword in [param] + info.get('related_params', [])):
                parameters.append(param)
        
        # Extract depth range
        depth_range = self._extract_depth_range(query_lower)
        
        # Extract spatial bounds
        spatial_bounds = self._extract_spatial_bounds(query_lower)
        
        # Extract temporal range
        temporal_range = self._extract_temporal_range(query_lower)
        
        # Determine analysis type
        analysis_type = self._determine_analysis_type(query_lower)
        
        # Identify physical processes
        physical_processes = []
        for process, keywords in self.physical_process_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                physical_processes.append(process)
        
        # Assess data quality requirements
        quality_requirements = self._assess_quality_requirements(query_lower)
        
        return OceanographicContext(
            parameters=parameters,
            depth_range=depth_range,
            spatial_bounds=spatial_bounds,
            temporal_range=temporal_range,
            analysis_type=analysis_type,
            physical_processes=physical_processes,
            data_quality_requirements=quality_requirements
        )
    
    def _extract_depth_range(self, query_lower: str) -> Optional[Tuple[float, float]]:
        """Extract depth/pressure range from query"""
        
        # Common depth references
        depth_patterns = {
            'surface': (0, 10),
            'shallow': (0, 100),
            'intermediate': (100, 1000),
            'deep': (1000, 4000),
            'abyssal': (4000, 6000),
            'mixed layer': (0, 100),
            'thermocline': (50, 200),
            'mesopelagic': (200, 1000),
            'bathypelagic': (1000, 4000)
        }
        
        for term, depth_range in depth_patterns.items():
            if term in query_lower:
                return depth_range
        
        # Extract specific depth values
        depth_matches = re.findall(r'(\d+)\s*(?:m|meter|dbar)', query_lower)
        if depth_matches:
            depths = [float(d) for d in depth_matches]
            if len(depths) == 1:
                return (0, depths[0])
            elif len(depths) >= 2:
                return (min(depths), max(depths))
        
        return None
    
    def _extract_spatial_bounds(self, query_lower: str) -> Optional[Dict[str, float]]:
        """Extract spatial boundaries from query"""
        
        # Check for regional references
        for region, bounds in self.regional_boundaries.items():
            if region.replace('_', ' ') in query_lower:
                lat_min, lat_max = bounds['lat_range']
                lon_min, lon_max = bounds['lon_range']
                return {
                    'lat_min': lat_min, 'lat_max': lat_max,
                    'lon_min': lon_min, 'lon_max': lon_max
                }
        
        # Extract specific coordinates
        coord_pattern = r'(-?\d+(?:\.\d+)?)\s*[°,]\s*(-?\d+(?:\.\d+)?)'
        coord_matches = re.findall(coord_pattern, query_lower)
        
        if coord_matches:
            # Simple bounding box from coordinates
            lats = [float(match[0]) for match in coord_matches]
            lons = [float(match[1]) for match in coord_matches]
            
            return {
                'lat_min': min(lats), 'lat_max': max(lats),
                'lon_min': min(lons), 'lon_max': max(lons)
            }
        
        return None
    
    def _extract_temporal_range(self, query_lower: str) -> Optional[Tuple[datetime, datetime]]:
        """Extract temporal range from query"""
        
        # Year extraction
        year_matches = re.findall(r'\b(20\d{2})\b', query_lower)
        if year_matches:
            years = [int(y) for y in year_matches]
            if len(years) == 1:
                year = years[0]
                return (datetime(year, 1, 1), datetime(year, 12, 31))
            elif len(years) >= 2:
                start_year, end_year = min(years), max(years)
                return (datetime(start_year, 1, 1), datetime(end_year, 12, 31))
        
        # Relative time references
        now = datetime.now()
        if 'last year' in query_lower:
            return (datetime(now.year - 1, 1, 1), datetime(now.year - 1, 12, 31))
        elif 'last month' in query_lower:
            if now.month == 1:
                last_month = datetime(now.year - 1, 12, 1)
            else:
                last_month = datetime(now.year, now.month - 1, 1)
            return (last_month, now)
        elif 'recent' in query_lower or 'latest' in query_lower:
            return (now - timedelta(days=90), now)
        
        return None
    
    def _determine_analysis_type(self, query_lower: str) -> str:
        """Determine the type of analysis required"""
        
        analysis_types = {
            'climatology': ['climate', 'climatology', 'long-term', 'average'],
            'variability': ['variability', 'variation', 'fluctuation', 'change'],
            'correlation': ['correlation', 'relationship', 'connection', 'association'],
            'time_series': ['time series', 'temporal', 'evolution', 'progression'],
            'spatial_analysis': ['spatial', 'geographic', 'distribution', 'pattern'],
            'extreme_events': ['extreme', 'maximum', 'minimum', 'peak', 'anomaly']
        }
        
        for analysis_type, keywords in analysis_types.items():
            if any(keyword in query_lower for keyword in keywords):
                return analysis_type
        
        return 'general'
    
    def _assess_quality_requirements(self, query_lower: str) -> str:
        """Assess data quality requirements"""
        
        if any(word in query_lower for word in ['research', 'scientific', 'precise', 'accurate']):
            return 'high'
        elif any(word in query_lower for word in ['quality', 'valid', 'reliable']):
            return 'medium'
        else:
            return 'standard'
    
    def _calculate_confidence(self, query_lower: str, intent: QueryIntent, 
                            context: OceanographicContext) -> float:
        """Calculate confidence in query classification"""
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on specific parameters
        if context.parameters:
            confidence += 0.2
        
        # Increase confidence based on specific depth/spatial/temporal context
        if context.depth_range:
            confidence += 0.1
        if context.spatial_bounds:
            confidence += 0.1
        if context.temporal_range:
            confidence += 0.1
        
        # Increase confidence based on physical processes identified
        if context.physical_processes:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _suggest_analysis_approach(self, intent: QueryIntent, complexity: ComplexityLevel,
                                 context: OceanographicContext) -> str:
        """Suggest the best analysis approach"""
        
        approach_map = {
            (QueryIntent.PROFILE_ANALYSIS, ComplexityLevel.BASIC): "Direct profile retrieval with visualization",
            (QueryIntent.PROFILE_ANALYSIS, ComplexityLevel.INTERMEDIATE): "Statistical profile analysis with derived properties",
            (QueryIntent.PROFILE_ANALYSIS, ComplexityLevel.ADVANCED): "Comparative profile analysis with physical interpretation",
            
            (QueryIntent.SPATIAL_MAPPING, ComplexityLevel.BASIC): "Simple spatial distribution mapping",
            (QueryIntent.SPATIAL_MAPPING, ComplexityLevel.INTERMEDIATE): "Spatial statistics with interpolation",
            (QueryIntent.SPATIAL_MAPPING, ComplexityLevel.ADVANCED): "Advanced spatial analysis with regional comparisons",
            
            (QueryIntent.TEMPORAL_TRENDS, ComplexityLevel.INTERMEDIATE): "Time series analysis with trend detection",
            (QueryIntent.TEMPORAL_TRENDS, ComplexityLevel.ADVANCED): "Advanced temporal analysis with statistical modeling",
            (QueryIntent.TEMPORAL_TRENDS, ComplexityLevel.EXPERT): "Predictive temporal modeling with uncertainty",
            
            (QueryIntent.ANOMALY_DETECTION, ComplexityLevel.ADVANCED): "Statistical anomaly detection with contextual analysis",
            (QueryIntent.ANOMALY_DETECTION, ComplexityLevel.EXPERT): "Machine learning-based anomaly detection with physical validation"
        }
        
        key = (intent, complexity)
        return approach_map.get(key, f"Custom {complexity.value} analysis for {intent.value}")
    
    def _identify_required_calculations(self, query_lower: str, intent: QueryIntent,
                                      context: OceanographicContext) -> List[str]:
        """Identify calculations required for the query"""
        
        calculations = []
        
        # Parameter-specific calculations
        if 'temperature' in context.parameters:
            if any(word in query_lower for word in ['potential', 'theta']):
                calculations.append('potential_temperature')
            if any(word in query_lower for word in ['conservative']):
                calculations.append('conservative_temperature')
        
        if 'salinity' in context.parameters:
            if any(word in query_lower for word in ['absolute']):
                calculations.append('absolute_salinity')
        
        if 'density' in context.parameters or 'potential' in query_lower:
            calculations.append('potential_density')
        
        # Analysis-specific calculations
        if intent == QueryIntent.PHYSICAL_PROPERTIES:
            calculations.extend(['buoyancy_frequency', 'mixed_layer_depth'])
        
        if 'stratification' in context.physical_processes:
            calculations.append('stratification_index')
        
        if intent == QueryIntent.STATISTICAL_SUMMARY:
            calculations.extend(['mean', 'std', 'percentiles'])
        
        return calculations
    
    def calculate_physical_properties(self, df: pd.DataFrame, 
                                    properties: List[str]) -> pd.DataFrame:
        """Calculate oceanographic physical properties"""
        
        result_df = df.copy()
        
        for prop in properties:
            try:
                if prop == 'potential_temperature' and 'temperature' in df.columns:
                    result_df['potential_temperature'] = self._calculate_potential_temperature(
                        df['temperature'], df.get('salinity', 35), df.get('pressure', 0)
                    )
                
                elif prop == 'potential_density' and all(col in df.columns for col in ['temperature', 'salinity']):
                    result_df['potential_density'] = self._calculate_potential_density(
                        df['temperature'], df['salinity'], df.get('pressure', 0)
                    )
                
                elif prop == 'buoyancy_frequency':
                    result_df['buoyancy_frequency'] = self._calculate_buoyancy_frequency(
                        result_df.get('potential_density', df.get('density')), 
                        df.get('pressure', df.get('depth', 0))
                    )
                
                elif prop == 'mixed_layer_depth':
                    result_df['mixed_layer_depth'] = self._calculate_mixed_layer_depth(df)
                
            except Exception as e:
                logger.warning(f"Could not calculate {prop}: {e}")
        
        return result_df
    
    def _calculate_potential_temperature(self, temperature: pd.Series, 
                                       salinity: pd.Series, pressure: pd.Series) -> pd.Series:
        """Calculate potential temperature (simplified)"""
        # Simplified calculation - in practice would use GSW library
        alpha = 2e-4  # Thermal expansion coefficient
        return temperature - alpha * pressure * temperature
    
    def _calculate_potential_density(self, temperature: pd.Series, 
                                   salinity: pd.Series, pressure: pd.Series) -> pd.Series:
        """Calculate potential density (simplified)"""
        # Simplified UNESCO equation of state
        density = (999.842594 + 
                  6.793952e-2 * temperature - 
                  9.095290e-3 * temperature**2 +
                  1.001685e-4 * temperature**3 +
                  8.24493e-1 * salinity -
                  4.0899e-3 * temperature * salinity)
        
        return density
    
    def _calculate_buoyancy_frequency(self, density: pd.Series, pressure: pd.Series) -> pd.Series:
        """Calculate buoyancy frequency (N^2)"""
        if len(density) < 2:
            return pd.Series([np.nan] * len(density))
        
        # Calculate density gradient
        rho_grad = density.diff() / pressure.diff()
        
        # Buoyancy frequency squared
        g = self.SEAWATER_CONSTANTS['gravity']
        n_squared = (-g / density) * rho_grad
        
        return np.sqrt(np.abs(n_squared))
    
    def _calculate_mixed_layer_depth(self, df: pd.DataFrame) -> pd.Series:
        """Calculate mixed layer depth (simplified approach)"""
        if 'temperature' not in df.columns or 'pressure' not in df.columns:
            return pd.Series([np.nan] * len(df))
        
        # Find depth where temperature differs by 0.2°C from surface
        surface_temp = df['temperature'].iloc[0]
        temp_diff = np.abs(df['temperature'] - surface_temp)
        
        try:
            mld_idx = temp_diff[temp_diff > 0.2].index[0]
            mld = df.loc[mld_idx, 'pressure']
        except (IndexError, KeyError):
            mld = np.nan
        
        return pd.Series([mld] * len(df))
    
    def generate_insights(self, query: str, results_df: pd.DataFrame, 
                         classification: QueryClassification) -> Dict[str, Any]:
        """Generate intelligent insights from query results"""
        
        insights = {
            'summary': self._generate_summary(results_df, classification),
            'key_findings': self._identify_key_findings(results_df, classification),
            'physical_interpretation': self._provide_physical_interpretation(results_df, classification),
            'data_quality_notes': self._assess_data_quality(results_df),
            'recommendations': self._generate_recommendations(classification),
            'visualization_suggestions': self._suggest_visualizations(classification)
        }
        
        return insights
    
    def _generate_summary(self, df: pd.DataFrame, classification: QueryClassification) -> str:
        """Generate executive summary of results"""
        
        n_records = len(df)
        parameters = classification.context.parameters
        
        summary = f"Analysis of {n_records:,} measurements"
        
        if parameters:
            summary += f" for {', '.join(parameters)}"
        
        if classification.context.depth_range:
            depth_min, depth_max = classification.context.depth_range
            summary += f" in depth range {depth_min}-{depth_max}m"
        
        if classification.context.spatial_bounds:
            summary += f" within specified geographic region"
        
        return summary
    
    def _identify_key_findings(self, df: pd.DataFrame, classification: QueryClassification) -> List[str]:
        """Identify key findings from the data"""
        
        findings = []
        
        # Statistical findings
        for param in classification.context.parameters:
            if param in df.columns:
                valid_data = df[param].dropna()
                if len(valid_data) > 0:
                    mean_val = valid_data.mean()
                    std_val = valid_data.std()
                    findings.append(f"{param.title()}: mean = {mean_val:.2f}, std = {std_val:.2f}")
        
        # Range findings
        if 'pressure' in df.columns:
            max_depth = df['pressure'].max()
            findings.append(f"Maximum depth surveyed: {max_depth:.1f} dbar")
        
        return findings
    
    def _provide_physical_interpretation(self, df: pd.DataFrame, 
                                       classification: QueryClassification) -> str:
        """Provide physical oceanographic interpretation"""
        
        interpretations = []
        
        # Temperature-specific interpretations
        if 'temperature' in df.columns:
            temp_range = df['temperature'].max() - df['temperature'].min()
            if temp_range > 10:
                interpretations.append("Large temperature range suggests significant vertical stratification")
        
        # Salinity interpretations
        if 'salinity' in df.columns:
            sal_std = df['salinity'].std()
            if sal_std > 0.5:
                interpretations.append("High salinity variability indicates strong mixing or freshwater influence")
        
        # Physical process interpretations
        processes = classification.context.physical_processes
        if 'mixing' in processes:
            interpretations.append("Analysis relevant to ocean mixing processes")
        if 'upwelling' in processes:
            interpretations.append("Data may show signatures of upwelling dynamics")
        
        return "; ".join(interpretations) if interpretations else "No specific physical interpretation available"
    
    def _assess_data_quality(self, df: pd.DataFrame) -> str:
        """Assess quality of the dataset"""
        
        total_records = len(df)
        
        quality_notes = []
        
        # Missing data assessment
        for col in df.columns:
            if col not in ['id', 'metadata_id']:
                missing_pct = df[col].isna().sum() / total_records * 100
                if missing_pct > 50:
                    quality_notes.append(f"High missing data in {col} ({missing_pct:.1f}%)")
                elif missing_pct > 20:
                    quality_notes.append(f"Moderate missing data in {col} ({missing_pct:.1f}%)")
        
        # Data range assessment
        for col in ['temperature', 'salinity', 'pressure']:
            if col in df.columns:
                valid_data = df[col].dropna()
                if len(valid_data) > 0:
                    param_info = self.parameter_relationships.get(col, {})
                    typical_ranges = param_info.get('typical_ranges', {})
                    
                    if typical_ranges:
                        min_val, max_val = valid_data.min(), valid_data.max()
                        # Check against typical ranges
                        surface_range = typical_ranges.get('surface', (float('-inf'), float('inf')))
                        deep_range = typical_ranges.get('deep', (float('-inf'), float('inf')))
                        
                        if min_val < min(surface_range[0], deep_range[0]) or max_val > max(surface_range[1], deep_range[1]):
                            quality_notes.append(f"Unusual {col} values detected - review recommended")
        
        if not quality_notes:
            return "Data quality appears good with reasonable coverage and values"
        else:
            return "; ".join(quality_notes)
    
    def _generate_recommendations(self, classification: QueryClassification) -> List[str]:
        """Generate analysis recommendations"""
        
        recommendations = []
        
        # Intent-based recommendations
        if classification.intent == QueryIntent.PROFILE_ANALYSIS:
            recommendations.append("Consider calculating derived properties (potential temperature, density)")
            recommendations.append("Visualize profiles with depth on y-axis, parameters on x-axis")
        
        elif classification.intent == QueryIntent.SPATIAL_MAPPING:
            recommendations.append("Create geographic distribution maps")
            recommendations.append("Consider interpolation for spatial continuity")
            
        elif classification.intent == QueryIntent.TEMPORAL_TRENDS:
            recommendations.append("Apply statistical trend analysis")
            recommendations.append("Consider seasonal decomposition")
        
        elif classification.intent == QueryIntent.ANOMALY_DETECTION:
            recommendations.append("Apply outlier detection methods")
            recommendations.append("Validate anomalies against physical oceanography principles")
        
        # Complexity-based recommendations
        if classification.complexity == ComplexityLevel.EXPERT:
            recommendations.append("Consider advanced statistical methods")
            recommendations.append("Validate results against published literature")
        
        # Parameter-specific recommendations
        if 'temperature' in classification.context.parameters and 'salinity' in classification.context.parameters:
            recommendations.append("Generate T-S diagrams for water mass analysis")
        
        return recommendations
    
    def _suggest_visualizations(self, classification: QueryClassification) -> List[str]:
        """Suggest appropriate visualizations"""
        
        viz_suggestions = []
        
        # Intent-based visualizations
        viz_map = {
            QueryIntent.PROFILE_ANALYSIS: ["Line plots (depth vs parameter)", "Scatter plots with depth coloring"],
            QueryIntent.SPATIAL_MAPPING: ["Geographic scatter plots", "Contour maps", "Heatmaps"],
            QueryIntent.TEMPORAL_TRENDS: ["Time series plots", "Seasonal decomposition plots"],
            QueryIntent.STATISTICAL_SUMMARY: ["Histograms", "Box plots", "Violin plots"],
            QueryIntent.COMPARATIVE_ANALYSIS: ["Side-by-side plots", "Difference plots"],
            QueryIntent.PHYSICAL_PROPERTIES: ["T-S diagrams", "Density profiles", "Stratification plots"]
        }
        
        base_suggestions = viz_map.get(classification.intent, ["Standard line plots"])
        viz_suggestions.extend(base_suggestions)
        
        # Parameter-specific visualizations
        params = classification.context.parameters
        if 'temperature' in params and 'salinity' in params:
            viz_suggestions.append("Temperature-Salinity (T-S) diagrams")
        
        if len(params) > 2:
            viz_suggestions.append("Multi-parameter correlation matrix")
        
        return viz_suggestions

# Example usage and testing functions
def test_oceanographic_intelligence():
    """Test the oceanographic intelligence engine"""
    
    print("🧠 Testing Oceanographic Intelligence Engine")
    print("=" * 50)
    
    engine = OceanographicIntelligenceEngine()
    
    # Test queries with different complexity levels
    test_queries = [
        "Show temperature profile for float 1900121",
        "What is the average surface temperature in the Arabian Sea?",
        "Compare thermocline depth between Arabian Sea and Bay of Bengal",
        "Identify unusual salinity anomalies in recent measurements",
        "Analyze seasonal temperature trends in the Indian Ocean",
        "Calculate potential density for deep water masses",
        "Find areas with strong vertical mixing",
        "Predict upwelling zones based on temperature gradients"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 40)
        
        classification = engine.classify_query(query)
        
        print(f"Intent: {classification.intent.value}")
        print(f"Complexity: {classification.complexity.value}")
        print(f"Confidence: {classification.confidence:.2f}")
        print(f"Parameters: {classification.context.parameters}")
        print(f"Physical Processes: {classification.context.physical_processes}")
        print(f"Depth Range: {classification.context.depth_range}")
        print(f"Approach: {classification.suggested_approach}")
        print(f"Required Calculations: {classification.required_calculations}")
        
        # Test insight generation with sample data
        sample_df = pd.DataFrame({
            'temperature': np.random.normal(15, 5, 100),
            'salinity': np.random.normal(35, 0.5, 100),
            'pressure': np.linspace(0, 2000, 100)
        })
        
        insights = engine.generate_insights(query, sample_df, classification)
        print(f"Summary: {insights['summary']}")
        print(f"Key Findings: {insights['key_findings'][:2]}")  # Show first 2 findings
        print(f"Visualizations: {insights['visualization_suggestions'][:2]}")  # Show first 2 suggestions

if __name__ == "__main__":
    test_oceanographic_intelligence()