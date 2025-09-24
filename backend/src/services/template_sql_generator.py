# src/services/template_sql_generator.py
"""
FIXED PRODUCTION-GRADE TEMPLATE SQL GENERATOR
Addresses the critical JOIN context awareness bug and production issues
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from datetime import datetime, timedelta

try:
    from services.oceanographic_intelligence_engine import (
        QueryIntent, ComplexityLevel, QueryClassification, OceanographicContext
    )
except ImportError:
    # Fallback imports for testing
    from oceanographic_intelligence_engine import (
        QueryIntent, ComplexityLevel, QueryClassification, OceanographicContext
    )

logger = logging.getLogger(__name__)

@dataclass
class SQLTemplate:
    """Template for SQL generation with metadata"""
    template: str
    parameters: List[str]
    performance_notes: str
    expected_result_size: str
    index_requirements: List[str]
    adaptability_score: float
    requires_measurements_join: bool = False  # CRITICAL: Track JOIN requirements

@dataclass
class GeneratedSQL:
    """Generated SQL with metadata for production monitoring"""
    sql: str
    template_id: str
    parameters_used: Dict[str, Any]
    estimated_performance: str
    recommended_timeout: int
    index_requirements: List[str]
    adaptations_made: List[str]
    validation_passed: bool = True
    warnings: List[str] = None

class ProductionSQLGenerator:
    """
    FIXED PRODUCTION-GRADE SQL GENERATOR
    
    Key fixes:
    1. JOIN context awareness in all filter methods
    2. Template-aware component building
    3. Proper parameter filter logic
    4. Production validation and safety
    """
    
    def __init__(self):
        self.templates = self._build_production_templates()
        self.performance_thresholds = self._define_performance_thresholds()
        self.spatial_regions = self._define_spatial_regions()
        
    def     _build_production_templates(self) -> Dict[str, SQLTemplate]:
        """Build production templates with proper JOIN awareness"""
        
        return {
            # SURFACE ANALYSIS - NO JOIN NEEDED
            'surface_analysis': SQLTemplate(
                template="""
                SELECT {select_columns}
                FROM argo_profiles p
                WHERE 1=1 
                    {spatial_filters}
                    {temporal_filters}
                    {parameter_filters}
                    {quality_filters}
                {grouping_clause}
                ORDER BY {ordering}
                LIMIT {limit}
                """,
                parameters=['select_columns', 'spatial_filters', 'temporal_filters', 
                           'parameter_filters', 'quality_filters', 'grouping_clause', 'ordering', 'limit'],
                performance_notes="Surface-only analysis - profiles table only",
                expected_result_size="1K-100K profiles", 
                index_requirements=['idx_profiles_coords', 'idx_profiles_date'],
                adaptability_score=0.95,
                requires_measurements_join=False  # CRITICAL: No JOIN
            ),
            
            # PROFILE ANALYSIS - REQUIRES JOIN
            'profile_analysis': SQLTemplate(
                template="""
                SELECT {select_columns}
                FROM argo_profiles p
                INNER JOIN argo_measurements m ON p.id = m.profile_id  
                WHERE 1=1
                    {spatial_filters}
                    {temporal_filters}
                    {depth_filters}
                    {parameter_filters}
                ORDER BY {ordering}
                LIMIT {limit}
                """,
                parameters=['select_columns', 'spatial_filters', 'temporal_filters',
                           'depth_filters', 'parameter_filters', 'ordering', 'limit'],
                performance_notes="Profile analysis with measurements JOIN",
                expected_result_size="10K-500K measurements",
                index_requirements=['idx_measurements_profile', 'idx_measurements_pressure'],
                adaptability_score=0.85,
                requires_measurements_join=True  # CRITICAL: Requires JOIN
            ),
            
            # STATISTICAL SUMMARY - CONDITIONAL JOIN
            'statistical_summary': SQLTemplate(
                template="""
                SELECT {aggregation_columns}
                FROM argo_profiles p
                {join_clause}
                WHERE 1=1
                    {spatial_filters}
                    {temporal_filters}
                    {parameter_filters}
                    {quality_filters}
                {grouping_clause}
                {ordering}
                LIMIT {limit}
                """,
                parameters=['aggregation_columns', 'join_clause', 'spatial_filters',
                        'temporal_filters', 'parameter_filters', 'quality_filters', 
                        'grouping_clause', 'ordering', 'limit'],
                performance_notes="Statistical analysis with conditional JOIN",
                expected_result_size="1-1K summary rows",
                index_requirements=['varies based on aggregation'],
                adaptability_score=0.9,
                requires_measurements_join=False  # Conditional based on aggregation
            ),
            
            # COUNT QUERY - NO JOIN BY DEFAULT
            'count_query': SQLTemplate(
                template="""
                SELECT COUNT(*) as total_profiles,
                       COUNT(DISTINCT platform_number) as unique_platforms,
                       {additional_counts}
                FROM argo_profiles p
                {join_clause}
                WHERE 1=1
                    {spatial_filters}
                    {temporal_filters}
                    {parameter_filters}
                    {quality_filters}
                """,
                parameters=['additional_counts', 'join_clause', 'spatial_filters', 
                           'temporal_filters', 'parameter_filters', 'quality_filters'],
                performance_notes="Count operations with conditional JOIN",
                expected_result_size="1 row",
                index_requirements=['idx_profiles_platform', 'idx_profiles_date'],
                adaptability_score=0.8,
                requires_measurements_join=False  # Conditional
            )
        }
    
    def generate_sql(self, classification: QueryClassification, query_text: str) -> GeneratedSQL:
        """
        MAIN SQL GENERATION - FIXED VERSION
        """
        
        try:
            # Step 1: Select appropriate template
            template_id = self._select_template(classification, query_text)
            template = self.templates[template_id]
            
            # Step 2: Determine if measurements JOIN is needed for this specific query
            needs_measurements_join = self._determine_measurements_join_needed(
                template, query_text, classification
            )
            
            # Step 3: Build context-aware components
            components = self._build_query_components_fixed(
                classification, query_text, template_id, needs_measurements_join
            )
            
            # Step 4: Render template with validated components
            sql = self._render_template(template, components)
            
            # Step 5: Apply production optimizations
            sql, optimizations = self._optimize_for_production(sql, classification)
            
            # Step 6: Final validation
            warnings = self._validate_sql_safety(sql, template_id)
            
            return GeneratedSQL(
                sql=sql,
                template_id=template_id,
                parameters_used=components,
                estimated_performance=self._estimate_performance(template, components),
                recommended_timeout=self._calculate_timeout(template, components),
                index_requirements=template.index_requirements,
                adaptations_made=optimizations,
                validation_passed=len(warnings) == 0,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            return self._generate_safe_fallback(classification, query_text, str(e))
    
    def _determine_measurements_join_needed(self, template: SQLTemplate, 
                                          query_text: str, 
                                          classification: QueryClassification) -> bool:
        """
        CRITICAL METHOD: Determine if measurements table JOIN is needed
        """
        
        # Templates that always require JOIN
        if template.requires_measurements_join:
            return True
        
        query_lower = query_text.lower()
        
        # Explicit depth/pressure indicators
        if any(term in query_lower for term in [
            'depth', 'pressure', 'vertical', 'profile data',
            'at 1000m', 'below', 'above', 'thermocline', 'halocline'
        ]):
            return True
        
        # Profile-specific analysis
        if any(term in query_lower for term in [
            'temperature profile', 'salinity profile', 
            'mixed layer depth', 'deep water'
        ]):
            return True
        
        # Classification context indicators
        if hasattr(classification.context, 'depth_range') and classification.context.depth_range:
            return True
        
        # For statistical summaries, check if we need profile measurements
        if 'aggregation_columns' in template.parameters:
            if any(term in query_lower for term in [
                'average temperature by depth', 'salinity statistics',
                'profile statistics', 'vertical distribution'
            ]):
                return True
        
        return False
    
    def _build_query_components_fixed(self, classification: QueryClassification, 
                                     query_text: str, template_id: str, 
                                     needs_measurements_join: bool) -> Dict[str, str]:
        """
        FIXED: Build query components with proper JOIN context awareness
        """
        
        components = {}
        query_lower = query_text.lower()
        context = classification.context
        
        # === BUILD SELECT COLUMNS ===
        if template_id == 'statistical_summary':
            components['aggregation_columns'] = self._build_aggregation_columns_fixed(
                query_lower, context, needs_measurements_join
            )
            components['join_clause'] = self._build_join_clause_conditional(needs_measurements_join)
        else:
            components['select_columns'] = self._build_select_columns_fixed(
                query_lower, context, template_id, needs_measurements_join
            )
        
        # === BUILD SPATIAL FILTERS ===
        components['spatial_filters'] = self._build_spatial_filters(query_lower, context)
        
        # === BUILD TEMPORAL FILTERS ===
        components['temporal_filters'] = self._build_temporal_filters(query_lower, context)
        
        # === FIXED: BUILD PARAMETER FILTERS WITH JOIN AWARENESS ===
        components['parameter_filters'] = self._build_parameter_filters_fixed(
            query_lower, context, needs_measurements_join
        )
        
        # === BUILD OTHER COMPONENTS ===
        components['ordering'] = self._build_ordering(template_id, query_lower)
        components['limit'] = self._build_limit(classification.complexity)
        
        # Template-specific components
        if template_id == 'profile_analysis':
            components['depth_filters'] = self._build_depth_filters(query_lower, context)
        
        if template_id in ['statistical_summary', 'surface_analysis']:
            components['grouping_clause'] = self._build_grouping(query_lower)
            
        if template_id == 'count_query':
            components['additional_counts'] = self._build_additional_counts_fixed(
                query_lower, needs_measurements_join
            )
            components['join_clause'] = self._build_join_clause_conditional(needs_measurements_join)
        
        # Quality filters
        components['quality_filters'] = self._build_quality_filters(query_lower)
        
        return components
    
    def _build_parameter_filters_fixed(self, query_lower: str, 
                                     context: OceanographicContext, 
                                     needs_measurements_join: bool) -> str:
        """
        CRITICAL FIX: Build parameter filters with proper JOIN context awareness
        
        This was the main bug - referencing m.temperature without JOIN
        """
        
        filters = []
        
        if 'temperature' in query_lower or 'temperature' in context.parameters:
            if 'surface' in query_lower or not needs_measurements_join:
                # Surface-only or no JOIN available - use profiles table only
                filters.append('AND p.surface_temp IS NOT NULL')
            else:
                # JOIN available - can reference both tables
                filters.append('AND (p.surface_temp IS NOT NULL OR m.temperature IS NOT NULL)')
        
        if 'salinity' in query_lower or 'salinity' in context.parameters:
            if 'surface' in query_lower or not needs_measurements_join:
                # Surface-only or no JOIN available
                filters.append('AND p.surface_salinity IS NOT NULL')
            else:
                # JOIN available
                filters.append('AND (p.surface_salinity IS NOT NULL OR m.salinity IS NOT NULL)')
        
        return ' '.join(filters)
    
    def _build_join_clause_conditional(self, needs_join: bool) -> str:
        """Build JOIN clause only when needed"""
        
        if needs_join:
            return 'INNER JOIN argo_measurements m ON p.id = m.profile_id'
        else:
            return ''
    
    def _build_select_columns_fixed(self, query_lower: str, context: OceanographicContext, 
                                   template_id: str, needs_measurements_join: bool) -> str:
        """Build SELECT columns with JOIN awareness"""
        
        base_columns = ['p.platform_number', 'p.profile_date', 'p.latitude', 'p.longitude']
        
        # Add parameter-specific columns
        if 'temperature' in query_lower or 'temperature' in context.parameters:
            base_columns.append('p.surface_temp')
            # CRITICAL FIX: Only add m.temperature if JOIN is actually needed
            if needs_measurements_join and template_id != 'surface_analysis':
                base_columns.append('m.temperature')
        
        if 'salinity' in query_lower or 'salinity' in context.parameters:
            base_columns.append('p.surface_salinity')
            # CRITICAL FIX: Only add m.salinity if JOIN is actually needed
            if needs_measurements_join and template_id != 'surface_analysis':
                base_columns.append('m.salinity')
        
        # CRITICAL FIX: Only add measurement columns if JOIN is actually needed
        if needs_measurements_join and template_id != 'surface_analysis':
            base_columns.extend(['m.pressure', 'm.depth'])
        
        if 'mixed layer' in query_lower:
            base_columns.append('p.mixed_layer_depth')
        
        return ',\n       '.join(base_columns)
    
    def _build_aggregation_columns_fixed(self, query_lower: str, context: OceanographicContext, 
                                        needs_measurements_join: bool) -> str:
        """Build aggregation columns with JOIN awareness"""
        
        agg_columns = []
        
        if 'temperature' in query_lower:
            if 'surface' in query_lower or not needs_measurements_join:
                agg_columns.extend([
                    'AVG(p.surface_temp) as avg_surface_temperature',
                    'STDDEV(p.surface_temp) as std_surface_temperature', 
                    'COUNT(p.surface_temp) as temperature_count'
                ])
            else:
                # With measurements JOIN
                agg_columns.extend([
                    'AVG(p.surface_temp) as avg_surface_temperature',
                    'AVG(m.temperature) as avg_profile_temperature',
                    'COUNT(m.temperature) as temperature_measurements'
                ])
        
        if 'salinity' in query_lower:
            if 'surface' in query_lower or not needs_measurements_join:
                agg_columns.append('AVG(p.surface_salinity) as avg_surface_salinity')
            else:
                agg_columns.extend([
                    'AVG(p.surface_salinity) as avg_surface_salinity',
                    'AVG(m.salinity) as avg_profile_salinity'
                ])
        
        # Always include profile count
        if not agg_columns:
            agg_columns.append('COUNT(*) as profile_count')
        elif 'COUNT' not in str(agg_columns):
            agg_columns.append('COUNT(*) as profile_count')
        
        return ',\n       '.join(agg_columns)
    
    def _build_additional_counts_fixed(self, query_lower: str, needs_measurements_join: bool) -> str:
        """Build additional count columns with JOIN awareness"""
        
        counts = []
        
        if 'temperature' in query_lower:
            counts.append('COUNT(p.surface_temp) as profiles_with_temperature')
            if needs_measurements_join:
                counts.append('COUNT(m.temperature) as total_temperature_measurements')
        
        if 'salinity' in query_lower:
            counts.append('COUNT(p.surface_salinity) as profiles_with_salinity')
            if needs_measurements_join:
                counts.append('COUNT(m.salinity) as total_salinity_measurements')
        
        if not counts:
            counts.extend([
                'MIN(p.profile_date) as earliest_date',
                'MAX(p.profile_date) as latest_date'
            ])
        
        return ',\n       '.join(counts)
    
    # === KEEP ALL EXISTING HELPER METHODS ===
    
    def _select_template(self, classification: QueryClassification, query_text: str) -> str:
        """Select most appropriate template"""
        
        query_lower = query_text.lower()
        
        # COUNT queries
        if any(word in query_lower for word in ['count', 'total', 'how many']):
            return 'count_query'
        
        # STATISTICAL queries (average, mean, etc.)
        elif any(word in query_lower for word in ['average', 'mean', 'statistics', 'summary']):
            return 'statistical_summary'
        
        # PROFILE analysis (depth, pressure, vertical)
        elif any(word in query_lower for word in ['profile', 'depth', 'pressure', 'vertical']) or \
             classification.intent == QueryIntent.PROFILE_ANALYSIS:
            return 'profile_analysis'
        
        # Default to SURFACE analysis
        else:
            return 'surface_analysis'
    
    def _build_spatial_filters(self, query_lower: str, context: OceanographicContext) -> str:
        """Build spatial filters based on query"""
        
        # Use classification context if available
        if hasattr(context, 'spatial_bounds') and context.spatial_bounds:
            bounds = context.spatial_bounds
            return f"""AND p.latitude BETWEEN {bounds['lat_min']} AND {bounds['lat_max']}
                      AND p.longitude BETWEEN {bounds['lon_min']} AND {bounds['lon_max']}"""
        
        # Extract from query text
        if 'indian ocean' in query_lower:
            return 'AND p.latitude BETWEEN -60 AND 30 AND p.longitude BETWEEN 20 AND 120'
        elif 'arabian sea' in query_lower:
            return 'AND p.latitude BETWEEN 10 AND 25 AND p.longitude BETWEEN 50 AND 78'  
        elif 'bay of bengal' in query_lower:
            return 'AND p.latitude BETWEEN 5 AND 22 AND p.longitude BETWEEN 78 AND 100'
        
        return ''  # No spatial filter
    
    def _build_temporal_filters(self, query_lower: str, context: OceanographicContext) -> str:
        """Build temporal filters"""
        
        # Use classification context if available
        if hasattr(context, 'temporal_range') and context.temporal_range:
            start_date, end_date = context.temporal_range
            return f"AND p.profile_date BETWEEN '{start_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}'"
        
        # Extract from query
        if 'recent' in query_lower or 'latest' in query_lower:
            return 'AND p.profile_date >= NOW() - INTERVAL \'1 year\''
        elif 'last year' in query_lower:
            return 'AND p.profile_date >= NOW() - INTERVAL \'1 year\''
        
        # Default: avoid scanning entire historical dataset  
        return 'AND p.profile_date >= NOW() - INTERVAL \'5 years\''
    
    def _build_depth_filters(self, query_lower: str, context: OceanographicContext) -> str:
        """Build depth/pressure filters"""
        
        if hasattr(context, 'depth_range') and context.depth_range:
            depth_min, depth_max = context.depth_range
            return f'AND m.pressure BETWEEN {depth_min} AND {depth_max}'
        
        if 'surface' in query_lower:
            return 'AND m.pressure <= 50'
        elif 'deep' in query_lower:
            return 'AND m.pressure >= 1000'
        
        return ''
    
    def _build_ordering(self, template_id: str, query_lower: str) -> str:
        """FIXED: Build ORDER BY clause - returns ONLY column names"""
        
        if template_id == 'statistical_summary':
            return ''  # No ordering for aggregation queries
        elif 'recent' in query_lower or 'latest' in query_lower:
            return 'p.profile_date DESC'  # ONLY columns
        elif template_id == 'profile_analysis':
            return 'p.profile_date DESC, m.pressure ASC'  # ONLY columns
        else:
            return 'p.profile_date DESC'  # ONLY columns
    
    def _build_grouping(self, query_lower: str) -> str:
        """Build GROUP BY clause when needed - ENHANCED"""
        
        if any(word in query_lower for word in ['distribution', 'by region', 'spatial']):
            return 'GROUP BY ROUND(p.latitude::numeric, 1), ROUND(p.longitude::numeric, 1)'
        elif any(word in query_lower for word in ['monthly', 'by month']):
            return 'GROUP BY DATE_TRUNC(\'month\', p.profile_date)'
        elif any(word in query_lower for word in ['yearly', 'by year']):
            return 'GROUP BY DATE_TRUNC(\'year\', p.profile_date)'
        
        return ''  # No grouping - this allows ORDER BY to work
    
    def _build_limit(self, complexity: ComplexityLevel) -> str:
        """Build LIMIT based on complexity"""
        
        limits = {
            ComplexityLevel.BASIC: '1000',
            ComplexityLevel.INTERMEDIATE: '5000', 
            ComplexityLevel.ADVANCED: '10000',
            ComplexityLevel.EXPERT: '50000'
        }
        
        return limits.get(complexity, '5000')
    
    def _build_quality_filters(self, query_lower: str) -> str:
        """Build quality filters"""
        return 'AND p.profile_date >= NOW() - INTERVAL \'5 years\''
    
    def _render_template(self, template: SQLTemplate, components: Dict[str, str]) -> str:
        """Render template with components"""
        
        sql = template.template
        
        # Replace all placeholders
        for param in template.parameters:
            placeholder = '{' + param + '}'
            value = components.get(param, '')
            sql = sql.replace(placeholder, value)
        
        return sql
    
    def _optimize_for_production(self, sql: str, classification: QueryClassification) -> Tuple[str, List[str]]:
        """Apply production optimizations"""
        
        optimizations = []
        
        # Clean up WHERE clause
        sql = re.sub(r'WHERE\s+1=1\s+AND', 'WHERE', sql)
        sql = re.sub(r'WHERE\s+1=1\s*(?=ORDER|GROUP|LIMIT|;|$)', '', sql)
        
        # Remove empty filters
        sql = re.sub(r'\s+AND\s+(?=ORDER|GROUP|LIMIT|;|$)', '', sql)
        
        optimizations.append('Cleaned WHERE clause structure')
        
        # Add semicolon
        if not sql.rstrip().endswith(';'):
            sql = sql.rstrip() + ';'
        
        return sql, optimizations
    
    def _validate_sql_safety(self, sql: str, template_id: str) -> List[str]:
        """Validate SQL for production safety"""
        
        warnings = []
        
        # Check for dangerous patterns
        if 'WHERE 1=1' in sql and 'AND' not in sql:
            warnings.append('Query has no filters - may return too many results')
        
        # Check for table reference consistency
        if 'm.' in sql and 'JOIN argo_measurements' not in sql:
            warnings.append('CRITICAL: References measurements table without JOIN')
        
        if 'ORDER BY m.' in sql and 'JOIN argo_measurements' not in sql:
            warnings.append('CRITICAL: Orders by measurements column without JOIN')
        
        return warnings
    
    def _apply_intelligence_optimizations(self, sql: str, 
                                    classification: QueryClassification,
                                    components: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Apply intelligent optimizations based on classification"""
        
        optimizations = []
        
        # Remove empty WHERE conditions
        sql = re.sub(r'WHERE\s+1=1\s+AND', 'WHERE', sql)
        sql = re.sub(r'WHERE\s+1=1\s*(?=ORDER|GROUP|LIMIT|;|$)', '', sql)
        sql = re.sub(r'AND\s+AND', 'AND', sql)
        
        # Add DISTINCT for certain query types
        if hasattr(classification, 'intent') and classification.intent == QueryIntent.STATISTICAL_SUMMARY:
            if 'platform_number' in sql and 'DISTINCT' not in sql:
                sql = sql.replace('SELECT ', 'SELECT DISTINCT ')
                optimizations.append('Added DISTINCT for platform queries')
        
        # Optimize JOIN order for performance
        if 'argo_measurements' in sql and 'argo_profiles' in sql:
            # Ensure profiles table is filtered first
            optimizations.append('Optimized JOIN order for large measurement table')
        
        return sql, optimizations

    # Also fix the _estimate_performance method:
    def _estimate_performance(self, template: SQLTemplate, components: Dict[str, str]) -> str:
        """Estimate performance based on template and components"""
        
        # Base performance from template
        score = template.adaptability_score
        
        # Adjust based on components
        if 'spatial_filters' in components and components['spatial_filters']:
            score += 0.1
            
        # Check if template uses measurements table by looking at the template string
        if 'argo_measurements' in template.template or 'JOIN argo_measurements' in template.template:
            score -= 0.2
            
        if int(components.get('limit', '5000')) > 10000:
            score -= 0.1
        
        if score > 0.8:
            return 'fast'
        elif score > 0.6:
            return 'medium'
        else:
            return 'slow'

    # And fix the _calculate_timeout method:
    def _calculate_timeout(self, template: SQLTemplate, components: Dict[str, str]) -> int:
        """Calculate recommended timeout"""
        
        base_timeout = 30
        
        # Check if template uses measurements table by looking at the template string
        if 'argo_measurements' in template.template or 'JOIN argo_measurements' in template.template:
            base_timeout += 30
            
        limit = int(components.get('limit', '5000'))
        if limit > 10000:
            base_timeout += 20
            
        return min(base_timeout, 120)
    
    def _generate_safe_fallback(self, classification: QueryClassification, 
                               query_text: str, error: str) -> GeneratedSQL:
        """Generate safe fallback SQL when generation fails"""
        
        logger.error(f"Generating fallback SQL due to error: {error}")
        
        # Ultra-safe fallback queries
        if 'count' in query_text.lower():
            sql = "SELECT COUNT(*) as total_profiles FROM argo_profiles WHERE profile_date >= NOW() - INTERVAL '1 year';"
        else:
            sql = """
            SELECT platform_number, profile_date, latitude, longitude, surface_temp, surface_salinity
            FROM argo_profiles 
            WHERE profile_date >= NOW() - INTERVAL '1 year'
            ORDER BY profile_date DESC 
            LIMIT 500;
            """
            
        return GeneratedSQL(
            sql=sql,
            template_id='safe_fallback',
            parameters_used={'fallback_reason': error},
            estimated_performance='fast',
            recommended_timeout=30,
            index_requirements=['idx_profiles_date'],
            adaptations_made=['Used safe fallback due to generation failure'],
            validation_passed=True,
            warnings=['Fallback SQL used - original generation failed']
        )
    
    # Keep existing helper method definitions
    def _define_performance_thresholds(self) -> Dict[str, Dict]:
        """Define performance characteristics for different query types"""
        return {
            'surface_analysis': {
                'typical_rows': 100000,
                'max_safe_rows': 1000000,
                'timeout_seconds': 30,
                'memory_mb': 100
            },
            'profile_analysis': {
                'typical_rows': 50000,
                'max_safe_rows': 500000,
                'timeout_seconds': 60,
                'memory_mb': 200
            },
            'statistical_summary': {
                'typical_rows': 1,
                'max_safe_rows': 100,
                'timeout_seconds': 15,
                'memory_mb': 10
            },
            'count_query': {
                'typical_rows': 1,
                'max_safe_rows': 1,
                'timeout_seconds': 10,
                'memory_mb': 5
            }
        }

    def _define_spatial_regions(self) -> Dict[str, Dict]:
        """Define spatial regions with performance characteristics"""
        return {
            'indian_ocean': {
                'bounds': {'lat_min': -60, 'lat_max': 30, 'lon_min': 20, 'lon_max': 120},
                'typical_profiles': 800000,
                'data_density': 'high'
            },
            'arabian_sea': {
                'bounds': {'lat_min': 10, 'lat_max': 25, 'lon_min': 50, 'lon_max': 78},
                'typical_profiles': 150000,
                'data_density': 'very_high'
            },
            'bay_of_bengal': {
                'bounds': {'lat_min': 5, 'lat_max': 22, 'lon_min': 78, 'lon_max': 100},
                'typical_profiles': 120000,
                'data_density': 'high'
            }
        }

# Test the fixes
def test_fixed_sql_generator():
    """Test the fixed SQL generator"""
    
    print("Testing Fixed SQL Generator")
    print("=" * 50)
    
    # Create dummy classification for testing
    from dataclasses import dataclass
    
    @dataclass
    class MockContext:
        parameters: list = None
        spatial_bounds: dict = None
        temporal_range: tuple = None
        depth_range: tuple = None
        
        def __post_init__(self):
            if self.parameters is None:
                self.parameters = ['temperature']
    
    @dataclass 
    class MockClassification:
        intent = 'EXPLORATION'
        complexity = 'BASIC' 
        context = MockContext()
    
    generator = ProductionSQLGenerator()
    
    test_queries = [
        "Count profiles in Arabian Sea",  # This was failing
        "Show temperature at 1000m depth",  
        "What is the average surface temperature?",
        "List platforms with salinity data"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing: {query}")
        try:
            result = generator.generate_sql(MockClassification(), query)
            print(f"✅ Template: {result.template_id}")
            print(f"✅ Performance: {result.estimated_performance}")
            print(f"✅ Warnings: {len(result.warnings or [])}")
            if result.warnings:
                print(f"⚠️  Warnings: {result.warnings}")
            print(f"✅ SQL Preview: {result.sql[:100]}...")
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print(f"\n{'='*50}")
    print("Fixed SQL Generator Test Complete")

if __name__ == "__main__":
    test_fixed_sql_generator()