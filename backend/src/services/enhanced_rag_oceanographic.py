# enhanced_rag_oceanographic.py - FIXED for production schema
# Location: src/services/enhanced_rag_oceanographic.py

import os
import sys
import re
import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, MetaData, inspect
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationSummaryBufferMemory
from dotenv import load_dotenv
import logging
import hashlib
from pathlib import Path
import warnings

# Fix import issues - add current directory to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Import our oceanographic intelligence engine with proper error handling
try:
    from oceanographic_intelligence_engine import (
        OceanographicIntelligenceEngine, 
        QueryClassification, 
        QueryIntent, 
        ComplexityLevel,
        OceanographicContext
    )
except ImportError:
    try:
        from .oceanographic_intelligence_engine import (
            OceanographicIntelligenceEngine, 
            QueryClassification, 
            QueryIntent, 
            ComplexityLevel,
            OceanographicContext
        )
    except ImportError as e:
        # Create dummy classes if import fails
        from enum import Enum
        from dataclasses import dataclass
        
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
        
        @dataclass
        class OceanographicContext:
            parameters: List[str]
            depth_range: Optional[Tuple[float, float]]
            spatial_bounds: Optional[Dict[str, float]]
            temporal_range: Optional[Tuple[datetime, datetime]]
            analysis_type: str
            physical_processes: List[str]
            data_quality_requirements: str
        
        @dataclass
        class QueryClassification:
            intent: QueryIntent
            complexity: ComplexityLevel
            context: OceanographicContext
            confidence: float
            suggested_approach: str
            required_calculations: List[str]
        
        class OceanographicIntelligenceEngine:
            def __init__(self, db_engine):
                self.engine = db_engine
            
            def classify_query(self, query: str) -> QueryClassification:
                return QueryClassification(
                    intent=QueryIntent.EXPLORATION,
                    complexity=ComplexityLevel.BASIC,
                    context=OceanographicContext(
                        parameters=['temperature', 'salinity', 'pressure'],
                        depth_range=None,
                        spatial_bounds=None,
                        temporal_range=None,
                        analysis_type='general',
                        physical_processes=[],
                        data_quality_requirements='standard'
                    ),
                    confidence=0.5,
                    suggested_approach="Basic data retrieval and analysis",
                    required_calculations=[]
                )
            
            def calculate_physical_properties(self, df, properties):
                return df
            
            def generate_insights(self, query, df, classification):
                return {
                    'summary': f'Analysis of {len(df)} records',
                    'key_findings': [f'Processed {len(df)} measurements'],
                    'physical_interpretation': 'Basic data analysis completed',
                    'data_quality_notes': 'Standard quality assessment',
                    'recommendations': ['Consider more detailed analysis'],
                    'visualization_suggestions': ['Basic plots recommended']
                }

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class EnhancedOceanographicRAG:
    """
    FIXED: Advanced RAG system for production ARGO schema
    Uses correct table names: argo_profiles and argo_measurements
    """
    
    def __init__(self, persist_directory: str = None, db_engine=None):
    # Database setup
        if db_engine is None:
            self.engine = create_engine(
                os.getenv('DATABASE_URL'),
                pool_size=15,
                max_overflow=25,
                pool_pre_ping=True,
                connect_args={"options": "-c timezone=UTC"}
            )
        else:
            self.engine = db_engine
        
        # DeepSeek LLM setup via OpenRouter
        # Docker Desktop Model Runner LLM setup
        try:
            self.llm = ChatOpenAI(
                model="ai/llama3.2:latest",  # Use the exact model name from your docker model list
                openai_api_base="http://localhost:12434/engines/llama.cpp/v1",  # Working endpoint
                openai_api_key="dummy",  # Not needed for local
                temperature=0.05,
                max_tokens=3000
            )
            logger.info("✅ Docker Desktop Model Runner connected successfully")
        except Exception as e:
            logger.warning(f"Failed to connect to Docker Desktop Model Runner: {e}")
            self.llm = None

        # Embeddings setup (prefer local HuggingFace)
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            logger.info("✅ Using local HuggingFace embeddings")
        except Exception as e:
            logger.warning(f"Failed to initialize embeddings: {e}")
            self.embeddings = None
        
        # Initialize oceanographic intelligence engine
        self.ocean_intelligence = OceanographicIntelligenceEngine(self.engine)
        
        # Vector store setup
        if persist_directory is None:
            persist_directory = os.path.join("storage", "chroma_db_oceanographic")
        
        self.vector_store = self._initialize_vector_store(persist_directory)
        
        # Database schema intelligence - FIXED for production schema
        self.schema_intelligence = self._build_schema_intelligence()
        
        # Conversation memory with oceanographic context
        if self.llm:
            self.memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=1500,
                return_messages=True
            )
        else:
            self.memory = None
        
        # Query performance cache
        self.query_cache = {}
        self.cache_dir = Path("query_cache_enhanced")
        self.cache_dir.mkdir(exist_ok=True)
        
        # FIXED: Oceanographic query patterns for production schema
        self.oceanographic_sql_patterns = self._build_sql_patterns()
    
    def _create_basic_oceanographic_vectorstore(self, persist_directory: str) -> Optional[Chroma]:
        if self.embeddings is None:
            return None
        
        from langchain.schema import Document
        
        # FIXED: Updated schema documentation for production tables
        basic_docs = [
            Document(
                page_content="""
                PRODUCTION ARGO Database Schema (Current):
                
                Primary Tables:
                - argo_profiles: Contains profile metadata (id, platform_number, cycle_number, profile_date, 
                  latitude, longitude, surface_temp, surface_salinity, max_pressure, n_levels, mixed_layer_depth)
                - argo_measurements: Contains measurement data (id, profile_id, pressure, temperature, salinity, depth) 
                - data_processing_log: Processing status and performance metrics
                
                Key Relationships:
                - JOIN argo_measurements m ON argo_profiles p WHERE m.profile_id = p.id
                
                Common Query Patterns:
                - Profile queries: SELECT m.pressure, m.temperature, m.salinity FROM argo_measurements m JOIN argo_profiles p ON m.profile_id = p.id
                - Surface analysis: SELECT p.surface_temp, p.surface_salinity FROM argo_profiles p
                - Statistical queries: Use aggregation functions with proper JOINs
                """,
                metadata={"type": "schema", "source": "production_database"}
            ),
            Document(
                page_content="""
                Production Schema Query Examples:
                
                Basic Profile Query:
                SELECT p.platform_number, p.profile_date, m.pressure, m.temperature, m.salinity
                FROM argo_profiles p 
                JOIN argo_measurements m ON p.id = m.profile_id
                WHERE p.platform_number = '1900121'
                ORDER BY m.pressure ASC;
                
                Surface Temperature Distribution:
                SELECT p.latitude, p.longitude, p.surface_temp, p.profile_date
                FROM argo_profiles p
                WHERE p.surface_temp IS NOT NULL
                ORDER BY p.profile_date DESC;
                
                Statistical Summary:
                SELECT COUNT(*) as total_profiles,
                       AVG(p.surface_temp) as avg_surface_temp,
                       AVG(p.surface_salinity) as avg_surface_salinity
                FROM argo_profiles p
                WHERE p.surface_temp IS NOT NULL;
                """,
                metadata={"type": "examples", "source": "production_queries"}
            ),
            Document(
                page_content="""
                Oceanographic Analysis Guidelines (Updated):
                - Temperature: Celsius, available as surface_temp in profiles table and temperature in measurements
                - Salinity: PSU, available as surface_salinity in profiles table and salinity in measurements  
                - Pressure: dbar, available in measurements table (depth conversion: depth ≈ pressure * 1.0194)
                - Mixed Layer Depth: Pre-calculated in profiles table as mixed_layer_depth
                
                Data Quality Features:
                - All coordinates validated (lat: -90 to 90, lon: -180 to 180)
                - Temperature range validated (-3 to 40°C)
                - Salinity range validated (0 to 50 PSU)
                - Processing status tracked in data_processing_log
                """,
                metadata={"type": "oceanography", "source": "production_features"}
            )
        ]
        
        try:
            vector_store = Chroma.from_documents(
                documents=basic_docs,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
            vector_store.persist()
            return vector_store
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            return None
    
    def _build_sql_patterns(self) -> Dict[str, Dict]:
        """FIXED: Build SQL patterns for production schema"""
        return {
            'profile_queries': {
                'basic_profile': """
                    SELECT p.platform_number, p.cycle_number, p.profile_date,
                           p.latitude, p.longitude,
                           m.pressure, m.temperature, m.salinity, m.depth
                    FROM argo_profiles p
                    JOIN argo_measurements m ON p.id = m.profile_id
                    WHERE p.platform_number = '{platform_number}'
                    AND m.pressure IS NOT NULL
                    ORDER BY m.pressure ASC
                    LIMIT 2000;
                """,
                'surface_analysis': """
                    SELECT p.platform_number, p.profile_date, p.latitude, p.longitude,
                           p.surface_temp, p.surface_salinity, p.mixed_layer_depth,
                           p.max_pressure, p.n_levels
                    FROM argo_profiles p
                    WHERE p.surface_temp IS NOT NULL
                    ORDER BY p.profile_date DESC
                    LIMIT 1000;
                """,
                'platform_summary': """
                    SELECT p.platform_number,
                           COUNT(*) as total_profiles,
                           MIN(p.profile_date) as first_profile,
                           MAX(p.profile_date) as last_profile,
                           AVG(p.surface_temp) as avg_surface_temp,
                           AVG(p.surface_salinity) as avg_surface_salinity
                    FROM argo_profiles p
                    WHERE p.platform_number = '{platform_number}'
                    GROUP BY p.platform_number;
                """
            },
            'statistical_queries': {
                'count_query': """
                    SELECT COUNT(*) as total_profiles,
                           COUNT(DISTINCT p.platform_number) as unique_platforms,
                           MIN(p.profile_date) as earliest_date,
                           MAX(p.profile_date) as latest_date
                    FROM argo_profiles p;
                """,
                'measurement_stats': """
                    SELECT 
                        COUNT(*) as total_measurements,
                        AVG(m.temperature) as mean_temperature,
                        STDDEV(m.temperature) as std_temperature,
                        AVG(m.salinity) as mean_salinity,
                        STDDEV(m.salinity) as std_salinity
                    FROM argo_measurements m
                    WHERE m.temperature IS NOT NULL AND m.salinity IS NOT NULL;
                """,
                'surface_stats': """
                    SELECT 
                        COUNT(*) as profiles_with_surface_data,
                        AVG(p.surface_temp) as mean_surface_temp,
                        STDDEV(p.surface_temp) as std_surface_temp,
                        AVG(p.surface_salinity) as mean_surface_salinity,
                        STDDEV(p.surface_salinity) as std_surface_salinity
                    FROM argo_profiles p
                    WHERE p.surface_temp IS NOT NULL AND p.surface_salinity IS NOT NULL;
                """
            },
            'spatial_queries': {
                'regional_analysis': """
                    SELECT p.latitude, p.longitude,
                           p.surface_temp, p.surface_salinity,
                           p.profile_date, p.platform_number
                    FROM argo_profiles p
                    WHERE p.latitude BETWEEN {lat_min} AND {lat_max}
                    AND p.longitude BETWEEN {lon_min} AND {lon_max}
                    AND p.surface_temp IS NOT NULL
                    ORDER BY p.profile_date DESC
                    LIMIT 2000;
                """,
                'spatial_distribution': """
                    SELECT p.latitude, p.longitude,
                           AVG(p.surface_temp) as avg_surface_temp,
                           AVG(p.surface_salinity) as avg_surface_salinity,
                           COUNT(*) as measurement_count
                    FROM argo_profiles p
                    WHERE p.surface_temp IS NOT NULL
                    GROUP BY p.latitude, p.longitude
                    HAVING COUNT(*) >= 1
                    ORDER BY p.latitude, p.longitude
                    LIMIT 5000;
                """
            }
        }
    
    def _fallback_context(self, classification: QueryClassification) -> str:
        """FIXED: Fallback context with correct schema"""
        
        fallback = [
            "PRODUCTION ARGO Database Schema:",
            "",
            "Main Tables:",
            "- argo_profiles: platform_number, cycle_number, profile_date, latitude, longitude, surface_temp, surface_salinity",
            "- argo_measurements: profile_id, pressure, temperature, salinity, depth (linked to argo_profiles.id)",
            "- data_processing_log: filename, processing_status, profiles_count, measurements_count",
            "",
            "Essential JOIN pattern:",
            "SELECT p.platform_number, p.profile_date, m.pressure, m.temperature, m.salinity",
            "FROM argo_profiles p",
            "JOIN argo_measurements m ON p.id = m.profile_id",
            "",
            "Surface data access:",
            "SELECT platform_number, surface_temp, surface_salinity, mixed_layer_depth",
            "FROM argo_profiles",
            "WHERE surface_temp IS NOT NULL",
            "",
            f"Query Intent: {classification.intent.value}",
            f"Suggested approach: {classification.suggested_approach}"
        ]
        
        return "\n".join(fallback)
    
    def generate_enhanced_sql(self, query: str, classification: QueryClassification) -> Optional[str]:
        """FIXED: Generate SQL for production schema"""
        
        # Get enhanced context
        context, context_meta = self.get_enhanced_context(query, classification)
        
        # Build system prompt with correct schema
        system_prompt =  f"""You are an expert oceanographer and PostgreSQL specialist with deep knowledge of ARGO float data analysis. You are powered by Llama 3.2, optimized for scientific and technical queries.

PRODUCTION DATABASE SCHEMA (CRITICAL - USE THESE TABLE NAMES):

PRODUCTION DATABASE SCHEMA (CRITICAL - USE THESE TABLE NAMES):
- argo_profiles: Main profiles table (id, platform_number, cycle_number, profile_date, latitude, longitude, surface_temp, surface_salinity, max_pressure, n_levels, mixed_layer_depth)
- argo_measurements: Measurements table (id, profile_id, pressure, temperature, salinity, depth)
- data_processing_log: Processing status table

CURRENT QUERY ANALYSIS:
Intent: {classification.intent.value}
Complexity: {classification.complexity.value}
Parameters: {', '.join(classification.context.parameters) if classification.context.parameters else 'General'}
Confidence: {classification.confidence:.2f}
Approach: {classification.suggested_approach}

{context}

CRITICAL SQL GENERATION RULES:
1. ALWAYS use table names: argo_profiles (alias 'p') and argo_measurements (alias 'm')
2. JOIN pattern: FROM argo_profiles p JOIN argo_measurements m ON p.id = m.profile_id
3. Surface data: Use p.surface_temp, p.surface_salinity from argo_profiles
4. Profile measurements: Use m.pressure, m.temperature, m.salinity from argo_measurements
5. Include appropriate WHERE clauses for data quality (IS NOT NULL)
6. Order results meaningfully (pressure ASC for profiles, profile_date DESC for time series)
7. Apply reasonable LIMIT clauses (1000-5000 depending on query type)
8. Use proper column names: profile_date (not date), surface_temp (not surface_temperature)

NEVER use these old table names: enhanced_floats_metadata, enhanced_measurements, floats_metadata

Return ONLY the optimized PostgreSQL query without explanation or markdown formatting."""
        
        # Build user prompt
        user_prompt = f"""USER QUERY: {query}

Generate the optimal PostgreSQL query using the PRODUCTION schema (argo_profiles, argo_measurements):"""
        
        if self.llm is None:
            return self._generate_fallback_sql(query, classification)
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm(messages)
            sql_query = self._clean_and_optimize_sql(response.content, classification)
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating enhanced SQL: {e}")
            return self._generate_fallback_sql(query, classification)
    
    def _generate_fallback_sql(self, query: str, classification: QueryClassification) -> str:
        """ENHANCED: Generate fallback SQL with correct table names and data types"""
        query_lower = query.lower()
        
        if 'count' in query_lower or 'total' in query_lower:
            return """
                SELECT COUNT(*) as total_profiles,
                    COUNT(DISTINCT platform_number) as unique_platforms
                FROM argo_profiles;
            """
        
        elif 'temperature' in query_lower and 'profile' in query_lower:
            return """
                SELECT p.platform_number, p.profile_date, p.latitude, p.longitude,
                    m.pressure, m.temperature
                FROM argo_profiles p
                JOIN argo_measurements m ON p.id = m.profile_id
                WHERE m.temperature IS NOT NULL
                ORDER BY p.profile_date DESC, m.pressure ASC
                LIMIT 1000;
            """
        
        elif 'surface' in query_lower and ('temperature' in query_lower or 'salinity' in query_lower):
            return """
                SELECT p.platform_number, p.profile_date, p.latitude, p.longitude,
                    p.surface_temp, p.surface_salinity, p.mixed_layer_depth
                FROM argo_profiles p
                WHERE p.surface_temp IS NOT NULL OR p.surface_salinity IS NOT NULL
                ORDER BY p.profile_date DESC
                LIMIT 1000;
            """
        
        elif 'salinity' in query_lower and 'average' in query_lower:
            return """
                SELECT AVG(p.surface_salinity) as avg_surface_salinity,
                    COUNT(p.surface_salinity) as measurement_count
                FROM argo_profiles p
                WHERE p.surface_salinity IS NOT NULL;
            """
        
        elif 'platform' in query_lower or any(char.isdigit() for char in query_lower):
            # FIXED: Extract platform number and properly quote it
            import re
            platform_match = re.search(r'(\d{7})', query_lower)
            platform_number = platform_match.group(1) if platform_match else '1900121'
            
            return f"""
                SELECT p.platform_number, p.profile_date, p.latitude, p.longitude,
                    p.surface_temp, p.surface_salinity
                FROM argo_profiles p
                WHERE p.platform_number = '{platform_number}'
                ORDER BY p.profile_date DESC
                LIMIT 100;
            """
        
        else:
            # Default comprehensive query
            return """
                SELECT p.platform_number, p.profile_date, p.latitude, p.longitude,
                    p.surface_temp, p.surface_salinity, p.max_pressure, p.n_levels
                FROM argo_profiles p
                WHERE p.surface_temp IS NOT NULL OR p.surface_salinity IS NOT NULL
                ORDER BY p.profile_date DESC
                LIMIT 500;
            """
    
    # def _attempt_query_fix(self, sql_query: str, error_msg: str) -> str:
    #     """ENHANCED: Auto-fix common SQL issues with correct table names"""
        
    #     error_lower = error_msg.lower()
        
    #     if "relation" in error_lower and "does not exist" in error_lower:
    #         # Fix old table names
    #         sql_query = sql_query.replace("enhanced_floats_metadata", "argo_profiles")
    #         sql_query = sql_query.replace("enhanced_measurements", "argo_measurements")
    #         sql_query = sql_query.replace("floats_metadata", "argo_profiles")
    #         sql_query = sql_query.replace("measurements", "argo_measurements")
            
    #         # Fix column names
    #         sql_query = sql_query.replace("metadata_id", "profile_id")
    #         sql_query = sql_query.replace("date", "profile_date")
    #         sql_query = sql_query.replace("surface_temperature", "surface_temp")
        
    #     if "column" in error_lower and "does not exist" in error_lower:
    #         # Fix column names
    #         sql_query = sql_query.replace("surface_temperature", "surface_temp")
    #         sql_query = sql_query.replace(".date", ".profile_date")
    #         sql_query = sql_query.replace("metadata_id", "profile_id")
        
    #     # FIX: Handle platform_number type casting issues
    #     if "operator does not exist" in error_lower and "character varying" in error_lower:
    #         import re
    #         # Find platform_number = numeric_value patterns and add quotes
    #         pattern = r"platform_number\s*=\s*(\d+)"
    #         matches = re.findall(pattern, sql_query)
            
    #         for match in matches:
    #             old_pattern = f"platform_number = {match}"
    #             new_pattern = f"platform_number = '{match}'"
    #             sql_query = sql_query.replace(old_pattern, new_pattern)
        
    #     return sql_query
    
    def _attempt_query_fix(self, sql_query: str, error_msg: str) -> str:
        """ENHANCED: Auto-fix common SQL issues with correct table names"""
        
        error_lower = error_msg.lower()
        corrected_sql = sql_query
        
        # Fix 1: Platform number quoting (MOST IMPORTANT)
        if "operator does not exist" in error_lower and "character varying" in error_lower:
            # Find all instances of platform_number = number pattern
            import re
            pattern = r"platform_number\s*=\s*(\d+)"
            matches = re.findall(pattern, corrected_sql)
            
            for match in matches:
                old_pattern = f"platform_number = {match}"
                new_pattern = f"platform_number = '{match}'"
                corrected_sql = corrected_sql.replace(old_pattern, new_pattern)
        
        # Fix 2: Other common issues
        if "relation" in error_lower and "does not exist" in error_lower:
            # Fix old table names
            corrected_sql = corrected_sql.replace("enhanced_floats_metadata", "argo_profiles")
            corrected_sql = corrected_sql.replace("enhanced_measurements", "argo_measurements")
            corrected_sql = corrected_sql.replace("floats_metadata", "argo_profiles")
            corrected_sql = corrected_sql.replace("measurements", "argo_measurements")
            
            # Fix column names
            corrected_sql = corrected_sql.replace("metadata_id", "profile_id")
            corrected_sql = corrected_sql.replace("date", "profile_date")
            corrected_sql = corrected_sql.replace("surface_temperature", "surface_temp")
        
        if "column" in error_lower and "does not exist" in error_lower:
            # Fix column names
            corrected_sql = corrected_sql.replace("surface_temperature", "surface_temp")
            corrected_sql = corrected_sql.replace(".date", ".profile_date")
            corrected_sql = corrected_sql.replace("metadata_id", "profile_id")
        
        return corrected_sql
    
    # Keep all other methods unchanged - they don't reference table names directly
    def _initialize_vector_store(self, persist_directory: str) -> Optional[Chroma]:
        """Initialize vector store with oceanographic domain knowledge"""
        
        if self.embeddings is None:
            logger.warning("Embeddings not available, skipping vector store initialization")
            return None
        
        try:
            if os.path.exists(persist_directory):
                vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                doc_count = vector_store._collection.count()
                logger.info(f"Loaded existing vector store with {doc_count} documents")
            else:
                logger.warning(f"Vector store not found at {persist_directory}")
                logger.info("Creating basic oceanographic vector store...")
                vector_store = self._create_basic_oceanographic_vectorstore(persist_directory)
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            logger.info("Creating fallback vector store...")
            return self._create_basic_oceanographic_vectorstore(persist_directory)
    
    def _build_schema_intelligence(self) -> Dict[str, Any]:
        """Build intelligent schema representation"""
        schema_intel = {
            'tables': {},
            'relationships': {},
            'optimized_joins': {},
            'common_patterns': {}
        }
        
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            
            for table in tables:
                columns = inspector.get_columns(table)
                foreign_keys = inspector.get_foreign_keys(table)
                indexes = inspector.get_indexes(table)
                
                column_categories = {
                    'identifiers': [],
                    'measurements': [],
                    'coordinates': [],
                    'temporal': [],
                    'quality': [],
                    'derived': [],
                    'metadata': []
                }
                
                for col in columns:
                    col_name = col['name'].lower()
                    
                    if 'id' in col_name or col_name in ['platform_number', 'cycle_number']:
                        column_categories['identifiers'].append(col['name'])
                    elif col_name in ['temperature', 'salinity', 'pressure', 'surface_temp', 'surface_salinity']:
                        column_categories['measurements'].append(col['name'])
                    elif col_name in ['latitude', 'longitude', 'location']:
                        column_categories['coordinates'].append(col['name'])
                    elif 'date' in col_name or 'time' in col_name:
                        column_categories['temporal'].append(col['name'])
                    elif 'qc' in col_name or 'flag' in col_name:
                        column_categories['quality'].append(col['name'])
                    elif col_name in ['mixed_layer_depth', 'max_pressure', 'n_levels']:
                        column_categories['derived'].append(col['name'])
                    else:
                        column_categories['metadata'].append(col['name'])
                
                schema_intel['tables'][table] = {
                    'columns': columns,
                    'column_categories': column_categories,
                    'foreign_keys': foreign_keys,
                    'indexes': indexes,
                    'primary_purpose': self._determine_table_purpose(table, column_categories)
                }
            
        except Exception as e:
            logger.error(f"Error building schema intelligence: {e}")
        
        return schema_intel
    
    def _determine_table_purpose(self, table_name: str, column_categories: Dict) -> str:
        """Determine the primary purpose of a table"""
        if 'profile' in table_name.lower():
            return 'profiles'
        elif 'measurement' in table_name.lower():
            return 'measurements'
        elif 'log' in table_name.lower():
            return 'monitoring'
        else:
            return 'reference'
    
    def get_enhanced_context(self, query: str, classification: QueryClassification, k: int = 5) -> Tuple[str, Dict]:
        """Get enhanced context using oceanographic intelligence"""
        
        try:
            if self.vector_store is None:
                return self._fallback_context(classification), {}
            
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            context_parts = []
            context_parts.append("=== OCEANOGRAPHIC DATABASE CONTEXT ===\n")
            
            for i, (doc, score) in enumerate(results, 1):
                relevance = max(0, (1 - score) * 100)
                
                if relevance > 25:
                    context_parts.append(f"--- Context {i} (Relevance: {relevance:.1f}%) ---")
                    context_parts.append(doc.page_content.strip())
                    context_parts.append("")
            
            # Add schema-specific context
            schema_context = self._get_schema_context(classification)
            if schema_context:
                context_parts.append("=== SCHEMA OPTIMIZATION HINTS ===")
                context_parts.append(schema_context)
            
            full_context = "\n".join(context_parts)
            return full_context, {}
            
        except Exception as e:
            logger.error(f"Error in enhanced context retrieval: {e}")
            return self._fallback_context(classification), {}
    
    def _get_schema_context(self, classification: QueryClassification) -> str:
        """Get schema-specific optimization hints"""
        
        schema_hints = []
        
        if classification.intent == QueryIntent.PROFILE_ANALYSIS:
            schema_hints.extend([
                "Profile Analysis Optimization:",
                "- JOIN argo_measurements m with argo_profiles p via profile_id",
                "- Order by pressure ASC for proper depth sequence",
                "- Filter by platform_number for specific floats"
            ])
        
        elif classification.intent == QueryIntent.SPATIAL_MAPPING:
            schema_hints.extend([
                "Spatial Analysis Optimization:",
                "- Use latitude, longitude from argo_profiles table",
                "- Surface data available as surface_temp, surface_salinity",
                "- Consider GROUP BY lat/lon for distribution analysis"
            ])
        
        return "\n".join(schema_hints) if schema_hints else ""
    
    def _clean_and_optimize_sql(self, sql_response: str, classification: QueryClassification) -> str:
        """Clean and optimize the generated SQL query"""
        
        # Remove markdown formatting
        sql_query = re.sub(r'```sql\s*', '', sql_response)
        sql_query = re.sub(r'```\s*', '', sql_query)
        sql_query = sql_query.strip()
        
        # Remove comments
        sql_query = re.sub(r'--.*\n', '\n', sql_query)
        sql_query = re.sub(r'/\*.*?\*/', '', sql_query, flags=re.DOTALL)
        
        # Clean whitespace
        sql_query = ' '.join(sql_query.split())
        sql_query = sql_query.rstrip(';')
        
        # Validate it's a SELECT query
        if not sql_query.upper().startswith('SELECT'):
            select_match = re.search(r'(SELECT.*?)(?:;|$)', sql_query, re.IGNORECASE | re.DOTALL)
            if select_match:
                sql_query = select_match.group(1).strip()
            else:
                raise ValueError("Invalid SQL query generated")
        
        # Add appropriate LIMIT based on intent
        if 'LIMIT' not in sql_query.upper():
            if classification.intent in [QueryIntent.STATISTICAL_SUMMARY, QueryIntent.SPATIAL_MAPPING]:
                sql_query += " LIMIT 5000"
            else:
                sql_query += " LIMIT 2000"
        
        # Add semicolon
        sql_query += ";"
        
        return sql_query
    
    def execute_enhanced_query(self, sql_query: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Execute SQL with enhanced error handling and optimization"""
        
        for attempt in range(max_retries):
            try:
                start_time = datetime.now()
                
                with self.engine.connect() as conn:
                    result = conn.execute(text(sql_query))
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Query executed successfully: {len(df)} rows in {execution_time:.2f}s")
                
                return df
                
            except Exception as e:
                logger.warning(f"Query execution attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    sql_query = self._attempt_query_fix(sql_query, str(e))
                    continue
                else:
                    logger.error(f"Final query execution failed: {e}")
        
        return None
    
    def process_oceanographic_query(self, natural_language_query: str) -> Dict[str, Any]:
        """Process query with full oceanographic intelligence pipeline"""
        
        logger.info(f"Processing oceanographic query: {natural_language_query}")
        start_time = datetime.now()
        
        try:
            # Step 1: Classify query with oceanographic intelligence
            classification = self.ocean_intelligence.classify_query(natural_language_query)
            
            logger.info(f"Query classified as: {classification.intent.value} ({classification.complexity.value})")
            
            # Step 2: Generate enhanced SQL
            sql_query = self.generate_enhanced_sql(natural_language_query, classification)
            
            if not sql_query:
                return {
                    'success': False,
                    'error': 'Failed to generate SQL query',
                    'query': natural_language_query,
                    'classification': classification.__dict__,
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            
            logger.info(f"Generated SQL: {sql_query}")
            
            # Step 3: Execute query
            results_df = self.execute_enhanced_query(sql_query)
            
            if results_df is None:
                return {
                    'success': False,
                    'error': 'Query execution failed',
                    'sql_query': sql_query,
                    'query': natural_language_query,
                    'classification': self._classification_to_dict(classification),
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            
            # Step 4: Calculate required physical properties
            if classification.required_calculations:
                results_df = self.ocean_intelligence.calculate_physical_properties(
                    results_df, classification.required_calculations
                )
            
            # Step 5: Generate intelligent insights
            insights = self.ocean_intelligence.generate_insights(
                natural_language_query, results_df, classification
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Build comprehensive response
            response = {
                'success': True,
                'query': natural_language_query,
                'sql_query': sql_query,
                'classification': self._classification_to_dict(classification),
                'results': results_df,
                'result_count': len(results_df),
                'columns': list(results_df.columns) if not results_df.empty else [],
                'insights': insights,
                'processing_time': processing_time,
                'data_types': {col: str(dtype) for col, dtype in results_df.dtypes.items()} if not results_df.empty else {}
            }
            
            logger.info(f"Query processed successfully: {len(results_df)} rows, {processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing oceanographic query: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': natural_language_query,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _classification_to_dict(self, classification: QueryClassification) -> Dict[str, Any]:
        """Convert classification object to dictionary for JSON serialization"""
        try:
            return {
                'intent': classification.intent.value,
                'complexity': classification.complexity.value,
                'confidence': classification.confidence,
                'parameters': classification.context.parameters,
                'depth_range': classification.context.depth_range,
                'spatial_bounds': classification.context.spatial_bounds,
                'temporal_range': [str(t) for t in classification.context.temporal_range] if classification.context.temporal_range else None,
                'physical_processes': classification.context.physical_processes,
                'suggested_approach': classification.suggested_approach,
                'required_calculations': classification.required_calculations
            }
        except Exception as e:
            logger.error(f"Error converting classification to dict: {e}")
            return {
                'intent': 'exploration',
                'complexity': 'basic',
                'confidence': 0.5,
                'parameters': [],
                'depth_range': None,
                'spatial_bounds': None,
                'temporal_range': None,
                'physical_processes': [],
                'suggested_approach': 'Basic analysis',
                'required_calculations': []
            }

def test_enhanced_oceanographic_rag():
    """Test the FIXED enhanced oceanographic RAG system"""
    
    logger.info("Testing FIXED Enhanced Oceanographic RAG System")
    logger.info("=" * 60)
    
    # Initialize system
    rag_system = EnhancedOceanographicRAG()
    
    # Test queries for production schema
    test_queries = [
        "How many profiles are in the database?",
        "Show me the latest temperature measurements",
        "What is the average surface temperature?",
        "Show surface salinity distribution",
        "Get profile data for platform '1900121'"
    ]
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n{i}. Testing Query: {query}")
        logger.info("-" * 50)
        
        try:
            result = rag_system.process_oceanographic_query(query)
            
            if result['success']:
                logger.info(f"✅ Success!")
                logger.info(f"   Intent: {result['classification']['intent']}")
                logger.info(f"   Results: {result['result_count']} rows")
                logger.info(f"   Time: {result['processing_time']:.2f}s")
                logger.info(f"   SQL: {result['sql_query'][:100]}...")
            else:
                logger.error(f"❌ Failed: {result['error']}")
                
        except Exception as e:
            logger.error(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_enhanced_oceanographic_rag()