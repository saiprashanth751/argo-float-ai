# enhanced_rag_oceanographic.py
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
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
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
                # Basic classification for fallback
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
    Advanced RAG system specifically designed for oceanographic data analysis.
    Integrates domain intelligence with sophisticated query understanding.
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
        
        # LLM setup with optimized parameters for oceanographic queries
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.05,  # Very low for scientific accuracy
                max_tokens=3000,
                google_api_key=os.getenv('GEMINI_API_KEY')
            )
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}")
            self.llm = None
        
        # Embeddings setup
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv('GEMINI_API_KEY')
            )
        except Exception as e:
            logger.warning(f"Failed to initialize embeddings: {e}")
            self.embeddings = None
        
        # Initialize oceanographic intelligence engine
        self.ocean_intelligence = OceanographicIntelligenceEngine(self.engine)
        
        # Vector store setup
        if persist_directory is None:
            persist_directory = os.path.join("storage", "chroma_db_oceanographic")
        
        self.vector_store = self._initialize_vector_store(persist_directory)
        
        # Database schema intelligence
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
        
        # Oceanographic query patterns for better SQL generation
        self.oceanographic_sql_patterns = self._build_sql_patterns()
    
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
    
    def _create_basic_oceanographic_vectorstore(self, persist_directory: str) -> Optional[Chroma]:
        """Create basic oceanographic vector store if none exists"""
        
        if self.embeddings is None:
            return None
        
        from langchain.schema import Document
        
        # Basic oceanographic documents
        basic_docs = [
            Document(
                page_content="""
                ARGO Float Database Schema:
                - enhanced_floats_metadata: Contains float metadata (platform_number, date, latitude, longitude, cycle_number)
                - enhanced_measurements: Contains measurement data (pressure, temperature, salinity) linked via metadata_id
                
                Key Relationships:
                - JOIN enhanced_measurements m ON enhanced_floats_metadata fm WHERE m.metadata_id = fm.id
                
                Common Query Patterns:
                - Profile queries: SELECT pressure, temperature, salinity ORDER BY pressure ASC
                - Spatial queries: Include latitude, longitude from metadata table
                - Statistical queries: Use GROUP BY with aggregate functions
                """,
                metadata={"type": "schema", "source": "database_structure"}
            ),
            Document(
                page_content="""
                Oceanographic Analysis Guidelines:
                - Temperature: Measured in Celsius, use for thermal analysis
                - Salinity: Measured in PSU, indicates water mass properties
                - Pressure: Measured in dbar, approximately equals depth in meters
                
                Physical Oceanography:
                - Surface layer: 0-100 dbar
                - Thermocline: 100-1000 dbar typically
                - Deep water: >1000 dbar
                - Mixed layer depth: determined by temperature/density gradients
                """,
                metadata={"type": "oceanography", "source": "domain_knowledge"}
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
            logger.error(f"Failed to create basic vector store: {e}")
            return None
    
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
                
                # Categorize columns by type and purpose
                column_categories = {
                    'identifiers': [],
                    'measurements': [],
                    'coordinates': [],
                    'temporal': [],
                    'quality': [],
                    'derived': []
                }
                
                for col in columns:
                    col_name = col['name'].lower()
                    col_type = str(col['type']).lower()
                    
                    if 'id' in col_name:
                        column_categories['identifiers'].append(col['name'])
                    elif col_name in ['temperature', 'salinity', 'pressure', 'density', 'oxygen']:
                        column_categories['measurements'].append(col['name'])
                    elif col_name in ['latitude', 'longitude', 'location']:
                        column_categories['coordinates'].append(col['name'])
                    elif 'date' in col_name or 'time' in col_name:
                        column_categories['temporal'].append(col['name'])
                    elif 'qc' in col_name or 'flag' in col_name:
                        column_categories['quality'].append(col['name'])
                    elif 'potential' in col_name or 'conservative' in col_name:
                        column_categories['derived'].append(col['name'])
                    else:
                        column_categories['measurements'].append(col['name'])
                
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
        if 'metadata' in table_name.lower():
            return 'metadata'
        elif 'measurement' in table_name.lower():
            return 'measurements'
        elif len(column_categories['measurements']) > len(column_categories['identifiers']):
            return 'measurements'
        else:
            return 'reference'
    
    def _build_sql_patterns(self) -> Dict[str, Dict]:
        """Build oceanographic-specific SQL patterns"""
        return {
            'profile_queries': {
                'basic_profile': """
                    SELECT m.pressure, m.temperature, m.salinity
                    FROM enhanced_measurements m
                    JOIN enhanced_floats_metadata fm ON m.metadata_id = fm.id
                    WHERE fm.platform_number = '{platform_number}'
                    AND m.pressure IS NOT NULL
                    ORDER BY m.pressure ASC
                    LIMIT 2000;
                """,
            },
            'spatial_queries': {
                'surface_distribution': """
                    SELECT fm.latitude, fm.longitude, 
                           AVG(m.{parameter}) as avg_{parameter},
                           COUNT(m.{parameter}) as data_points
                    FROM enhanced_measurements m
                    JOIN enhanced_floats_metadata fm ON m.metadata_id = fm.id
                    WHERE m.pressure <= {surface_pressure}
                    AND m.{parameter} IS NOT NULL
                    GROUP BY fm.latitude, fm.longitude
                    HAVING COUNT(m.{parameter}) >= 3
                    ORDER BY fm.latitude, fm.longitude
                    LIMIT 5000;
                """,
            },
            'statistical_queries': {
                'parameter_statistics': """
                    SELECT 
                        COUNT(*) as total_measurements,
                        AVG(m.{parameter}) as mean_{parameter},
                        STDDEV(m.{parameter}) as std_{parameter},
                        MIN(m.{parameter}) as min_{parameter},
                        MAX(m.{parameter}) as max_{parameter}
                    FROM enhanced_measurements m
                    WHERE m.{parameter} IS NOT NULL;
                """,
            }
        }
    
    def get_enhanced_context(self, query: str, classification: QueryClassification, k: int = 5) -> Tuple[str, Dict]:
        """Get enhanced context using oceanographic intelligence"""
        
        try:
            if self.vector_store is None:
                return self._fallback_context(classification), {}
            
            # Get relevant context from vector store
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
                "- Always JOIN measurements with metadata via metadata_id",
                "- Order by pressure ASC for proper depth sequence",
                "- Filter by platform_number for specific floats"
            ])
        
        elif classification.intent == QueryIntent.SPATIAL_MAPPING:
            schema_hints.extend([
                "Spatial Analysis Optimization:",
                "- Include latitude, longitude from floats_metadata",
                "- Use appropriate spatial filtering WHERE clauses",
                "- Consider GROUP BY lat/lon for distribution analysis"
            ])
        
        return "\n".join(schema_hints) if schema_hints else ""
    
    def _fallback_context(self, classification: QueryClassification) -> str:
        """Provide fallback context when vector store is unavailable"""
        
        fallback = [
            "ARGO Float Database Schema (Fallback Context):",
            "",
            "Tables:",
            "- enhanced_floats_metadata: platform_number, date, latitude, longitude, cycle_number",
            "- enhanced_measurements: pressure, temperature, salinity (linked via metadata_id)",
            "",
            "Essential JOIN pattern:",
            "SELECT m.pressure, m.temperature, m.salinity, fm.latitude, fm.longitude",
            "FROM enhanced_measurements m",
            "JOIN enhanced_floats_metadata fm ON m.metadata_id = fm.id",
            "",
            f"Query Intent: {classification.intent.value}",
            f"Suggested approach: {classification.suggested_approach}"
        ]
        
        return "\n".join(fallback)
    
    def generate_enhanced_sql(self, query: str, classification: QueryClassification) -> Optional[str]:
        """Generate oceanographically-intelligent SQL queries"""
        
        # Get enhanced context
        context, context_meta = self.get_enhanced_context(query, classification)
        
        # Build system prompt
        system_prompt = f"""You are an expert oceanographer and PostgreSQL specialist with deep knowledge of ARGO float data analysis.

CURRENT QUERY ANALYSIS:
Intent: {classification.intent.value}
Complexity: {classification.complexity.value}
Parameters: {', '.join(classification.context.parameters) if classification.context.parameters else 'General'}
Confidence: {classification.confidence:.2f}
Approach: {classification.suggested_approach}

{context}

CRITICAL SQL GENERATION RULES:
1. ALWAYS use proper JOINs between enhanced_measurements and enhanced_floats_metadata
2. Use meaningful table aliases: 'm' for measurements, 'fm' for floats_metadata
3. Include appropriate WHERE clauses for data quality (IS NOT NULL)
4. Order results meaningfully (pressure ASC for profiles, date ASC for time series)
5. Apply reasonable LIMIT clauses (2000 for basic queries, 10000 for aggregations)
6. Use appropriate aggregate functions for statistical queries
7. Include geographic context (lat/lon) for spatial queries
8. Handle temporal filtering with proper date functions

Return ONLY the optimized PostgreSQL query without explanation or markdown formatting.
Ensure query is syntactically correct and optimized for the ARGO dataset."""
        
        # Build user prompt
        user_prompt = f"""USER QUERY: {query}

Generate the optimal PostgreSQL query for this oceanographic analysis:"""
        
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
        """Generate fallback SQL when LLM is not available"""
        
        query_lower = query.lower()
        
        # Basic patterns based on query content
        if 'count' in query_lower or 'total' in query_lower or 'how many' in query_lower:
            return """
                SELECT COUNT(*) as total_count
                FROM enhanced_floats_metadata fm
                JOIN enhanced_measurements m ON fm.id = m.metadata_id
                WHERE m.pressure IS NOT NULL
                AND m.temperature IS NOT NULL
                AND m.salinity IS NOT NULL
                LIMIT 1;
            """
        
        elif 'temperature' in query_lower and 'profile' in query_lower:
            return """
                SELECT m.pressure, m.temperature, fm.latitude, fm.longitude
                FROM enhanced_measurements m
                JOIN enhanced_floats_metadata fm ON m.metadata_id = fm.id
                WHERE m.temperature IS NOT NULL
                AND m.pressure IS NOT NULL
                ORDER BY m.pressure ASC
                LIMIT 1000;
            """
        
        elif 'salinity' in query_lower and 'average' in query_lower:
            return """
                SELECT AVG(m.salinity) as avg_salinity,
                       COUNT(m.salinity) as measurement_count
                FROM enhanced_measurements m
                WHERE m.salinity IS NOT NULL
                LIMIT 1;
            """
        
        elif 'spatial' in query_lower or 'distribution' in query_lower:
            return """
                SELECT fm.latitude, fm.longitude,
                       AVG(m.temperature) as avg_temperature,
                       AVG(m.salinity) as avg_salinity,
                       COUNT(m.id) as measurement_count
                FROM enhanced_floats_metadata fm
                JOIN enhanced_measurements m ON fm.id = m.metadata_id
                WHERE m.temperature IS NOT NULL
                AND m.salinity IS NOT NULL
                GROUP BY fm.latitude, fm.longitude
                ORDER BY fm.latitude, fm.longitude
                LIMIT 1000;
            """
        
        else:
            # Default query
            return """
                SELECT fm.platform_number, fm.latitude, fm.longitude,
                       m.pressure, m.temperature, m.salinity
                FROM enhanced_floats_metadata fm
                JOIN enhanced_measurements m ON fm.id = m.metadata_id
                WHERE m.pressure IS NOT NULL
                AND m.temperature IS NOT NULL
                AND m.salinity IS NOT NULL
                ORDER BY fm.platform_number, m.pressure ASC
                LIMIT 1000;
            """
    
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
                sql_query += " LIMIT 10000"
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
    
    def _attempt_query_fix(self, sql_query: str, error_msg: str) -> str:
        """Attempt to automatically fix common SQL issues"""
        
        error_lower = error_msg.lower()
        
        if "column" in error_lower and "does not exist" in error_lower:
            sql_query = sql_query.replace("temp ", "temperature ")
            sql_query = sql_query.replace("sal ", "salinity ")
            sql_query = sql_query.replace("pres ", "pressure ")
        
        if "relation" in error_lower and "does not exist" in error_lower:
            sql_query = sql_query.replace("measurements ", "enhanced_measurements ")
            sql_query = sql_query.replace("floats_metadata ", "enhanced_floats_metadata ")
        
        return sql_query
    
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
    """Test the enhanced oceanographic RAG system"""
    
    logger.info("Testing Enhanced Oceanographic RAG System")
    logger.info("=" * 60)
    
    # Initialize system
    rag_system = EnhancedOceanographicRAG()
    
    # Test queries
    test_queries = [
        "Show temperature measurements for float 1900121",
        "What is the average salinity in the database?",
        "Show surface temperature distribution",
        "Calculate temperature statistics by depth layers"
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
            else:
                logger.error(f"❌ Failed: {result['error']}")
                
        except Exception as e:
            logger.error(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_enhanced_oceanographic_rag()