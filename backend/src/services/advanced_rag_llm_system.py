# advanced_rag_llm_system.py
import os
import re
import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy import create_engine, text, MetaData, inspect
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationSummaryBufferMemory
from dotenv import load_dotenv
import pandas as pd
import logging
from datetime import datetime
import numpy as np
from pathlib import Path
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class AdvancedRAGEnhancedLLM:
    """Advanced RAG-Enhanced LLM with conversation memory, query optimization, and intelligent routing"""
    
    def __init__(self, persist_directory: str = "chroma_db_advanced"):
        # Database connection with connection pooling
        self.engine = create_engine(
            os.getenv('DATABASE_URL'),
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            connect_args={"options": "-c timezone=UTC"}
        )
        
        # Enhanced LLM with better parameters
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            max_tokens=2048,
            google_api_key=os.getenv('GEMINI_API_KEY')
        )
        
        # Embeddings for RAG
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv('GEMINI_API_KEY')
        )
        
        # Load vector store
        self.vector_store = self._load_vector_store(os.path.join("backend", "storage", persist_directory))
        
        # Database metadata and schema caching
        self.inspector = inspect(self.engine)
        self.schema_cache = self._build_schema_cache()
        
        # Conversation memory
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1000,
            return_messages=True
        )
        
        # Query cache for performance
        self.query_cache = {}
        self.cache_dir = Path("query_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def _load_vector_store(self, persist_directory: str) -> Optional[Chroma]:
        """Load the Chroma vector store with error handling"""
        try:
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            doc_count = vector_store._collection.count()
            logger.info(f"Loaded vector store with {doc_count} documents")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            logger.info("Please run advanced_rag_setup.py first")
            return None
    
    def _build_schema_cache(self) -> Dict:
        """Build and cache database schema information"""
        schema_cache = {}
        
        try:
            tables = self.inspector.get_table_names()
            for table in tables:
                columns = self.inspector.get_columns(table)
                foreign_keys = self.inspector.get_foreign_keys(table)
                indexes = self.inspector.get_indexes(table)
                
                schema_cache[table] = {
                    'columns': columns,
                    'foreign_keys': foreign_keys,
                    'indexes': indexes,
                    'column_names': [col['name'] for col in columns],
                    'nullable_columns': [col['name'] for col in columns if col['nullable']],
                    'numeric_columns': [col['name'] for col in columns if 'numeric' in str(col['type']).lower() or 'float' in str(col['type']).lower() or 'int' in str(col['type']).lower()]
                }
                
            logger.info(f"Cached schema for {len(tables)} tables")
        except Exception as e:
            logger.error(f"Error building schema cache: {e}")
        
        return schema_cache
    
    def _get_query_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _load_query_cache(self, cache_key: str) -> Optional[Dict]:
        """Load cached query result"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                # Check if cache is still fresh (24 hours)
                cache_time = datetime.fromisoformat(cached['timestamp'])
                if (datetime.now() - cache_time).total_seconds() < 86400:
                    return cached
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
        return None
    
    def _save_query_cache(self, cache_key: str, data: Dict) -> None:
        """Save query result to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            data['timestamp'] = datetime.now().isoformat()
            with open(cache_file, 'w') as f:
                json.dump(data, f, default=str, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    def get_intelligent_context(self, query: str, k: int = 5) -> Tuple[str, Dict]:
        """Get intelligent context using multiple strategies"""
        if self.vector_store is None:
            return "", {}
        
        try:
            # Multi-strategy context retrieval
            contexts = []
            metadata_summary = {}
            
            # Strategy 1: Direct similarity search
            direct_results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Strategy 2: Query expansion for better context
            expanded_queries = self._expand_query(query)
            expanded_results = []
            
            for exp_query in expanded_queries:
                exp_results = self.vector_store.similarity_search_with_score(exp_query, k=2)
                expanded_results.extend(exp_results)
            
            # Combine and deduplicate results
            all_results = direct_results + expanded_results
            seen_content = set()
            unique_results = []
            
            for doc, score in all_results:
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_results.append((doc, score))
            
            # Sort by relevance and take top k
            unique_results.sort(key=lambda x: x[1])
            top_results = unique_results[:k]
            
            if not top_results:
                return "No relevant context found.", {}
            
            # Format context with relevance scores
            context_parts = []
            for i, (doc, score) in enumerate(top_results, 1):
                relevance = max(0, (1 - score) * 100)
                
                if relevance > 30:  # Only include reasonably relevant results
                    context_parts.append(f"--- Context {i} (Relevance: {relevance:.1f}%) ---")
                    context_parts.append(doc.page_content.strip())
                    
                    # Collect metadata
                    meta = doc.metadata
                    source_type = meta.get('type', 'unknown')
                    if source_type not in metadata_summary:
                        metadata_summary[source_type] = 0
                    metadata_summary[source_type] += 1
                    
                    context_parts.append(f"[Source: {meta.get('source', 'unknown')}, Type: {source_type}]")
                    context_parts.append("")
            
            full_context = "Relevant Database Context:\n\n" + "\n".join(context_parts)
            return full_context, metadata_summary
            
        except Exception as e:
            logger.error(f"Error in intelligent context retrieval: {e}")
            return "", {}
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with domain-specific synonyms and related terms"""
        expansions = []
        
        # Domain-specific expansions
        query_lower = query.lower()
        
        # Oceanography term mappings
        term_mappings = {
            'temperature': ['temp', 'thermal', 'T', 'celsius', 'degree'],
            'salinity': ['salt', 'S', 'PSU', 'conductivity'],
            'pressure': ['depth', 'P', 'dbar', 'level'],
            'profile': ['vertical', 'depth profile', 'cast'],
            'float': ['platform', 'buoy', 'argo'],
            'surface': ['0m', '0 dbar', 'sea surface', 'sst'],
            'deep': ['bottom', 'abyssal', 'deep water', '>1000'],
            'spatial': ['geographic', 'map', 'location', 'lat lon'],
            'temporal': ['time series', 'seasonal', 'trend']
        }
        
        for term, synonyms in term_mappings.items():
            if term in query_lower:
                for synonym in synonyms:
                    if synonym not in query_lower:
                        expanded = query_lower.replace(term, synonym)
                        if expanded != query_lower:
                            expansions.append(expanded)
        
        # Add related concept queries
        if 'profile' in query_lower:
            expansions.extend(['vertical measurement', 'depth analysis'])
        if 'spatial' in query_lower or 'map' in query_lower:
            expansions.extend(['geographic distribution', 'location data'])
        if 'quality' in query_lower:
            expansions.extend(['missing data', 'data validation'])
        
        return expansions[:3]  # Limit expansions
    
    def get_enhanced_schema_info(self) -> str:
        """Get enhanced schema information with relationships"""
        schema_info = []
        
        for table, info in self.schema_cache.items():
            table_info = [f"\nTable: {table}"]
            
            # Columns with types and constraints
            col_details = []
            for col in info['columns']:
                col_detail = f"  {col['name']} ({col['type']}"
                if not col['nullable']:
                    col_detail += ", NOT NULL"
                if col.get('default'):
                    col_detail += f", DEFAULT {col['default']}"
                col_detail += ")"
                col_details.append(col_detail)
            
            table_info.append("Columns:")
            table_info.extend(col_details)
            
            # Foreign key relationships
            if info['foreign_keys']:
                table_info.append("Relationships:")
                for fk in info['foreign_keys']:
                    fk_info = f"  {', '.join(fk['constrained_columns'])} -> {fk['referred_table']}.{', '.join(fk['referred_columns'])}"
                    table_info.append(fk_info)
            
            # Indexed columns for performance hints
            if info['indexes']:
                indexed_cols = []
                for idx in info['indexes']:
                    indexed_cols.extend(idx['column_names'])
                if indexed_cols:
                    table_info.append(f"Indexed columns: {', '.join(set(indexed_cols))}")
            
            # Data type summary
            numeric_cols = info['numeric_columns']
            if numeric_cols:
                table_info.append(f"Numeric columns (for aggregation): {', '.join(numeric_cols)}")
            
            schema_info.append("\n".join(table_info))
        
        return "\n".join(schema_info)
    
    def classify_query_intent(self, query: str) -> Dict[str, Any]:
        """Classify the intent and complexity of the query"""
        query_lower = query.lower()
        
        intent = {
            'type': 'unknown',
            'complexity': 'simple',
            'parameters': [],
            'aggregation': False,
            'spatial': False,
            'temporal': False,
            'specific_float': False,
            'requires_join': False
        }
        
        # Detect query type
        if any(word in query_lower for word in ['profile', 'depth', 'vertical']):
            intent['type'] = 'profile'
            intent['requires_join'] = True
            
        elif any(word in query_lower for word in ['map', 'spatial', 'geographic', 'latitude', 'longitude']):
            intent['type'] = 'spatial'
            intent['spatial'] = True
            intent['requires_join'] = True
            
        elif any(word in query_lower for word in ['average', 'mean', 'sum', 'count', 'max', 'min', 'statistics']):
            intent['type'] = 'statistical'
            intent['aggregation'] = True
            intent['complexity'] = 'medium'
            
        elif any(word in query_lower for word in ['time', 'seasonal', 'trend', 'date', 'year', 'month']):
            intent['type'] = 'temporal'
            intent['temporal'] = True
            intent['complexity'] = 'medium'
            
        # Detect specific parameters
        parameter_keywords = {
            'temperature': ['temperature', 'temp', 'thermal'],
            'salinity': ['salinity', 'salt', 'psu'],
            'pressure': ['pressure', 'depth', 'dbar']
        }
        
        for param, keywords in parameter_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                intent['parameters'].append(param)
        
        # Detect specific float reference
        if re.search(r'\b\d{7}\b', query) or 'float' in query_lower:
            intent['specific_float'] = True
        
        # Detect complexity indicators
        complexity_indicators = ['group by', 'join', 'subquery', 'having', 'window']
        if any(indicator in query_lower for indicator in complexity_indicators):
            intent['complexity'] = 'complex'
        
        return intent
    
    def generate_optimized_sql(self, natural_language_query: str) -> Optional[str]:
        """Generate optimized SQL using RAG context and query classification"""
        
        # Get intelligent context
        rag_context, context_meta = self.get_intelligent_context(natural_language_query)
        
        # Classify query intent
        intent = self.classify_query_intent(natural_language_query)
        
        # Get enhanced schema
        schema_info = self.get_enhanced_schema_info()
        
        # Build context-aware system prompt
        system_prompt = self._build_system_prompt(schema_info, rag_context, intent, context_meta)
        
        # Build user prompt with examples
        user_prompt = self._build_user_prompt(natural_language_query, intent)
        
        try:
            # Add conversation context
            conversation_context = self.memory.chat_memory.messages[-4:] if self.memory.chat_memory.messages else []
            
            messages = [SystemMessage(content=system_prompt)]
            
            # Add relevant conversation history
            if conversation_context:
                messages.append(HumanMessage(content="Previous conversation context for reference:"))
                for msg in conversation_context:
                    messages.append(msg)
            
            messages.append(HumanMessage(content=user_prompt))
            
            response = self.llm(messages)
            sql_query = self._clean_and_validate_sql(response.content, intent)
            
            # Update conversation memory
            self.memory.chat_memory.add_user_message(natural_language_query)
            self.memory.chat_memory.add_ai_message(f"Generated SQL: {sql_query}")
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return None
    
    def _build_system_prompt(self, schema_info: str, rag_context: str, intent: Dict, context_meta: Dict) -> str:
        """Build comprehensive system prompt"""
        
        base_prompt = f"""You are an expert oceanographer and PostgreSQL analyst specializing in ARGO float data analysis.

DATASET CONTEXT:
{schema_info}

{rag_context}

QUERY CLASSIFICATION:
Type: {intent['type']}, Complexity: {intent['complexity']}
Parameters: {', '.join(intent['parameters']) if intent['parameters'] else 'None detected'}
Requires JOIN: {intent['requires_join']}
Spatial Analysis: {intent['spatial']}
Temporal Analysis: {intent['temporal']}

CRITICAL SQL GENERATION RULES:
1. ALWAYS use proper JOINs: measurements m JOIN floats_metadata fm ON m.metadata_id = fm.id
2. Use table aliases (m for measurements, fm for floats_metadata) for clarity
3. Add appropriate WHERE clauses for data filtering
4. Include ORDER BY for meaningful result ordering
5. Use LIMIT for large datasets (default 1000, use 10000 for aggregations)
6. Handle NULL values with IS NOT NULL or COALESCE()
7. For spatial queries: Include latitude, longitude from floats_metadata
8. For profile queries: Order by pressure ASC (shallow to deep)
9. For statistical queries: Use appropriate GROUP BY and aggregation functions
10. For temporal queries: Use date functions and proper time filtering

PERFORMANCE OPTIMIZATIONS:
- Use indexed columns when possible
- Avoid SELECT * in production queries
- Use DISTINCT only when necessary
- Consider using EXISTS instead of IN for subqueries

OCEANOGRAPHIC BEST PRACTICES:"""
        
        # Add specific guidance based on intent
        if intent['type'] == 'profile':
            base_prompt += """
- Profile queries should order by pressure ASC (surface to depth)
- Include pressure, temperature, salinity for complete profiles
- Consider filtering for specific pressure ranges if needed"""
        
        elif intent['type'] == 'spatial':
            base_prompt += """
- Spatial queries should include latitude, longitude coordinates
- Consider adding geographic bounds for performance
- Surface data typically uses pressure < 10 dbar"""
        
        elif intent['type'] == 'statistical':
            base_prompt += """
- Use appropriate aggregate functions (AVG, MIN, MAX, COUNT, STDDEV)
- Group by relevant dimensions (pressure levels, time periods, regions)
- Consider using HAVING clause for post-aggregation filtering"""
        
        base_prompt += """

Return ONLY the SQL query without explanation, comments, or markdown formatting.
Ensure the query is syntactically correct and optimized for the ARGO dataset structure."""
        
        return base_prompt
    
    def _build_user_prompt(self, query: str, intent: Dict) -> str:
        """Build context-aware user prompt with examples"""
        
        examples = ""
        
        # Add relevant examples based on query type
        if intent['type'] == 'profile':
            examples = """
EXAMPLE for profile queries:
Query: "Show temperature profile for float 1900121"
SQL: SELECT fm.platform_number, m.pressure, m.temperature 
     FROM measurements m 
     JOIN floats_metadata fm ON m.metadata_id = fm.id 
     WHERE fm.platform_number = '1900121' AND m.temperature IS NOT NULL 
     ORDER BY m.pressure ASC LIMIT 1000;
"""
        
        elif intent['type'] == 'spatial':
            examples = """
EXAMPLE for spatial queries:
Query: "Show surface salinity distribution"
SQL: SELECT fm.latitude, fm.longitude, AVG(m.salinity) as avg_salinity
     FROM measurements m
     JOIN floats_metadata fm ON m.metadata_id = fm.id
     WHERE m.pressure < 10 AND m.salinity IS NOT NULL
     GROUP BY fm.latitude, fm.longitude
     ORDER BY fm.latitude, fm.longitude LIMIT 1000;
"""
        
        elif intent['type'] == 'statistical':
            examples = """
EXAMPLE for statistical queries:
Query: "What is the average temperature by depth level?"
SQL: SELECT ROUND(m.pressure/100)*100 as pressure_level, 
            COUNT(*) as measurement_count,
            AVG(m.temperature) as avg_temperature
     FROM measurements m
     WHERE m.temperature IS NOT NULL AND m.pressure IS NOT NULL
     GROUP BY ROUND(m.pressure/100)*100
     ORDER BY pressure_level LIMIT 1000;
"""
        
        user_prompt = f"""{examples}

Now convert this oceanography question to optimized PostgreSQL:

QUESTION: {query}

Generate ONLY the SQL query that best answers this question using the ARGO dataset structure."""
        
        return user_prompt
    
    def _clean_and_validate_sql(self, query: str, intent: Dict) -> str:
        """Clean and validate the generated SQL query"""
        # Remove markdown formatting
        query = re.sub(r'```sql\s*', '', query)
        query = re.sub(r'```\s*', '', query)
        query = query.strip()
        
        # Remove comments and extra whitespace
        query = re.sub(r'--.*\n', '\n', query)
        query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
        query = ' '.join(query.split())
        
        # Remove any existing semicolons first
        query = query.rstrip(';')
        
        # Ensure it's a SELECT query
        if not query.upper().startswith('SELECT'):
            select_match = re.search(r'(SELECT.*?)(?=;|$)', query, re.IGNORECASE | re.DOTALL)
            if select_match:
                query = select_match.group(1).strip()
            else:
                raise ValueError("Generated query is not a SELECT statement")
        
        # Add appropriate LIMIT if missing
        if 'LIMIT' not in query.upper():
            if intent['aggregation'] or 'GROUP BY' in query.upper():
                query += " LIMIT 10000"
            else:
                query += " LIMIT 1000"
        
        # Now add semicolon at the very end
        query += ";"
    
        return query
    
    def execute_query_with_retry(self, sql_query: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Execute SQL query with retry logic and error handling"""
        
        for attempt in range(max_retries):
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(text(sql_query))
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    
                    # Log successful execution
                    logger.info(f"Query executed successfully: {len(df)} rows returned")
                    return df
                    
            except Exception as e:
                logger.warning(f"Query execution attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    # Try to fix common issues
                    if "does not exist" in str(e).lower():
                        logger.info("Attempting to fix column/table reference issues...")
                        # Could implement auto-correction logic here
                    continue
                else:
                    logger.error(f"Final query execution failed: {e}")
                    logger.error(f"Problematic query: {sql_query}")
        
        return None
    
    def process_advanced_query(self, natural_language_query: str) -> Dict[str, Any]:
        """Process query with advanced features including caching and conversation context"""
        
        # Check cache first
        cache_key = self._get_query_cache_key(natural_language_query)
        cached_result = self._load_query_cache(cache_key)
        
        if cached_result:
            logger.info("Returning cached result")
            return cached_result
        
        logger.info(f"Processing query: {natural_language_query}")
        
        # Generate SQL with RAG enhancement
        start_time = datetime.now()
        sql_query = self.generate_optimized_sql(natural_language_query)
        
        if not sql_query:
            return {
                'success': False,
                'error': 'Could not generate SQL query',
                'query': natural_language_query,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
        
        logger.info(f"Generated SQL: {sql_query}")
        
        # Execute query with retry
        results_df = self.execute_query_with_retry(sql_query)
        
        if results_df is None:
            return {
                'success': False,
                'error': 'Error executing SQL query',
                'sql_query': sql_query,
                'query': natural_language_query,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
        
        # Prepare result
        result = {
            'success': True,
            'sql_query': sql_query,
            'results': results_df,
            'result_count': len(results_df),
            'query': natural_language_query,
            'processing_time': (datetime.now() - start_time).total_seconds(),
            'columns': list(results_df.columns) if not results_df.empty else [],
            'data_types': {col: str(dtype) for col, dtype in results_df.dtypes.items()} if not results_df.empty else {}
        }
        
        # Cache successful results
        if result['success'] and result['result_count'] > 0:
            cache_data = result.copy()
            # Convert DataFrame to dict for JSON serialization
            if not results_df.empty:
                cache_data['results'] = results_df.to_dict('records')
            self._save_query_cache(cache_key, cache_data)
        
        return result

def test_advanced_rag_system():
    """Comprehensive test suite for the advanced RAG system"""
    logger.info("🚀 Testing Advanced RAG-Enhanced LLM System")
    logger.info("=" * 70)
    
    rag_llm = AdvancedRAGEnhancedLLM()
    
    # Test queries with different complexity levels
    test_queries = [
        # Simple profile queries
        ("Show temperature profile for float 1900121", "profile", "simple"),
        ("Get salinity measurements for platform 1900122", "profile", "simple"),
        
        # Statistical queries
        ("What is the average temperature by pressure level?", "statistical", "medium"),
        ("Show temperature and salinity statistics for the dataset", "statistical", "medium"),
        ("Count measurements by institution", "statistical", "simple"),
        
        # Spatial queries
        ("Show surface temperature distribution", "spatial", "medium"),
        ("Map salinity values at 100 dbar depth", "spatial", "medium"),
        ("Find all measurements in the Arabian Sea region", "spatial", "complex"),
        
        # Temporal queries
        ("Show seasonal temperature trends", "temporal", "complex"),
        ("Get monthly average salinity for 2023", "temporal", "medium"),
        
        # Complex analytical queries
        ("Compare temperature profiles between different projects", "analytical", "complex"),
        ("Find floats with unusual salinity readings", "analytical", "complex")
    ]
    
    results_summary = []
    
    for query, expected_type, expected_complexity in test_queries:
        logger.info(f"\n🔍 Testing Query: {query}")
        logger.info(f"Expected: {expected_type} ({expected_complexity})")
        logger.info("-" * 50)
        
        try:
            # Test query classification
            intent = rag_llm.classify_query_intent(query)
            logger.info(f"Classified as: {intent['type']} ({intent['complexity']})")
            logger.info(f"Parameters: {intent['parameters']}")
            
            # Process full query
            result = rag_llm.process_advanced_query(query)
            
            if result['success']:
                logger.info(f"✅ Success! Processing time: {result['processing_time']:.2f}s")
                logger.info(f"📊 Results: {result['result_count']} rows, {len(result['columns'])} columns")
                
                if result['result_count'] > 0 and len(result['results']) > 0:
                    # Show sample of results
                    if isinstance(result['results'], list):
                        sample = result['results'][:3] if len(result['results']) > 3 else result['results']
                        logger.info(f"Sample results: {sample}")
                    else:
                        logger.info(f"First few results:\n{result['results'].head(3).to_string()}")
                
                results_summary.append({
                    'query': query,
                    'status': 'success',
                    'rows': result['result_count'],
                    'time': result['processing_time']
                })
                
            else:
                logger.error(f"❌ Failed: {result['error']}")
                logger.error(f"SQL: {result.get('sql_query', 'N/A')}")
                
                results_summary.append({
                    'query': query,
                    'status': 'failed',
                    'error': result['error'],
                    'time': result['processing_time']
                })
            
        except Exception as e:
            logger.error(f"❌ Exception: {e}")
            results_summary.append({
                'query': query,
                'status': 'exception',
                'error': str(e),
                'time': 0
            })
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("="*70)
    
    successful = len([r for r in results_summary if r['status'] == 'success'])
    total = len(results_summary)
    
    logger.info(f"Success Rate: {successful}/{total} ({successful/total*100:.1f}%)")
    logger.info(f"Average Processing Time: {np.mean([r['time'] for r in results_summary]):.2f}s")
    
    logger.info("\nDetailed Results:")
    for result in results_summary:
        status_icon = "✅" if result['status'] == 'success' else "❌"
        logger.info(f"{status_icon} {result['query'][:50]}... - {result['status']}")

if __name__ == "__main__":
    test_advanced_rag_system()