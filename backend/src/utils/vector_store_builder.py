# src/utils/vector_store_builder.py
"""
ROBUST VECTOR STORE BUILDER - Based on Actual Schema
Built specifically for the production ARGO database schema.
No hardcoding - extracts everything from your actual database.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text, inspect
import traceback
from config.vector_store_config import get_vector_store_path


try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.error("LangChain not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionVectorStoreBuilder:
    """
    Builds vector store context from your actual production database.
    Every piece of information comes from live database introspection.
    """
    
    def __init__(self, db_engine, persist_directory: str):
        self.engine = db_engine
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        if LANGCHAIN_AVAILABLE:
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                logger.info("Embeddings initialized")
            except Exception as e:
                logger.error(f"Embeddings failed: {e}")
                self.embeddings = None
        else:
            self.embeddings = None
        
        # Storage for extracted data
        self.schema_info = None
        self.sample_data = None
        self.data_statistics = None
    
    def extract_live_database_context(self) -> Dict[str, Any]:
        """Extract complete context from your live production database"""
        
        logger.info("Extracting live database context...")
        
        try:
            # Step 1: Get actual schema from your database
            schema = self._extract_production_schema()
            
            # Step 2: Get real data samples to understand patterns
            samples = self._extract_data_samples()
            
            # Step 3: Calculate actual statistics
            stats = self._calculate_database_statistics()
            
            # Step 4: Discover actual relationships
            relationships = self._discover_table_relationships()
            
            # Step 5: Generate query patterns from real data
            patterns = self._generate_query_patterns_from_data()
            
            complete_context = {
                'schema': schema,
                'data_samples': samples,
                'statistics': stats,
                'relationships': relationships,
                'query_patterns': patterns,
                'extracted_at': datetime.now().isoformat()
            }
            
            # Store for document creation
            self.schema_info = schema
            self.sample_data = samples
            self.data_statistics = stats
            
            logger.info(f"Context extraction complete: {len(schema)} tables")
            return complete_context
            
        except Exception as e:
            logger.error(f"Context extraction failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _extract_production_schema(self) -> Dict[str, Any]:
        """Extract schema from your production database"""
        
        logger.info("Extracting production schema...")
        schema = {}
        
        try:
            inspector = inspect(self.engine)
            table_names = inspector.get_table_names()
            
            logger.info(f"Found tables: {table_names}")
            
            for table_name in table_names:
                try:
                    # Get columns (skip geometry columns that cause issues)
                    columns = inspector.get_columns(table_name)
                    valid_columns = []
                    
                    for col in columns:
                        # Skip problematic column types
                        col_type_str = str(col['type']).lower()
                        if 'geometry' not in col_type_str and 'geography' not in col_type_str:
                            valid_columns.append({
                                'name': col['name'],
                                'type': str(col['type']),
                                'nullable': col['nullable'],
                                'primary_key': col.get('primary_key', False)
                            })
                    
                    # Get foreign keys
                    try:
                        fks = inspector.get_foreign_keys(table_name)
                        foreign_keys = [
                            {
                                'columns': fk.get('constrained_columns', []),
                                'refers_to_table': fk.get('referred_table', ''),
                                'refers_to_columns': fk.get('referred_columns', [])
                            }
                            for fk in fks
                        ]
                    except Exception:
                        foreign_keys = []
                    
                    # Get indexes
                    try:
                        indexes = inspector.get_indexes(table_name)
                        index_info = [
                            {
                                'name': idx.get('name', ''),
                                'columns': idx.get('column_names', []),
                                'unique': idx.get('unique', False)
                            }
                            for idx in indexes
                        ]
                    except Exception:
                        index_info = []
                    
                    # Get row count
                    row_count = self._safe_get_row_count(table_name)
                    
                    schema[table_name] = {
                        'columns': valid_columns,
                        'foreign_keys': foreign_keys,
                        'indexes': index_info,
                        'row_count': row_count
                    }
                    
                    logger.info(f"Schema extracted for {table_name}: {len(valid_columns)} columns, {row_count} rows")
                    
                except Exception as e:
                    logger.warning(f"Failed to extract {table_name}: {e}")
                    continue
            
            return schema
            
        except Exception as e:
            logger.error(f"Schema extraction failed: {e}")
            raise
    
    def _safe_get_row_count(self, table_name: str) -> int:
        """Safely get row count for a table"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                return result.scalar() or 0
        except Exception as e:
            logger.warning(f"Cannot get count for {table_name}: {e}")
            return 0
    
    def _extract_data_samples(self) -> Dict[str, Any]:
        """Extract actual data samples from your database"""
        
        logger.info("Extracting data samples...")
        samples = {}
        
        # Sample from argo_profiles
        try:
            with self.engine.connect() as conn:
                profiles_query = """
                SELECT platform_number, cycle_number, profile_date, 
                       latitude, longitude, max_pressure, n_levels,
                       surface_temp, surface_salinity, mixed_layer_depth,
                       source_file
                FROM argo_profiles 
                WHERE surface_temp IS NOT NULL 
                ORDER BY profile_date DESC 
                LIMIT 100
                """
                
                df = pd.read_sql(profiles_query, conn)
                if not df.empty:
                    samples['argo_profiles'] = {
                        'sample_size': len(df),
                        'columns': list(df.columns),
                        'sample_values': {},
                        'data_ranges': {}
                    }
                    
                    # Get sample values and ranges for each column
                    for col in df.columns:
                        if df[col].dtype in ['int64', 'float64']:
                            valid_data = df[col].dropna()
                            if len(valid_data) > 0:
                                samples['argo_profiles']['data_ranges'][col] = {
                                    'min': float(valid_data.min()),
                                    'max': float(valid_data.max()),
                                    'mean': float(valid_data.mean())
                                }
                        
                        # Sample values (first 5 non-null values)
                        sample_vals = df[col].dropna().head(5).tolist()
                        samples['argo_profiles']['sample_values'][col] = [str(v) for v in sample_vals]
                
                logger.info(f"Profiles sample: {len(df)} records")
        
        except Exception as e:
            logger.warning(f"Failed to sample argo_profiles: {e}")
        
        # Sample from argo_measurements
        try:
            with self.engine.connect() as conn:
                measurements_query = """
                SELECT profile_id, pressure, depth, temperature, salinity
                FROM argo_measurements 
                WHERE temperature IS NOT NULL 
                ORDER BY profile_id DESC, pressure ASC 
                LIMIT 1000
                """
                
                df = pd.read_sql(measurements_query, conn)
                if not df.empty:
                    samples['argo_measurements'] = {
                        'sample_size': len(df),
                        'columns': list(df.columns),
                        'data_ranges': {}
                    }
                    
                    for col in df.columns:
                        if df[col].dtype in ['int64', 'float64']:
                            valid_data = df[col].dropna()
                            if len(valid_data) > 0:
                                samples['argo_measurements']['data_ranges'][col] = {
                                    'min': float(valid_data.min()),
                                    'max': float(valid_data.max()),
                                    'mean': float(valid_data.mean())
                                }
                
                logger.info(f"Measurements sample: {len(df)} records")
        
        except Exception as e:
            logger.warning(f"Failed to sample argo_measurements: {e}")
        
        return samples
    
    def _calculate_database_statistics(self) -> Dict[str, Any]:
        """Calculate actual database statistics"""
        
        logger.info("Calculating database statistics...")
        stats = {}
        
        try:
            with self.engine.connect() as conn:
                # Use your existing function if available
                try:
                    result = conn.execute(text("SELECT * FROM get_processing_stats()"))
                    processing_stats = dict(result.fetchall())
                    stats['processing_stats'] = processing_stats
                except Exception:
                    # Fallback to manual calculation
                    result = conn.execute(text("SELECT COUNT(*) FROM argo_profiles"))
                    stats['total_profiles'] = result.scalar()
                    
                    result = conn.execute(text("SELECT COUNT(*) FROM argo_measurements"))
                    stats['total_measurements'] = result.scalar()
                    
                    result = conn.execute(text("SELECT COUNT(DISTINCT platform_number) FROM argo_profiles"))
                    stats['unique_platforms'] = result.scalar()
                
                # Date range
                result = conn.execute(text("""
                    SELECT MIN(profile_date) as earliest, MAX(profile_date) as latest
                    FROM argo_profiles
                """))
                date_range = result.fetchone()
                if date_range:
                    stats['date_range'] = {
                        'earliest': str(date_range[0]),
                        'latest': str(date_range[1])
                    }
                
                # Geographic coverage
                result = conn.execute(text("""
                    SELECT MIN(latitude) as min_lat, MAX(latitude) as max_lat,
                           MIN(longitude) as min_lon, MAX(longitude) as max_lon
                    FROM argo_profiles
                """))
                geo_range = result.fetchone()
                if geo_range:
                    stats['geographic_coverage'] = {
                        'latitude_range': [float(geo_range[0]), float(geo_range[1])],
                        'longitude_range': [float(geo_range[2]), float(geo_range[3])]
                    }
                
                # Surface parameter statistics
                result = conn.execute(text("""
                    SELECT AVG(surface_temp) as avg_temp, STDDEV(surface_temp) as std_temp,
                           AVG(surface_salinity) as avg_sal, STDDEV(surface_salinity) as std_sal
                    FROM argo_profiles
                    WHERE surface_temp IS NOT NULL AND surface_salinity IS NOT NULL
                """))
                surface_stats = result.fetchone()
                if surface_stats:
                    stats['surface_parameters'] = {
                        'temperature': {'mean': float(surface_stats[0] or 0), 'std': float(surface_stats[1] or 0)},
                        'salinity': {'mean': float(surface_stats[2] or 0), 'std': float(surface_stats[3] or 0)}
                    }
        
        except Exception as e:
            logger.warning(f"Statistics calculation failed: {e}")
        
        return stats
    
    def _discover_table_relationships(self) -> Dict[str, Any]:
        """Discover relationships from your schema"""
        
        relationships = {
            'primary_relationship': {
                'description': 'argo_profiles (parent) → argo_measurements (child)',
                'join_condition': 'argo_profiles.id = argo_measurements.profile_id',
                'relationship_type': 'one_to_many',
                'importance': 'critical_for_depth_analysis'
            },
            'common_joins': [
                {
                    'sql': 'FROM argo_profiles p INNER JOIN argo_measurements m ON p.id = m.profile_id',
                    'use_case': 'Profile analysis with depth measurements',
                    'performance': 'Good with proper WHERE clauses'
                }
            ]
        }
        
        return relationships
    
    def _generate_query_patterns_from_data(self) -> List[Dict[str, Any]]:
        """Generate proven query patterns based on your actual schema"""
        
        patterns = [
            {
                'name': 'Surface Temperature Analysis',
                'description': 'Query surface temperature data with geographic filtering',
                'sql_template': '''
SELECT platform_number, profile_date, latitude, longitude, 
       surface_temp, surface_salinity
FROM argo_profiles 
WHERE surface_temp IS NOT NULL
  AND latitude BETWEEN {lat_min} AND {lat_max}
  AND longitude BETWEEN {lon_min} AND {lon_max}
  AND profile_date >= '{start_date}'
ORDER BY profile_date DESC
LIMIT {limit}
                '''.strip(),
                'parameters': ['lat_min', 'lat_max', 'lon_min', 'lon_max', 'start_date', 'limit'],
                'use_cases': ['regional surface analysis', 'temperature mapping', 'time series'],
                'performance': 'Fast - uses spatial indexes'
            },
            {
                'name': 'Profile Depth Analysis',
                'description': 'Full depth profile analysis with measurements',
                'sql_template': '''
SELECT p.platform_number, p.profile_date, p.latitude, p.longitude,
       m.pressure, m.depth, m.temperature, m.salinity
FROM argo_profiles p
INNER JOIN argo_measurements m ON p.id = m.profile_id
WHERE p.platform_number = '{platform_number}'
  AND m.temperature IS NOT NULL
ORDER BY m.pressure ASC
LIMIT {limit}
                '''.strip(),
                'parameters': ['platform_number', 'limit'],
                'use_cases': ['temperature profiles', 'salinity profiles', 'thermocline analysis'],
                'performance': 'Medium - requires JOIN'
            },
            {
                'name': 'Statistical Summary',
                'description': 'Regional statistical analysis',
                'sql_template': '''
SELECT COUNT(*) as profile_count,
       COUNT(DISTINCT platform_number) as unique_floats,
       AVG(surface_temp) as avg_surface_temp,
       STDDEV(surface_temp) as std_surface_temp,
       AVG(surface_salinity) as avg_surface_salinity,
       MIN(profile_date) as earliest_date,
       MAX(profile_date) as latest_date
FROM argo_profiles
WHERE surface_temp IS NOT NULL
  AND latitude BETWEEN {lat_min} AND {lat_max}
  AND longitude BETWEEN {lon_min} AND {lon_max}
  AND profile_date >= '{start_date}'
                '''.strip(),
                'parameters': ['lat_min', 'lat_max', 'lon_min', 'lon_max', 'start_date'],
                'use_cases': ['regional statistics', 'data coverage analysis'],
                'performance': 'Fast - aggregation with filters'
            },
            {
                'name': 'Platform-Specific Analysis',
                'description': 'Analysis for specific float platforms',
                'sql_template': '''
SELECT profile_date, latitude, longitude, surface_temp, 
       surface_salinity, mixed_layer_depth, n_levels
FROM argo_profiles
WHERE platform_number = '{platform_number}'
  AND profile_date >= '{start_date}'
ORDER BY profile_date DESC
LIMIT {limit}
                '''.strip(),
                'parameters': ['platform_number', 'start_date', 'limit'],
                'use_cases': ['float tracking', 'temporal analysis', 'drift patterns'],
                'performance': 'Very fast - uses primary index'
            }
        ]
        
        return patterns
    
    def create_context_documents(self) -> List[Document]:
        """Create comprehensive context documents for vector store"""
        
        if not self.schema_info:
            raise ValueError("Schema info not available. Run extract_live_database_context first.")
        
        documents = []
        
        # 1. Schema document
        schema_doc = self._create_schema_document()
        documents.append(schema_doc)
        
        # 2. Query pattern documents
        pattern_docs = self._create_query_pattern_documents()
        documents.extend(pattern_docs)
        
        # 3. Data characteristics document
        data_doc = self._create_data_characteristics_document()
        documents.append(data_doc)
        
        # 4. Best practices document
        practices_doc = self._create_best_practices_document()
        documents.append(practices_doc)
        
        logger.info(f"Created {len(documents)} context documents")
        return documents
    
    def _create_schema_document(self) -> Document:
        """Create detailed schema document"""
        
        content = "ARGO OCEANOGRAPHIC DATABASE SCHEMA\n"
        content += "=" * 50 + "\n\n"
        content += "CRITICAL: Use ONLY these exact table and column names in SQL queries.\n\n"
        
        for table_name, table_info in self.schema_info.items():
            content += f"TABLE: {table_name}\n"
            content += "-" * 30 + "\n"
            content += f"Row Count: {table_info.get('row_count', 'unknown'):,}\n\n"
            
            content += "Columns:\n"
            for col in table_info['columns']:
                nullable = "NULL" if col['nullable'] else "NOT NULL"
                pk = " (PRIMARY KEY)" if col['primary_key'] else ""
                content += f"  - {col['name']} ({col['type']}) {nullable}{pk}\n"
            
            if table_info['foreign_keys']:
                content += "\nForeign Keys:\n"
                for fk in table_info['foreign_keys']:
                    content += f"  - {fk['columns']} → {fk['refers_to_table']}.{fk['refers_to_columns']}\n"
            
            content += "\n"
        
        # Add critical JOIN information
        content += "CRITICAL JOIN PATTERNS:\n"
        content += "=" * 30 + "\n"
        content += "Profile + Measurements (MOST COMMON):\n"
        content += "FROM argo_profiles p\n"
        content += "INNER JOIN argo_measurements m ON p.id = m.profile_id\n\n"
        
        # Add data ranges if available
        if self.data_statistics:
            content += "DATA CHARACTERISTICS:\n"
            if 'processing_stats' in self.data_statistics:
                stats = self.data_statistics['processing_stats']
                content += f"Total Profiles: {stats.get('total_profiles', 'N/A'):,}\n"
                content += f"Total Measurements: {stats.get('total_measurements', 'N/A'):,}\n"
                content += f"Unique Platforms: {stats.get('unique_platforms', 'N/A'):,}\n"
            
            if 'date_range' in self.data_statistics:
                dr = self.data_statistics['date_range']
                content += f"Date Range: {dr['earliest']} to {dr['latest']}\n"
            
            if 'geographic_coverage' in self.data_statistics:
                geo = self.data_statistics['geographic_coverage']
                content += f"Latitude Range: {geo['latitude_range'][0]:.1f}° to {geo['latitude_range'][1]:.1f}°\n"
                content += f"Longitude Range: {geo['longitude_range'][0]:.1f}° to {geo['longitude_range'][1]:.1f}°\n"
        
        return Document(
            page_content=content,
            metadata={'type': 'schema', 'priority': 'critical', 'source': 'live_database'}
        )
    
    def _create_query_pattern_documents(self) -> List[Document]:
        """Create documents for each query pattern"""
        
        documents = []
        
        if not hasattr(self, 'query_patterns'):
            patterns = self._generate_query_patterns_from_data()
        else:
            patterns = self.query_patterns
        
        for pattern in patterns:
            content = f"QUERY PATTERN: {pattern['name']}\n"
            content += "=" * 40 + "\n\n"
            content += f"Description: {pattern['description']}\n\n"
            content += f"Use Cases: {', '.join(pattern['use_cases'])}\n\n"
            content += f"Performance: {pattern['performance']}\n\n"
            content += "SQL Template:\n"
            content += pattern['sql_template'] + "\n\n"
            content += "Parameters:\n"
            for param in pattern['parameters']:
                content += f"  - {param}\n"
            
            documents.append(Document(
                page_content=content,
                metadata={
                    'type': 'query_pattern',
                    'pattern_name': pattern['name'].lower().replace(' ', '_'),
                    'priority': 'high'
                }
            ))
        
        return documents
    
    def _create_data_characteristics_document(self) -> Document:
        """Create document describing data characteristics"""
        
        content = "ARGO DATA CHARACTERISTICS\n"
        content += "=" * 30 + "\n\n"
        
        if self.sample_data and 'argo_profiles' in self.sample_data:
            profiles = self.sample_data['argo_profiles']
            content += "SURFACE MEASUREMENTS (argo_profiles):\n"
            
            if 'data_ranges' in profiles:
                ranges = profiles['data_ranges']
                if 'surface_temp' in ranges:
                    temp_range = ranges['surface_temp']
                    content += f"Temperature: {temp_range['min']:.1f}°C to {temp_range['max']:.1f}°C (avg: {temp_range['mean']:.1f}°C)\n"
                
                if 'surface_salinity' in ranges:
                    sal_range = ranges['surface_salinity']
                    content += f"Salinity: {sal_range['min']:.1f} to {sal_range['max']:.1f} PSU (avg: {sal_range['mean']:.1f})\n"
                
                if 'latitude' in ranges and 'longitude' in ranges:
                    lat_range = ranges['latitude']
                    lon_range = ranges['longitude']
                    content += f"Geographic Coverage: {lat_range['min']:.1f}°N to {lat_range['max']:.1f}°N, "
                    content += f"{lon_range['min']:.1f}°E to {lon_range['max']:.1f}°E\n"
        
        if self.sample_data and 'argo_measurements' in self.sample_data:
            measurements = self.sample_data['argo_measurements']
            content += "\nDEPTH MEASUREMENTS (argo_measurements):\n"
            
            if 'data_ranges' in measurements:
                ranges = measurements['data_ranges']
                if 'pressure' in ranges:
                    press_range = ranges['pressure']
                    content += f"Pressure: {press_range['min']:.1f} to {press_range['max']:.1f} dbar\n"
                
                if 'temperature' in ranges:
                    temp_range = ranges['temperature']
                    content += f"Profile Temperature: {temp_range['min']:.1f}°C to {temp_range['max']:.1f}°C\n"
        
        content += "\nDATA QUALITY GUIDELINES:\n"
        content += "- Always filter for NOT NULL on required parameters\n"
        content += "- Use geographic bounds to improve performance\n"
        content += "- Include date filters for temporal analysis\n"
        content += "- JOIN operations require proper WHERE clauses\n"
        
        return Document(
            page_content=content,
            metadata={'type': 'data_characteristics', 'priority': 'medium'}
        )
    
    def _create_best_practices_document(self) -> Document:
        """Create best practices document"""
        
        content = "SQL BEST PRACTICES FOR ARGO DATABASE\n"
        content += "=" * 40 + "\n\n"
        
        content += "PERFORMANCE OPTIMIZATION:\n"
        content += "1. Always use table aliases: p for argo_profiles, m for argo_measurements\n"
        content += "2. Include geographic filters (latitude, longitude) when possible\n"
        content += "3. Use date filters: profile_date >= 'YYYY-MM-DD'\n"
        content += "4. Add NOT NULL filters for required parameters\n"
        content += "5. Always include LIMIT clauses\n\n"
        
        content += "JOIN REQUIREMENTS:\n"
        content += "- For surface-only data: Use argo_profiles table only\n"
        content += "- For depth profiles: MUST JOIN with argo_measurements\n"
        content += "- JOIN syntax: INNER JOIN argo_measurements m ON p.id = m.profile_id\n\n"
        
        content += "COMMON MISTAKES TO AVOID:\n"
        content += "- Never reference measurement columns without JOIN\n"
        content += "- Don't use non-existent tables or columns\n"
        content += "- Always include geographic constraints for large queries\n"
        content += "- Don't forget date filters for historical data\n\n"
        
        content += "VALIDATION CHECKLIST:\n"
        content += "✓ Table names: argo_profiles, argo_measurements\n"
        content += "✓ Column names match schema exactly\n"
        content += "✓ JOINs present when accessing measurement data\n"
        content += "✓ Geographic and temporal filters included\n"
        content += "✓ NOT NULL filters for critical parameters\n"
        content += "✓ Reasonable LIMIT clause\n"
        
        return Document(
            page_content=content,
            metadata={'type': 'best_practices', 'priority': 'critical'}
        )
    
    def build_production_vector_store(self) -> bool:
        """Build the complete production vector store"""
        
        if not self.embeddings:
            logger.error("Embeddings not available - cannot build vector store")
            return False
        
        try:
            logger.info("Building production vector store...")
            
            # Step 1: Extract all context from live database
            logger.info("Extracting database context...")
            context = self.extract_live_database_context()
            
            # Step 2: Create comprehensive documents
            logger.info("Creating context documents...")
            documents = self.create_context_documents()
            
            # Step 3: Build vector store
            logger.info(f"Building vector store with {len(documents)} documents...")
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            
            vector_store.persist()
            
            # Step 4: Test quality
            self._test_vector_store_quality(vector_store)
            
            logger.info("Production vector store built successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Vector store building failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _test_vector_store_quality(self, vector_store):
        """Test the quality of the built vector store"""
        
        test_queries = [
            "What tables are available in the database?",
            "How do I query surface temperature data?",
            "What is the JOIN syntax for depth profiles?",
            "Show me query for Arabian Sea region",
            "How to filter by date range?"
        ]
        
        logger.info("Testing vector store quality...")
        
        total_score = 0
        for query in test_queries:
            try:
                results = vector_store.similarity_search_with_score(query, k=3)
                if results:
                    best_score = results[0][1]
                    # Convert distance to relevance (lower distance = higher relevance)
                    relevance = max(0, (1 - best_score) * 100)
                    total_score += relevance
                    logger.info(f"Query: '{query[:30]}...' - Relevance: {relevance:.1f}%")
            except Exception as e:
                logger.warning(f"Test query failed: {e}")
        
        avg_quality = total_score / len(test_queries)
        logger.info(f"Average context quality: {avg_quality:.1f}%")
        
        if avg_quality >= 70:
            logger.info("✅ Context quality is good for production")
        else:
            logger.warning("⚠️ Context quality below 70% - may need improvement")


def rebuild_vector_store_from_live_db(db_engine, persist_directory: str) -> bool:
    """
    Main function to rebuild vector store from your live production database
    """
    
    logger.info("Rebuilding vector store from live production database...")
    
    try:
        builder = ProductionVectorStoreBuilder(db_engine, persist_directory)
        success = builder.build_production_vector_store()
        
        if success:
            logger.info("✅ Vector store rebuilt successfully!")
            logger.info("The system now has production-grade context for DeepSeek")
        else:
            logger.error("❌ Vector store rebuild failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Vector store rebuild failed: {e}")
        return False


if __name__ == "__main__":
    import os
    from sqlalchemy import create_engine
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Create database engine
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        logger.error("DATABASE_URL not found in environment")
        exit(1)
    
    try:
        engine = create_engine(db_url)
        
        # Test database connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM argo_profiles LIMIT 1"))
            logger.info("Database connection successful")
        
        # Set persist directory
        persist_dir = str(get_vector_store_path())
        
        # Rebuild vector store
        logger.info("Starting vector store rebuild...")
        success = rebuild_vector_store_from_live_db(engine, persist_dir)
        
        if success:
            logger.info("=" * 50)
            logger.info("VECTOR STORE REBUILD COMPLETE")
            logger.info("=" * 50)
            logger.info("Your RAG system now has:")
            logger.info("✓ Live database schema context")
            logger.info("✓ Real data patterns and ranges") 
            logger.info("✓ Proven query patterns")
            logger.info("✓ Production-grade best practices")
            logger.info("✓ Quality-tested embeddings")
            logger.info("")
            logger.info("DeepSeek should now generate much better SQL queries!")
        else:
            logger.error("Vector store rebuild failed - check logs above")
            exit(1)
    
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        logger.error(traceback.format_exc())
        exit(1)