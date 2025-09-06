# advanced_rag_setup.py
import os
import json
import logging
from typing import List, Dict, Any
from sqlalchemy import create_engine, text, inspect
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class AdvancedRAGSetup:
    """Setup class for creating the vector store for advanced RAG system"""
    
    def __init__(self, persist_directory: str = None):
        if persist_directory is None:
            persist_directory = os.path.join("backend", "storage", "chroma_db_advanced")
        # Database connection
        self.engine = create_engine(
            os.getenv('DATABASE_URL'),
            connect_args={"options": "-c timezone=UTC"}
        )
        
        # Embeddings for RAG
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv('GEMINI_API_KEY')
        )
        
        self.persist_directory = persist_directory
        self.inspector = inspect(self.engine)
    
    def get_database_schema_documents(self) -> List[Document]:
        """Generate documents for database schema information"""
        documents = []
        
        try:
            tables = self.inspector.get_table_names()
            logger.info(f"Found {len(tables)} tables in database")
            
            for table in tables:
                # Get table information
                columns = self.inspector.get_columns(table)
                foreign_keys = self.inspector.get_foreign_keys(table)
                indexes = self.inspector.get_indexes(table)
                
                # Build schema description
                schema_content = []
                schema_content.append(f"Table: {table}")
                schema_content.append("Columns:")
                
                for col in columns:
                    col_info = f"  {col['name']} ({col['type']})"
                    if not col['nullable']:
                        col_info += " NOT NULL"
                    if col.get('default'):
                        col_info += f" DEFAULT {col['default']}"
                    schema_content.append(col_info)
                
                # Add relationships
                if foreign_keys:
                    schema_content.append("Foreign Key Relationships:")
                    for fk in foreign_keys:
                        fk_info = f"  {', '.join(fk['constrained_columns'])} → {fk['referred_table']}({', '.join(fk['referred_columns'])})"
                        schema_content.append(fk_info)
                
                # Add indexes
                if indexes:
                    schema_content.append("Indexes:")
                    for idx in indexes:
                        if not idx.get('unique', False):
                            idx_info = f"  INDEX: {', '.join(idx['column_names'])}"
                        else:
                            idx_info = f"  UNIQUE INDEX: {', '.join(idx['column_names'])}"
                        schema_content.append(idx_info)
                
                # Create document
                schema_text = "\n".join(schema_content)
                document = Document(
                    page_content=schema_text,
                    metadata={
                        "type": "schema",
                        "source": "database_schema",
                        "table": table,
                        "content_type": "table_structure"
                    }
                )
                documents.append(document)
                
                # Create additional context documents for each table
                self._create_table_context_documents(table, columns, documents)
            
        except Exception as e:
            logger.error(f"Error generating schema documents: {e}")
        
        return documents
    
    def _create_table_context_documents(self, table: str, columns: List[Dict], documents: List[Document]) -> None:
        """Create additional context documents for better retrieval"""
        
        # Document 1: Column descriptions and usage
        column_details = []
        numeric_cols = []
        text_cols = []
        date_cols = []
        
        for col in columns:
            col_type = str(col['type']).lower()
            column_details.append(f"{col['name']}: {col['type']} {'(NOT NULL)' if not col['nullable'] else ''}")
            
            if any(numeric in col_type for numeric in ['int', 'float', 'numeric', 'double', 'real']):
                numeric_cols.append(col['name'])
            elif any(text in col_type for text in ['char', 'text', 'varchar']):
                text_cols.append(col['name'])
            elif any(date in col_type for date in ['date', 'time', 'timestamp']):
                date_cols.append(col['name'])
        
        # Usage context document
        usage_content = [
            f"Table: {table} - Usage Guide",
            "Columns available:",
            *column_details,
            "",
            "Data Analysis Tips:"
        ]
        
        if numeric_cols:
            usage_content.append(f"- Numeric columns for calculations: {', '.join(numeric_cols)}")
        if text_cols:
            usage_content.append(f"- Text columns for filtering: {', '.join(text_cols)}")
        if date_cols:
            usage_content.append(f"- Date/time columns for temporal analysis: {', '.join(date_cols)}")
        
        usage_doc = Document(
            page_content="\n".join(usage_content),
            metadata={
                "type": "usage_guide",
                "source": "table_analysis",
                "table": table,
                "content_type": "usage_tips"
            }
        )
        documents.append(usage_doc)
        
        # Document 2: Query examples
        if table == 'measurements':
            example_content = [
                f"Table: {table} - Query Examples",
                "Common analytical queries for oceanography data:",
                "",
                "1. Profile queries:",
                "   SELECT pressure, temperature, salinity FROM measurements WHERE metadata_id = X ORDER BY pressure ASC",
                "   - Use for vertical profile analysis",
                "",
                "2. Statistical queries:",
                "   SELECT AVG(temperature), AVG(salinity) FROM measurements WHERE pressure BETWEEN 0 AND 100",
                "   - Use for aggregate analysis by depth ranges",
                "",
                "3. Spatial queries:",
                "   JOIN with floats_metadata for latitude/longitude information",
                "   - Essential for geographic distribution analysis"
            ]
            
            example_doc = Document(
                page_content="\n".join(example_content),
                metadata={
                    "type": "query_examples",
                    "source": "best_practices",
                    "table": table,
                    "content_type": "examples"
                }
            )
            documents.append(example_doc)
    
    def get_sample_data_documents(self) -> List[Document]:
        """Generate documents with sample data statistics"""
        documents = []
        
        try:
            # Get sample statistics from key tables
            tables_to_sample = ['measurements', 'floats_metadata']
            
            for table in tables_to_sample:
                # Get basic statistics
                count_query = f"SELECT COUNT(*) as total_count FROM {table}"
                with self.engine.connect() as conn:
                    result = conn.execute(text(count_query))
                    total_count = result.scalar()
                
                # Get column statistics for numeric columns
                stats_content = [f"Table: {table} - Data Statistics", f"Total records: {total_count:,}"]
                
                # Get numeric column ranges
                columns = self.inspector.get_columns(table)
                numeric_columns = [col['name'] for col in columns 
                                 if any(numeric in str(col['type']).lower() 
                                       for numeric in ['int', 'float', 'numeric', 'double', 'real'])]
                
                if numeric_columns:
                    stats_content.append("\nNumeric Column Ranges:")
                    for col in numeric_columns[:5]:  # Limit to first 5 numeric columns
                        try:
                            range_query = f"""
                                SELECT MIN({col}), MAX({col}), AVG({col}) 
                                FROM {table} 
                                WHERE {col} IS NOT NULL
                            """
                            with self.engine.connect() as conn:
                                result = conn.execute(text(range_query))
                                min_val, max_val, avg_val = result.fetchone()
                            
                            stats_content.append(f"  {col}: {min_val:.2f} to {max_val:.2f} (avg: {avg_val:.2f})")
                        except:
                            stats_content.append(f"  {col}: Could not compute statistics")
                
                # Create statistics document
                stats_doc = Document(
                    page_content="\n".join(stats_content),
                    metadata={
                        "type": "data_statistics",
                        "source": "sample_data",
                        "table": table,
                        "content_type": "statistics"
                    }
                )
                documents.append(stats_doc)
                
        except Exception as e:
            logger.error(f"Error generating sample data documents: {e}")
        
        return documents
    
    def get_oceanography_context_documents(self) -> List[Document]:
        """Generate documents with oceanography domain knowledge"""
        documents = []
        
        # Oceanography terminology and concepts
        oceanography_content = [
            "Oceanography Data Analysis - Domain Knowledge",
            "",
            "Key Parameters:",
            "- Temperature: Measured in degrees Celsius, critical for thermal analysis",
            "- Salinity: Measured in PSU (Practical Salinity Units), indicates salt content",
            "- Pressure: Measured in dbar (decibars), correlates with depth",
            "",
            "Common Analysis Types:",
            "1. Vertical Profiles: Analysis of parameters vs depth/pressure",
            "2. Spatial Distribution: Geographic patterns of ocean parameters",
            "3. Temporal Trends: Changes over time, seasonal variations",
            "4. Statistical Summaries: Averages, ranges, distributions",
            "",
            "ARGO Float Specifics:",
            "- Floats are identified by platform_number (7-digit codes)",
            "- Measurements are linked to metadata via metadata_id",
            "- Typical analysis involves JOINs between measurements and floats_metadata",
            "- Spatial analysis requires latitude/longitude from floats_metadata",
            "",
            "Best Practices:",
            "- Filter for valid data: use IS NOT NULL for critical parameters",
            "- Order profiles by pressure ASC (surface to depth)",
            "- Use appropriate aggregation for statistical queries",
            "- Include spatial coordinates for geographic context"
        ]
        
        oceanography_doc = Document(
            page_content="\n".join(oceanography_content),
            metadata={
                "type": "domain_knowledge",
                "source": "oceanography_expert",
                "content_type": "background"
            }
        )
        documents.append(oceanography_doc)
        
        # Query pattern examples
        query_patterns = [
            "Common Query Patterns for ARGO Data Analysis",
            "",
            "Profile Queries:",
            "SELECT pressure, temperature, salinity FROM measurements JOIN floats_metadata ON metadata_id WHERE platform_number = X ORDER BY pressure ASC",
            "- Used for: Vertical structure analysis, parameter relationships with depth",
            "",
            "Spatial Queries:",
            "SELECT latitude, longitude, AVG(temperature) FROM measurements JOIN floats_metadata GROUP BY latitude, longitude",
            "- Used for: Geographic distribution maps, regional comparisons",
            "",
            "Statistical Queries:",
            "SELECT pressure_level, COUNT(*), AVG(temperature) FROM measurements GROUP BY pressure_level",
            "- Used for: Data summaries, aggregation by depth ranges",
            "",
            "Temporal Queries:",
            "SELECT date, AVG(temperature) FROM measurements JOIN floats_metadata GROUP BY date ORDER BY date",
            "- Used for: Time series analysis, trend detection"
        ]
        
        patterns_doc = Document(
            page_content="\n".join(query_patterns),
            metadata={
                "type": "query_patterns",
                "source": "analysis_patterns",
                "content_type": "examples"
            }
        )
        documents.append(patterns_doc)
        
        return documents
    
    def create_vector_store(self) -> bool:
        """Create and populate the vector store"""
        try:
            logger.info("Starting vector store creation...")
            
            # Collect all documents
            all_documents = []
            
            # Schema documents
            schema_docs = self.get_database_schema_documents()
            logger.info(f"Generated {len(schema_docs)} schema documents")
            all_documents.extend(schema_docs)
            
            # Sample data documents
            sample_docs = self.get_sample_data_documents()
            logger.info(f"Generated {len(sample_docs)} sample data documents")
            all_documents.extend(sample_docs)
            
            # Oceanography context documents
            context_docs = self.get_oceanography_context_documents()
            logger.info(f"Generated {len(context_docs)} context documents")
            all_documents.extend(context_docs)
            
            logger.info(f"Total documents to index: {len(all_documents)}")
            
            # Create vector store
            vector_store = Chroma.from_documents(
                documents=all_documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Persist the vector store
            vector_store.persist()
            
            # Verify creation
            doc_count = vector_store._collection.count()
            logger.info(f"✅ Vector store created successfully with {doc_count} documents")
            logger.info(f"📁 Persisted to: {self.persist_directory}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create vector store: {e}")
            return False
    
    def verify_vector_store(self) -> bool:
        """Verify the vector store was created correctly"""
        try:
            vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            
            doc_count = vector_store._collection.count()
            logger.info(f"Vector store verification: {doc_count} documents found")
            
            # Test a simple query
            test_results = vector_store.similarity_search("temperature profile", k=2)
            if test_results:
                logger.info("✅ Vector store is working correctly")
                return True
            else:
                logger.warning("⚠️  Vector store created but no results returned")
                return False
                
        except Exception as e:
            logger.error(f"❌ Vector store verification failed: {e}")
            return False

def main():
    """Main function to set up the advanced RAG system"""
    logger.info("🚀 Setting up Advanced RAG System")
    logger.info("=" * 50)
    
    # Check environment variables
    required_vars = ['DATABASE_URL', 'GEMINI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        logger.info("Please check your .env file")
        return
    
    # Create setup instance
    rag_setup = AdvancedRAGSetup()
    
    # Create vector store
    success = rag_setup.create_vector_store()
    
    if success:
        # Verify the vector store
        verification = rag_setup.verify_vector_store()
        
        if verification:
            logger.info("🎉 Advanced RAG setup completed successfully!")
            logger.info("You can now run advanced_rag_llm_system.py")
        else:
            logger.warning("⚠️  Setup completed but verification failed")
    else:
        logger.error("❌ Advanced RAG setup failed")

if __name__ == "__main__":
    main()