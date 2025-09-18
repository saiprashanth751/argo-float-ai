"""
Create mcp_tools_core.py with essential base classes:
"""

# mcp_tools_core.py - Essential base classes for MCP tools
import logging
import sqlite3
from typing import Dict, List, Optional, Any
from pathlib import Path
from sqlalchemy import text

logger = logging.getLogger(__name__)

class MCPToolsManager:
    """Core MCP Tools Manager for agent system"""
    
    def __init__(self, db_engine):
        self.db_engine = db_engine
        self.initialized = True
        
        # Initialize storage paths
        self.storage_path = Path("storage")
        self.knowledge_db_path = self.storage_path / "knowledge" / "oceanographic.db"
        
        # Create storage directories
        self.storage_path.mkdir(exist_ok=True)
        self.knowledge_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._setup_core_components()
        
        logger.info("MCP Tools Manager initialized successfully")
    
    def _setup_core_components(self):
        """Setup core components"""
        try:
            # Test database connection
            with self.db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.db_available = True
        except Exception as e:
            logger.warning(f"Database connection issue: {e}")
            self.db_available = False
    
    def get_database_connection(self):
        """Get database connection"""
        return self.db_engine
    
    def is_ready(self) -> bool:
        """Check if tools manager is ready"""
        return self.initialized and self.db_available

class DatabaseExplorerTool:
    """Database exploration tool for agents"""
    
    def __init__(self, db_engine, tools_manager):
        self.db_engine = db_engine
        self.tools_manager = tools_manager
        
        # Cache for schema information
        self.schema_cache = {}
        
        logger.info("Database Explorer Tool initialized")
    
    def explore_database_schema(self, table_name: str = None, include_sample_data: bool = False) -> str:
        """Explore database schema with caching"""
        
        cache_key = f"schema_{table_name or 'all'}_{include_sample_data}"
        
        if cache_key in self.schema_cache:
            return self.schema_cache[cache_key]
        
        try:
            if table_name:
                result = self._get_table_schema(table_name, include_sample_data)
            else:
                result = self._get_all_tables()
            
            self.schema_cache[cache_key] = result
            return result
            
        except Exception as e:
            error_msg = f"Schema exploration failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _get_table_schema(self, table_name: str, include_sample_data: bool) -> str:
        """Get schema for specific table"""
        
        schema_query = """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = :table_name
        ORDER BY ordinal_position
        """
        
        with self.db_engine.connect() as conn:
            result = conn.execute(text(schema_query), {'table_name': table_name})
            columns = result.fetchall()
        
        if not columns:
            return f"Table '{table_name}' not found or no columns available"
        
        schema_info = f"Schema for table '{table_name}':\n"
        schema_info += "-" * 40 + "\n"
        
        for col in columns:
            nullable = "NULL" if col[2] == "YES" else "NOT NULL"
            default = f" DEFAULT {col[3]}" if col[3] else ""
            schema_info += f"  {col[0]} ({col[1]}) {nullable}{default}\n"
        
        # Add sample data if requested
        if include_sample_data:
            sample_query = f"SELECT * FROM {table_name} LIMIT 5"
            try:
                with self.db_engine.connect() as conn:
                    sample_result = conn.execute(text(sample_query))
                    sample_rows = sample_result.fetchall()
                
                if sample_rows:
                    schema_info += "\nSample data:\n"
                    schema_info += "-" * 20 + "\n"
                    for row in sample_rows[:3]:  # Show only 3 rows
                        schema_info += f"  {dict(row)}\n"
            except Exception as e:
                schema_info += f"\nSample data unavailable: {str(e)}\n"
        
        return schema_info
    
    def _get_all_tables(self) -> str:
        """Get all available tables"""
        
        tables_query = """
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
        """
        
        with self.db_engine.connect() as conn:
            result = conn.execute(text(tables_query))
            tables = result.fetchall()
        
        if not tables:
            return "No tables found in the database"
        
        tables_info = "Available tables in database:\n"
        tables_info += "=" * 40 + "\n"
        
        for table in tables:
            tables_info += f"  {table[0]} ({table[1]})\n"
        
        tables_info += f"\nTotal tables: {len(tables)}\n"
        
        # Add recommendations
        tables_info += "\nRecommendations:\n"
        tables_info += "- Use 'argo_profiles' for surface parameters and metadata\n"
        tables_info += "- Use 'argo_measurements' for detailed depth profiles\n"
        tables_info += "- JOIN tables on profile_id for comprehensive analysis\n"
        
        return tables_info

class SQLValidatorTool:
    """SQL validation and optimization tool"""
    
    def __init__(self, db_engine, tools_manager):
        self.db_engine = db_engine
        self.tools_manager = tools_manager
        
        # Validation patterns
        self.validation_patterns = {
            'dangerous_operations': ['DELETE', 'DROP', 'TRUNCATE', 'UPDATE', 'ALTER'],
            'required_keywords': ['SELECT', 'FROM'],
            'performance_hints': ['LIMIT', 'WHERE', 'INDEX']
        }
        
        logger.info("SQL Validator Tool initialized")
    
    def validate_sql_query(self, sql_query: str, explain_plan: bool = False) -> str:
        """Comprehensive SQL validation"""
        
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'suggestions': [],
            'performance_notes': []
        }
        
        try:
            # Basic syntax validation
            sql_cleaned = sql_query.strip()
            
            if not sql_cleaned:
                validation_result['is_valid'] = False
                return "Error: Empty SQL query provided"
            
            # Check for dangerous operations
            sql_upper = sql_cleaned.upper()
            for dangerous_op in self.validation_patterns['dangerous_operations']:
                if dangerous_op in sql_upper:
                    validation_result['is_valid'] = False
                    validation_result['warnings'].append(f"Dangerous operation detected: {dangerous_op}")
            
            # Check for required keywords
            for required in self.validation_patterns['required_keywords']:
                if required not in sql_upper:
                    validation_result['is_valid'] = False
                    validation_result['warnings'].append(f"Missing required keyword: {required}")
            
            if not validation_result['is_valid']:
                return self._format_validation_result(validation_result)
            
            # Syntax validation using database
            try:
                validation_query = f"EXPLAIN (FORMAT TEXT) {sql_cleaned}"
                with self.db_engine.connect() as conn:
                    conn.execute(text(validation_query))
                
                validation_result['suggestions'].append("SQL syntax is valid")
                
            except Exception as e:
                validation_result['is_valid'] = False
                validation_result['warnings'].append(f"Syntax error: {str(e)}")
            
            # Performance suggestions
            if 'LIMIT' not in sql_upper:
                validation_result['performance_notes'].append("Consider adding LIMIT clause for large datasets")
            
            if 'WHERE' not in sql_upper:
                validation_result['performance_notes'].append("Consider adding WHERE clause for better performance")
            
            return self._format_validation_result(validation_result)
            
        except Exception as e:
            return f"Validation error: {str(e)}"
    
    def _format_validation_result(self, result: Dict[str, Any]) -> str:
        """Format validation result"""
        
        output = f"SQL Validation Result: {'VALID' if result['is_valid'] else 'INVALID'}\n"
        output += "=" * 50 + "\n"
        
        if result['warnings']:
            output += "WARNINGS:\n"
            for warning in result['warnings']:
                output += f"  - {warning}\n"
            output += "\n"
        
        if result['suggestions']:
            output += "SUGGESTIONS:\n"
            for suggestion in result['suggestions']:
                output += f"  - {suggestion}\n"
            output += "\n"
        
        if result['performance_notes']:
            output += "PERFORMANCE NOTES:\n"
            for note in result['performance_notes']:
                output += f"  - {note}\n"
        
        return output