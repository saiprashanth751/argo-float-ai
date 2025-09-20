# backend\src\utils\database_manager.py
"""
PRODUCTION DATABASE CONNECTION MANAGER
Single source of truth for database connections across the entire system
"""

import os
from sqlalchemy import create_engine, text  # Added text import
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import logging
from dotenv import load_dotenv
from typing import Generator

logger = logging.getLogger(__name__)

class DatabaseConnectionManager:
    _instance = None
    _engine = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnectionManager, cls).__new__(cls)
            cls._initialize_engine()
        return cls._instance
    
    @classmethod
    def _initialize_engine(cls):
        """Initialize the database engine with production settings"""
        try:
            load_dotenv()
            
            db_url = os.getenv('DATABASE_URL')
            if not db_url:
                raise ValueError("DATABASE_URL environment variable not set")
            
            cls._engine = create_engine(
                db_url,
                pool_size=20,
                max_overflow=30,
                pool_timeout=30,
                pool_recycle=1800,
                pool_pre_ping=True,
                connect_args={
                    "connect_timeout": 10,
                    "options": "-c timezone=UTC -c statement_timeout=30000"
                }
            )
            
            # Test connection immediately - FIXED: Use text() for SQL expression
            with cls._engine.connect() as conn:
                conn.execute(text("SELECT 1"))  # Wrap with text()
            
            logger.info("Production database engine initialized successfully")
            
        except Exception as e:
            logger.critical(f"Failed to initialize database engine: {e}")
            raise
    
    @classmethod
    def get_engine(cls) -> Engine:
        """Get the shared database engine instance"""
        if cls._engine is None:
            cls._initialize_engine()
        return cls._engine
    
    @classmethod
    @contextmanager
    def get_session(cls) -> Generator:
        """Get a database session with automatic cleanup"""
        session = sessionmaker(bind=cls.get_engine())()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

# Global access point
def get_db_engine() -> Engine:
    return DatabaseConnectionManager.get_engine()

@contextmanager
def get_db_session():
    with DatabaseConnectionManager.get_session() as session:
        yield session