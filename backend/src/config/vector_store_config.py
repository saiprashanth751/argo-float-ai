# backend/src/config/vector_store_config.py
"""SINGLE SOURCE OF TRUTH for Vector Store Configuration"""

import os
from pathlib import Path
from typing import Optional

# ABSOLUTE PATH - SINGLE SOURCE OF TRUTH
VECTOR_STORE_BASE_DIR = Path(__file__).parent.parent / "storage"
VECTOR_STORE_PATH = VECTOR_STORE_BASE_DIR / "chroma_db_oceanographic"

def ensure_vector_store_path() -> Path:
    """Ensure the vector store directory exists"""
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    return VECTOR_STORE_PATH

def get_vector_store_path() -> Path:
    """Get the single source vector store path"""
    return VECTOR_STORE_PATH

# Validation
def validate_vector_store_location():
    """Validate that all components use the same path"""
    actual_path = get_vector_store_path()
    expected_path = VECTOR_STORE_PATH
    
    if actual_path != expected_path:
        raise ValueError(f"Vector store path mismatch! Expected: {expected_path}, Got: {actual_path}")
    
    return actual_path