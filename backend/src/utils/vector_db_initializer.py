# vector_db_initializer_enhanced_local.py
# Updated version for local Docker Desktop Model Runner setup
# Uses only local HuggingFace embeddings - no external API calls

import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import hashlib

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Primary embedding option - HuggingFace for local capability
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OceanographicVectorDBInitializer:
    """Local initializer for Docker Desktop Model Runner with HuggingFace embeddings"""
    
    def __init__(self, 
                 persist_directory: str = "../storage/chroma_db_oceanographic",
                 embedding_provider: str = "huggingface"):
        self.persist_directory = persist_directory
        self.embedding_provider = embedding_provider
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.embeddings = self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize local HuggingFace embeddings only"""
        
        if not HUGGINGFACE_AVAILABLE:
            logger.error("HuggingFace embeddings not available!")
            logger.info("Install with: pip install sentence-transformers")
            return None
        
        try:
            logger.info("Initializing local HuggingFace embeddings (free, no API calls)")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            # Test embedding to ensure it works
            test_embed = embeddings.embed_query("test oceanographic data")
            logger.info(f"✅ HuggingFace embeddings initialized successfully (dimension: {len(test_embed)})")
            return embeddings
        except Exception as e:
            logger.error(f"HuggingFace embeddings failed: {e}")
            return None
    
    def create_comprehensive_knowledge_base(self, max_documents: Optional[int] = None) -> bool:
        """Create knowledge base with optional document limit"""
        
        if not self.embeddings:
            logger.error("Cannot create knowledge base without embeddings model")
            return False
        
        logger.info("Creating comprehensive oceanographic knowledge base...")
        
        try:
            # Create documents in priority order
            documents = []
            
            # Critical documents first
            critical_docs = self._create_schema_documents()
            documents.extend(critical_docs)
            logger.info(f"Added {len(critical_docs)} schema documents")
            
            if max_documents is None or len(documents) < max_documents:
                query_docs = self._create_query_pattern_documents()
                documents.extend(query_docs)
                logger.info(f"Added {len(query_docs)} query pattern documents")
            
            if max_documents is None or len(documents) < max_documents:
                ocean_docs = self._create_oceanographic_knowledge_documents()
                if max_documents:
                    remaining = max_documents - len(documents)
                    ocean_docs = ocean_docs[:remaining]
                documents.extend(ocean_docs)
                logger.info(f"Added {len(ocean_docs)} oceanographic knowledge documents")
            
            if max_documents is None or len(documents) < max_documents:
                calc_docs = self._create_calculation_documents()
                if max_documents:
                    remaining = max_documents - len(documents)
                    calc_docs = calc_docs[:remaining]
                documents.extend(calc_docs)
                logger.info(f"Added {len(calc_docs)} calculation documents")
            
            logger.info(f"Total documents to process: {len(documents)}")
            
            # Check if database already exists with same content
            if self._check_existing_database(documents):
                logger.info("✅ Database already exists with same content")
                return True
            
            # Create vector store with progress tracking
            logger.info("Starting vector store creation...")
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            vector_store.persist()
            logger.info(f"✅ Vector database created successfully at {self.persist_directory}")
            
            # Save metadata about the database
            self._save_database_metadata(documents)
            
            # Verify creation
            doc_count = vector_store._collection.count()
            logger.info(f"Vector database contains {doc_count} document chunks")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create knowledge base: {e}")
            return False
    
    def _check_existing_database(self, documents: List[Document]) -> bool:
        """Check if database already exists with same content"""
        try:
            metadata_file = Path(self.persist_directory) / "db_metadata.json"
            if not metadata_file.exists():
                return False
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Create hash of current documents
            content_hash = self._create_content_hash(documents)
            
            return metadata.get('content_hash') == content_hash
            
        except Exception:
            return False
    
    def _create_content_hash(self, documents: List[Document]) -> str:
        """Create hash of document content for change detection"""
        content = "".join([doc.page_content for doc in documents])
        return hashlib.md5(content.encode()).hexdigest()
    
    def _save_database_metadata(self, documents: List[Document]):
        """Save metadata about the database"""
        try:
            metadata = {
                'created_at': datetime.now().isoformat(),
                'document_count': len(documents),
                'content_hash': self._create_content_hash(documents),
                'embedding_provider': 'huggingface_local',
                'model_setup': 'docker_desktop_llama'
            }
            
            metadata_file = Path(self.persist_directory) / "db_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not save metadata: {e}")
    
    def create_minimal_knowledge_base(self) -> bool:
        """Create a minimal knowledge base for quick setup"""
        logger.info("Creating minimal knowledge base (local setup friendly)...")
        return self.create_comprehensive_knowledge_base(max_documents=5)
    
    def _create_schema_documents(self) -> List[Document]:
        """Create documents about database schema"""
        
        return [
            Document(
                page_content="""
                ARGO Production Database Schema (PostgreSQL)
                
                Primary Tables:
                1. argo_profiles - Profile metadata and surface measurements
                - id: Primary key (SERIAL)
                - platform_number: Float identifier (VARCHAR(20), indexed)
                - cycle_number: Profile cycle (INTEGER) 
                - profile_date: Measurement date/time (TIMESTAMP, indexed)
                - latitude, longitude: Position (DOUBLE PRECISION, indexed together)
                - surface_temp: Surface temperature in °C (DOUBLE PRECISION)
                - surface_salinity: Surface salinity in PSU (DOUBLE PRECISION)
                - max_pressure: Maximum depth reached in dbar (DOUBLE PRECISION)
                - n_levels: Number of measurement levels (INTEGER)
                - mixed_layer_depth: Calculated MLD in meters (DOUBLE PRECISION)
                - source_file: Original NetCDF file (VARCHAR(255))
                
                2. argo_measurements - Vertical profile measurements
                - id: Primary key (BIGSERIAL)
                - profile_id: Foreign key to argo_profiles.id (INTEGER, indexed)
                - pressure: Measurement pressure in dbar (DOUBLE PRECISION, indexed)
                - depth: Approximate depth in meters (DOUBLE PRECISION)
                - temperature: Temperature in °C (DOUBLE PRECISION)
                - salinity: Salinity in PSU (DOUBLE PRECISION)
                - Composite index on (temperature, salinity) for T-S analysis
                """,
                metadata={"type": "schema", "source": "production_database", "priority": "critical"}
            )
        ]
    
    def _create_query_pattern_documents(self) -> List[Document]:
        """Create documents about query patterns"""
        
        return [
            Document(
                page_content="""
                Essential Query Patterns for ARGO Database
                
                1. Profile Analysis (Temperature/Salinity vs Depth):
                SELECT p.platform_number, p.profile_date, p.latitude, p.longitude,
                    m.pressure, m.temperature, m.salinity, m.depth
                FROM argo_profiles p
                JOIN argo_measurements m ON p.id = m.profile_id
                WHERE p.platform_number = '{platform_id}'
                ORDER BY m.pressure ASC;
                
                2. Surface Analysis (Geographic Distribution):
                SELECT p.platform_number, p.profile_date, p.latitude, p.longitude,
                    p.surface_temp, p.surface_salinity, p.mixed_layer_depth
                FROM argo_profiles p
                WHERE p.surface_temp IS NOT NULL
                AND p.latitude BETWEEN {lat_min} AND {lat_max}
                ORDER BY p.profile_date DESC;
                """,
                metadata={"type": "query_patterns", "source": "production_usage", "priority": "high"}
            ),
            Document(
                page_content="""
                Advanced Query Patterns for ARGO Database
                
                3. Depth Profile Analysis:
                SELECT p.platform_number, p.profile_date,
                    m.pressure, m.temperature, m.salinity,
                    CASE 
                        WHEN m.pressure < 200 THEN 'Surface'
                        WHEN m.pressure < 1000 THEN 'Intermediate' 
                        ELSE 'Deep'
                    END as depth_layer
                FROM argo_profiles p
                JOIN argo_measurements m ON p.id = m.profile_id
                WHERE p.latitude BETWEEN {lat_min} AND {lat_max}
                AND p.longitude BETWEEN {lon_min} AND {lon_max}
                ORDER BY p.profile_date DESC, m.pressure ASC;
                
                4. Time Series Analysis:
                SELECT DATE_TRUNC('month', p.profile_date) as month,
                    AVG(p.surface_temp) as avg_surface_temp,
                    AVG(p.surface_salinity) as avg_surface_salinity,
                    COUNT(*) as profile_count
                FROM argo_profiles p
                WHERE p.profile_date >= '{start_date}'
                AND p.latitude BETWEEN {lat_min} AND {lat_max}
                GROUP BY DATE_TRUNC('month', p.profile_date)
                ORDER BY month;
                """,
                metadata={"type": "query_patterns", "source": "advanced_analysis", "priority": "high"}
            )
        ]
    
    def _create_oceanographic_knowledge_documents(self) -> List[Document]:
        """Create oceanographic domain knowledge documents"""
        
        return [
            Document(
                page_content="""
                Oceanographic Parameters and Physical Properties
                
                Temperature:
                - Range: Surface (15-30°C tropics, 0-15°C polar), Deep (1-4°C globally)
                - Physical significance: Controls density, mixing, biological activity
                - Measurement accuracy: ±0.002°C for ARGO CTDs
                - Vertical structure: Thermocline separates warm surface from cold deep water
                
                Salinity:
                - Range: Surface (32-37 PSU typical, extremes 30-42 PSU)
                - Physical significance: Controls density, water mass identification
                - Measurement accuracy: ±0.002 PSU for ARGO CTDs
                - Global patterns: Higher in subtropical gyres, lower at high latitudes
                
                Pressure/Depth:
                - Conversion: ~10 dbar = 10 m depth (varies with latitude)
                - ARGO range: Surface to 2000+ dbar
                - Standard levels: 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1750, 2000 dbar
                """,
                metadata={"type": "oceanography", "source": "physical_properties", "priority": "critical"}
            ),
            Document(
                page_content="""
                Ocean Water Masses and Circulation
                
                Major Water Masses (identifiable by T-S properties):
                - Surface Water: Warm (>15°C), variable salinity (32-37 PSU)
                - Intermediate Water: Cool (4-15°C), salinity minima/maxima
                - Deep Water: Cold (<4°C), high salinity (34.6-34.7 PSU)
                - Bottom Water: Very cold (<2°C), highest salinity
                
                Circulation Patterns:
                - Surface: Wind-driven, seasonal variation
                - Deep: Thermohaline circulation, slow (~mm/s)
                - Vertical: Mixing processes, seasonal thermocline
                
                Regional Characteristics:
                - North Atlantic: Deep water formation, high salinity
                - North Pacific: Fresher surface, distinct intermediate water
                - Southern Ocean: Circumpolar current, water mass formation
                - Tropical: Warm surface layer, strong thermocline
                """,
                metadata={"type": "oceanography", "source": "water_masses", "priority": "medium"}
            ),
            Document(
                page_content="""
                Seasonal and Geographic Variations
                
                Seasonal Cycles:
                - Temperature: Surface amplitude 2-10°C depending on latitude
                - Mixed layer depth: Shallow in summer (10-50m), deep in winter (50-200m)
                - Thermocline: Strong in summer, weak in winter
                
                Geographic Patterns:
                - Equatorial: Minimal seasonal variation, upwelling zones
                - Subtropical: Strong seasonal cycles, high surface salinity
                - Polar: Large seasonal temperature range, ice effects
                - Coastal: Influenced by land, upwelling, river input
                
                Frontal Systems:
                - Gulf Stream: Sharp temperature/salinity gradients
                - Antarctic Circumpolar Current: Multiple fronts
                - Equatorial: Complex current systems
                """,
                metadata={"type": "oceanography", "source": "variability", "priority": "medium"}
            )
        ]
    
    def _create_calculation_documents(self) -> List[Document]:
        """Create documents about oceanographic calculations"""
        
        return [
            Document(
                page_content="""
                Essential Oceanographic Calculations for ARGO Data
                
                1. Mixed Layer Depth (MLD) Calculation Methods:
                
                a) Temperature Criterion:
                MLD = depth where T = T_ref - 0.2°C
                Where T_ref = temperature at 10m depth
                
                b) Density Criterion:
                MLD = depth where σθ = σθ_ref + 0.03 kg/m³
                Where σθ_ref = potential density at 10m depth
                
                Implementation Steps:
                1. Find reference value at 10m (or shallowest measurement)
                2. Calculate threshold value
                3. Interpolate to find exact depth where threshold is crossed
                4. Handle cases where threshold is never reached
                
                SQL Implementation:
                ```sql
                WITH ref_values AS (
                  SELECT profile_id, 
                         first_value(temperature) OVER (ORDER BY pressure) as t_ref,
                         first_value(pressure) OVER (ORDER BY pressure) as p_ref
                  FROM argo_measurements 
                  WHERE pressure >= 10
                ),
                mld_calc AS (
                  SELECT m.profile_id, m.pressure,
                         CASE WHEN m.temperature <= (r.t_ref - 0.2) 
                              THEN m.pressure 
                              ELSE NULL END as mld_pressure
                  FROM argo_measurements m
                  JOIN ref_values r ON m.profile_id = r.profile_id
                )
                SELECT profile_id, MIN(mld_pressure) as mixed_layer_depth
                FROM mld_calc 
                GROUP BY profile_id;
                ```
                """,
                metadata={"type": "calculations", "source": "mixed_layer", "priority": "critical"}
            ),
            Document(
                page_content="""
                Density and Water Mass Calculations
                
                2. Potential Density (σθ) Calculation:
                Purpose: Density referenced to surface pressure (removes pressure effects)
                Formula: σθ = ρ(S,θ,0) - 1000 kg/m³
                Where θ = potential temperature, S = salinity
                
                Range: Typically 20-28 kg/m³ (surface to deep)
                
                3. Potential Temperature (θ):
                Purpose: Temperature a water parcel would have if moved adiabatically to surface
                Accounts for: Adiabatic heating/cooling with pressure changes
                
                4. Brunt-Väisälä Frequency (N²):
                Purpose: Measure of water column stability
                Formula: N² = -(g/ρ) × (dρ/dz)
                Interpretation: High N² = stable stratification
                
                5. Temperature-Salinity (T-S) Analysis:
                Purpose: Water mass identification and mixing analysis
                Method: Plot T vs S, identify characteristic curves
                Applications: Track water mass origins, mixing processes
                
                SQL for Potential Density Approximation:
                ```sql
                SELECT profile_id, pressure, temperature, salinity,
                       -- Simplified potential density calculation
                       999.842594 + 6.793952e-2*temperature - 9.095290e-3*temperature^2 
                       + 1.001685e-4*temperature^3 - 1.120083e-6*temperature^4 
                       + 6.536332e-9*temperature^5 + salinity*(8.24493e-1 - 4.0899e-3*temperature 
                       + 7.6438e-5*temperature^2 - 8.2467e-7*temperature^3 + 5.3875e-9*temperature^4) - 1000 as sigma_theta
                FROM argo_measurements;
                ```
                """,
                metadata={"type": "calculations", "source": "density_calcs", "priority": "critical"}
            ),
            Document(
                page_content="""
                Statistical Analysis and Quality Control
                
                6. Data Quality Assessment:
                
                Temperature QC Checks:
                - Range test: -2.5°C < T < 40°C
                - Gradient test: |dT/dz| < 10°C per 100m
                - Spike test: |T[n] - 0.5*(T[n-1] + T[n+1])| < 2°C
                
                Salinity QC Checks:
                - Range test: 2 < S < 41 PSU
                - Gradient test: |dS/dz| < 5 PSU per 100m
                - Density inversion check: σθ should generally increase with depth
                
                7. Interpolation Methods:
                
                Standard Depth Interpolation:
                Purpose: Compare profiles at standard depths
                Method: Linear interpolation between measured points
                Standard depths: 0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1750, 2000m
                
                SQL for Interpolation:
                ```sql
                WITH standard_depths AS (
                  SELECT unnest(ARRAY[0,10,20,30,50,75,100,125,150,200,250,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1750,2000]) as std_depth
                ),
                interpolated AS (
                  SELECT p.id as profile_id, sd.std_depth,
                         -- Linear interpolation of temperature
                         (m1.temperature + (m2.temperature - m1.temperature) * 
                          (sd.std_depth - m1.depth) / (m2.depth - m1.depth)) as interp_temp
                  FROM argo_profiles p
                  CROSS JOIN standard_depths sd
                  JOIN argo_measurements m1 ON p.id = m1.profile_id AND m1.depth <= sd.std_depth
                  JOIN argo_measurements m2 ON p.id = m2.profile_id AND m2.depth > sd.std_depth
                  WHERE m1.depth = (SELECT MAX(depth) FROM argo_measurements WHERE profile_id = p.id AND depth <= sd.std_depth)
                    AND m2.depth = (SELECT MIN(depth) FROM argo_measurements WHERE profile_id = p.id AND depth > sd.std_depth)
                )
                SELECT * FROM interpolated;
                ```
                """,
                metadata={"type": "calculations", "source": "qc_stats", "priority": "high"}
            )
        ]
    
    def test_vector_database(self) -> bool:
        """Test the created vector database"""
        
        if not self.embeddings:
            logger.error("Cannot test without embeddings model")
            return False
        
        try:
            # Load existing vector database
            vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            
            # Test queries
            test_queries = [
                "How to query ARGO temperature profiles?",
                "What is mixed layer depth calculation?",
                "Database schema for ARGO profiles table"
            ]
            
            logger.info("Testing vector database with multiple queries...")
            for query in test_queries:
                results = vector_store.similarity_search_with_score(query, k=2)
                logger.info(f"Query: {query}")
                
                for i, (doc, score) in enumerate(results, 1):
                    relevance = (1 - score) * 100
                    doc_type = doc.metadata.get('type', 'unknown')
                    logger.info(f"  {i}. {doc_type} (relevance: {relevance:.1f}%)")
            
            doc_count = vector_store._collection.count()
            logger.info(f"✅ Vector database test completed successfully!")
            logger.info(f"Total documents in database: {doc_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Vector database test failed: {e}")
            return False


def initialize_production_vector_db(mode: str = "minimal"):
    """Initialize the production vector database for Docker Desktop Model Runner
    
    Args:
        mode: "minimal" for quick setup, "full" for complete database
    """
    
    logger.info("🚀 Initializing Production Vector Database (Docker Desktop Llama Ready)")
    logger.info("=" * 50)
    
    # Check available embedding providers
    if not HUGGINGFACE_AVAILABLE:
        logger.error("❌ HuggingFace embeddings not available!")
        logger.info("Install with: pip install sentence-transformers")
        return False
    
    logger.info("Available embedding provider: HuggingFace (free, local) - RECOMMENDED")
    logger.info("Using local embeddings - no API calls required")
    
    # Initialize with HuggingFace only
    initializer = OceanographicVectorDBInitializer(embedding_provider="huggingface")
    
    if not initializer.embeddings:
        logger.error("❌ HuggingFace embeddings not available. Please install sentence-transformers.")
        return False
    
    # Create knowledge base
    if mode == "minimal":
        success = initializer.create_minimal_knowledge_base()
    else:
        success = initializer.create_comprehensive_knowledge_base()
    
    if success:
        logger.info("✅ Knowledge base created successfully")
        
        # Test the database
        test_success = initializer.test_vector_database()
        
        if test_success:
            logger.info("✅ Vector database initialization completed successfully")
            logger.info("Ready for use with Docker Desktop Model Runner (Llama 3.2)")
            return True
        else:
            logger.error("❌ Vector database testing failed")
            return False
    else:
        logger.error("❌ Failed to create knowledge base")
        return False


if __name__ == "__main__":
    import sys
    
    # Allow command line argument for mode
    mode = "minimal" if len(sys.argv) <= 1 else sys.argv[1]
    
    if mode not in ["minimal", "full"]:
        print("Usage: python vector_db_initializer_enhanced_local.py [minimal|full]")
        print("  minimal: Create small database (quick setup)")
        print("  full: Create complete database")
        sys.exit(1)
    
    print(f"Mode: {mode}")
    success = initialize_production_vector_db(mode)
    sys.exit(0 if success else 1)