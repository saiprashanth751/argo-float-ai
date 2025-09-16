# create_supabase_tables.py
import os
import psycopg2
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def create_tables():
    """Create tables in Supabase database"""
    
    # Get connection from environment
    DATABASE_URL = os.getenv('DATABASE_URL')
    
    if not DATABASE_URL:
        logger.error("DATABASE_URL not found in environment variables")
        return False
    
    logger.info("Connecting to Supabase database...")
    
    try:
        # Connect to Supabase
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        logger.info("Connected successfully!")
        
        # Create tables SQL (simplified from your init.sql)
        create_tables_sql = """
        -- Enable PostGIS extension for spatial data
        CREATE EXTENSION IF NOT EXISTS postgis;
        
        -- Drop existing tables if they exist
        DROP TABLE IF EXISTS enhanced_measurements CASCADE;
        DROP TABLE IF EXISTS enhanced_floats_metadata CASCADE;
        DROP TABLE IF EXISTS float_summary_stats CASCADE;
        
        -- Create floats metadata table
        CREATE TABLE enhanced_floats_metadata (
            id SERIAL PRIMARY KEY,
            platform_number VARCHAR(50) NOT NULL,
            cycle_number INTEGER,
            date TIMESTAMP,
            latitude DOUBLE PRECISION,
            longitude DOUBLE PRECISION,
            location GEOGRAPHY(POINT, 4326),
            
            -- Metadata fields
            project_name VARCHAR(200),
            pi_name VARCHAR(200), 
            institution VARCHAR(200),
            wmo_inst_type VARCHAR(50),
            data_mode VARCHAR(50),
            source_file VARCHAR(500),
            
            -- Oceanographic parameters
            max_pressure DOUBLE PRECISION,
            min_pressure DOUBLE PRECISION,
            mixed_layer_depth DOUBLE PRECISION,
            thermocline_depth DOUBLE PRECISION,
            surface_temperature DOUBLE PRECISION,
            surface_salinity DOUBLE PRECISION,
            bottom_temperature DOUBLE PRECISION,
            bottom_salinity DOUBLE PRECISION,
            
            -- Data quality indicators
            n_measurements INTEGER DEFAULT 0,
            has_bgc_data BOOLEAN DEFAULT FALSE,
            
            -- Processing metadata
            processed_at TIMESTAMP DEFAULT NOW(),
            processing_version VARCHAR(20) DEFAULT '3.0-supabase'
        );
        
        -- Create measurements table
        CREATE TABLE enhanced_measurements (
            id SERIAL PRIMARY KEY,
            metadata_id INTEGER REFERENCES enhanced_floats_metadata(id) ON DELETE CASCADE,
            pressure DOUBLE PRECISION NOT NULL,
            depth DOUBLE PRECISION,
            
            -- Core parameters
            temperature DOUBLE PRECISION,
            salinity DOUBLE PRECISION,
            
            -- Derived parameters
            potential_temperature DOUBLE PRECISION,
            conservative_temperature DOUBLE PRECISION,
            absolute_salinity DOUBLE PRECISION,
            density DOUBLE PRECISION,
            
            -- BGC parameters
            oxygen DOUBLE PRECISION,
            chlorophyll DOUBLE PRECISION,
            backscatter_700 DOUBLE PRECISION,
            nitrate DOUBLE PRECISION,
            ph_in_situ DOUBLE PRECISION,
            
            -- Quality control flags
            pressure_qc INTEGER,
            temperature_qc INTEGER,
            salinity_qc INTEGER
        );
        
        -- Create indexes for performance
        CREATE INDEX idx_metadata_platform ON enhanced_floats_metadata (platform_number);
        CREATE INDEX idx_metadata_date ON enhanced_floats_metadata (date);
        CREATE INDEX idx_metadata_location ON enhanced_floats_metadata USING GIST (location);
        CREATE INDEX idx_measurements_metadata ON enhanced_measurements (metadata_id);
        CREATE INDEX idx_measurements_pressure ON enhanced_measurements (pressure);
        
        -- Function to automatically set location from lat/lon
        CREATE OR REPLACE FUNCTION update_location_geography() RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.latitude IS NOT NULL AND NEW.longitude IS NOT NULL THEN
                NEW.location = ST_MakePoint(NEW.longitude, NEW.latitude)::geography;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        
        -- Trigger to automatically update location
        CREATE TRIGGER trigger_update_location_geography
            BEFORE INSERT OR UPDATE ON enhanced_floats_metadata
            FOR EACH ROW
            EXECUTE FUNCTION update_location_geography();
        """
        
        logger.info("Creating tables...")
        cursor.execute(create_tables_sql)
        conn.commit()
        
        logger.info("✅ Tables created successfully!")
        
        # Verify tables were created
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE 'enhanced_%'
        """)
        
        tables = cursor.fetchall()
        logger.info(f"Created tables: {[table[0] for table in tables]}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        return False

def test_connection():
    """Test connection to Supabase"""
    
    DATABASE_URL = os.getenv('DATABASE_URL')
    
    if not DATABASE_URL:
        logger.error("DATABASE_URL not found in environment variables")
        return False
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Test query
        cursor.execute("SELECT current_database(), version();")
        result = cursor.fetchone()
        
        logger.info(f"✅ Connected to database: {result[0]}")
        logger.info(f"PostgreSQL version: {result[1][:50]}...")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing Supabase connection...")
    
    if test_connection():
        logger.info("Creating tables...")
        if create_tables():
            logger.info("🎉 Setup completed successfully!")
        else:
            logger.error("❌ Table creation failed")
    else:
        logger.error("❌ Connection test failed")