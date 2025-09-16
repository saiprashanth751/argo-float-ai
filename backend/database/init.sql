-- production_init.sql
-- Complete production-optimized database schema for ARGO data processing
-- Designed for high-performance bulk operations with 980+ files using PostgreSQL COPY

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Drop existing tables for clean restart (matches processor exactly)
DROP TABLE IF EXISTS argo_measurements CASCADE;
DROP TABLE IF EXISTS argo_profiles CASCADE;
DROP TABLE IF EXISTS data_processing_log CASCADE;
DROP TABLE IF EXISTS enhanced_floats_metadata CASCADE;
DROP TABLE IF EXISTS enhanced_measurements CASCADE;
DROP TABLE IF EXISTS float_summary_stats CASCADE;
DROP TABLE IF EXISTS regional_ocean_stats CASCADE;
DROP TABLE IF EXISTS argo_gridded_data CASCADE;

-- ==========================================
-- PRODUCTION TABLES (EXACT MATCH TO PROCESSOR)
-- ==========================================

-- Main profiles table (optimized for PostgreSQL COPY operations)
CREATE TABLE argo_profiles (
    id SERIAL PRIMARY KEY,
    platform_number VARCHAR(20) NOT NULL,
    cycle_number INTEGER,
    profile_date TIMESTAMP NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    
    -- Profile characteristics (exact match to processor)
    max_pressure DOUBLE PRECISION,
    n_levels INTEGER DEFAULT 0,
    surface_temp DOUBLE PRECISION,
    surface_salinity DOUBLE PRECISION,
    max_temp DOUBLE PRECISION,
    min_temp DOUBLE PRECISION,
    temp_at_1000m DOUBLE PRECISION,
    mixed_layer_depth DOUBLE PRECISION,
    
    -- Metadata (exact match to processor)
    source_file VARCHAR(255) NOT NULL,
    processed_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT unique_profile UNIQUE(platform_number, cycle_number, profile_date),
    CONSTRAINT valid_coordinates CHECK (
        latitude BETWEEN -90 AND 90 AND 
        longitude BETWEEN -180 AND 180 AND
        NOT (latitude = 0 AND longitude = 0)
    ),
    CONSTRAINT valid_pressure CHECK (max_pressure IS NULL OR max_pressure > 0),
    CONSTRAINT valid_levels CHECK (n_levels >= 0),
    CONSTRAINT valid_temperatures CHECK (
        (surface_temp IS NULL OR surface_temp BETWEEN -3 AND 40) AND
        (max_temp IS NULL OR max_temp BETWEEN -3 AND 40) AND
        (min_temp IS NULL OR min_temp BETWEEN -3 AND 40) AND
        (temp_at_1000m IS NULL OR temp_at_1000m BETWEEN -3 AND 40)
    ),
    CONSTRAINT valid_salinity CHECK (surface_salinity IS NULL OR surface_salinity BETWEEN 0 AND 50),
    CONSTRAINT valid_mld CHECK (mixed_layer_depth IS NULL OR mixed_layer_depth >= 0)
);

-- Measurements table (optimized for bulk COPY inserts)
CREATE TABLE argo_measurements (
    id BIGSERIAL PRIMARY KEY,
    profile_id INTEGER NOT NULL REFERENCES argo_profiles(id) ON DELETE CASCADE,
    pressure DOUBLE PRECISION NOT NULL,
    depth DOUBLE PRECISION,
    temperature DOUBLE PRECISION,
    salinity DOUBLE PRECISION,
    
    CONSTRAINT valid_pressure_meas CHECK (pressure > 0),
    CONSTRAINT valid_depth_meas CHECK (depth IS NULL OR depth >= 0),
    CONSTRAINT valid_temperature_meas CHECK (temperature IS NULL OR temperature BETWEEN -3 AND 40),
    CONSTRAINT valid_salinity_meas CHECK (salinity IS NULL OR salinity BETWEEN 0 AND 50),
    CONSTRAINT has_measurement CHECK (temperature IS NOT NULL OR salinity IS NOT NULL)
);

-- Processing log for monitoring (exact match to processor)
CREATE TABLE data_processing_log (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) UNIQUE NOT NULL,
    processing_status VARCHAR(20) NOT NULL,
    profiles_count INTEGER DEFAULT 0,
    measurements_count INTEGER DEFAULT 0,
    error_message TEXT,
    processing_time_seconds DOUBLE PRECISION,
    processed_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT valid_status CHECK (processing_status IN ('SUCCESS', 'FAILED', 'SKIPPED')),
    CONSTRAINT valid_counts CHECK (profiles_count >= 0 AND measurements_count >= 0),
    CONSTRAINT valid_processing_time CHECK (processing_time_seconds IS NULL OR processing_time_seconds >= 0)
);

-- ==========================================
-- PERFORMANCE INDEXES (OPTIMIZED FOR BULK OPERATIONS)
-- ==========================================

-- Primary performance indexes (created after bulk loading for speed)
CREATE INDEX idx_profiles_platform ON argo_profiles (platform_number);
CREATE INDEX idx_profiles_date ON argo_profiles (profile_date DESC);
CREATE INDEX idx_profiles_coords ON argo_profiles (latitude, longitude);
CREATE INDEX idx_profiles_surface_temp ON argo_profiles (surface_temp) WHERE surface_temp IS NOT NULL;
CREATE INDEX idx_profiles_surface_sal ON argo_profiles (surface_salinity) WHERE surface_salinity IS NOT NULL;
CREATE INDEX idx_profiles_max_pressure ON argo_profiles (max_pressure) WHERE max_pressure IS NOT NULL;
CREATE INDEX idx_profiles_mld ON argo_profiles (mixed_layer_depth) WHERE mixed_layer_depth IS NOT NULL;

-- Measurement indexes (created after bulk loading)
CREATE INDEX idx_measurements_profile ON argo_measurements (profile_id);
CREATE INDEX idx_measurements_pressure ON argo_measurements (pressure);
CREATE INDEX idx_measurements_temp_sal ON argo_measurements (temperature, salinity) 
    WHERE temperature IS NOT NULL AND salinity IS NOT NULL;
CREATE INDEX idx_measurements_temp ON argo_measurements (temperature) WHERE temperature IS NOT NULL;
CREATE INDEX idx_measurements_sal ON argo_measurements (salinity) WHERE salinity IS NOT NULL;

-- Processing log indexes
CREATE INDEX idx_log_status ON data_processing_log (processing_status);
CREATE INDEX idx_log_processed_at ON data_processing_log (processed_at DESC);
CREATE INDEX idx_log_filename ON data_processing_log (filename);

-- ==========================================
-- BACKWARD COMPATIBILITY TABLES (FOR EXISTING SYSTEM)
-- ==========================================

-- Enhanced floats metadata table (for backward compatibility)
CREATE TABLE enhanced_floats_metadata (
    id SERIAL PRIMARY KEY,
    platform_number VARCHAR(20) NOT NULL,
    cycle_number INTEGER NOT NULL,
    date TIMESTAMP NOT NULL,
    latitude DOUBLE PRECISION NOT NULL CHECK (latitude >= -90 AND latitude <= 90),
    longitude DOUBLE PRECISION NOT NULL CHECK (longitude >= -180 AND longitude <= 180),
    location GEOGRAPHY(POINT, 4326),
    project_name VARCHAR(100) DEFAULT 'Unknown',
    pi_name VARCHAR(100) DEFAULT 'Unknown', 
    institution VARCHAR(100) DEFAULT 'Unknown',
    wmo_inst_type VARCHAR(10) DEFAULT 'Unknown',
    data_mode CHAR(1) CHECK (data_mode IN ('R', 'A', 'D')) DEFAULT 'R',
    source_file VARCHAR(255) NOT NULL,
    max_pressure DOUBLE PRECISION CHECK (max_pressure >= 0),
    min_pressure DOUBLE PRECISION CHECK (min_pressure >= 0),
    mixed_layer_depth DOUBLE PRECISION CHECK (mixed_layer_depth >= 0),
    thermocline_depth DOUBLE PRECISION CHECK (thermocline_depth >= 0),
    halocline_depth DOUBLE PRECISION CHECK (halocline_depth >= 0),
    surface_temperature DOUBLE PRECISION CHECK (surface_temperature >= -3 AND surface_temperature <= 40),
    surface_salinity DOUBLE PRECISION CHECK (surface_salinity >= 0 AND surface_salinity <= 50),
    bottom_temperature DOUBLE PRECISION CHECK (bottom_temperature >= -3 AND bottom_temperature <= 40),
    bottom_salinity DOUBLE PRECISION CHECK (bottom_salinity >= 0 AND bottom_salinity <= 50),
    profile_quality_flag INTEGER DEFAULT 0,
    n_measurements INTEGER DEFAULT 0 CHECK (n_measurements >= 0),
    has_bgc_data BOOLEAN DEFAULT FALSE,
    processed_at TIMESTAMP DEFAULT NOW(),
    processing_version VARCHAR(10) DEFAULT '2.0',
    UNIQUE(platform_number, cycle_number, date),
    CHECK (min_pressure <= max_pressure OR min_pressure IS NULL OR max_pressure IS NULL)
);

-- Enhanced measurements table (for backward compatibility)
CREATE TABLE enhanced_measurements (
    id BIGSERIAL PRIMARY KEY,
    metadata_id INTEGER NOT NULL REFERENCES enhanced_floats_metadata(id) ON DELETE CASCADE,
    pressure DOUBLE PRECISION NOT NULL CHECK (pressure > 0),
    depth DOUBLE PRECISION CHECK (depth >= 0),
    temperature DOUBLE PRECISION CHECK (temperature >= -3 AND temperature <= 40),
    salinity DOUBLE PRECISION CHECK (salinity >= 0 AND salinity <= 50),
    potential_temperature DOUBLE PRECISION,
    conservative_temperature DOUBLE PRECISION,
    absolute_salinity DOUBLE PRECISION,
    density DOUBLE PRECISION CHECK (density > 900 AND density < 1100),
    potential_density DOUBLE PRECISION CHECK (potential_density > 900 AND potential_density < 1100),
    buoyancy_frequency DOUBLE PRECISION,
    oxygen DOUBLE PRECISION CHECK (oxygen >= 0 AND oxygen <= 800),
    oxygen_saturation DOUBLE PRECISION CHECK (oxygen_saturation >= 0 AND oxygen_saturation <= 150),
    chlorophyll DOUBLE PRECISION CHECK (chlorophyll >= 0 AND chlorophyll <= 100),
    chlorophyll_fluorescence DOUBLE PRECISION CHECK (chlorophyll_fluorescence >= 0),
    backscatter_700 DOUBLE PRECISION CHECK (backscatter_700 >= 0),
    backscatter_532 DOUBLE PRECISION CHECK (backscatter_532 >= 0),
    cdom DOUBLE PRECISION CHECK (cdom >= 0),
    nitrate DOUBLE PRECISION CHECK (nitrate >= 0 AND nitrate <= 100),
    ph_in_situ DOUBLE PRECISION CHECK (ph_in_situ >= 6 AND ph_in_situ <= 9),
    downwelling_par DOUBLE PRECISION CHECK (downwelling_par >= 0),
    pressure_qc INTEGER DEFAULT 1 CHECK (pressure_qc BETWEEN 0 AND 9),
    temperature_qc INTEGER DEFAULT 1 CHECK (temperature_qc BETWEEN 0 AND 9),
    salinity_qc INTEGER DEFAULT 1 CHECK (salinity_qc BETWEEN 0 AND 9),
    oxygen_qc INTEGER DEFAULT 1 CHECK (oxygen_qc BETWEEN 0 AND 9),
    chlorophyll_qc INTEGER DEFAULT 1 CHECK (chlorophyll_qc BETWEEN 0 AND 9)
);

-- Backward compatibility indexes
CREATE INDEX idx_enhanced_metadata_platform ON enhanced_floats_metadata (platform_number);
CREATE INDEX idx_enhanced_metadata_date ON enhanced_floats_metadata (date DESC);
CREATE INDEX idx_enhanced_metadata_coords ON enhanced_floats_metadata (latitude, longitude);
CREATE INDEX idx_enhanced_measurements_metadata ON enhanced_measurements (metadata_id);
CREATE INDEX idx_enhanced_measurements_pressure ON enhanced_measurements (pressure);

-- ==========================================
-- SPATIAL FUNCTIONALITY
-- ==========================================

-- Add geography columns and triggers
ALTER TABLE argo_profiles ADD COLUMN location GEOGRAPHY(POINT, 4326);

-- Function to update geography from coordinates
CREATE OR REPLACE FUNCTION update_location_geography() RETURNS TRIGGER AS $$
BEGIN
    NEW.location = ST_Point(NEW.longitude, NEW.latitude)::geography;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update location
CREATE TRIGGER trigger_update_location_geography_profiles
    BEFORE INSERT OR UPDATE ON argo_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_location_geography();

CREATE TRIGGER trigger_update_location_geography_enhanced
    BEFORE INSERT OR UPDATE ON enhanced_floats_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_location_geography();

-- Function to bulk update existing geography data
CREATE OR REPLACE FUNCTION bulk_update_geography() RETURNS void AS $$
BEGIN
    UPDATE argo_profiles 
    SET location = ST_Point(longitude, latitude)::geography 
    WHERE location IS NULL;
    
    UPDATE enhanced_floats_metadata 
    SET location = ST_Point(longitude, latitude)::geography 
    WHERE location IS NULL;
END;
$$ LANGUAGE plpgsql;

-- Spatial indexes (created after bulk geography update)
-- CREATE INDEX idx_profiles_location ON argo_profiles USING GIST (location);
-- CREATE INDEX idx_enhanced_metadata_location ON enhanced_floats_metadata USING GIST (location);

-- ==========================================
-- HIGH-PERFORMANCE QUERY FUNCTIONS
-- ==========================================

-- Function to get profiles within radius (optimized for production)
CREATE OR REPLACE FUNCTION get_profiles_near_point(
    center_lat DOUBLE PRECISION,
    center_lon DOUBLE PRECISION, 
    radius_km DOUBLE PRECISION,
    limit_count INTEGER DEFAULT 1000
) RETURNS TABLE (
    id INTEGER,
    platform_number VARCHAR(20),
    profile_date TIMESTAMP,
    distance_km DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p.id,
        p.platform_number,
        p.profile_date,
        ST_Distance(
            ST_Point(center_lon, center_lat)::geography,
            ST_Point(p.longitude, p.latitude)::geography
        ) / 1000.0 as distance_km
    FROM argo_profiles p
    WHERE ST_DWithin(
        ST_Point(p.longitude, p.latitude)::geography,
        ST_Point(center_lon, center_lat)::geography,
        radius_km * 1000
    )
    ORDER BY distance_km
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Function for temperature profile analysis
CREATE OR REPLACE FUNCTION get_temperature_profile(
    target_platform VARCHAR(20),
    target_cycle INTEGER DEFAULT NULL
) RETURNS TABLE (
    pressure DOUBLE PRECISION,
    temperature DOUBLE PRECISION,
    depth DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.pressure,
        m.temperature,
        m.depth
    FROM argo_measurements m
    JOIN argo_profiles p ON m.profile_id = p.id
    WHERE p.platform_number = target_platform
    AND (target_cycle IS NULL OR p.cycle_number = target_cycle)
    AND m.temperature IS NOT NULL
    ORDER BY p.profile_date DESC, m.pressure ASC
    LIMIT 5000;
END;
$$ LANGUAGE plpgsql;

-- Function for salinity profile analysis
CREATE OR REPLACE FUNCTION get_salinity_profile(
    target_platform VARCHAR(20),
    target_cycle INTEGER DEFAULT NULL
) RETURNS TABLE (
    pressure DOUBLE PRECISION,
    salinity DOUBLE PRECISION,
    depth DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.pressure,
        m.salinity,
        m.depth
    FROM argo_measurements m
    JOIN argo_profiles p ON m.profile_id = p.id
    WHERE p.platform_number = target_platform
    AND (target_cycle IS NULL OR p.cycle_number = target_cycle)
    AND m.salinity IS NOT NULL
    ORDER BY p.profile_date DESC, m.pressure ASC
    LIMIT 5000;
END;
$$ LANGUAGE plpgsql;

-- Function for spatial temperature distribution
CREATE OR REPLACE FUNCTION get_surface_temperature_distribution(
    lat_min DOUBLE PRECISION,
    lat_max DOUBLE PRECISION,
    lon_min DOUBLE PRECISION, 
    lon_max DOUBLE PRECISION,
    days_back INTEGER DEFAULT 30
) RETURNS TABLE (
    platform_number VARCHAR(20),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    surface_temp DOUBLE PRECISION,
    profile_date TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p.platform_number,
        p.latitude,
        p.longitude,
        p.surface_temp,
        p.profile_date
    FROM argo_profiles p
    WHERE p.latitude BETWEEN lat_min AND lat_max
    AND p.longitude BETWEEN lon_min AND lon_max
    AND p.surface_temp IS NOT NULL
    AND p.profile_date >= NOW() - INTERVAL '1 day' * days_back
    ORDER BY p.profile_date DESC;
END;
$$ LANGUAGE plpgsql;

-- ==========================================
-- PRODUCTION MONITORING VIEWS
-- ==========================================

-- Processing status summary
CREATE VIEW processing_status_summary AS
SELECT 
    processing_status,
    COUNT(*) as file_count,
    SUM(profiles_count) as total_profiles,
    SUM(measurements_count) as total_measurements,
    AVG(processing_time_seconds) as avg_processing_time,
    SUM(processing_time_seconds) as total_processing_time
FROM data_processing_log
GROUP BY processing_status;

-- Platform summary
CREATE VIEW platform_summary AS
SELECT 
    platform_number,
    COUNT(*) as total_profiles,
    MIN(profile_date) as first_profile,
    MAX(profile_date) as last_profile,
    AVG(latitude) as avg_latitude,
    AVG(longitude) as avg_longitude,
    AVG(surface_temp) as avg_surface_temp,
    AVG(surface_salinity) as avg_surface_salinity,
    AVG(mixed_layer_depth) as avg_mld
FROM argo_profiles
GROUP BY platform_number
ORDER BY total_profiles DESC;

-- Recent data summary
CREATE VIEW recent_data_summary AS
SELECT 
    COUNT(*) as profiles_last_30_days,
    COUNT(DISTINCT platform_number) as active_platforms,
    AVG(surface_temp) as avg_surface_temp,
    AVG(surface_salinity) as avg_surface_salinity,
    MIN(profile_date) as earliest_date,
    MAX(profile_date) as latest_date
FROM argo_profiles
WHERE profile_date >= NOW() - INTERVAL '30 days';

-- ==========================================
-- DATA SYNCHRONIZATION FUNCTIONS
-- ==========================================

-- Function to sync production data to legacy schema for backward compatibility
CREATE OR REPLACE FUNCTION sync_to_legacy_schema() RETURNS void AS $$
BEGIN
    -- Insert/update enhanced_floats_metadata from argo_profiles
    INSERT INTO enhanced_floats_metadata (
        platform_number, cycle_number, date, latitude, longitude, location,
        source_file, max_pressure, min_pressure, mixed_layer_depth,
        surface_temperature, surface_salinity, n_measurements, processed_at
    )
    SELECT 
        p.platform_number, 
        p.cycle_number, 
        p.profile_date, 
        p.latitude, 
        p.longitude, 
        p.location,
        p.source_file, 
        p.max_pressure, 
        0 as min_pressure, 
        p.mixed_layer_depth,
        p.surface_temp, 
        p.surface_salinity, 
        p.n_levels,
        p.processed_at
    FROM argo_profiles p
    ON CONFLICT (platform_number, cycle_number, date) 
    DO UPDATE SET
        latitude = EXCLUDED.latitude,
        longitude = EXCLUDED.longitude,
        location = EXCLUDED.location,
        max_pressure = EXCLUDED.max_pressure,
        mixed_layer_depth = EXCLUDED.mixed_layer_depth,
        surface_temperature = EXCLUDED.surface_temperature,
        surface_salinity = EXCLUDED.surface_salinity,
        n_measurements = EXCLUDED.n_measurements,
        processed_at = EXCLUDED.processed_at;

    -- Insert/update enhanced_measurements from argo_measurements
    INSERT INTO enhanced_measurements (
        metadata_id, pressure, depth, temperature, salinity
    )
    SELECT 
        efm.id, 
        m.pressure, 
        m.depth, 
        m.temperature, 
        m.salinity
    FROM argo_measurements m
    JOIN argo_profiles p ON m.profile_id = p.id
    JOIN enhanced_floats_metadata efm ON (
        p.platform_number = efm.platform_number 
        AND p.cycle_number = efm.cycle_number 
        AND p.profile_date = efm.date
    )
    ON CONFLICT (metadata_id, pressure) 
    DO UPDATE SET
        temperature = EXCLUDED.temperature,
        salinity = EXCLUDED.salinity,
        depth = EXCLUDED.depth;

    -- Update measurement counts
    UPDATE enhanced_floats_metadata efm
    SET n_measurements = (
        SELECT COUNT(*)
        FROM enhanced_measurements em
        WHERE em.metadata_id = efm.id
    );
END;
$$ LANGUAGE plpgsql;

-- Function to create spatial indexes after bulk loading
CREATE OR REPLACE FUNCTION create_spatial_indexes() RETURNS void AS $$
BEGIN
    -- Update all geography columns first
    PERFORM bulk_update_geography();
    
    -- Create spatial indexes
    CREATE INDEX IF NOT EXISTS idx_profiles_location ON argo_profiles USING GIST (location);
    CREATE INDEX IF NOT EXISTS idx_enhanced_metadata_location ON enhanced_floats_metadata USING GIST (location);
    
    RAISE NOTICE 'Spatial indexes created successfully';
END;
$$ LANGUAGE plpgsql;

-- ==========================================
-- PRODUCTION UTILITIES
-- ==========================================

-- Function to optimize database after bulk loading
CREATE OR REPLACE FUNCTION optimize_after_bulk_load() RETURNS void AS $$
BEGIN
    -- Update table statistics
    ANALYZE argo_profiles;
    ANALYZE argo_measurements;
    ANALYZE data_processing_log;
    
    -- Create spatial indexes
    PERFORM create_spatial_indexes();
    
    -- Sync to legacy schema for backward compatibility
    PERFORM sync_to_legacy_schema();
    
    -- Update legacy table statistics
    ANALYZE enhanced_floats_metadata;
    ANALYZE enhanced_measurements;
    
    RAISE NOTICE 'Database optimization completed';
END;
$$ LANGUAGE plpgsql;

-- Function to get processing statistics
CREATE OR REPLACE FUNCTION get_processing_stats() RETURNS TABLE (
    metric VARCHAR,
    value BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 'total_profiles'::VARCHAR, COUNT(*)::BIGINT FROM argo_profiles
    UNION ALL
    SELECT 'total_measurements'::VARCHAR, COUNT(*)::BIGINT FROM argo_measurements
    UNION ALL
    SELECT 'unique_platforms'::VARCHAR, COUNT(DISTINCT platform_number)::BIGINT FROM argo_profiles
    UNION ALL
    SELECT 'files_processed'::VARCHAR, COUNT(*)::BIGINT FROM data_processing_log WHERE processing_status = 'SUCCESS'
    UNION ALL
    SELECT 'files_failed'::VARCHAR, COUNT(*)::BIGINT FROM data_processing_log WHERE processing_status = 'FAILED';
END;
$$ LANGUAGE plpgsql;

-- ==========================================
-- FINAL SETUP AND PERMISSIONS
-- ==========================================

-- Grant all permissions to argo_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO argo_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO argo_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO argo_user;

-- Set optimal PostgreSQL settings for bulk operations
-- (These should also be set in postgresql.conf)
-- shared_buffers = 256MB (or 25% of RAM)
-- work_mem = 50MB (for sorting operations)
-- maintenance_work_mem = 1GB (for CREATE INDEX)
-- checkpoint_timeout = 30min
-- max_wal_size = 2GB

-- Final status message
DO $$
BEGIN
    RAISE NOTICE '===============================================';
    RAISE NOTICE 'PRODUCTION ARGO DATABASE INITIALIZED';
    RAISE NOTICE '===============================================';
    RAISE NOTICE 'Main tables: argo_profiles, argo_measurements';
    RAISE NOTICE 'Monitoring: data_processing_log';
    RAISE NOTICE 'Backward compatibility: enhanced_floats_metadata, enhanced_measurements';
    RAISE NOTICE 'Optimized for PostgreSQL COPY bulk operations';
    RAISE NOTICE 'Ready for processing 980+ files';
    RAISE NOTICE '===============================================';
    
    -- Show table counts
    PERFORM get_processing_stats();
END $$;