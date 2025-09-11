-- init.sql - Enhanced database initialization for ARGO data processing
-- This file is automatically executed when PostgreSQL container starts

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create enhanced floats metadata table with spatial support
CREATE TABLE IF NOT EXISTS enhanced_floats_metadata (
    id SERIAL PRIMARY KEY,
    
    -- Core ARGO identifiers
    platform_number VARCHAR(20) NOT NULL,
    cycle_number INTEGER NOT NULL,
    date TIMESTAMP NOT NULL,
    
    -- Geospatial data (WGS84)
    latitude DOUBLE PRECISION NOT NULL CHECK (latitude >= -90 AND latitude <= 90),
    longitude DOUBLE PRECISION NOT NULL CHECK (longitude >= -180 AND longitude <= 180),
    location GEOGRAPHY(POINT, 4326),
    
    -- Enhanced metadata
    project_name VARCHAR(100) DEFAULT 'Unknown',
    pi_name VARCHAR(100) DEFAULT 'Unknown', 
    institution VARCHAR(100) DEFAULT 'Unknown',
    wmo_inst_type VARCHAR(10) DEFAULT 'Unknown',
    data_mode CHAR(1) CHECK (data_mode IN ('R', 'A', 'D')) DEFAULT 'R',
    source_file VARCHAR(255) NOT NULL,
    
    -- Pressure range
    max_pressure DOUBLE PRECISION CHECK (max_pressure >= 0),
    min_pressure DOUBLE PRECISION CHECK (min_pressure >= 0),
    
    -- Derived oceanographic parameters
    mixed_layer_depth DOUBLE PRECISION CHECK (mixed_layer_depth >= 0),
    thermocline_depth DOUBLE PRECISION CHECK (thermocline_depth >= 0),
    halocline_depth DOUBLE PRECISION CHECK (halocline_depth >= 0),
    
    -- Surface conditions
    surface_temperature DOUBLE PRECISION CHECK (surface_temperature >= -3 AND surface_temperature <= 40),
    surface_salinity DOUBLE PRECISION CHECK (surface_salinity >= 0 AND surface_salinity <= 50),
    
    -- Bottom conditions  
    bottom_temperature DOUBLE PRECISION CHECK (bottom_temperature >= -3 AND bottom_temperature <= 40),
    bottom_salinity DOUBLE PRECISION CHECK (bottom_salinity >= 0 AND bottom_salinity <= 50),
    
    -- Data quality and processing info
    profile_quality_flag INTEGER DEFAULT 0,
    n_measurements INTEGER DEFAULT 0 CHECK (n_measurements >= 0),
    has_bgc_data BOOLEAN DEFAULT FALSE,
    
    -- Processing metadata
    processed_at TIMESTAMP DEFAULT NOW(),
    processing_version VARCHAR(10) DEFAULT '2.0',
    
    -- Constraints
    UNIQUE(platform_number, cycle_number, date),
    CHECK (min_pressure <= max_pressure)
);

-- Enhanced measurements table with all parameters
CREATE TABLE IF NOT EXISTS enhanced_measurements (
    id BIGSERIAL PRIMARY KEY,
    metadata_id INTEGER NOT NULL REFERENCES enhanced_floats_metadata(id) ON DELETE CASCADE,
    
    -- Core pressure coordinate
    pressure DOUBLE PRECISION NOT NULL CHECK (pressure > 0),
    depth DOUBLE PRECISION CHECK (depth >= 0),
    
    -- Core parameters (adjusted values preferred)
    temperature DOUBLE PRECISION CHECK (temperature >= -3 AND temperature <= 40),
    salinity DOUBLE PRECISION CHECK (salinity >= 0 AND salinity <= 50),
    
    -- Derived thermodynamic parameters
    potential_temperature DOUBLE PRECISION,
    conservative_temperature DOUBLE PRECISION,
    absolute_salinity DOUBLE PRECISION,
    density DOUBLE PRECISION CHECK (density > 900 AND density < 1100),
    potential_density DOUBLE PRECISION CHECK (potential_density > 900 AND potential_density < 1100),
    buoyancy_frequency DOUBLE PRECISION,
    
    -- Biogeochemical parameters
    oxygen DOUBLE PRECISION CHECK (oxygen >= 0 AND oxygen <= 800), -- micromol/kg
    oxygen_saturation DOUBLE PRECISION CHECK (oxygen_saturation >= 0 AND oxygen_saturation <= 150), -- %
    chlorophyll DOUBLE PRECISION CHECK (chlorophyll >= 0 AND chlorophyll <= 100), -- mg/m³
    chlorophyll_fluorescence DOUBLE PRECISION CHECK (chlorophyll_fluorescence >= 0),
    backscatter_700 DOUBLE PRECISION CHECK (backscatter_700 >= 0), -- 1/m
    backscatter_532 DOUBLE PRECISION CHECK (backscatter_532 >= 0), -- 1/m
    cdom DOUBLE PRECISION CHECK (cdom >= 0), -- 1/m
    nitrate DOUBLE PRECISION CHECK (nitrate >= 0 AND nitrate <= 100), -- micromol/kg
    ph_in_situ DOUBLE PRECISION CHECK (ph_in_situ >= 6 AND ph_in_situ <= 9),
    downwelling_par DOUBLE PRECISION CHECK (downwelling_par >= 0), -- micromol quanta/m²/s
    
    -- Quality control flags (ARGO standard: 1=good, 2=probably good, 3=probably bad, 4=bad, etc.)
    pressure_qc INTEGER DEFAULT 1 CHECK (pressure_qc BETWEEN 0 AND 9),
    temperature_qc INTEGER DEFAULT 1 CHECK (temperature_qc BETWEEN 0 AND 9),
    salinity_qc INTEGER DEFAULT 1 CHECK (salinity_qc BETWEEN 0 AND 9),
    oxygen_qc INTEGER DEFAULT 1 CHECK (oxygen_qc BETWEEN 0 AND 9),
    chlorophyll_qc INTEGER DEFAULT 1 CHECK (chlorophyll_qc BETWEEN 0 AND 9)
);

-- Pre-computed summary statistics table for fast aggregations
CREATE TABLE IF NOT EXISTS float_summary_stats (
    platform_number VARCHAR(20) PRIMARY KEY,
    total_profiles INTEGER NOT NULL DEFAULT 0,
    date_range_start TIMESTAMP,
    date_range_end TIMESTAMP,
    
    -- Geographical bounds
    latitude_min DOUBLE PRECISION,
    latitude_max DOUBLE PRECISION, 
    longitude_min DOUBLE PRECISION,
    longitude_max DOUBLE PRECISION,
    
    -- Oceanographic summaries
    avg_mixed_layer_depth DOUBLE PRECISION,
    avg_surface_temp DOUBLE PRECISION,
    avg_surface_salinity DOUBLE PRECISION,
    
    -- Capabilities
    has_bgc_sensors BOOLEAN DEFAULT FALSE,
    max_depth_sampled DOUBLE PRECISION,
    
    -- Metadata
    last_updated TIMESTAMP DEFAULT NOW(),
    
    CHECK (latitude_min <= latitude_max),
    CHECK (longitude_min <= longitude_max),
    CHECK (date_range_start <= date_range_end)
);

-- Regional aggregation table for query optimization
CREATE TABLE IF NOT EXISTS regional_ocean_stats (
    id SERIAL PRIMARY KEY,
    region_name VARCHAR(50) NOT NULL,
    region_bounds GEOGRAPHY(POLYGON, 4326) NOT NULL,
    
    -- Time period
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    
    -- Aggregated statistics
    avg_surface_temp DOUBLE PRECISION,
    avg_surface_salinity DOUBLE PRECISION,
    avg_mixed_layer_depth DOUBLE PRECISION,
    total_profiles INTEGER DEFAULT 0,
    
    last_updated TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(region_name, start_date, end_date)
);

-- ==========================================
-- INDEXES FOR PERFORMANCE
-- ==========================================

-- Spatial indexes
CREATE INDEX IF NOT EXISTS idx_enhanced_metadata_location 
    ON enhanced_floats_metadata USING GIST (location);

CREATE INDEX IF NOT EXISTS idx_enhanced_metadata_lat_lon 
    ON enhanced_floats_metadata (latitude, longitude);

-- Temporal indexes  
CREATE INDEX IF NOT EXISTS idx_enhanced_metadata_date 
    ON enhanced_floats_metadata (date DESC);

CREATE INDEX IF NOT EXISTS idx_enhanced_metadata_date_location 
    ON enhanced_floats_metadata (date DESC, latitude, longitude);

-- Platform and identification indexes
CREATE INDEX IF NOT EXISTS idx_enhanced_metadata_platform 
    ON enhanced_floats_metadata (platform_number);

CREATE INDEX IF NOT EXISTS idx_enhanced_metadata_platform_cycle 
    ON enhanced_floats_metadata (platform_number, cycle_number);

-- Oceanographic parameter indexes
CREATE INDEX IF NOT EXISTS idx_enhanced_metadata_surface_temp 
    ON enhanced_floats_metadata (surface_temperature) WHERE surface_temperature IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_enhanced_metadata_mld 
    ON enhanced_floats_metadata (mixed_layer_depth) WHERE mixed_layer_depth IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_enhanced_metadata_bgc 
    ON enhanced_floats_metadata (has_bgc_data) WHERE has_bgc_data = true;

-- Measurements table indexes
CREATE INDEX IF NOT EXISTS idx_enhanced_measurements_metadata 
    ON enhanced_measurements (metadata_id);

CREATE INDEX IF NOT EXISTS idx_enhanced_measurements_pressure 
    ON enhanced_measurements (pressure);

CREATE INDEX IF NOT EXISTS idx_enhanced_measurements_temp_sal 
    ON enhanced_measurements (temperature, salinity) 
    WHERE temperature IS NOT NULL AND salinity IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_enhanced_measurements_density 
    ON enhanced_measurements (density) WHERE density IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_enhanced_measurements_oxygen 
    ON enhanced_measurements (oxygen) WHERE oxygen IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_enhanced_measurements_chlorophyll 
    ON enhanced_measurements (chlorophyll) WHERE chlorophyll IS NOT NULL;

-- Composite index for profile queries
CREATE INDEX IF NOT EXISTS idx_enhanced_measurements_profile_analysis 
    ON enhanced_measurements (metadata_id, pressure, temperature, salinity);

-- Regional stats spatial index
CREATE INDEX IF NOT EXISTS idx_regional_stats_bounds 
    ON regional_ocean_stats USING GIST (region_bounds);

-- ==========================================
-- FUNCTIONS FOR COMMON CALCULATIONS
-- ==========================================

-- Function to calculate distance between two points (Great Circle distance)
CREATE OR REPLACE FUNCTION calculate_distance_km(
    lat1 DOUBLE PRECISION, 
    lon1 DOUBLE PRECISION, 
    lat2 DOUBLE PRECISION, 
    lon2 DOUBLE PRECISION
) RETURNS DOUBLE PRECISION AS $$
BEGIN
    RETURN ST_Distance(
        ST_MakePoint(lon1, lat1)::geography,
        ST_MakePoint(lon2, lat2)::geography
    ) / 1000.0; -- Convert meters to kilometers
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to get profiles within radius of a point
CREATE OR REPLACE FUNCTION get_profiles_within_radius(
    center_lat DOUBLE PRECISION,
    center_lon DOUBLE PRECISION, 
    radius_km DOUBLE PRECISION
) RETURNS TABLE (
    id INTEGER,
    platform_number VARCHAR(20),
    date TIMESTAMP,
    distance_km DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.id,
        m.platform_number,
        m.date,
        calculate_distance_km(center_lat, center_lon, m.latitude, m.longitude) as distance_km
    FROM enhanced_floats_metadata m
    WHERE ST_DWithin(
        location,
        ST_MakePoint(center_lon, center_lat)::geography,
        radius_km * 1000 -- Convert km to meters
    )
    ORDER BY distance_km;
END;
$$ LANGUAGE plpgsql;

-- Function to update regional statistics
CREATE OR REPLACE FUNCTION update_regional_statistics() RETURNS VOID AS $$
BEGIN
    -- This function can be called periodically to update regional_ocean_stats
    -- Implementation depends on specific regional boundaries
    
    -- Example for Indian Ocean regions (simplified)
    INSERT INTO regional_ocean_stats (
        region_name, region_bounds, start_date, end_date,
        avg_surface_temp, avg_surface_salinity, avg_mixed_layer_depth, total_profiles
    )
    SELECT 
        'Arabian_Sea' as region_name,
        ST_GeogFromText('POLYGON((50 5, 78 5, 78 28, 50 28, 50 5))') as region_bounds,
        DATE_TRUNC('month', MIN(date)) as start_date,
        DATE_TRUNC('month', MAX(date)) as end_date,
        AVG(surface_temperature) as avg_surface_temp,
        AVG(surface_salinity) as avg_surface_salinity,
        AVG(mixed_layer_depth) as avg_mixed_layer_depth,
        COUNT(*) as total_profiles
    FROM enhanced_floats_metadata
    WHERE ST_Within(location, ST_GeogFromText('POLYGON((50 5, 78 5, 78 28, 50 28, 50 5))'))
    AND date >= DATE_TRUNC('month', NOW() - INTERVAL '1 month')
    ON CONFLICT (region_name, start_date, end_date) 
    DO UPDATE SET
        avg_surface_temp = EXCLUDED.avg_surface_temp,
        avg_surface_salinity = EXCLUDED.avg_surface_salinity,
        avg_mixed_layer_depth = EXCLUDED.avg_mixed_layer_depth,
        total_profiles = EXCLUDED.total_profiles,
        last_updated = NOW();
END;
$$ LANGUAGE plpgsql;

-- ==========================================
-- VIEWS FOR COMMON QUERIES
-- ==========================================

-- View for recent surface conditions
CREATE OR REPLACE VIEW recent_surface_conditions AS
SELECT 
    platform_number,
    date,
    latitude,
    longitude,
    surface_temperature,
    surface_salinity,
    mixed_layer_depth,
    has_bgc_data,
    calculate_distance_km(0, 77, latitude, longitude) as distance_from_indian_coast_km
FROM enhanced_floats_metadata
WHERE date >= NOW() - INTERVAL '30 days'
ORDER BY date DESC;

-- View for BGC-enabled floats
CREATE OR REPLACE VIEW bgc_floats_summary AS
SELECT 
    m.platform_number,
    m.date,
    m.latitude,
    m.longitude,
    m.surface_temperature,
    m.surface_salinity,
    COUNT(mes.id) FILTER (WHERE mes.oxygen IS NOT NULL) as oxygen_measurements,
    COUNT(mes.id) FILTER (WHERE mes.chlorophyll IS NOT NULL) as chlorophyll_measurements,
    COUNT(mes.id) FILTER (WHERE mes.nitrate IS NOT NULL) as nitrate_measurements
FROM enhanced_floats_metadata m
LEFT JOIN enhanced_measurements mes ON m.id = mes.metadata_id
WHERE m.has_bgc_data = true
GROUP BY m.id, m.platform_number, m.date, m.latitude, m.longitude, 
         m.surface_temperature, m.surface_salinity
ORDER BY m.date DESC;

-- ==========================================
-- TRIGGERS FOR DATA MAINTENANCE
-- ==========================================

-- Trigger to automatically update location geography from lat/lon
CREATE OR REPLACE FUNCTION update_location_geography() RETURNS TRIGGER AS $$
BEGIN
    NEW.location = ST_MakePoint(NEW.longitude, NEW.latitude)::geography;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_location_geography
    BEFORE INSERT OR UPDATE ON enhanced_floats_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_location_geography();

-- Trigger to update summary statistics when new profiles are added
CREATE OR REPLACE FUNCTION update_float_summary() RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO float_summary_stats (
        platform_number, total_profiles, date_range_start, date_range_end,
        latitude_min, latitude_max, longitude_min, longitude_max,
        avg_mixed_layer_depth, avg_surface_temp, avg_surface_salinity, has_bgc_sensors
    )
    SELECT 
        NEW.platform_number,
        1,
        NEW.date,
        NEW.date,
        NEW.latitude,
        NEW.latitude,
        NEW.longitude,
        NEW.longitude,
        NEW.mixed_layer_depth,
        NEW.surface_temperature,
        NEW.surface_salinity,
        NEW.has_bgc_data
    ON CONFLICT (platform_number) 
    DO UPDATE SET
        total_profiles = float_summary_stats.total_profiles + 1,
        date_range_end = GREATEST(float_summary_stats.date_range_end, NEW.date),
        date_range_start = LEAST(float_summary_stats.date_range_start, NEW.date),
        latitude_min = LEAST(float_summary_stats.latitude_min, NEW.latitude),
        latitude_max = GREATEST(float_summary_stats.latitude_max, NEW.latitude),
        longitude_min = LEAST(float_summary_stats.longitude_min, NEW.longitude),
        longitude_max = GREATEST(float_summary_stats.longitude_max, NEW.longitude),
        has_bgc_sensors = float_summary_stats.has_bgc_sensors OR NEW.has_bgc_data,
        last_updated = NOW();
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_float_summary
    AFTER INSERT ON enhanced_floats_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_float_summary();

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO argo_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO argo_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO argo_user;

-- Create indexes after initial data load (these can be created later if needed)
-- VACUUM ANALYZE enhanced_floats_metadata;
-- VACUUM ANALYZE enhanced_measurements;

-- Final status message
DO $$
BEGIN
    RAISE NOTICE 'Enhanced ARGO database schema initialized successfully!';
    RAISE NOTICE 'Spatial indexing enabled with PostGIS';
    RAISE NOTICE 'Ready for advanced oceanographic data processing';
END $$;