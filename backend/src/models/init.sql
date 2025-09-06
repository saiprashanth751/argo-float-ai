-- init.sql
CREATE TABLE IF NOT EXISTS floats_metadata (
    id SERIAL PRIMARY KEY,
    platform_number VARCHAR(50),
    cycle_number INTEGER,
    date TIMESTAMP,
    latitude FLOAT,
    longitude FLOAT,
    project_name VARCHAR(255),
    pi_name VARCHAR(255),
    institution VARCHAR(255),
    wmo_inst_type VARCHAR(100),
    data_mode VARCHAR(50),
    source_file VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS measurements (
    id SERIAL PRIMARY KEY,
    metadata_id INTEGER REFERENCES floats_metadata(id),
    pressure FLOAT,
    temperature FLOAT,
    salinity FLOAT,
    oxygen FLOAT,
    chlorophyll FLOAT,
    backscatter FLOAT
);

CREATE INDEX IF NOT EXISTS idx_measurements_metadata ON measurements(metadata_id);
CREATE INDEX IF NOT EXISTS idx_measurements_pressure ON measurements(pressure);
CREATE INDEX IF NOT EXISTS idx_measurements_temp ON measurements(temperature);
CREATE INDEX IF NOT EXISTS idx_measurements_salinity ON measurements(salinity);