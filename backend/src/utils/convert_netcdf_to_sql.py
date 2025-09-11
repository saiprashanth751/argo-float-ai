# fixed_production_argo_processor.py
import xarray as xr
import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np
from datetime import datetime
import os   
import glob
from tqdm import tqdm
import re
import logging
from dotenv import load_dotenv
import json
import time
from pathlib import Path
import traceback
from contextlib import contextmanager
import psutil
import gc

load_dotenv()

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('argo_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionArgoProcessor:
    def __init__(self, db_engine):
        self.engine = db_engine
        self.batch_size = 1000  # Reduced for stability
        self.processed_files = set()
        self.failed_files = []
        self.success_count = 0
        self.failed_count = 0  # ADDED: Dedicated failed counter
        self.skip_count = 0
        
        # Progress tracking
        self.progress_file = 'processing_progress.json'
        self.load_progress()

    def load_progress(self):
        """Load previous processing progress"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.processed_files = set(data.get('processed_files', []))
                    self.failed_files = data.get('failed_files', [])
                    # Calculate failed_count from failed_files list
                    self.failed_count = len(self.failed_files)
                logger.info(f"Loaded progress: {len(self.processed_files)} processed, {self.failed_count} failed")
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}")

    def save_progress(self):
        """Save processing progress"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump({
                    'processed_files': list(self.processed_files),
                    'failed_files': self.failed_files,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save progress: {e}")

    def create_robust_schema(self):
        """Create production-ready database schema"""
        schema_sql = """
        -- Drop existing tables if they exist (for clean restart)
        DROP TABLE IF EXISTS enhanced_measurements CASCADE;
        DROP TABLE IF EXISTS enhanced_floats_metadata CASCADE;
        DROP TABLE IF EXISTS float_summary_stats CASCADE;
        DROP TABLE IF EXISTS processing_log CASCADE;

        -- Enhanced metadata table
        CREATE TABLE enhanced_floats_metadata (
            id SERIAL PRIMARY KEY,
            platform_number VARCHAR(50) NOT NULL,
            cycle_number INTEGER NOT NULL,
            date TIMESTAMP NOT NULL,
            latitude DOUBLE PRECISION NOT NULL,
            longitude DOUBLE PRECISION NOT NULL,
            
            -- Geospatial column for efficient spatial queries
            location GEOGRAPHY(POINT, 4326),
            
            -- Enhanced metadata with generous field sizes
            project_name VARCHAR(200),
            pi_name VARCHAR(200),
            institution VARCHAR(200),
            wmo_inst_type VARCHAR(50),
            data_mode VARCHAR(50),
            source_file VARCHAR(500),
            
            -- Derived oceanographic parameters
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
            processing_version VARCHAR(20) DEFAULT '3.0-production'
        );

        -- Enhanced measurements table
        CREATE TABLE enhanced_measurements (
            id SERIAL PRIMARY KEY,
            metadata_id INTEGER REFERENCES enhanced_floats_metadata(id) ON DELETE CASCADE,
            pressure DOUBLE PRECISION NOT NULL,
            depth DOUBLE PRECISION,
            
            -- Core parameters
            temperature DOUBLE PRECISION,
            salinity DOUBLE PRECISION,
            
            -- Derived parameters (will be calculated if GSW available)
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

        -- Processing log table
        CREATE TABLE processing_log (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(500),
            status VARCHAR(50),
            error_message TEXT,
            measurements_count INTEGER,
            processing_time_seconds DOUBLE PRECISION,
            processed_at TIMESTAMP DEFAULT NOW()
        );

        -- Essential indexes for performance
        CREATE INDEX idx_metadata_platform ON enhanced_floats_metadata (platform_number);
        CREATE INDEX idx_metadata_date ON enhanced_floats_metadata (date);
        CREATE INDEX idx_metadata_location ON enhanced_floats_metadata USING GIST (location);
        CREATE INDEX idx_measurements_metadata ON enhanced_measurements (metadata_id);
        CREATE INDEX idx_measurements_pressure ON enhanced_measurements (pressure);
        CREATE INDEX idx_processing_log_status ON processing_log (status);
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(schema_sql))
            conn.commit()
        
        logger.info("Production database schema created")

    def check_file_validity(self, file_path):
        """Check if NetCDF file is valid before processing"""
        try:
            # Quick validity check
            with xr.open_dataset(file_path) as ds:
                # Check if basic structure exists
                if not ds.variables:
                    return False, "No variables found"
                return True, "Valid"
        except Exception as e:
            return False, str(e)

    def safe_extract_scalar(self, data, default_value=None):
        """Safely extract scalar values from various data types"""
        try:
            if data is None:
                return default_value
            
            if np.isscalar(data):
                if isinstance(data, (int, float, np.number)):
                    if np.isnan(data):
                        return default_value
                    return float(data)
                else:
                    return str(data)
            
            # Handle arrays
            if hasattr(data, 'values'):
                values = data.values
            else:
                values = np.asarray(data)
            
            # Flatten and find first valid value
            flat_values = values.flatten()
            
            # Handle masked arrays
            if hasattr(flat_values, 'mask'):
                valid_values = flat_values[~flat_values.mask]
                if len(valid_values) == 0:
                    return default_value
                first_valid = valid_values[0]
            else:
                # Handle regular arrays - find first non-NaN value
                if flat_values.dtype.kind in 'fc':  # float or complex
                    finite_mask = np.isfinite(flat_values)
                    if not np.any(finite_mask):
                        return default_value
                    first_valid = flat_values[finite_mask][0]
                else:
                    # Non-numeric data
                    if len(flat_values) == 0:
                        return default_value
                    first_valid = flat_values[0]
            
            # Convert to appropriate type
            if isinstance(first_valid, (int, float, np.number)):
                return float(first_valid)
            else:
                return str(first_valid)
                
        except Exception as e:
            logger.debug(f"Error extracting scalar: {e}")
            return default_value

    def safe_convert_to_float(self, value):
        """Safely convert any value to float, returning None if not possible"""
        try:
            if value is None:
                return None
            
            # Handle string representations
            if isinstance(value, str):
                value = value.strip()
                if value.lower() in ['', 'nan', 'none', 'null']:
                    return None
                return float(value)
            
            # Handle numeric types
            if isinstance(value, (int, float, np.number)):
                if np.isnan(value) or np.isinf(value):
                    return None
                return float(value)
            
            # Try direct conversion
            return float(value)
            
        except (ValueError, TypeError, OverflowError):
            return None

    def clean_string_field(self, value, max_length=None):
        """Clean and truncate string fields"""
        if value is None:
            return "Unknown"
        
        try:
            # Convert to string
            str_value = str(value)
            
            # Remove common problematic characters
            str_value = str_value.replace("b'", "").replace("'", "").replace('"', '')
            str_value = re.sub(r'[\[\]()]', '', str_value)
            
            # Strip whitespace
            str_value = str_value.strip()
            
            # Handle empty strings
            if not str_value or str_value.lower() in ['nan', 'none', 'null', '']:
                return "Unknown"
            
            # Truncate if needed
            if max_length and len(str_value) > max_length:
                str_value = str_value[:max_length-3] + "..."
            
            return str_value
            
        except Exception:
            return "Unknown"

    def extract_basic_metadata(self, ds, file_path):
        """Extract basic metadata from dataset with robust error handling"""
        metadata = {}
        
        try:
            # Platform number - try multiple sources
            platform_vars = ['platform_number', 'PLATFORM_NUMBER']
            platform_value = None
            for var_name in platform_vars:
                if var_name in ds.variables:
                    platform_value = self.safe_extract_scalar(ds[var_name], "Unknown")
                    break
            if not platform_value:
                platform_value = ds.attrs.get('PLATFORM_NUMBER', 'Unknown')
            metadata['platform_number'] = self.clean_string_field(platform_value, 50)
            
            # Cycle number - with safe conversion
            cycle_vars = ['cycle_number', 'CYCLE_NUMBER']
            cycle_value = 0
            for var_name in cycle_vars:
                if var_name in ds.variables:
                    raw_cycle = self.safe_extract_scalar(ds[var_name], 0)
                    cycle_float = self.safe_convert_to_float(raw_cycle)
                    cycle_value = int(cycle_float) if cycle_float is not None else 0
                    break
            metadata['cycle_number'] = cycle_value
            
            # Date - FIXED: Handle string/numeric conversion properly
            date_vars = ['juld', 'JULD']
            date_value = datetime.now()
            
            for var_name in date_vars:
                if var_name in ds.variables:
                    raw_juld = self.safe_extract_scalar(ds[var_name])
                    juld_float = self.safe_convert_to_float(raw_juld)
                    
                    # Only process if we have a valid numeric value
                    if juld_float is not None and juld_float > 0:
                        try:
                            reference_date = pd.to_datetime('1950-01-01')
                            date_value = reference_date + pd.Timedelta(days=juld_float)
                            break
                        except Exception as e:
                            logger.debug(f"Date conversion error for {juld_float}: {e}")
                            continue
            
            metadata['date'] = date_value
            
            # Coordinates - with safe conversion
            lat_vars = ['latitude', 'LATITUDE']
            lon_vars = ['longitude', 'LONGITUDE']
            
            latitude = 0.0
            for var_name in lat_vars:
                if var_name in ds.variables:
                    raw_lat = self.safe_extract_scalar(ds[var_name], 0.0)
                    lat_float = self.safe_convert_to_float(raw_lat)
                    latitude = lat_float if lat_float is not None else 0.0
                    break
            
            longitude = 0.0
            for var_name in lon_vars:
                if var_name in ds.variables:
                    raw_lon = self.safe_extract_scalar(ds[var_name], 0.0)
                    lon_float = self.safe_convert_to_float(raw_lon)
                    longitude = lon_float if lon_float is not None else 0.0
                    break
            
            metadata['latitude'] = latitude
            metadata['longitude'] = longitude
            
            # Additional metadata
            metadata['project_name'] = self.clean_string_field(ds.attrs.get('PROJECT_NAME'), 200)
            metadata['pi_name'] = self.clean_string_field(ds.attrs.get('PI_NAME'), 200)
            metadata['institution'] = self.clean_string_field(ds.attrs.get('institution'), 200)
            metadata['wmo_inst_type'] = self.clean_string_field(ds.attrs.get('WMO_INST_TYPE'), 50)
            metadata['data_mode'] = self.clean_string_field(ds.attrs.get('DATA_MODE'), 50)
            metadata['source_file'] = os.path.basename(file_path)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {e}")
            traceback.print_exc()
            raise

    def extract_measurements(self, ds):
        """Extract measurement data from dataset with robust error handling"""
        measurements = []
        
        try:
            # Find pressure variable
            pressure_vars = ['pres_adjusted', 'PRES_ADJUSTED', 'pres', 'PRES']
            pressure_var = None
            for var_name in pressure_vars:
                if var_name in ds.variables:
                    pressure_var = ds[var_name]
                    break
            
            if pressure_var is None:
                logger.warning("No pressure variable found")
                return measurements
            
            # Find temperature and salinity variables
            temp_vars = ['temp_adjusted', 'TEMP_ADJUSTED', 'temp', 'TEMP']
            sal_vars = ['psal_adjusted', 'PSAL_ADJUSTED', 'psal', 'PSAL']
            
            temp_var = None
            for var_name in temp_vars:
                if var_name in ds.variables:
                    temp_var = ds[var_name]
                    break
            
            sal_var = None
            for var_name in sal_vars:
                if var_name in ds.variables:
                    sal_var = ds[var_name]
                    break
            
            # Get data arrays
            pressure_data = pressure_var.values
            temp_data = temp_var.values if temp_var is not None else None
            sal_data = sal_var.values if sal_var is not None else None
            
            # Handle different data structures
            if pressure_data.ndim == 1:
                # Single profile
                for i, pressure in enumerate(pressure_data):
                    pressure_float = self.safe_convert_to_float(pressure)
                    if pressure_float is None or pressure_float <= 0:
                        continue
                    
                    measurement = {
                        'pressure': pressure_float,
                        'temperature': None,
                        'salinity': None
                    }
                    
                    if temp_data is not None and i < len(temp_data):
                        temp_float = self.safe_convert_to_float(temp_data[i])
                        if temp_float is not None:
                            measurement['temperature'] = temp_float
                    
                    if sal_data is not None and i < len(sal_data):
                        sal_float = self.safe_convert_to_float(sal_data[i])
                        if sal_float is not None:
                            measurement['salinity'] = sal_float
                    
                    measurements.append(measurement)
            
            elif pressure_data.ndim == 2:
                # Multiple profiles
                n_profiles, n_levels = pressure_data.shape
                
                for prof_idx in range(n_profiles):
                    for level_idx in range(n_levels):
                        pressure = pressure_data[prof_idx, level_idx]
                        pressure_float = self.safe_convert_to_float(pressure)
                        
                        if pressure_float is None or pressure_float <= 0:
                            continue
                        
                        measurement = {
                            'pressure': pressure_float,
                            'temperature': None,
                            'salinity': None
                        }
                        
                        if temp_data is not None:
                            temp_float = self.safe_convert_to_float(temp_data[prof_idx, level_idx])
                            if temp_float is not None:
                                measurement['temperature'] = temp_float
                        
                        if sal_data is not None:
                            sal_float = self.safe_convert_to_float(sal_data[prof_idx, level_idx])
                            if sal_float is not None:
                                measurement['salinity'] = sal_float
                        
                        measurements.append(measurement)
            
            logger.debug(f"Extracted {len(measurements)} measurements")
            return measurements
            
        except Exception as e:
            logger.error(f"Error extracting measurements: {e}")
            traceback.print_exc()
            return []

    def calculate_profile_stats(self, measurements):
        """Calculate basic profile statistics"""
        stats = {}
        
        if not measurements:
            return stats
        
        try:
            pressures = [m['pressure'] for m in measurements]
            temperatures = [m['temperature'] for m in measurements if m['temperature'] is not None]
            salinities = [m['salinity'] for m in measurements if m['salinity'] is not None]
            
            stats['max_pressure'] = max(pressures) if pressures else None
            stats['min_pressure'] = min(pressures) if pressures else None
            stats['surface_temperature'] = temperatures[0] if temperatures else None
            stats['surface_salinity'] = salinities[0] if salinities else None
            stats['bottom_temperature'] = temperatures[-1] if temperatures else None
            stats['bottom_salinity'] = salinities[-1] if salinities else None
            
        except Exception as e:
            logger.debug(f"Error calculating profile stats: {e}")
        
        return stats

    @contextmanager
    def database_transaction(self):
        """Context manager for database transactions"""
        conn = self.engine.connect()
        trans = conn.begin()
        try:
            yield conn
            trans.commit()
        except Exception:
            trans.rollback()
            raise
        finally:
            conn.close()

    def process_single_file(self, file_path):
        """Process a single NetCDF file with comprehensive error handling"""
        start_time = time.time()
        filename = os.path.basename(file_path)
        
        # Skip if already processed
        if filename in self.processed_files:
            logger.debug(f"Skipping already processed: {filename}")
            self.skip_count += 1
            return True
        
        logger.info(f"Processing: {filename}")
        
        try:
            # Check file validity first
            is_valid, error_msg = self.check_file_validity(file_path)
            if not is_valid:
                self.log_processing_result(filename, "INVALID", error_msg, 0, time.time() - start_time)
                self.failed_files.append({'file': filename, 'error': error_msg})
                self.failed_count += 1  # ADDED: Increment failed counter
                return False
            
            # Open dataset
            with xr.open_dataset(file_path) as ds:
                # Extract metadata
                metadata = self.extract_basic_metadata(ds, file_path)
                
                # Extract measurements
                measurements = self.extract_measurements(ds)
                
                if not measurements:
                    error_msg = "No valid measurements found"
                    self.log_processing_result(filename, "NO_DATA", error_msg, 0, time.time() - start_time)
                    self.failed_files.append({'file': filename, 'error': error_msg})
                    self.failed_count += 1  # ADDED: Increment failed counter
                    return False
                
                # Calculate profile statistics
                profile_stats = self.calculate_profile_stats(measurements)
                metadata.update(profile_stats)
                metadata['n_measurements'] = len(measurements)
                
                # Insert into database
                with self.database_transaction() as conn:
                    # Insert metadata
                    result = conn.execute(text("""
                        INSERT INTO enhanced_floats_metadata 
                        (platform_number, cycle_number, date, latitude, longitude, location,
                         project_name, pi_name, institution, wmo_inst_type, data_mode, source_file,
                         max_pressure, min_pressure, surface_temperature, surface_salinity,
                         bottom_temperature, bottom_salinity, n_measurements)
                        VALUES (:platform_number, :cycle_number, :date, :latitude, :longitude, 
                               ST_GeogFromText(:location), :project_name, :pi_name, :institution,
                               :wmo_inst_type, :data_mode, :source_file, :max_pressure, :min_pressure,
                               :surface_temperature, :surface_salinity, :bottom_temperature, 
                               :bottom_salinity, :n_measurements)
                        RETURNING id
                    """), {
                        **metadata,
                        'location': f'POINT({metadata["longitude"]} {metadata["latitude"]})'
                    })
                    
                    metadata_id = result.scalar()
                    
                    # Insert measurements in batches
                    for i in range(0, len(measurements), self.batch_size):
                        batch = measurements[i:i + self.batch_size]
                        for measurement in batch:
                            measurement['metadata_id'] = metadata_id
                        
                        df = pd.DataFrame(batch)
                        df.to_sql('enhanced_measurements', conn, if_exists='append', 
                                index=False, method='multi')
                
                # Log success
                processing_time = time.time() - start_time
                self.log_processing_result(filename, "SUCCESS", None, len(measurements), processing_time)
                self.processed_files.add(filename)
                self.success_count += 1
                
                logger.info(f"Completed: {filename} ({len(measurements)} measurements, {processing_time:.1f}s)")
                return True
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)[:500]  # Truncate long error messages
            
            logger.error(f"Error processing {filename}: {error_msg}")
            traceback.print_exc()
            self.log_processing_result(filename, "ERROR", error_msg, 0, processing_time)
            self.failed_files.append({'file': filename, 'error': error_msg})
            self.failed_count += 1  # ADDED: Increment failed counter
            return False

    def log_processing_result(self, filename, status, error_message, count, processing_time):
        """Log processing result to database"""
        try:
            with self.database_transaction() as conn:
                conn.execute(text("""
                    INSERT INTO processing_log 
                    (filename, status, error_message, measurements_count, processing_time_seconds)
                    VALUES (:filename, :status, :error_message, :count, :processing_time)
                """), {
                    'filename': filename,
                    'status': status,
                    'error_message': error_message,
                    'count': count,
                    'processing_time': processing_time
                })
        except Exception as e:
            logger.error(f"Could not log processing result: {e}")

    def process_files_sequentially(self, file_list):
        """Process files one by one to avoid resource issues"""
        total_files = len(file_list)
        
        logger.info(f"Starting sequential processing of {total_files} files")
        logger.info(f"Previously processed: {len(self.processed_files)}, failed: {self.failed_count}")  # CHANGED: Use failed_count
        
        # Filter out already processed files
        remaining_files = [f for f in file_list if os.path.basename(f) not in self.processed_files]
        
        logger.info(f"Remaining to process: {len(remaining_files)} files")
        
        progress_bar = tqdm(remaining_files, desc="Processing files")
        
        for i, file_path in enumerate(progress_bar):
            # Update progress bar
            progress_bar.set_postfix({
                'success': self.success_count,
                'failed': self.failed_count,  # CHANGED: Use failed_count
                'skipped': self.skip_count
            })
            
            # Process file
            self.process_single_file(file_path)
            
            # Save progress every 50 files
            if (i + 1) % 50 == 0:
                self.save_progress()
                # Force garbage collection to free memory
                gc.collect()
                
                # Check memory usage
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 85:
                    logger.warning(f"High memory usage: {memory_percent}%. Pausing for cleanup.")
                    time.sleep(2)
                    gc.collect()
        
        # Final progress save
        self.save_progress()
        
        return {
            'success': self.success_count,
            'failed': self.failed_count,  # CHANGED: Use failed_count
            'skipped': self.skip_count,
            'total': total_files
        }

    def generate_final_report(self, results):
        """Generate comprehensive processing report"""
        logger.info("=" * 60)
        logger.info("FINAL PROCESSING REPORT")
        logger.info("=" * 60)
        
        logger.info(f"Total files found: {results['total']}")
        logger.info(f"Successfully processed: {results['success']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"Skipped (already processed): {results['skipped']}")
        
        processed_attempts = results['total'] - results['skipped']
        success_rate = (results['success'] / processed_attempts) * 100 if processed_attempts > 0 else 0
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        # Database summary
        try:
            with self.engine.connect() as conn:
                metadata_count = conn.execute(text("SELECT COUNT(*) FROM enhanced_floats_metadata")).scalar()
                measurements_count = conn.execute(text("SELECT COUNT(*) FROM enhanced_measurements")).scalar()
                
                logger.info(f"Database summary:")
                logger.info(f"  - Float profiles: {metadata_count:,}")
                logger.info(f"  - Measurements: {measurements_count:,}")
        except Exception as e:
            logger.error(f"Could not get database summary: {e}")
        
        # Top failure reasons
        if self.failed_files:
            logger.info(f"\nTop failure reasons:")
            from collections import Counter
            error_types = Counter([f['error'][:100] for f in self.failed_files[-100:]])  # Last 100 failures
            for error, count in error_types.most_common(5):
                logger.info(f"  - {error}: {count} files")

def main():
    """Main processing function"""
    logger.info("ARGO Production NetCDF Processor Starting (FIXED VERSION)")
    logger.info("=" * 60)
    
    try:
        # Database connection with production settings
        engine = create_engine(
            os.getenv('DATABASE_URL'),
            pool_size=5,  # Conservative pool size
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections every hour
            echo=False
        )
        
        # Initialize processor
        processor = ProductionArgoProcessor(engine)
        
        # Create schema
        logger.info("Creating database schema...")
        processor.create_robust_schema()
        
        # Find all NetCDF files
        search_patterns = [
            "../../data/indian_ocean/raw/*.nc",
            "../../data/indian_ocean/raw/**/*.nc",
            "../../data/*.nc"
        ]
        
        all_files = []
        for pattern in search_patterns:
            files = glob.glob(pattern, recursive=True)
            all_files.extend(files)
        
        # Remove duplicates and sort
        unique_files = list(set(all_files))
        unique_files.sort()
        
        logger.info(f"Found {len(unique_files)} unique NetCDF files")
        
        if not unique_files:
            logger.error("No NetCDF files found! Check your data directory paths.")
            return
        
        # Process all files
        results = processor.process_files_sequentially(unique_files)
        
        # Generate final report
        processor.generate_final_report(results)
        
        logger.info("Processing completed!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()