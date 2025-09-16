# production_argo_processor.py
"""
Production-grade ARGO processor designed for 980+ files
Uses PostgreSQL COPY operations for maximum performance
"""

import xarray as xr
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import os
import glob
from tqdm import tqdm
import logging
from pathlib import Path
import tempfile
import csv
from typing import List, Dict, Tuple, Optional
import time
import psycopg2
from io import StringIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionArgoProcessor:
    def __init__(self, db_engine, batch_size=5000):
        self.engine = db_engine
        self.batch_size = batch_size
        self.stats = {
            'files_processed': 0,
            'profiles_inserted': 0,
            'measurements_inserted': 0,
            'processing_time': 0,
            'files_failed': 0
        }
        
    def create_production_schema(self):
        """Create production-optimized schema"""
        schema_sql = """
        -- Drop existing tables
        DROP TABLE IF EXISTS argo_measurements CASCADE;
        DROP TABLE IF EXISTS argo_profiles CASCADE;
        DROP TABLE IF EXISTS data_processing_log CASCADE;

        -- Optimized profiles table
        CREATE TABLE argo_profiles (
            id SERIAL PRIMARY KEY,
            platform_number VARCHAR(20) NOT NULL,
            cycle_number INTEGER,
            profile_date TIMESTAMP NOT NULL,
            latitude DOUBLE PRECISION NOT NULL,
            longitude DOUBLE PRECISION NOT NULL,
            
            -- Profile characteristics
            max_pressure DOUBLE PRECISION,
            n_levels INTEGER DEFAULT 0,
            surface_temp DOUBLE PRECISION,
            surface_salinity DOUBLE PRECISION,
            max_temp DOUBLE PRECISION,
            min_temp DOUBLE PRECISION,
            temp_at_1000m DOUBLE PRECISION,
            mixed_layer_depth DOUBLE PRECISION,
            
            -- Metadata
            source_file VARCHAR(255) NOT NULL,
            processed_at TIMESTAMP DEFAULT NOW(),
            
            UNIQUE(platform_number, cycle_number, profile_date)
        );

        -- Measurements table
        CREATE TABLE argo_measurements (
            id BIGSERIAL PRIMARY KEY,
            profile_id INTEGER REFERENCES argo_profiles(id) ON DELETE CASCADE,
            pressure DOUBLE PRECISION NOT NULL,
            depth DOUBLE PRECISION,
            temperature DOUBLE PRECISION,
            salinity DOUBLE PRECISION
        );

        -- Processing log
        CREATE TABLE data_processing_log (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) UNIQUE NOT NULL,
            processing_status VARCHAR(20),
            profiles_count INTEGER DEFAULT 0,
            measurements_count INTEGER DEFAULT 0,
            error_message TEXT,
            processing_time_seconds DOUBLE PRECISION,
            processed_at TIMESTAMP DEFAULT NOW()
        );

        -- Performance indexes
        CREATE INDEX idx_profiles_platform ON argo_profiles (platform_number);
        CREATE INDEX idx_profiles_date ON argo_profiles (profile_date);
        CREATE INDEX idx_profiles_coords ON argo_profiles (latitude, longitude);
        CREATE INDEX idx_measurements_profile ON argo_measurements (profile_id);
        CREATE INDEX idx_measurements_pressure ON argo_measurements (pressure);
        CREATE INDEX idx_measurements_temp_sal ON argo_measurements (temperature, salinity);
        CREATE INDEX idx_log_status ON data_processing_log (processing_status);
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(schema_sql))
            conn.commit()
        logger.info("Production schema created successfully")

    def process_file_to_dataframes(self, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process NetCDF file and return DataFrames ready for bulk insert"""
        
        filename = os.path.basename(file_path)
        profiles_data = []
        measurements_data = []
        
        with xr.open_dataset(file_path) as ds:
            dims = dict(ds.sizes)
            n_prof = dims.get('N_PROF', 0)
            n_levels = dims.get('N_LEVELS', 0)
            
            for prof_idx in range(n_prof):
                try:
                    # Extract profile metadata
                    platform_num = self._safe_extract_string(ds, 'PLATFORM_NUMBER', prof_idx)
                    cycle_num = self._safe_extract_number(ds, 'CYCLE_NUMBER', prof_idx, 0)
                    
                    # Date conversion
                    juld = self._safe_extract_number(ds, 'JULD', prof_idx)
                    profile_date = self._convert_juld_to_datetime(juld)
                    
                    latitude = self._safe_extract_number(ds, 'LATITUDE', prof_idx, 0.0)
                    longitude = self._safe_extract_number(ds, 'LONGITUDE', prof_idx, 0.0)
                    
                    # Skip invalid coordinates
                    if abs(latitude) > 90 or abs(longitude) > 180 or (latitude == 0 and longitude == 0):
                        continue
                    
                    # Extract all measurements for this profile
                    profile_measurements = []
                    for level_idx in range(n_levels):
                        pressure = self._safe_extract_number(ds, 'PRES', (prof_idx, level_idx))
                        if pressure is None or pressure <= 0:
                            continue
                        
                        temp = self._safe_extract_number(ds, 'TEMP', (prof_idx, level_idx))
                        temp_adj = self._safe_extract_number(ds, 'TEMP_ADJUSTED', (prof_idx, level_idx))
                        temperature = temp_adj if temp_adj is not None else temp
                        
                        psal = self._safe_extract_number(ds, 'PSAL', (prof_idx, level_idx))
                        psal_adj = self._safe_extract_number(ds, 'PSAL_ADJUSTED', (prof_idx, level_idx))
                        salinity = psal_adj if psal_adj is not None else psal
                        
                        if temperature is not None or salinity is not None:
                            profile_measurements.append({
                                'pressure': pressure,
                                'depth': pressure * 1.0194,
                                'temperature': temperature,
                                'salinity': salinity
                            })
                    
                    if len(profile_measurements) < 3:
                        continue
                    
                    # Calculate statistics
                    temperatures = [m['temperature'] for m in profile_measurements if m['temperature'] is not None]
                    salinities = [m['salinity'] for m in profile_measurements if m['salinity'] is not None]
                    pressures = [m['pressure'] for m in profile_measurements]
                    
                    profile_record = {
                        'platform_number': platform_num,
                        'cycle_number': int(cycle_num) if cycle_num else None,
                        'profile_date': profile_date,
                        'latitude': latitude,
                        'longitude': longitude,
                        'max_pressure': max(pressures),
                        'n_levels': len(profile_measurements),
                        'surface_temp': temperatures[0] if temperatures else None,
                        'surface_salinity': salinities[0] if salinities else None,
                        'max_temp': max(temperatures) if temperatures else None,
                        'min_temp': min(temperatures) if temperatures else None,
                        'temp_at_1000m': self._find_temp_at_depth(profile_measurements, 1000),
                        'mixed_layer_depth': self._calculate_mld_simple(profile_measurements),
                        'source_file': filename
                    }
                    
                    profiles_data.append(profile_record)
                    
                    # Add measurements with temporary profile index
                    for measurement in profile_measurements:
                        measurement['profile_temp_idx'] = len(profiles_data) - 1
                        measurements_data.append(measurement)
                
                except Exception as e:
                    logger.warning(f"Error processing profile {prof_idx} in {filename}: {e}")
                    continue
        
        profiles_df = pd.DataFrame(profiles_data)
        measurements_df = pd.DataFrame(measurements_data)
        
        return profiles_df, measurements_df

    def bulk_insert_profiles(self, profiles_df: pd.DataFrame) -> List[int]:
        """Bulk insert profiles using PostgreSQL COPY"""
        
        if profiles_df.empty:
            return []
        
        # Get raw psycopg2 connection
        raw_conn = self.engine.raw_connection()
        cursor = raw_conn.cursor()
        
        try:
            # Prepare data for COPY
            profiles_df_clean = profiles_df.copy()
            profiles_df_clean = profiles_df_clean.fillna('')  # Replace NaN with empty string
            
            # Create CSV buffer
            output = StringIO()
            profiles_df_clean.to_csv(output, sep='\t', header=False, index=False, 
                                   date_format='%Y-%m-%d %H:%M:%S')
            output.seek(0)
            
            # Use COPY to insert data
            cursor.copy_from(
                output, 
                'argo_profiles',
                columns=list(profiles_df_clean.columns),
                sep='\t',
                null=''
            )
            
            # Get the inserted IDs
            cursor.execute("""
                SELECT id FROM argo_profiles 
                WHERE source_file = %s 
                ORDER BY id DESC 
                LIMIT %s
            """, (profiles_df.iloc[0]['source_file'], len(profiles_df)))
            
            profile_ids = [row[0] for row in cursor.fetchall()]
            profile_ids.reverse()  # Reverse to match insertion order
            
            raw_conn.commit()
            logger.info(f"Bulk inserted {len(profile_ids)} profiles")
            
            return profile_ids
            
        except Exception as e:
            raw_conn.rollback()
            logger.error(f"Error in bulk profile insert: {e}")
            raise
        finally:
            cursor.close()
            raw_conn.close()

    def bulk_insert_measurements(self, measurements_df: pd.DataFrame, profile_ids: List[int]) -> int:
        """Bulk insert measurements using PostgreSQL COPY"""
        
        if measurements_df.empty or not profile_ids:
            return 0
        
        # Map temporary profile indices to actual profile IDs
        id_mapping = {i: profile_ids[i] for i in range(len(profile_ids))}
        
        # Replace temp indices with actual profile IDs
        measurements_df['profile_id'] = measurements_df['profile_temp_idx'].map(id_mapping)
        
        # Remove rows where mapping failed and temporary column
        measurements_df = measurements_df.dropna(subset=['profile_id'])
        measurements_df = measurements_df.drop('profile_temp_idx', axis=1)
        measurements_df['profile_id'] = measurements_df['profile_id'].astype(int)
        
        if measurements_df.empty:
            return 0
        
        # Get raw psycopg2 connection
        raw_conn = self.engine.raw_connection()
        cursor = raw_conn.cursor()
        
        try:
            # Prepare data for COPY
            measurements_clean = measurements_df.copy()
            measurements_clean = measurements_clean.fillna('')
            
            # Reorder columns to match table structure
            column_order = ['profile_id', 'pressure', 'depth', 'temperature', 'salinity']
            measurements_clean = measurements_clean[column_order]
            
            # Create CSV buffer
            output = StringIO()
            measurements_clean.to_csv(output, sep='\t', header=False, index=False)
            output.seek(0)
            
            # Use COPY to insert data
            cursor.copy_from(
                output,
                'argo_measurements',
                columns=column_order,
                sep='\t',
                null=''
            )
            
            raw_conn.commit()
            
            measurements_count = len(measurements_clean)
            logger.info(f"Bulk inserted {measurements_count} measurements")
            
            return measurements_count
            
        except Exception as e:
            raw_conn.rollback()
            logger.error(f"Error in bulk measurements insert: {e}")
            raise
        finally:
            cursor.close()
            raw_conn.close()

    def process_single_file(self, file_path: str) -> Dict:
        """Process a single file with bulk operations"""
        
        filename = os.path.basename(file_path)
        start_time = time.time()
        
        try:
            # Check if already processed
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT processing_status FROM data_processing_log 
                    WHERE filename = :filename AND processing_status = 'SUCCESS'
                """), {'filename': filename})
                
                if result.fetchone():
                    return {'status': 'SKIPPED', 'message': 'Already processed'}
            
            # Process file to DataFrames
            profiles_df, measurements_df = self.process_file_to_dataframes(file_path)
            
            if profiles_df.empty:
                return {'status': 'SKIPPED', 'message': 'No valid profiles found'}
            
            # Bulk insert profiles
            profile_ids = self.bulk_insert_profiles(profiles_df)
            
            # Bulk insert measurements
            measurements_count = self.bulk_insert_measurements(measurements_df, profile_ids)
            
            processing_time = time.time() - start_time
            
            # Log success
            self._log_result(filename, 'SUCCESS', len(profile_ids), measurements_count, None, processing_time)
            
            return {
                'status': 'SUCCESS',
                'profiles_inserted': len(profile_ids),
                'measurements_inserted': measurements_count,
                'processing_time': processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            self._log_result(filename, 'FAILED', 0, 0, error_msg, processing_time)
            
            return {
                'status': 'FAILED',
                'error': error_msg,
                'processing_time': processing_time
            }

    def process_all_files(self, data_directory: str):
        """Process all NetCDF files in directory"""
        
        logger.info(f"Starting production processing of: {data_directory}")
        start_time = time.time()
        
        # Find all NetCDF files
        all_files = []
        for pattern in [f"{data_directory}/**/*.nc", f"{data_directory}/*.nc"]:
            all_files.extend(glob.glob(pattern, recursive=True))
        
        all_files = list(set(all_files))  # Remove duplicates
        logger.info(f"Found {len(all_files)} unique NetCDF files")
        
        for file_path in tqdm(all_files, desc="Processing files"):
            try:
                result = self.process_single_file(file_path)
                
                if result['status'] == 'SUCCESS':
                    self.stats['files_processed'] += 1
                    self.stats['profiles_inserted'] += result['profiles_inserted']
                    self.stats['measurements_inserted'] += result['measurements_inserted']
                    
                elif result['status'] == 'FAILED':
                    self.stats['files_failed'] += 1
                    logger.error(f"Failed to process {os.path.basename(file_path)}: {result['error']}")
                
            except Exception as e:
                self.stats['files_failed'] += 1
                logger.error(f"Exception processing {file_path}: {e}")
        
        self.stats['processing_time'] = time.time() - start_time
        return self.stats

    def _log_result(self, filename, status, profiles_count, measurements_count, error_msg, processing_time):
        """Log processing result"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO data_processing_log 
                    (filename, processing_status, profiles_count, measurements_count, 
                     error_message, processing_time_seconds)
                    VALUES (:filename, :status, :profiles, :measurements, :error, :time)
                    ON CONFLICT (filename) DO UPDATE SET
                        processing_status = EXCLUDED.processing_status,
                        profiles_count = EXCLUDED.profiles_count,
                        measurements_count = EXCLUDED.measurements_count,
                        error_message = EXCLUDED.error_message,
                        processing_time_seconds = EXCLUDED.processing_time_seconds,
                        processed_at = NOW()
                """), {
                    'filename': filename,
                    'status': status,
                    'profiles': profiles_count,
                    'measurements': measurements_count,
                    'error': error_msg,
                    'time': processing_time
                })
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging result: {e}")

    def _find_temp_at_depth(self, measurements: List[Dict], target_depth: float) -> Optional[float]:
        """Find temperature closest to target depth"""
        best_temp = None
        min_diff = float('inf')
        
        for m in measurements:
            if m['temperature'] is not None and m['pressure']:
                depth_diff = abs(m['pressure'] - target_depth)
                if depth_diff < min_diff:
                    min_diff = depth_diff
                    best_temp = m['temperature']
        
        return best_temp

    def _calculate_mld_simple(self, measurements: List[Dict]) -> Optional[float]:
        """Calculate mixed layer depth"""
        if len(measurements) < 3:
            return None
        
        sorted_measurements = sorted(measurements, key=lambda x: x['pressure'])
        
        surface_temp = None
        for m in sorted_measurements[:3]:
            if m['temperature'] is not None:
                surface_temp = m['temperature']
                break
        
        if surface_temp is None:
            return None
        
        for m in sorted_measurements[1:]:
            if m['temperature'] is not None:
                temp_diff = abs(m['temperature'] - surface_temp)
                if temp_diff >= 0.2:
                    return m['pressure'] * 1.0194
        
        return None

    def _safe_extract_string(self, ds, var_name, index, default="Unknown"):
        """Extract string safely"""
        try:
            if var_name not in ds.variables:
                return default
            
            value = ds[var_name].values[index]
            
            if hasattr(value, 'decode'):
                return value.decode('utf-8').strip()
            elif isinstance(value, bytes):
                return value.decode('utf-8').strip()
            return str(value).strip()
        except:
            return default

    def _safe_extract_number(self, ds, var_name, index, default=None):
        """Extract number safely"""
        try:
            if var_name not in ds.variables:
                return default
            
            if isinstance(index, tuple):
                value = ds[var_name].values[index]
            else:
                value = ds[var_name].values[index]
            
            if isinstance(value, (np.ndarray, list)):
                value = value.item() if hasattr(value, 'item') else value[0]
            
            if value in [99999.0, 999999.0, -999.0]:
                return default
            
            if np.isnan(value) or np.isinf(value):
                return default
            
            return float(value)
        except:
            return default

    def _convert_juld_to_datetime(self, juld_value):
        """Convert Julian day to datetime"""
        if juld_value is None or np.isnan(juld_value):
            return datetime.now()
        try:
            reference_date = datetime(1950, 1, 1)
            return reference_date + timedelta(days=juld_value)
        except:
            return datetime.now()

    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        logger.info("=" * 60)
        logger.info("PRODUCTION ARGO PROCESSING REPORT")
        logger.info("=" * 60)
        
        processing_rate = self.stats['files_processed'] / (self.stats['processing_time'] / 60) if self.stats['processing_time'] > 0 else 0
        
        logger.info(f"Files Processed: {self.stats['files_processed']}")
        logger.info(f"Files Failed: {self.stats['files_failed']}")
        logger.info(f"Profiles Inserted: {self.stats['profiles_inserted']:,}")
        logger.info(f"Measurements Inserted: {self.stats['measurements_inserted']:,}")
        logger.info(f"Total Processing Time: {self.stats['processing_time']:.2f} seconds")
        logger.info(f"Processing Rate: {processing_rate:.1f} files/minute")
        
        # Database verification
        try:
            with self.engine.connect() as conn:
                profile_count = conn.execute(text("SELECT COUNT(*) FROM argo_profiles")).scalar()
                measurement_count = conn.execute(text("SELECT COUNT(*) FROM argo_measurements")).scalar()
                
                logger.info(f"\nDatabase Verification:")
                logger.info(f"  - Total Profiles: {profile_count:,}")
                logger.info(f"  - Total Measurements: {measurement_count:,}")
                logger.info(f"  - Avg Measurements/Profile: {measurement_count/profile_count:.1f}" if profile_count > 0 else "  - No profiles found")
                
                # Platform statistics
                platform_stats = conn.execute(text("""
                    SELECT COUNT(DISTINCT platform_number) as unique_platforms,
                           MIN(profile_date) as earliest_date,
                           MAX(profile_date) as latest_date
                    FROM argo_profiles
                """)).fetchone()
                
                if platform_stats:
                    logger.info(f"  - Unique Platforms: {platform_stats[0]}")
                    logger.info(f"  - Date Range: {platform_stats[1]} to {platform_stats[2]}")
                
        except Exception as e:
            logger.error(f"Error in database verification: {e}")


# Usage example
if __name__ == "__main__":
    from sqlalchemy import create_engine
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Create processor
    engine = create_engine(
        os.getenv('DATABASE_URL'),
        pool_size=10,
        max_overflow=20
    )
    
    processor = ProductionArgoProcessor(engine, batch_size=1000)
    
    # Create schema
    processor.create_production_schema()
    
    # Process files
    stats = processor.process_all_files("../../data")
    
    # Generate report
    processor.generate_performance_report()