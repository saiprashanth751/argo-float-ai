# # fixed_argo_processor.py (comprehensive_argo_workflow.py)
# """
# Fixed ARGO data processing pipeline that addresses the identified issues
# """

# import os
# import sys
# import glob
# import json
# import time
# import logging
# import traceback
# from pathlib import Path
# from datetime import datetime
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import psutil
# import gc

# # Import required libraries with error handling
# try:
#     import xarray as xr
#     import netCDF4 as nc
#     from sqlalchemy import create_engine, text
#     REQUIRED_LIBS_AVAILABLE = True
# except ImportError as e:
#     print(f"Missing required libraries: {e}")
#     print("Install with: pip install xarray netcdf4 sqlalchemy psycopg2-binary")
#     REQUIRED_LIBS_AVAILABLE = False

# # Setup logging
# def setup_logging():
#     log_filename = f"argo_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_filename),
#             logging.StreamHandler(sys.stdout)
#         ]
#     )
#     return logging.getLogger(__name__)

# logger = setup_logging()

# class FixedArgoProcessor:
#     def __init__(self, data_dir="../../data/indian_ocean/raw"):
#         self.data_dir = data_dir
#         self.output_dir = "processing_output"
#         self.batch_size = 500  # Reduced batch size
#         self.processed_files = set()
#         self.failed_files = []
#         self.success_count = 0
#         self.create_directories()
        
#     def create_directories(self):
#         """Create necessary directories"""
#         dirs = [
#             self.output_dir,
#             f"{self.output_dir}/logs",
#             f"{self.output_dir}/diagnostics",
#             f"{self.output_dir}/quarantine"
#         ]
#         for dir_path in dirs:
#             os.makedirs(dir_path, exist_ok=True)
    
#     def check_prerequisites(self):
#         """Check if all prerequisites are met"""
#         issues = []
        
#         # Check libraries
#         if not REQUIRED_LIBS_AVAILABLE:
#             issues.append("Missing required Python libraries")
        
#         # Check data directory
#         if not os.path.exists(self.data_dir):
#             issues.append(f"Data directory not found: {self.data_dir}")
        
#         # Check for NetCDF files
#         nc_files = glob.glob(f"{self.data_dir}/*.nc")
#         if not nc_files:
#             issues.append("No NetCDF files found in data directory")
        
#         # Check database connectivity (optional)
#         try:
#             self.test_database_connection()
#         except Exception as e:
#             issues.append(f"Database connection issue: {e}")
        
#         if issues:
#             logger.error("Prerequisites not met:")
#             for issue in issues:
#                 logger.error(f"  - {issue}")
#             return False
        
#         logger.info("All prerequisites met")
#         return True
    
#     def test_database_connection(self):
#         """Test database connection"""
#         try:
#             engine = create_engine(
#                 'postgresql://argo_user:argo_password@localhost:5432/argo_data',
#                 pool_pre_ping=True
#             )
#             with engine.connect() as conn:
#                 conn.execute(text("SELECT 1"))
#             logger.info("Database connection successful")
#             return engine
#         except Exception as e:
#             logger.warning(f"Database connection failed: {e}")
#             logger.info("Will save data to CSV files instead")
#             return None
    
#     def create_schema_if_needed(self, engine):
#         """Create database schema if database is available"""
#         if not engine:
#             return
        
#         schema_sql = """
#         -- Create tables if they don't exist
#         CREATE TABLE IF NOT EXISTS enhanced_floats_metadata (
#             id SERIAL PRIMARY KEY,
#             platform_number VARCHAR(50) NOT NULL,
#             cycle_number INTEGER NOT NULL,
#             date TIMESTAMP NOT NULL,
#             latitude DOUBLE PRECISION NOT NULL,
#             longitude DOUBLE PRECISION NOT NULL,
#             project_name VARCHAR(200),
#             pi_name VARCHAR(200),
#             institution VARCHAR(200),
#             source_file VARCHAR(500),
#             max_pressure DOUBLE PRECISION,
#             min_pressure DOUBLE PRECISION,
#             surface_temperature DOUBLE PRECISION,
#             surface_salinity DOUBLE PRECISION,
#             n_measurements INTEGER DEFAULT 0,
#             processed_at TIMESTAMP DEFAULT NOW()
#         );

#         CREATE TABLE IF NOT EXISTS enhanced_measurements (
#             id SERIAL PRIMARY KEY,
#             metadata_id INTEGER REFERENCES enhanced_floats_metadata(id) ON DELETE CASCADE,
#             pressure DOUBLE PRECISION NOT NULL,
#             depth DOUBLE PRECISION,
#             temperature DOUBLE PRECISION,
#             salinity DOUBLE PRECISION
#         );

#         CREATE TABLE IF NOT EXISTS processing_log (
#             id SERIAL PRIMARY KEY,
#             filename VARCHAR(500),
#             status VARCHAR(50),
#             error_message TEXT,
#             measurements_count INTEGER,
#             processing_time_seconds DOUBLE PRECISION,
#             processed_at TIMESTAMP DEFAULT NOW()
#         );
        
#         -- Create indexes if they don't exist
#         CREATE INDEX IF NOT EXISTS idx_metadata_platform ON enhanced_floats_metadata (platform_number);
#         CREATE INDEX IF NOT EXISTS idx_metadata_date ON enhanced_floats_metadata (date);
#         CREATE INDEX IF NOT EXISTS idx_measurements_metadata ON enhanced_measurements (metadata_id);
#         """
        
#         try:
#             with engine.connect() as conn:
#                 conn.execute(text(schema_sql))
#                 conn.commit()
#             logger.info("Database schema ready")
#         except Exception as e:
#             logger.error(f"Schema creation failed: {e}")
#             raise
    
#     def safe_extract_value(self, data, default=None):
#         """Safely extract scalar values from NetCDF variables"""
#         try:
#             if data is None:
#                 return default
            
#             # Handle different data types
#             if np.isscalar(data):
#                 if isinstance(data, (np.floating, float)) and np.isnan(data):
#                     return default
#                 return data
            
#             # Handle arrays
#             if hasattr(data, 'values'):
#                 values = data.values
#             else:
#                 values = np.asarray(data)
            
#             # Flatten and get first valid value
#             flat_values = values.flatten()
            
#             # Handle masked arrays
#             if hasattr(flat_values, 'mask'):
#                 valid_values = flat_values[~flat_values.mask]
#                 if len(valid_values) == 0:
#                     return default
#                 return valid_values[0]
            
#             # Handle regular arrays with NaN
#             if flat_values.dtype.kind in ['f', 'c']:  # float or complex
#                 valid_mask = ~np.isnan(flat_values)
#                 valid_values = flat_values[valid_mask]
#                 if len(valid_values) == 0:
#                     return default
#                 return valid_values[0]
            
#             # For other types, just return first element
#             return flat_values[0] if len(flat_values) > 0 else default
            
#         except Exception as e:
#             logger.debug(f"Error extracting value: {e}")
#             return default
    
#     def clean_string_value(self, value, max_length=None):
#         """Clean string values for database storage"""
#         if value is None:
#             return "Unknown"
        
#         try:
#             # Convert to string
#             str_value = str(value)
            
#             # Handle byte strings
#             if str_value.startswith("b'") and str_value.endswith("'"):
#                 str_value = str_value[2:-1]
            
#             # Clean up
#             str_value = str_value.strip()
            
#             # Handle empty or invalid values
#             if not str_value or str_value.lower() in ['nan', 'none', 'null']:
#                 return "Unknown"
            
#             # Truncate if needed
#             if max_length and len(str_value) > max_length:
#                 str_value = str_value[:max_length-3] + "..."
            
#             return str_value
#         except:
#             return "Unknown"
    
#     def process_single_file(self, file_path):
#         """Process a single NetCDF file with improved error handling"""
#         start_time = time.time()
#         filename = os.path.basename(file_path)
        
#         try:
#             # Open dataset
#             with xr.open_dataset(file_path) as ds:
#                 # Extract metadata
#                 metadata = self.extract_metadata(ds, file_path)
                
#                 # Extract measurements
#                 measurements = self.extract_measurements(ds)
                
#                 if not measurements:
#                     logger.warning(f"No measurements found in {filename}")
#                     return False, "No measurements"
                
#                 # Add measurement count to metadata
#                 metadata['n_measurements'] = len(measurements)
                
#                 processing_time = time.time() - start_time
#                 logger.info(f"Processed {filename}: {len(measurements)} measurements ({processing_time:.1f}s)")
                
#                 return True, {'metadata': metadata, 'measurements': measurements}
                
#         except Exception as e:
#             processing_time = time.time() - start_time
#             error_msg = str(e)[:500]
#             logger.error(f"Error processing {filename}: {error_msg}")
#             return False, error_msg
    
#     def extract_metadata(self, ds, file_path):
#         """Extract metadata from dataset"""
#         metadata = {}
        
#         # Platform number
#         platform_vars = ['PLATFORM_NUMBER', 'platform_number']
#         platform_value = None
#         for var_name in platform_vars:
#             if var_name in ds.variables:
#                 platform_value = self.safe_extract_value(ds[var_name])
#                 break
#         metadata['platform_number'] = self.clean_string_value(platform_value, 50)
        
#         # Cycle number
#         cycle_vars = ['CYCLE_NUMBER', 'cycle_number']
#         cycle_value = 0
#         for var_name in cycle_vars:
#             if var_name in ds.variables:
#                 cycle_value = self.safe_extract_value(ds[var_name], 0)
#                 break
#         metadata['cycle_number'] = int(float(cycle_value)) if cycle_value else 0
        
#         # Date
#         date_vars = ['JULD', 'juld']
#         date_value = datetime.now()
#         for var_name in date_vars:
#             if var_name in ds.variables:
#                 juld_val = self.safe_extract_value(ds[var_name])
#                 if juld_val and juld_val > 0:
#                     try:
#                         reference_date = pd.to_datetime('1950-01-01')
#                         date_value = reference_date + pd.Timedelta(days=float(juld_val))
#                         break
#                     except:
#                         continue
#         metadata['date'] = date_value
        
#         # Coordinates
#         lat_vars = ['LATITUDE', 'latitude']
#         lon_vars = ['LONGITUDE', 'longitude']
        
#         latitude = 0.0
#         for var_name in lat_vars:
#             if var_name in ds.variables:
#                 latitude = self.safe_extract_value(ds[var_name], 0.0)
#                 break
        
#         longitude = 0.0
#         for var_name in lon_vars:
#             if var_name in ds.variables:
#                 longitude = self.safe_extract_value(ds[var_name], 0.0)
#                 break
        
#         metadata['latitude'] = float(latitude)
#         metadata['longitude'] = float(longitude)
        
#         # Additional metadata from attributes
#         metadata['project_name'] = self.clean_string_value(ds.attrs.get('PROJECT_NAME'), 200)
#         metadata['pi_name'] = self.clean_string_value(ds.attrs.get('PI_NAME'), 200)
#         metadata['institution'] = self.clean_string_value(ds.attrs.get('institution'), 200)
#         metadata['source_file'] = os.path.basename(file_path)
        
#         return metadata
    
#     def extract_measurements(self, ds):
#         """Extract measurement data from dataset"""
#         measurements = []
        
#         try:
#             # Find pressure variable
#             pressure_vars = ['PRES_ADJUSTED', 'pres_adjusted', 'PRES', 'pres']
#             pressure_var = None
#             for var_name in pressure_vars:
#                 if var_name in ds.variables:
#                     pressure_var = ds[var_name]
#                     break
            
#             if pressure_var is None:
#                 return measurements
            
#             # Find temperature and salinity variables
#             temp_vars = ['TEMP_ADJUSTED', 'temp_adjusted', 'TEMP', 'temp']
#             sal_vars = ['PSAL_ADJUSTED', 'psal_adjusted', 'PSAL', 'psal']
            
#             temp_var = None
#             for var_name in temp_vars:
#                 if var_name in ds.variables:
#                     temp_var = ds[var_name]
#                     break
            
#             sal_var = None
#             for var_name in sal_vars:
#                 if var_name in ds.variables:
#                     sal_var = ds[var_name]
#                     break
            
#             # Get data arrays
#             pressure_data = pressure_var.values
#             temp_data = temp_var.values if temp_var is not None else None
#             sal_data = sal_var.values if sal_var is not None else None
            
#             # Handle different array shapes
#             if pressure_data.ndim == 1:
#                 # Single profile
#                 for i in range(len(pressure_data)):
#                     pressure = pressure_data[i]
                    
#                     # Skip invalid pressure values
#                     if (hasattr(pressure, 'mask') and pressure.mask) or np.isnan(pressure) or pressure <= 0:
#                         continue
                    
#                     measurement = {
#                         'pressure': float(pressure),
#                         'temperature': None,
#                         'salinity': None
#                     }
                    
#                     # Add temperature if available
#                     if temp_data is not None and i < len(temp_data):
#                         temp_val = temp_data[i]
#                         if not ((hasattr(temp_val, 'mask') and temp_val.mask) or np.isnan(temp_val)):
#                             measurement['temperature'] = float(temp_val)
                    
#                     # Add salinity if available
#                     if sal_data is not None and i < len(sal_data):
#                         sal_val = sal_data[i]
#                         if not ((hasattr(sal_val, 'mask') and sal_val.mask) or np.isnan(sal_val)):
#                             measurement['salinity'] = float(sal_val)
                    
#                     measurements.append(measurement)
            
#             elif pressure_data.ndim == 2:
#                 # Multiple profiles
#                 n_profiles, n_levels = pressure_data.shape
                
#                 for prof_idx in range(n_profiles):
#                     for level_idx in range(n_levels):
#                         pressure = pressure_data[prof_idx, level_idx]
                        
#                         # Skip invalid pressure values
#                         if (hasattr(pressure, 'mask') and pressure.mask) or np.isnan(pressure) or pressure <= 0:
#                             continue
                        
#                         measurement = {
#                             'pressure': float(pressure),
#                             'temperature': None,
#                             'salinity': None
#                         }
                        
#                         # Add temperature if available
#                         if temp_data is not None:
#                             temp_val = temp_data[prof_idx, level_idx]
#                             if not ((hasattr(temp_val, 'mask') and temp_val.mask) or np.isnan(temp_val)):
#                                 measurement['temperature'] = float(temp_val)
                        
#                         # Add salinity if available
#                         if sal_data is not None:
#                             sal_val = sal_data[prof_idx, level_idx]
#                             if not ((hasattr(sal_val, 'mask') and sal_val.mask) or np.isnan(sal_val)):
#                                 measurement['salinity'] = float(sal_val)
                        
#                         measurements.append(measurement)
            
#         except Exception as e:
#             logger.error(f"Error extracting measurements: {e}")
        
#         return measurements
    
#     def save_to_csv(self, all_metadata, all_measurements):
#         """Save data to CSV files as fallback"""
#         try:
#             # Save metadata
#             metadata_df = pd.DataFrame(all_metadata)
#             metadata_file = f"{self.output_dir}/argo_metadata.csv"
#             metadata_df.to_csv(metadata_file, index=False)
#             logger.info(f"Metadata saved to: {metadata_file}")
            
#             # Save measurements
#             measurements_df = pd.DataFrame(all_measurements)
#             measurements_file = f"{self.output_dir}/argo_measurements.csv"
#             measurements_df.to_csv(measurements_file, index=False)
#             logger.info(f"Measurements saved to: {measurements_file}")
            
#             return True
#         except Exception as e:
#             logger.error(f"Error saving to CSV: {e}")
#             return False
    
#     def save_to_database(self, engine, all_metadata, all_measurements):
#         """Save data to database"""
#         try:
#             # Save metadata
#             metadata_df = pd.DataFrame(all_metadata)
#             metadata_df.to_sql('enhanced_floats_metadata', engine, if_exists='append', 
#                              index=False, method='multi')
            
#             # Save measurements (in batches)
#             measurements_df = pd.DataFrame(all_measurements)
            
#             # Add metadata_id to measurements (simplified approach)
#             # In production, you'd need proper foreign key handling
#             for i, measurement_batch in enumerate(np.array_split(measurements_df, 
#                                                                 len(measurements_df) // self.batch_size + 1)):
#                 if not measurement_batch.empty:
#                     measurement_batch.to_sql('enhanced_measurements', engine, if_exists='append', 
#                                            index=False, method='multi')
            
#             logger.info("Data saved to database successfully")
#             return True
#         except Exception as e:
#             logger.error(f"Error saving to database: {e}")
#             return False
    
#     def process_all_files(self):
#         """Process all NetCDF files"""
#         if not self.check_prerequisites():
#             return False
        
#         # Get database engine (optional)
#         engine = self.test_database_connection()
#         if engine:
#             self.create_schema_if_needed(engine)
        
#         # Find all NetCDF files
#         file_patterns = [
#             f"{self.data_dir}/*.nc",
#             f"{self.data_dir}/**/*.nc"
#         ]
        
#         all_files = []
#         for pattern in file_patterns:
#             files = glob.glob(pattern, recursive=True)
#             all_files.extend(files)
        
#         unique_files = sorted(list(set(all_files)))
#         logger.info(f"Found {len(unique_files)} NetCDF files to process")
        
#         if not unique_files:
#             logger.error("No NetCDF files found!")
#             return False
        
#         # Process files
#         all_metadata = []
#         all_measurements = []
        
#         progress_bar = tqdm(unique_files, desc="Processing files")
        
#         for file_path in progress_bar:
#             success, result = self.process_single_file(file_path)
            
#             if success:
#                 # Add file index to measurements for metadata linking
#                 for measurement in result['measurements']:
#                     measurement['file_index'] = len(all_metadata)
                
#                 all_metadata.append(result['metadata'])
#                 all_measurements.extend(result['measurements'])
#                 self.success_count += 1
#             else:
#                 self.failed_files.append({'file': os.path.basename(file_path), 'error': result})
            
#             # Update progress bar
#             progress_bar.set_postfix({
#                 'success': self.success_count,
#                 'failed': len(self.failed_files),
#                 'measurements': len(all_measurements)
#             })
            
#             # Memory management
#             if len(all_metadata) % 100 == 0:
#                 gc.collect()
        
#         # Save results
#         logger.info(f"Processing complete. Saving {len(all_metadata)} profiles with {len(all_measurements)} measurements")
        
#         # Try database first, fallback to CSV
#         if engine:
#             if not self.save_to_database(engine, all_metadata, all_measurements):
#                 logger.info("Database save failed, falling back to CSV")
#                 self.save_to_csv(all_metadata, all_measurements)
#         else:
#             self.save_to_csv(all_metadata, all_measurements)
        
#         # Generate summary report
#         self.generate_summary_report(len(unique_files), len(all_metadata), len(all_measurements))
        
#         return True
    
#     def generate_summary_report(self, total_files, metadata_count, measurements_count):
#         """Generate processing summary report"""
#         logger.info("=" * 60)
#         logger.info("PROCESSING SUMMARY REPORT")
#         logger.info("=" * 60)
        
#         logger.info(f"Total files found: {total_files}")
#         logger.info(f"Successfully processed: {self.success_count}")
#         logger.info(f"Failed: {len(self.failed_files)}")
        
#         if total_files > 0:
#             success_rate = (self.success_count / total_files) * 100
#             logger.info(f"Success rate: {success_rate:.1f}%")
        
#         logger.info(f"Total profiles: {metadata_count}")
#         logger.info(f"Total measurements: {measurements_count:,}")
        
#         if metadata_count > 0:
#             avg_measurements = measurements_count / metadata_count
#             logger.info(f"Average measurements per profile: {avg_measurements:.1f}")
        
#         # Top failure reasons
#         if self.failed_files:
#             logger.info("\nFailure reasons:")
#             error_counts = {}
#             for failed in self.failed_files:
#                 error_key = failed['error'][:100]  # Truncate for grouping
#                 error_counts[error_key] = error_counts.get(error_key, 0) + 1
            
#             for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
#                 logger.info(f"  {count:3d}x: {error}")
        
#         logger.info(f"\nOutput directory: {os.path.abspath(self.output_dir)}")

# def main():
#     """Main execution function"""
#     print("Fixed ARGO Data Processing Pipeline")
#     print("=" * 50)
    
#     # Get data directory from user
#     data_dir = input("Enter data directory path (default: ../../data/indian_ocean/raw): ").strip()
#     if not data_dir:
#         data_dir = "../../data/indian_ocean/raw"
    
#     # Initialize processor
#     processor = FixedArgoProcessor(data_dir)
    
#     # Run processing
#     try:
#         success = processor.process_all_files()
        
#         if success:
#             print("\n" + "=" * 50)
#             print("PROCESSING COMPLETED SUCCESSFULLY!")
#             print("=" * 50)
#             print(f"Check output in: {processor.output_dir}")
#         else:
#             print("\n" + "=" * 50)
#             print("PROCESSING FAILED")
#             print("=" * 50)
#             print("Check logs for details")
    
#     except KeyboardInterrupt:
#         print("\nProcessing interrupted by user")
#     except Exception as e:
#         print(f"Processing error: {e}")
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()