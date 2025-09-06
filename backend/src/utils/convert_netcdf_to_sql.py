# convert_netcdf_to_sql.py
import xarray as xr
import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np
from datetime import datetime
import os
import glob
from tqdm import tqdm
import re

def get_database_engine(db_type="postgresql"):
    """Get database engine with Docker PostgreSQL"""
    
    if db_type == "postgresql":
        try:
            # Docker PostgreSQL connection
            engine = create_engine('postgresql://argo_user:argo_password@localhost:5432/argo_data')
            
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            print("✅ Connected to PostgreSQL in Docker")
            return engine
            
        except Exception as e:
            print(f"❌ Could not connect to PostgreSQL: {e}")
            print("💡 Make sure Docker is running: docker-compose up -d")
            return None
    
    else:
        raise ValueError("db_type must be 'postgresql'")

def extract_scalar_from_array(data):
    """Safely extract a scalar value from potentially multi-dimensional data"""
    try:
        if np.isscalar(data):
            return float(data)
        
        # Handle numpy arrays
        if hasattr(data, 'values'):
            values = data.values
        else:
            values = data
            
        if np.isscalar(values):
            return float(values)
        
        # Flatten and get valid values
        flat_values = np.asarray(values).flatten()
        
        # Remove NaN and masked values
        if hasattr(flat_values, 'mask'):  # Masked array
            valid_values = flat_values[~flat_values.mask]
        else:
            valid_values = flat_values[~np.isnan(flat_values)]
        
        if len(valid_values) > 0:
            return float(valid_values[0])
        else:
            return 0.0
            
    except Exception as e:
        print(f"Warning: Could not extract scalar from data: {e}")
        return 0.0

def batch_insert_measurements(engine, measurements_data, batch_size=1000):
    """Insert measurements in batches for better performance"""
    if not measurements_data:
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(measurements_data)
    
    # Insert in batches
    total_batches = (len(df) // batch_size) + 1
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch.to_sql('measurements', engine, if_exists='append', index=False, method='multi')
    
    print(f"📊 Inserted {len(df)} measurements")

def convert_netcdf_to_sql(file_path, engine):
    """Convert NetCDF to PostgreSQL with bulk operations"""
    print(f"🔍 Processing: {os.path.basename(file_path)}")
    
    try:
        ds = xr.open_dataset(file_path)
        
        # Extract metadata with better error handling
        def safe_get_attr(ds, attr_name, default=None):
            try:
                value = ds.attrs.get(attr_name, default)
                if isinstance(value, bytes):
                    return value.decode('utf-8').strip()
                elif isinstance(value, str):
                    return value.strip()
                return value
            except:
                return default

        def safe_get_variable(ds, var_names):
            """Get first available variable from a list of possible names"""
            for var_name in var_names:
                if var_name in ds.variables:
                    return ds[var_name]
            return None
        
        # Extract platform number
        platform_var = safe_get_variable(ds, ['platform_number', 'PLATFORM_NUMBER'])
        if platform_var is not None:
            platform_number = str(platform_var.values).strip()
            # Handle array of platform numbers
            if '[' in platform_number or 'array' in platform_number.lower():
                try:
                    numbers = re.findall(r'\d+', platform_number)
                    platform_number = numbers[0] if numbers else 'Unknown'
                except:
                    platform_number = 'Unknown'
        else:
            platform_number = safe_get_attr(ds, 'PLATFORM_NUMBER', 'Unknown')
        
        # Extract cycle number
        cycle_var = safe_get_variable(ds, ['cycle_number', 'CYCLE_NUMBER'])
        if cycle_var is not None:
            cycle_number = int(extract_scalar_from_array(cycle_var))
        else:
            cycle_number = 0
        
        # Handle date
        date_creation = datetime.now()
        
        # Try juld (Julian day) variable
        juld_var = safe_get_variable(ds, ['juld', 'JULD'])
        if juld_var is not None:
            try:
                juld_value = extract_scalar_from_array(juld_var)
                if juld_value > 0:
                    reference_date = pd.to_datetime('1950-01-01')
                    date_creation = reference_date + pd.Timedelta(days=juld_value)
            except:
                pass
        
        # Extract coordinates
        lat_var = safe_get_variable(ds, ['latitude', 'LATITUDE', 'lat', 'LAT'])
        latitude = extract_scalar_from_array(lat_var) if lat_var is not None else 0.0
        
        lon_var = safe_get_variable(ds, ['longitude', 'LONGITUDE', 'lon', 'LON'])
        longitude = extract_scalar_from_array(lon_var) if lon_var is not None else 0.0
        
        # Build metadata dictionary
        metadata = {
            'platform_number': platform_number,
            'cycle_number': cycle_number,
            'date': date_creation,
            'latitude': latitude,
            'longitude': longitude,
            'project_name': safe_get_attr(ds, 'PROJECT_NAME', 'Unknown'),
            'pi_name': safe_get_attr(ds, 'PI_NAME', 'Unknown'),
            'institution': safe_get_attr(ds, 'institution', 'Unknown'),
            'wmo_inst_type': safe_get_attr(ds, 'WMO_INST_TYPE', 'Unknown'),
            'data_mode': safe_get_attr(ds, 'DATA_MODE', 'Unknown'),
            'source_file': os.path.basename(file_path)
        }
        
        # Clean up metadata values
        for key, value in metadata.items():
            if isinstance(value, str):
                value = value.replace("b'", "").replace("'", "").replace('"', '')
                if value.startswith('[') and value.endswith(']'):
                    value = value[1:-1].split()[0] if value[1:-1].strip() else 'Unknown'
                metadata[key] = value.strip()
        
        print(f"📋 Extracted metadata: Platform {metadata['platform_number']}, Cycle {metadata['cycle_number']}")
        print(f"📍 Location: {metadata['latitude']:.4f}, {metadata['longitude']:.4f}")
        
        # Insert metadata into floats_metadata table
        with engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO floats_metadata 
                (platform_number, cycle_number, date, latitude, longitude, 
                 project_name, pi_name, institution, wmo_inst_type, data_mode, source_file)
                VALUES (:platform_number, :cycle_number, :date, :latitude, :longitude, 
                       :project_name, :pi_name, :institution, :wmo_inst_type, :data_mode, :source_file)
                RETURNING id
            """), metadata)
            metadata_id = result.scalar()
            conn.commit()
        
        print(f"✅ Metadata inserted with ID: {metadata_id}")
        
        # Prepare measurements data
        measurements_data = []
        
        # Find measurement variables
        pressure_var = safe_get_variable(ds, ['pres_adjusted', 'PRES_ADJUSTED', 'pres', 'PRES', 'pressure'])
        temp_var = safe_get_variable(ds, ['temp_adjusted', 'TEMP_ADJUSTED', 'temp', 'TEMP', 'temperature'])
        sal_var = safe_get_variable(ds, ['psal_adjusted', 'PSAL_ADJUSTED', 'psal', 'PSAL', 'salinity'])
        
        if pressure_var is not None:
            print(f"📊 Found measurement variables:")
            print(f"  Pressure: {pressure_var.name} {pressure_var.shape}")
            if temp_var is not None:
                print(f"  Temperature: {temp_var.name} {temp_var.shape}")
            if sal_var is not None:
                print(f"  Salinity: {sal_var.name} {sal_var.shape}")
            
            # Get data arrays
            pressure_data = pressure_var.values
            temp_data = temp_var.values if temp_var is not None else None
            sal_data = sal_var.values if sal_var is not None else None
            
            # Handle different data structures
            if pressure_data.ndim == 2:
                n_profiles, n_levels = pressure_data.shape
                print(f"🔢 Processing {n_profiles} profiles with up to {n_levels} levels each")
                
                for prof_idx in tqdm(range(n_profiles), desc="Processing profiles"):
                    for level_idx in range(n_levels):
                        pressure = pressure_data[prof_idx, level_idx]
                        
                        # Skip invalid measurements
                        if (np.ma.is_masked(pressure) or np.isnan(pressure) or pressure <= 0):
                            continue
                        
                        measurement = {
                            'metadata_id': metadata_id,
                            'pressure': float(pressure),
                            'temperature': None,
                            'salinity': None,
                            'oxygen': None,
                            'chlorophyll': None,
                            'backscatter': None
                        }
                        
                        # Add temperature if available
                        if temp_data is not None:
                            temp_val = temp_data[prof_idx, level_idx]
                            if not (np.ma.is_masked(temp_val) or np.isnan(temp_val)):
                                measurement['temperature'] = float(temp_val)
                        
                        # Add salinity if available
                        if sal_data is not None:
                            sal_val = sal_data[prof_idx, level_idx]
                            if not (np.ma.is_masked(sal_val) or np.isnan(sal_val)):
                                measurement['salinity'] = float(sal_val)
                        
                        measurements_data.append(measurement)
            
            # Batch insert
            batch_insert_measurements(engine, measurements_data)
        
        else:
            print("⚠️ No pressure data found in file")
        
        ds.close()
        return True
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function with Docker PostgreSQL"""
    try:
        # Use PostgreSQL with Docker
        engine = get_database_engine("postgresql")
        if engine is None:
            return
        
        # Find and process files
        netcdf_files = glob.glob("../../data/*.nc")
        
        if not netcdf_files:
            print("❌ No NetCDF files found in data/ directory")
            return
        
        print(f"📁 Found {len(netcdf_files)} NetCDF files")
        
        successful = 0
        for file_path in tqdm(netcdf_files, desc="Processing files"):
            if convert_netcdf_to_sql(file_path, engine):
                successful += 1
        
        print(f"\n🎉 Conversion completed: {successful}/{len(netcdf_files)} successful")
        
        # Show summary
        with engine.connect() as conn:
            meta_count = conn.execute(text("SELECT COUNT(*) FROM floats_metadata")).scalar()
            meas_count = conn.execute(text("SELECT COUNT(*) FROM measurements")).scalar()
            print(f"📊 Database now has {meta_count} float profiles and {meas_count} measurements")
            
    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    main()