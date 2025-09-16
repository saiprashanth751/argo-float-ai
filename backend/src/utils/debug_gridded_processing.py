# debug_gridded_processing.py
"""
Debugger to understand why gridded data processing is failing
"""

import xarray as xr
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

def analyze_netcdf_files(data_directory):
    """
    Analyze NetCDF files to understand their structure and why gridded processing might fail
    """
    
    # Find all NetCDF files
    file_patterns = [
        f"{data_directory}/**/*.nc",
        f"{data_directory}/*.nc"
    ]
    
    all_files = []
    for pattern in file_patterns:
        all_files.extend(glob.glob(pattern, recursive=True))
    
    print(f"Found {len(all_files)} NetCDF files")
    
    file_analysis = {
        'profile_files': [],
        'gridded_files': [],
        'unknown_files': [],
        'error_files': []
    }
    
    for i, file_path in enumerate(all_files[:10]):  # Analyze first 10 files
        print(f"\nAnalyzing file {i+1}: {os.path.basename(file_path)}")
        print("-" * 50)
        
        try:
            with xr.open_dataset(file_path) as ds:
                # Get dimensions and variables
                dims = dict(ds.sizes)  # Use .sizes instead of .dims
                variables = list(ds.variables.keys())
                coords = list(ds.coords.keys())
                
                print(f"Dimensions: {dims}")
                print(f"Coordinates: {coords}")
                print(f"Variables: {variables[:10]}...")  # Show first 10 variables
                
                # Analyze file type
                has_n_prof = 'N_PROF' in dims
                has_n_levels = 'N_LEVELS' in dims
                has_platform_number = any('PLATFORM' in var.upper() for var in variables)
                has_cycle_number = any('CYCLE' in var.upper() for var in variables)
                
                # Check for gridded data structure
                has_lat_lon_dims = ('latitude' in dims or 'lat' in dims) and ('longitude' in dims or 'lon' in dims)
                has_time_dim = 'time' in dims or 'TIME' in dims
                
                print(f"Profile indicators: N_PROF={has_n_prof}, N_LEVELS={has_n_levels}, PLATFORM={has_platform_number}")
                print(f"Gridded indicators: lat/lon={has_lat_lon_dims}, time={has_time_dim}")
                
                if has_n_prof and has_n_levels and has_platform_number:
                    file_type = 'PROFILE'
                    file_analysis['profile_files'].append(file_path)
                    
                    # Show sample profile data
                    print(f"Sample profile data structure:")
                    if 'TEMP' in variables:
                        temp_shape = ds['TEMP'].shape
                        print(f"  TEMP shape: {temp_shape}")
                    if 'PSAL' in variables:
                        psal_shape = ds['PSAL'].shape  
                        print(f"  PSAL shape: {psal_shape}")
                        
                elif has_lat_lon_dims and has_time_dim:
                    file_type = 'GRIDDED'
                    file_analysis['gridded_files'].append(file_path)
                    
                    # Show sample gridded data
                    print(f"Gridded data structure:")
                    if 'latitude' in coords:
                        lat_size = len(ds['latitude'])
                        lon_size = len(ds['longitude'])
                    elif 'lat' in coords:
                        lat_size = len(ds['lat'])
                        lon_size = len(ds['lon'])
                    else:
                        lat_size = lon_size = 0
                        
                    time_size = len(ds['time']) if 'time' in coords else len(ds['TIME']) if 'TIME' in coords else 0
                    
                    print(f"  Grid size: {lat_size} x {lon_size} x {time_size}")
                    
                    # Check for temperature/salinity variables
                    temp_vars = [v for v in variables if 'temp' in v.lower() or v.upper() in ['TEMP', 'TEMPERATURE']]
                    sal_vars = [v for v in variables if 'sal' in v.lower() or v.upper() in ['PSAL', 'SALINITY']]
                    
                    print(f"  Temperature variables: {temp_vars}")
                    print(f"  Salinity variables: {sal_vars}")
                    
                    # Check actual data values
                    if temp_vars:
                        temp_var = ds[temp_vars[0]]
                        print(f"  {temp_vars[0]} shape: {temp_var.shape}")
                        print(f"  {temp_vars[0]} dims: {temp_var.dims}")
                        
                        # Check for valid data
                        valid_count = np.sum(~np.isnan(temp_var.values))
                        total_count = np.prod(temp_var.shape)
                        print(f"  Valid temperature values: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
                
                else:
                    file_type = 'UNKNOWN'
                    file_analysis['unknown_files'].append(file_path)
                    
                print(f"File type: {file_type}")
                
        except Exception as e:
            print(f"ERROR analyzing file: {e}")
            file_analysis['error_files'].append(file_path)
    
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Profile files: {len(file_analysis['profile_files'])}")
    print(f"Gridded files: {len(file_analysis['gridded_files'])}")
    print(f"Unknown files: {len(file_analysis['unknown_files'])}")
    print(f"Error files: {len(file_analysis['error_files'])}")
    
    return file_analysis

def debug_specific_gridded_file(file_path):
    """
    Deep dive into a specific gridded file to understand the data structure
    """
    
    print(f"Deep analysis of: {os.path.basename(file_path)}")
    print("=" * 60)
    
    try:
        with xr.open_dataset(file_path) as ds:
            print("Dataset overview:")
            print(ds)
            
            print(f"\nCoordinates:")
            for coord in ds.coords:
                coord_data = ds[coord]
                print(f"  {coord}: {coord_data.shape} - {coord_data.dtype}")
                if len(coord_data) <= 10:
                    print(f"    Values: {coord_data.values}")
                else:
                    print(f"    Range: {coord_data.min().values} to {coord_data.max().values}")
            
            print(f"\nData variables:")
            for var in ds.data_vars:
                var_data = ds[var]
                print(f"  {var}: {var_data.shape} - {var_data.dtype}")
                print(f"    Dims: {var_data.dims}")
                
                # Check for actual data
                if var_data.dtype.kind in ['f', 'i']:  # float or int
                    valid_count = np.sum(~np.isnan(var_data.values))
                    total_count = np.prod(var_data.shape)
                    print(f"    Valid values: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
                    
                    if valid_count > 0:
                        print(f"    Value range: {np.nanmin(var_data.values):.3f} to {np.nanmax(var_data.values):.3f}")
    
    except Exception as e:
        print(f"Error in deep analysis: {e}")

if __name__ == "__main__":
    # Run analysis on your data directory
    data_directory = "../../data/indian_ocean/raw"  # Adjust this path
    
    if not os.path.exists(data_directory):
        print(f"Data directory not found: {data_directory}")
        print("Please update the path to your NetCDF files")
    else:
        analysis = analyze_netcdf_files(data_directory)
        
        # If we found gridded files, analyze one in detail
        if analysis['gridded_files']:
            print(f"\n{'='*60}")
            print("DETAILED GRIDDED FILE ANALYSIS")  
            print(f"{'='*60}")
            debug_specific_gridded_file(analysis['gridded_files'][0])