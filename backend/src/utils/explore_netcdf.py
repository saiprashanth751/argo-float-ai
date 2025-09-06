import xarray as xr
import pandas as pd
import os

def explore_netcdf(file_path):
    """Explore the structure of a NetCDF file"""
    print(f"Exploring: {file_path}")
    
    # Open the NetCDF file
    ds = xr.open_dataset(file_path)
    
    print("\n=== DATASET DIMENSIONS ===")
    print(ds.dims)
    
    print("\n=== DATASET VARIABLES ===")
    for var_name in ds.variables:
        var = ds.variables[var_name]
        print(f"{var_name}: {var.dims} - {var.attrs.get('long_name', 'No description')}")
    
    print("\n=== GLOBAL ATTRIBUTES ===")
    for attr_name in ds.attrs:
        print(f"{attr_name}: {ds.attrs[attr_name]}")
    
    # Show a sample of the data
    print("\n=== SAMPLE DATA ===")
    if 'TEMP' in ds.variables:
        print("Temperature data sample:")
        print(ds['TEMP'].values[:5])  # First 5 values
    
    if 'PSAL' in ds.variables:
        print("Salinity data sample:")
        print(ds['PSAL'].values[:5])
    
    ds.close()

if __name__ == "__main__":
    file_path = "backend/files/nodc_1900121_prof.nc"  # Update if different
    explore_netcdf(file_path)