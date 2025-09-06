# backend/src/utils/download_sample.py
import os
import shutil

def setup_sample_data(nc_file_path):
    """
    Setup function to use existing NetCDF file
    
    Args:
        nc_file_path (str): Path to your existing .nc file
    
    Returns:
        str: Path to the data file ready for use
    """
    # Create backend/files directory if it doesn't exist
    data_dir = os.path.join("backend", "files")
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if the provided file exists
    if not os.path.exists(nc_file_path):
        raise FileNotFoundError(f"NetCDF file not found: {nc_file_path}")
    
    # Define target path in backend/files directory
    filename = os.path.basename(nc_file_path)
    target_path = os.path.join(data_dir, filename)
    
    # If file is not already in backend/files directory, copy it there
    if os.path.abspath(nc_file_path) != os.path.abspath(target_path):
        print(f"Copying {nc_file_path} to {target_path}...")
        shutil.copy2(nc_file_path, target_path)
        print(f"File copied successfully to: {target_path}")
    else:
        print(f"Using existing file: {target_path}")
    
    return target_path

def get_sample_data_path(filename=None):
    """
    Get the path to sample data file
    
    Args:
        filename (str, optional): Specific filename to look for in backend/files directory
    
    Returns:
        str: Path to the data file
    """
    data_dir = os.path.join("backend", "files")
    
    if filename:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            return file_path
        else:
            raise FileNotFoundError(f"File not found: {file_path}")
    
    # If no filename specified, look for any .nc file in backend/files directory
    if os.path.exists(data_dir):
        nc_files = [f for f in os.listdir(data_dir) if f.endswith('.nc')]
        if nc_files:
            return os.path.join(data_dir, nc_files[0])
    
    raise FileNotFoundError("No NetCDF files found in backend/files directory")

if __name__ == "__main__":
    # Example usage - replace with your actual file path
    your_nc_file = "backend/files/nodc_1900121_prof.nc"  # Updated path
    
    try:
        # Setup the data file
        data_file = setup_sample_data(your_nc_file)
        print(f"Data file ready: {data_file}")
        
        # Verify file can be found
        found_file = get_sample_data_path()
        print(f"Sample data path: {found_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please update 'your_nc_file' variable with the correct path to your .nc file")