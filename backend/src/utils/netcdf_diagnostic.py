# improved_netcdf_diagnostic.py
import xarray as xr
import numpy as np
import os
import glob
import pandas as pd
from collections import defaultdict, Counter
import json
from pathlib import Path
import traceback
import netCDF4 as nc
from datetime import datetime
import warnings
import pickle
warnings.filterwarnings('ignore')

class ImprovedFileDiagnostics:
    def __init__(self):
        self.diagnostics_results = []
        self.issue_patterns = defaultdict(list)
        self.fix_suggestions = defaultdict(list)
        
    def safe_serialize(self, obj):
        """Safely convert objects to serializable format"""
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self.safe_serialize(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(k): self.safe_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            if obj.size < 10:  # Small arrays
                return obj.tolist()
            else:
                return f"Array shape: {obj.shape}, dtype: {obj.dtype}"
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return str(obj)
    
    def diagnose_file_structure(self, file_path):
        """Deep dive into file structure and identify issues"""
        filename = os.path.basename(file_path)
        diagnosis = {
            'filename': filename,
            'file_path': str(file_path),
            'file_size_mb': round(os.path.getsize(file_path) / (1024*1024), 2),
            'issues': [],
            'warnings': [],
            'structure_info': {},
            'variables_info': {},
            'attributes_info': {},
            'data_quality': {},
            'fix_suggestions': []
        }
        
        try:
            # First, try with netCDF4 directly for low-level info
            with nc.Dataset(file_path, 'r') as ds:
                diagnosis['nc4_readable'] = True
                diagnosis['structure_info']['dimensions'] = {str(k): int(v.size) for k, v in ds.dimensions.items()}
                diagnosis['structure_info']['variables'] = list(ds.variables.keys())
                diagnosis['structure_info']['global_attrs'] = len(ds.ncattrs())
        except Exception as e:
            diagnosis['nc4_readable'] = False
            diagnosis['issues'].append(f"NetCDF4 reading error: {str(e)[:100]}")
            
        # Try with xarray
        try:
            with xr.open_dataset(file_path) as ds:
                diagnosis['xarray_readable'] = True
                self._analyze_dataset_structure(ds, diagnosis)
                self._analyze_variables(ds, diagnosis)
                self._analyze_attributes(ds, diagnosis)
                self._analyze_data_quality(ds, diagnosis)
                
        except Exception as e:
            diagnosis['xarray_readable'] = False
            diagnosis['issues'].append(f"xarray reading error: {str(e)[:100]}")
            self._analyze_reading_failure(file_path, diagnosis, str(e))
        
        # Generate fix suggestions based on issues
        self._generate_fix_suggestions(diagnosis)
        
        return diagnosis
    
    def _analyze_dataset_structure(self, ds, diagnosis):
        """Analyze the structure of the dataset"""
        try:
            diagnosis['structure_info']['data_vars'] = len(ds.data_vars)
            diagnosis['structure_info']['coords'] = len(ds.coords)
            diagnosis['structure_info']['dims'] = {str(k): int(v) for k, v in ds.dims.items()}
            
            # Check for expected ARGO dimensions
            expected_dims = ['N_PROF', 'N_LEVELS', 'STRING256', 'STRING64', 'STRING32']
            missing_dims = [dim for dim in expected_dims if dim not in ds.dims]
            if missing_dims:
                diagnosis['warnings'].append(f"Missing expected dimensions: {missing_dims}")
            
            # Check data variable shapes (convert to serializable format)
            var_shapes = {}
            for var_name, var in ds.data_vars.items():
                var_shapes[str(var_name)] = list(var.shape)
            diagnosis['structure_info']['var_shapes'] = var_shapes
            
        except Exception as e:
            diagnosis['issues'].append(f"Structure analysis error: {str(e)[:100]}")
    
    def _analyze_variables(self, ds, diagnosis):
        """Analyze variables and their properties"""
        try:
            # Core ARGO variables to check
            core_vars = {
                'pressure': ['PRES', 'pres', 'PRES_ADJUSTED', 'pres_adjusted'],
                'temperature': ['TEMP', 'temp', 'TEMP_ADJUSTED', 'temp_adjusted'],
                'salinity': ['PSAL', 'psal', 'PSAL_ADJUSTED', 'psal_adjusted'],
                'platform': ['PLATFORM_NUMBER', 'platform_number'],
                'cycle': ['CYCLE_NUMBER', 'cycle_number'],
                'date': ['JULD', 'juld'],
                'latitude': ['LATITUDE', 'latitude'],
                'longitude': ['LONGITUDE', 'longitude']
            }
            
            var_availability = {}
            for param_type, var_names in core_vars.items():
                found_vars = []
                for var_name in var_names:
                    if var_name in ds.variables:
                        var = ds[var_name]
                        found_vars.append({
                            'name': str(var_name),
                            'shape': list(var.shape),
                            'dtype': str(var.dtype),
                            'has_fill_value': hasattr(var, '_FillValue'),
                            'has_data': var.size > 0
                        })
                var_availability[param_type] = found_vars
            
            diagnosis['variables_info'] = var_availability
            
            # Check for critical missing variables
            critical_missing = []
            for param_type, vars_list in var_availability.items():
                if not vars_list and param_type in ['pressure', 'platform', 'latitude', 'longitude']:
                    critical_missing.append(param_type)
            
            if critical_missing:
                diagnosis['issues'].append(f"Critical variables missing: {critical_missing}")
            
            # Check BGC variables
            bgc_vars = ['DOXY', 'CHLA', 'BBP700', 'NITRATE', 'PH_IN_SITU_TOTAL']
            found_bgc = [var for var in bgc_vars if var in ds.variables]
            diagnosis['variables_info']['bgc_available'] = found_bgc
            
        except Exception as e:
            diagnosis['issues'].append(f"Variable analysis error: {str(e)[:100]}")
    
    def _analyze_attributes(self, ds, diagnosis):
        """Analyze global attributes"""
        try:
            attrs_info = {}
            
            # Key attributes to check
            key_attrs = [
                'DATA_MODE', 'PLATFORM_NUMBER', 'PROJECT_NAME', 'PI_NAME',
                'institution', 'WMO_INST_TYPE', 'date_creation'
            ]
            
            for attr in key_attrs:
                if attr in ds.attrs:
                    value = ds.attrs[attr]
                    # Convert to string and handle different types
                    if isinstance(value, bytes):
                        try:
                            value_str = value.decode('utf-8')
                        except:
                            value_str = str(value)
                    else:
                        value_str = str(value)
                        
                    attrs_info[attr] = {
                        'value': value_str[:100],  # Truncate for display
                        'type': type(value).__name__,
                        'length': len(value_str)
                    }
                    
                    # Check for problematic values
                    if attr == 'DATA_MODE' and len(value_str) > 10:
                        diagnosis['issues'].append(f"DATA_MODE too long: {len(value_str)} chars")
                    
                    if isinstance(value, bytes):
                        diagnosis['warnings'].append(f"Attribute {attr} is bytes type")
                else:
                    attrs_info[attr] = {'missing': True}
            
            diagnosis['attributes_info'] = attrs_info
            
        except Exception as e:
            diagnosis['issues'].append(f"Attributes analysis error: {str(e)[:100]}")
    
    def _analyze_data_quality(self, ds, diagnosis):
        """Analyze data quality issues"""
        try:
            quality_info = {}
            
            # Check pressure data
            pressure_vars = ['PRES', 'pres', 'PRES_ADJUSTED', 'pres_adjusted']
            for var_name in pressure_vars:
                if var_name in ds.variables:
                    pres_data = ds[var_name].values
                    
                    # Handle different numpy versions and data types
                    try:
                        if pres_data.dtype.kind in ['U', 'S']:  # String types
                            quality_info[var_name] = {
                                'total_points': int(pres_data.size),
                                'valid_points': 'N/A - string data',
                                'data_type': 'string'
                            }
                        else:
                            # Numeric data
                            pres_float = pres_data.astype(np.float64, errors='ignore')
                            valid_mask = np.isfinite(pres_float)
                            valid_points = int(np.sum(valid_mask))
                            
                            quality_info[var_name] = {
                                'total_points': int(pres_data.size),
                                'valid_points': valid_points,
                                'negative_values': int(np.sum(pres_float < 0)) if valid_points > 0 else 0,
                                'zero_values': int(np.sum(pres_float == 0)) if valid_points > 0 else 0,
                                'max_value': float(np.nanmax(pres_float)) if valid_points > 0 else None,
                                'min_value': float(np.nanmin(pres_float)) if valid_points > 0 else None
                            }
                            
                            # Quality checks
                            valid_ratio = valid_points / pres_data.size
                            if valid_ratio < 0.1:
                                diagnosis['issues'].append(f"{var_name}: Very low valid data ratio ({valid_ratio:.2%})")
                            
                            if quality_info[var_name]['negative_values'] > 0:
                                diagnosis['warnings'].append(f"{var_name}: Has {quality_info[var_name]['negative_values']} negative values")
                                
                    except Exception as e:
                        quality_info[var_name] = {
                            'error': f"Could not analyze: {str(e)[:50]}",
                            'total_points': int(pres_data.size),
                            'dtype': str(pres_data.dtype)
                        }
                    
                    break  # Only analyze the first pressure variable found
            
            # Check coordinate data
            for coord_name in ['LATITUDE', 'LONGITUDE', 'latitude', 'longitude']:
                if coord_name in ds.variables:
                    coord_data = ds[coord_name].values
                    coord_key = coord_name.lower().replace('latitude', 'lat').replace('longitude', 'lon')
                    
                    try:
                        if coord_data.size > 0 and coord_data.dtype.kind in 'fc':
                            coord_float = coord_data.astype(np.float64)
                            value = float(np.nanmean(coord_float)) if np.any(np.isfinite(coord_float)) else None
                            quality_info[coord_key] = value
                            
                            # Sanity checks
                            if value is not None:
                                if 'lat' in coord_key and (value < -90 or value > 90):
                                    diagnosis['issues'].append(f"Invalid latitude: {value:.2f}")
                                elif 'lon' in coord_key and (value < -180 or value > 180):
                                    diagnosis['issues'].append(f"Invalid longitude: {value:.2f}")
                        else:
                            quality_info[coord_key] = f"Non-numeric or empty: {coord_data.dtype}"
                    except Exception as e:
                        quality_info[coord_key] = f"Error: {str(e)[:50]}"
            
            diagnosis['data_quality'] = quality_info
            
        except Exception as e:
            diagnosis['issues'].append(f"Data quality analysis error: {str(e)[:100]}")
    
    def _analyze_reading_failure(self, file_path, diagnosis, error_msg):
        """Analyze why a file failed to read"""
        try:
            # Check file size
            if diagnosis['file_size_mb'] < 0.01:
                diagnosis['issues'].append("File is extremely small (< 10KB)")
                diagnosis['fix_suggestions'].append("File may be corrupted or incomplete")
            
            # Check file extension and naming
            if not file_path.endswith('.nc'):
                diagnosis['issues'].append("File doesn't have .nc extension")
            
            # Try to identify specific error patterns
            if "NetCDF: Not a valid ID" in error_msg:
                diagnosis['issues'].append("File is not a valid NetCDF file")
                diagnosis['fix_suggestions'].append("File may be corrupted, truncated, or not actually a NetCDF file")
            
            elif "NetCDF: HDF error" in error_msg:
                diagnosis['issues'].append("HDF5/NetCDF internal error")
                diagnosis['fix_suggestions'].append("File may have HDF5 corruption or incompatible format version")
            
            elif "permission" in error_msg.lower():
                diagnosis['issues'].append("File permission error")
                diagnosis['fix_suggestions'].append("Check file permissions")
            
            # Try reading just the header
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(16)
                    if header[:3] == b'CDF':
                        diagnosis['warnings'].append("File has CDF signature but still fails to read")
                    elif header[:4] == b'\x89HDF' or header[:8] == b'\x89HDF\r\n\x1a\n':
                        diagnosis['warnings'].append("File has HDF5 signature")
                    else:
                        diagnosis['issues'].append(f"Unrecognized file header: {header[:8]}")
            except:
                diagnosis['issues'].append("Cannot read file header")
                
        except Exception as e:
            diagnosis['issues'].append(f"Failure analysis error: {str(e)[:100]}")
    
    def _generate_fix_suggestions(self, diagnosis):
        """Generate specific fix suggestions based on identified issues"""
        suggestions = []
        
        # Based on readability
        if not diagnosis.get('xarray_readable', True):
            suggestions.append("File cannot be read by xarray - consider skipping or manual repair")
        
        # Based on structure issues
        if any('Critical variables missing' in issue for issue in diagnosis.get('issues', [])):
            suggestions.append("Skip processing - missing essential ARGO variables")
        
        # Based on data quality
        data_quality = diagnosis.get('data_quality', {})
        for var_name, quality in data_quality.items():
            if isinstance(quality, dict) and quality.get('valid_points') == 0:
                suggestions.append(f"No valid data in {var_name} - skip this variable")
        
        # Based on attributes
        attrs_info = diagnosis.get('attributes_info', {})
        if any(attr.get('length', 0) > 50 for attr in attrs_info.values() if isinstance(attr, dict)):
            suggestions.append("Truncate long attribute values before database insertion")
        
        diagnosis['fix_suggestions'].extend(suggestions)
    
    def diagnose_multiple_files(self, file_pattern, max_files=100):
        """Diagnose multiple files and identify patterns"""
        files = glob.glob(file_pattern, recursive=True)
        
        if max_files:
            files = files[:max_files]
        
        print(f"Diagnosing {len(files)} files...")
        
        results = []
        for i, file_path in enumerate(files):
            print(f"[{i+1:3d}/{len(files)}] {os.path.basename(file_path)}")
            
            try:
                diagnosis = self.diagnose_file_structure(file_path)
                results.append(diagnosis)
                self.diagnostics_results.append(diagnosis)
                
                # Collect issue patterns
                for issue in diagnosis['issues']:
                    self.issue_patterns[issue].append(os.path.basename(file_path))
                    
            except Exception as e:
                error_diagnosis = {
                    'filename': os.path.basename(file_path),
                    'file_path': str(file_path),
                    'file_size_mb': 0,
                    'fatal_error': str(e)[:200],
                    'issues': [f"Fatal analysis error: {str(e)[:100]}"],
                    'xarray_readable': False,
                    'nc4_readable': False
                }
                results.append(error_diagnosis)
                self.diagnostics_results.append(error_diagnosis)
                print(f"    ERROR: {str(e)[:100]}")
        
        return results
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive diagnostic report"""
        if not self.diagnostics_results:
            print("No diagnostic results available")
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE ARGO FILE DIAGNOSTIC REPORT")
        print("="*80)
        
        total_files = len(self.diagnostics_results)
        readable_files = sum(1 for d in self.diagnostics_results if d.get('xarray_readable', False))
        
        print(f"\nOVERALL SUMMARY:")
        print(f"Total files analyzed: {total_files}")
        print(f"Successfully readable: {readable_files} ({readable_files/total_files*100:.1f}%)")
        print(f"Failed to read: {total_files - readable_files} ({(total_files-readable_files)/total_files*100:.1f}%)")
        
        # Most common issues
        print(f"\nMOST COMMON ISSUES:")
        if self.issue_patterns:
            for issue, files in sorted(self.issue_patterns.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
                print(f"  {len(files):3d} files: {issue}")
        else:
            print("  No issues detected!")
        
        # File size analysis
        file_sizes = [d.get('file_size_mb', 0) for d in self.diagnostics_results if 'file_size_mb' in d]
        if file_sizes:
            print(f"\nFILE SIZE ANALYSIS:")
            print(f"  Average: {np.mean(file_sizes):.2f} MB")
            print(f"  Median: {np.median(file_sizes):.2f} MB")
            print(f"  Min: {np.min(file_sizes):.2f} MB")
            print(f"  Max: {np.max(file_sizes):.2f} MB")
        
        # Variable availability analysis
        print(f"\nVARIABLE AVAILABILITY (readable files only):")
        readable_results = [d for d in self.diagnostics_results if d.get('xarray_readable', False)]
        
        if readable_results:
            var_counts = defaultdict(int)
            for result in readable_results:
                vars_info = result.get('variables_info', {})
                for param_type, vars_list in vars_info.items():
                    if vars_list and param_type != 'bgc_available':
                        var_counts[param_type] += 1
            
            for param_type, count in sorted(var_counts.items()):
                percentage = count / len(readable_results) * 100
                print(f"  {param_type}: {count}/{len(readable_results)} ({percentage:.1f}%)")
        
        # BGC data availability
        if readable_results:
            bgc_files = sum(1 for d in readable_results if d.get('variables_info', {}).get('bgc_available', []))
            print(f"\nBGC DATA AVAILABILITY:")
            print(f"  Files with BGC variables: {bgc_files}/{len(readable_results)} ({bgc_files/len(readable_results)*100:.1f}%)")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        print("1. Files with reading errors should be:")
        print("   - Checked for corruption")
        print("   - Skipped in processing pipeline")
        print("   - Reported to data provider if systematic")
        
        if any("DATA_MODE too long" in issue for issue in self.issue_patterns.keys()):
            print("2. Expand DATA_MODE field in database schema")
        
        if any("missing" in issue.lower() for issue in self.issue_patterns.keys()):
            print("3. Implement fallback handling for missing critical variables")
        
        print("4. Consider processing files in order of reliability:")
        print("   - Process smaller, complete files first")
        print("   - Handle problematic files separately")
        
        # Export detailed results
        self.export_detailed_results()
    
    def export_detailed_results(self, filename='argo_diagnostics_detailed.json'):
        """Export detailed results to JSON with improved serialization"""
        try:
            # Create a completely serializable version of results
            serializable_results = []
            for result in self.diagnostics_results:
                clean_result = self.safe_serialize(result)
                serializable_results.append(clean_result)
            
            # Convert issue_patterns to serializable format
            serializable_issues = {}
            for issue, files in self.issue_patterns.items():
                serializable_issues[str(issue)] = list(files)
            
            export_data = {
                'diagnostics_results': serializable_results,
                'issue_patterns': serializable_issues,
                'summary': {
                    'total_files': len(self.diagnostics_results),
                    'readable_files': sum(1 for d in self.diagnostics_results if d.get('xarray_readable', False)),
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }
            
            # Write JSON file
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            print(f"\nDetailed results exported to: {filename}")
            
            # Also create a CSV summary for easy analysis
            self.export_csv_summary()
            
        except Exception as e:
            print(f"Could not export JSON results: {e}")
            # Try to export as pickle as fallback
            try:
                pickle_filename = filename.replace('.json', '.pkl')
                with open(pickle_filename, 'wb') as f:
                    pickle.dump({
                        'diagnostics_results': self.diagnostics_results,
                        'issue_patterns': dict(self.issue_patterns)
                    }, f)
                print(f"Results exported as pickle to: {pickle_filename}")
            except Exception as e2:
                print(f"Could not export pickle results either: {e2}")
    
    def export_csv_summary(self, filename='argo_diagnostics_summary.csv'):
        """Export a summary CSV for easy analysis"""
        try:
            summary_data = []
            for result in self.diagnostics_results:
                row = {
                    'filename': result.get('filename', 'unknown'),
                    'file_size_mb': result.get('file_size_mb', 0),
                    'xarray_readable': result.get('xarray_readable', False),
                    'nc4_readable': result.get('nc4_readable', False),
                    'num_issues': len(result.get('issues', [])),
                    'num_warnings': len(result.get('warnings', [])),
                    'has_pressure': bool(result.get('variables_info', {}).get('pressure', [])),
                    'has_temperature': bool(result.get('variables_info', {}).get('temperature', [])),
                    'has_salinity': bool(result.get('variables_info', {}).get('salinity', [])),
                    'has_bgc': bool(result.get('variables_info', {}).get('bgc_available', [])),
                    'main_issues': '; '.join(result.get('issues', [])[:3])  # First 3 issues
                }
                summary_data.append(row)
            
            df = pd.DataFrame(summary_data)
            df.to_csv(filename, index=False)
            print(f"Summary CSV exported to: {filename}")
            
        except Exception as e:
            print(f"Could not export CSV summary: {e}")

def main():
    """Main diagnostic function"""
    print("ARGO NetCDF Improved File Diagnostics")
    print("="*50)
    
    # Initialize diagnostics
    diagnostics = ImprovedFileDiagnostics()
    
    # File patterns to check
    patterns = [
        "../../data/indian_ocean/raw/*.nc",
        "../../data/indian_ocean/raw/**/*.nc"
    ]
    
    all_files = []
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
    
    unique_files = list(set(all_files))
    print(f"Found {len(unique_files)} unique files")
    
    if not unique_files:
        print("No files found! Check your data directory paths.")
        return
    
    # Diagnose files (limit for initial run)
    max_files = min(100, len(unique_files))  # Analyze up to 100 files initially
    print(f"Analyzing first {max_files} files...")
    
    try:
        results = diagnostics.diagnose_multiple_files("../../data/indian_ocean/raw/*.nc", max_files)
        diagnostics.generate_comprehensive_report()
        
    except KeyboardInterrupt:
        print("\nDiagnostics interrupted by user")
        if diagnostics.diagnostics_results:
            diagnostics.generate_comprehensive_report()
    
    except Exception as e:
        print(f"Error during diagnostics: {e}")
        traceback.print_exc()
        # Try to generate report with whatever data we have
        if diagnostics.diagnostics_results:
            print("\nAttempting to generate partial report...")
            diagnostics.generate_comprehensive_report()

if __name__ == "__main__":
    main()