# file_repair_utility.py
import os
import shutil
import glob
import json
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArgoFileRepairUtility:
    def __init__(self):
        self.repair_log = []
        self.backup_dir = "file_backups"
        self.repaired_dir = "repaired_files"
        
        # Create directories
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(self.repaired_dir, exist_ok=True)
    
    def create_backup(self, file_path):
        """Create backup of original file"""
        try:
            filename = os.path.basename(file_path)
            backup_path = os.path.join(self.backup_dir, filename)
            shutil.copy2(file_path, backup_path)
            return backup_path
        except Exception as e:
            logger.error(f"Could not create backup for {file_path}: {e}")
            return None
    
    def check_file_integrity(self, file_path):
        """Check basic file integrity"""
        issues = []
        
        try:
            # Check file size
            size = os.path.getsize(file_path)
            if size == 0:
                issues.append("File is empty (0 bytes)")
            elif size < 1000:  # Very small file
                issues.append(f"File is very small ({size} bytes)")
            
            # Check file header
            with open(file_path, 'rb') as f:
                header = f.read(16)
                
                if not header:
                    issues.append("Cannot read file header")
                elif len(header) < 4:
                    issues.append("File header too short")
                elif header[:3] != b'CDF' and header[:4] != b'\x89HDF':
                    issues.append(f"Unrecognized file format (header: {header[:8]})")
            
            return issues
        except Exception as e:
            return [f"Error checking file integrity: {str(e)}"]
    
    def attempt_file_repair(self, file_path):
        """Attempt to repair common file issues"""
        filename = os.path.basename(file_path)
        logger.info(f"Attempting repair of: {filename}")
        
        repair_actions = []
        success = False
        
        try:
            # Check integrity first
            issues = self.check_file_integrity(file_path)
            
            if "File is empty (0 bytes)" in issues:
                repair_actions.append("Cannot repair empty file - marked for deletion")
                return False, repair_actions
            
            # Create backup
            backup_path = self.create_backup(file_path)
            if backup_path:
                repair_actions.append(f"Created backup: {backup_path}")
            
            # Try to repair file permissions
            try:
                os.chmod(file_path, 0o644)
                repair_actions.append("Fixed file permissions")
            except:
                pass
            
            # Check if file is actually readable after permission fix
            try:
                with open(file_path, 'rb') as f:
                    # Try to read the first 1KB
                    data = f.read(1024)
                    if data:
                        repair_actions.append("File is readable after permission fix")
                        success = True
            except Exception as e:
                repair_actions.append(f"File still unreadable: {str(e)}")
            
            # If file has header issues, try to detect actual format
            if any("Unrecognized file format" in issue for issue in issues):
                try:
                    with open(file_path, 'rb') as f:
                        header = f.read(32)  # Read more header data
                        
                    # Check for various NetCDF signatures
                    if b'netcdf' in header.lower() or b'hdf' in header.lower():
                        repair_actions.append("Detected NetCDF-like content despite header issues")
                        success = True
                    else:
                        repair_actions.append("File does not appear to be NetCDF format")
                        
                except Exception as e:
                    repair_actions.append(f"Could not analyze file content: {str(e)}")
            
        except Exception as e:
            repair_actions.append(f"Repair attempt failed: {str(e)}")
        
        # Log repair attempt
        self.repair_log.append({
            'filename': filename,
            'original_path': file_path,
            'backup_path': backup_path,
            'success': success,
            'actions': repair_actions,
            'timestamp': datetime.now().isoformat()
        })
        
        return success, repair_actions
    
    def quarantine_problematic_files(self, file_list, quarantine_dir="quarantined_files"):
        """Move problematic files to quarantine directory"""
        os.makedirs(quarantine_dir, exist_ok=True)
        quarantined = []
        
        for file_path in file_list:
            try:
                filename = os.path.basename(file_path)
                quarantine_path = os.path.join(quarantine_dir, filename)
                
                # Move file to quarantine
                shutil.move(file_path, quarantine_path)
                quarantined.append((file_path, quarantine_path))
                logger.info(f"Quarantined: {filename}")
                
            except Exception as e:
                logger.error(f"Could not quarantine {file_path}: {e}")
        
        return quarantined
    
    def batch_repair_files(self, file_pattern, max_files=None):
        """Attempt to repair multiple files"""
        files = glob.glob(file_pattern, recursive=True)
        
        if max_files:
            files = files[:max_files]
        
        logger.info(f"Starting batch repair of {len(files)} files")
        
        results = {
            'total_files': len(files),
            'successful_repairs': 0,
            'failed_repairs': 0,
            'unrepairable_files': []
        }
        
        for i, file_path in enumerate(files):
            logger.info(f"[{i+1}/{len(files)}] Processing: {os.path.basename(file_path)}")
            
            success, actions = self.attempt_file_repair(file_path)
            
            if success:
                results['successful_repairs'] += 1
                logger.info(f"  SUCCESS: {'; '.join(actions)}")
            else:
                results['failed_repairs'] += 1
                results['unrepairable_files'].append(file_path)
                logger.warning(f"  FAILED: {'; '.join(actions)}")
        
        return results
    
    def cleanup_empty_files(self, file_pattern):
        """Remove empty files that cannot be repaired"""
        files = glob.glob(file_pattern, recursive=True)
        removed_files = []
        
        for file_path in files:
            try:
                if os.path.getsize(file_path) == 0:
                    logger.info(f"Removing empty file: {os.path.basename(file_path)}")
                    os.remove(file_path)
                    removed_files.append(file_path)
            except Exception as e:
                logger.error(f"Could not remove {file_path}: {e}")
        
        return removed_files
    
    def generate_repair_report(self):
        """Generate comprehensive repair report"""
        print("\n" + "="*60)
        print("FILE REPAIR REPORT")
        print("="*60)
        
        if not self.repair_log:
            print("No repair attempts logged")
            return
        
        total_attempts = len(self.repair_log)
        successful = sum(1 for log in self.repair_log if log['success'])
        failed = total_attempts - successful
        
        print(f"Total repair attempts: {total_attempts}")
        print(f"Successful repairs: {successful} ({successful/total_attempts*100:.1f}%)")
        print(f"Failed repairs: {failed} ({failed/total_attempts*100:.1f}%)")
        
        # Most common repair actions
        all_actions = []
        for log in self.repair_log:
            all_actions.extend(log['actions'])
        
        from collections import Counter
        action_counts = Counter(all_actions)
        
        print(f"\nMost common repair actions:")
        for action, count in action_counts.most_common(10):
            print(f"  {count:3d}x: {action}")
        
        # Files that couldn't be repaired
        failed_files = [log['filename'] for log in self.repair_log if not log['success']]
        if failed_files:
            print(f"\nFiles that could not be repaired ({len(failed_files)}):")
            for filename in failed_files[:20]:  # Show first 20
                print(f"  - {filename}")
            if len(failed_files) > 20:
                print(f"  ... and {len(failed_files) - 20} more")
        
        # Export repair log
        self.export_repair_log()
    
    def export_repair_log(self, filename='file_repair_log.json'):
        """Export repair log to JSON"""
        try:
            with open(filename, 'w') as f:
                json.dump({
                    'repair_log': self.repair_log,
                    'export_timestamp': datetime.now().isoformat()
                }, f, indent=2)
            print(f"\nRepair log exported to: {filename}")
        except Exception as e:
            print(f"Could not export repair log: {e}")

def main():
    """Main repair function"""
    print("ARGO File Repair Utility")
    print("="*40)
    
    repair_util = ArgoFileRepairUtility()
    
    # File pattern for ARGO data
    file_pattern = "../../data/indian_ocean/raw/*.nc"
    
    print(f"Scanning for files: {file_pattern}")
    files = glob.glob(file_pattern)
    print(f"Found {len(files)} files")
    
    if not files:
        print("No files found! Check your data directory path.")
        return
    
    # Menu for repair options
    print("\nRepair Options:")
    print("1. Check file integrity only")
    print("2. Attempt repairs (max 50 files)")
    print("3. Clean up empty files")
    print("4. Quarantine problematic files")
    print("5. Full repair workflow")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    try:
        if choice == '1':
            # Check integrity only
            print("\nChecking file integrity...")
            problematic_files = []
            
            for file_path in files[:100]:  # Check first 100
                issues = repair_util.check_file_integrity(file_path)
                if issues:
                    print(f"ISSUES in {os.path.basename(file_path)}: {'; '.join(issues)}")
                    problematic_files.append(file_path)
            
            print(f"\nFound {len(problematic_files)} files with issues")
        
        elif choice == '2':
            # Attempt repairs
            print("\nAttempting repairs on up to 50 files...")
            results = repair_util.batch_repair_files(file_pattern, max_files=50)
            
            print(f"\nRepair Results:")
            print(f"  Total files: {results['total_files']}")
            print(f"  Successful repairs: {results['successful_repairs']}")
            print(f"  Failed repairs: {results['failed_repairs']}")
            
            repair_util.generate_repair_report()
        
        elif choice == '3':
            # Clean up empty files
            print("\nCleaning up empty files...")
            removed = repair_util.cleanup_empty_files(file_pattern)
            print(f"Removed {len(removed)} empty files")
        
        elif choice == '4':
            # Quarantine problematic files
            print("\nIdentifying problematic files for quarantine...")
            problematic_files = []
            
            for file_path in files:
                issues = repair_util.check_file_integrity(file_path)
                if any("empty" in issue.lower() or "unrecognized" in issue.lower() for issue in issues):
                    problematic_files.append(file_path)
            
            if problematic_files:
                confirm = input(f"Quarantine {len(problematic_files)} problematic files? (y/N): ")
                if confirm.lower() == 'y':
                    quarantined = repair_util.quarantine_problematic_files(problematic_files)
                    print(f"Quarantined {len(quarantined)} files")
            else:
                print("No files identified for quarantine")
        
        elif choice == '5':
            # Full workflow
            print("\nRunning full repair workflow...")
            
            # Step 1: Check integrity
            print("Step 1: Checking file integrity...")
            problematic_files = []
            for file_path in files:
                issues = repair_util.check_file_integrity(file_path)
                if issues:
                    problematic_files.append(file_path)
            
            print(f"Found {len(problematic_files)} problematic files")
            
            # Step 2: Attempt repairs
            if problematic_files:
                print("Step 2: Attempting repairs...")
                repair_results = repair_util.batch_repair_files(file_pattern, max_files=100)
                
                # Step 3: Quarantine unrepairable files
                if repair_results['unrepairable_files']:
                    print("Step 3: Quarantining unrepairable files...")
                    quarantined = repair_util.quarantine_problematic_files(repair_results['unrepairable_files'])
                    print(f"Quarantined {len(quarantined)} unrepairable files")
            
            # Step 4: Clean up empty files
            print("Step 4: Cleaning up empty files...")
            removed = repair_util.cleanup_empty_files(file_pattern)
            print(f"Removed {len(removed)} empty files")
            
            # Step 5: Generate final report
            repair_util.generate_repair_report()
            
            print(f"\nFull workflow completed!")
            print(f"Check the repair log and backup directory for details")
        
        else:
            print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        repair_util.generate_repair_report()
    
    except Exception as e:
        print(f"Error during repair operation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()