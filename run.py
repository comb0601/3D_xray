#!/usr/bin/env python3
import argparse
import subprocess
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process patient data through multiple steps")
    parser.add_argument("--id", help="Patient ID to process")
    parser.add_argument("--raw_folder", help="Path to folder containing raw files (default from Python script: data/raw_voxel)")
    parser.add_argument("--list", action="store_true", help="List all available raw files in the folder and exit")
    parser.add_argument("--vis", action="store_true", help="Enable visualization in visual_hull.py")
    
    return parser.parse_args()

def run_command(command):
    print(f"Running: {' '.join(command)}")
    try:
        result = subprocess.run(command, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error executing command: {e}")
        return False

def main():
    from license.license_check import check_license
    check_license()
    args = parse_arguments()
    
    # Check if --list is specified
    if args.list:
        print("Listing all raw files...")
        raw_folder_args = ["--raw_folder", args.raw_folder] if args.raw_folder else []
        command = ["python", "1_raw_to_npy.py"] + raw_folder_args + ["--list", "--id", "dummy"]
        run_command(command)
        return 0
    
    # Check if patient ID is provided
    if not args.id:
        print("Error: Patient ID is required. Use --id option.")
        print("Use --help for usage information")
        return 1
    
    print(f"Starting processing for patient ID: {args.id}")
    
    # Step 1: Run raw_to_npy.py
    print("Step 1: Converting raw file to NPY and OBJ formats...")
    raw_folder_args = ["--raw_folder", args.raw_folder] if args.raw_folder else []
    step1_command = ["python", "1_raw_to_npy.py"] + raw_folder_args + ["--id", args.id]
    if not run_command(step1_command):
        print("Voxel conversion failed. Stopping.")
        return 1
    
    print("Step 1 completed successfully.")
    
    # Step 2: Run detection_2d.py
    print("Step 2: Running 2D detection...")
    step2_command = ["python", "2_detection_2d.py", "--input", f"data/raw_image/{args.id}"]
    if not run_command(step2_command):
        print("2D Detector failed. Stopping.")
        return 1
    
    print("Step 2 completed successfully.")
    
    # Step 3: Run visual_hull.py with visualization flag
    print("Step 3: Running visual hull processing...")
    visualization = "True" if args.vis else "False"
    print(f"Visualization {'enabled' if args.vis else 'disabled'}")
    step3_command = ["python", "3_visual_hull.py", "--name", args.id, "--visualization", visualization]
    if not run_command(step3_command):
        print("Error: Visual Hull failed. Stopping.")
        return 1
    
    print("Step 3 completed successfully.")
    print(f"All processing steps completed successfully for patient ID: {args.id}")
    return 0

if __name__ == "__main__":
    sys.exit(main())