import os
import sys
import argparse
import numpy as np
import glob
import re

def find_raw_file_by_id(raw_folder, patient_id):
    """
    Find a raw file that contains the patient ID.
    Only selects *_L_*.raw files and ignores *_H_*.raw files.
    
    Args:
        raw_folder (str): Path to the folder containing raw files
        patient_id (str): Patient ID to search for (without trailing zeros)
    
    Returns:
        str: Path to the found raw file or None if not found
    """
    # Check if folder exists
    if not os.path.exists(raw_folder):
        print(f"Error: Raw folder does not exist: {raw_folder}")
        print(f"Current working directory: {os.getcwd()}")
        return None
        
    # Search for all raw files
    raw_files = glob.glob(os.path.join(raw_folder, "*.raw"))
    
    if not raw_files:
        print(f"Error: No .raw files found in {raw_folder}")
        return None
        
    print(f"Found {len(raw_files)} .raw files in {raw_folder}")
    
    # Look for files with the pattern where ID is in filename before "00_L_"
    for raw_file in raw_files:
        filename = os.path.basename(raw_file)
        print(f"Checking file: {filename}")
        
        # Look for the ID in the filename
        if patient_id + "00_L_" in filename:
            print(f"Found matching L file with ID {patient_id}: {filename}")
            return raw_file
    
    # If no L file found, log that information
    print(f"No file with ID '{patient_id}' followed by '00_L_' found")
    
    return None

def process_raw_file(raw_file_path, output_folder, patient_id):
    """
    Process a raw file and create obj and npy output files as point clouds.
    Uses the same method as the reference code but maintains the requested filename format.
    
    Args:
        raw_file_path (str): Path to the raw file
        output_folder (str): Path to the output folder for both obj and npy files
        patient_id (str): Patient ID to use for output filenames
    """
    import numpy as np
    import os
    import re
    
    if not os.path.exists(raw_file_path):
        print(f"Error: Raw file {raw_file_path} does not exist.")
        return False
    
    filename = os.path.basename(raw_file_path)
    
    # First pattern
    match = None
    pattern1 = r'_L_(\d+)x(\d+)x(\d+)_(\d+)u\.raw$'
    try:
        match = re.search(pattern1, filename)
    except Exception as e:
        print(f"Error with pattern1: {e}")
    
    # If first pattern doesn't match, try alternative
    if not match:
        pattern2 = r'_L_(\d+)x(\d+)x(\d+)x(\d+)u\.raw$'
        try:
            match = re.search(pattern2, filename)
        except Exception as e:
            print(f"Error with pattern2: {e}")
    
    if not match:
        print(f"Error: Could not parse dimensions from filename: {filename}")
        return False
    
    # Extract dimensions from the filename
    width = int(match.group(1))
    height = int(match.group(2))
    depth = int(match.group(3))
    bit_depth = int(match.group(4))
    
    # Determine numpy data type based on bit depth
    if bit_depth == 8:
        dtype = np.uint8
    elif bit_depth == 16:
        dtype = np.uint16
    else:
        print(f"Error: Unsupported bit depth: {bit_depth}")
        return False
    
    print(f"Processing file: {filename}")
    print(f"Dimensions: {width}x{height}x{depth}, {bit_depth}-bit")
    
    try:
        # Read the raw file into a numpy array
        with open(raw_file_path, 'rb') as f:
            data = np.fromfile(f, dtype=dtype)
        
        # Reshape the data according to the dimensions
        volume = data.reshape(depth, height, width)
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Extract the ID from the filename
        file_id = patient_id
        try:
            # Extract the full ID (all digits) before "_L_"
            full_id_match = re.search(r'[A-Za-z]*(\d+)_L_', filename)
            if full_id_match:
                full_id = full_id_match.group(1)
                # Check if it ends with "00" and extract the 5 digits before it
                if full_id.endswith('00'):
                    file_id = full_id[-7:-2]  # 5 digits before the trailing "00"
                else:
                    # Fallback: take the last 5 digits
                    file_id = full_id[-5:]
            print(f"Extracted ID from filename: {file_id}")
        except Exception as e:
            print(f"Warning: Could not extract ID from filename, using provided ID: {e}")
        
        # Define output filenames with consistent naming pattern (including dimensions for both)
        dimensions_str = f"{width}x{height}x{depth}"
        obj_file = os.path.join(output_folder, f"{file_id}_{dimensions_str}.obj")
        npy_file = os.path.join(output_folder, f"{file_id}_{dimensions_str}.npy")
        
        # Use fixed threshold of 64 as in the reference code
        threshold = 64
        print(f"Using threshold value: {threshold}")
        
        # Create a point cloud based on intensity threshold, exactly like the reference code
        indices = np.where(volume > threshold)
        points = np.stack(indices, axis=-1)
        
        # Calculate the center of the volume
        center = np.array(volume.shape) / 2
        
        # Shift the points to have the origin at the center
        shifted_points = points - center
        
        # Convert to float64 to match the desired output
        shifted_points = shifted_points.astype(np.float64)
        
        # Save as npy file (point cloud format)
        print(f"Saving point cloud .npy file to: {npy_file}")
        print(f"Point cloud shape: {shifted_points.shape}")
        print(f"Point cloud dtype: {shifted_points.dtype}")
        np.save(npy_file, shifted_points)
        
        # Create and save point cloud OBJ file - no comments, just vertices
        print(f"Saving point cloud .obj file to: {obj_file}")
        with open(obj_file, 'w') as f:
            # Write all vertices in one go using join (like the reference code)
            f.write('\n'.join(f"v {point[0]} {point[1]} {point[2]}" for point in shifted_points) + '\n')
        
        print("Processing completed successfully.")
        return True
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return False


def main():
    from license.license_check import check_license
    check_license()
    parser = argparse.ArgumentParser(description='Convert raw image files to obj and npy formats')
    parser.add_argument('--raw_folder', default='data/raw_voxel', help='Folder containing raw files (default: data/raw_voxel)')
    parser.add_argument('--id', required=True, help='Patient ID to process')
    parser.add_argument('--list', action='store_true', help='List all available raw files')
    
    args = parser.parse_args()
    
    # If --list is specified, just list all raw files in the folder
    if args.list:
        raw_files = glob.glob(os.path.join(args.raw_folder, "*.raw"))
        if not raw_files:
            print(f"No raw files found in {args.raw_folder}")
            return
        print(f"Available raw files in {args.raw_folder}:")
        for f in raw_files:
            print(f" - {os.path.basename(f)}")
        return
    
    # Create output folder based on patient ID
    output_folder = os.path.join(args.raw_folder, args.id)
    
    # Find the raw file matching the patient ID
    raw_file = find_raw_file_by_id(args.raw_folder, args.id)
    
    if not raw_file:
        print(f"Error: No suitable raw file found for patient ID {args.id}")
        # List a few raw files to help with debugging
        raw_files = glob.glob(os.path.join(args.raw_folder, "*.raw"))
        if raw_files:
            print("\nAvailable raw files (showing up to 5):")
            for f in raw_files[:5]:
                print(f" - {os.path.basename(f)}")
            print(f"Use --list to see all available files")
        sys.exit(1)
    
    # Process the found raw file
    success = process_raw_file(raw_file, output_folder, args.id)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()