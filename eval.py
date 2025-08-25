import os
import json
import argparse
import cv2
import numpy as np
import glob
import re


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_3d_bboxes(bbox_dir):
    """
    Load all 3D bounding boxes from JSON files in the specified directory.
    
    Args:
        bbox_dir: Directory containing 3D bbox JSON files
    
    Returns:
        dict: Dictionary mapping class names to 3D bbox coordinates
    """
    bboxes = {}
    
    if not os.path.exists(bbox_dir):
        print(f"No 3D bounding box directory found: {bbox_dir}")
        return bboxes
    
    # Sort files for consistent ordering
    json_files = sorted(glob.glob(os.path.join(bbox_dir, "*.json")))
    
    for json_file in json_files:
        class_name = os.path.basename(json_file).replace('.json', '')
        with open(json_file, 'r') as f:
            coords = json.load(f)
            bboxes[class_name] = np.array(coords)
    
    return bboxes


def load_2d_detections(json_file):
    """
    Load 2D detections from a JSON file.
    
    Args:
        json_file: Path to JSON file containing 2D detections
    
    Returns:
        list: List of detection dictionaries with bbox, label, and score
    """
    if not os.path.exists(json_file):
        return []
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    detections = []
    for item in data.get('data', []):
        detections.append({
            'bbox': item['bbox'],
            'label': item['label'],
            'score': item.get('score', 1.0)
        })
    
    return detections


# ============================================================================
# Projection Functions
# ============================================================================

def project_3d_to_2d_with_y(points_3d, fanbeam, image_shape, visual_dsd=1100):
    """
    Project 3D points to 2D image coordinates using camera calibration.
    X is perspective projected, Y is orthographic (independent).
    
    Args:
        points_3d: Nx3 array of 3D points [x, y, z]
        fanbeam: [Tx, Tz, theta, DSD] calibration parameters
        image_shape: (height, width) of the image
        visual_dsd: Visual source-to-detector distance offset
    
    Returns:
        np.array: Nx2 array of 2D points in image coordinates
    """
    H, W = image_shape
    
    # Extract camera parameters
    Tx, Tz, theta, DSD = fanbeam
    theta = theta * 0.01  # Convert to radians
    
    # Camera rotation matrix
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])
    
    # Camera translation vector
    T = np.array([Tx, 0, Tz])
    
    # Camera center in world coordinates
    cam_center = -R.T @ T
    cam_center[1] = 0  # Y position at ground level
    
    # Effective source-to-detector distance
    effective_dsd = DSD + visual_dsd
    
    # Image scaling and center calculation
    scale_factor = (DSD + visual_dsd) / DSD
    resized_w = int(W * scale_factor)
    resized_h = int(H * scale_factor)
    Cx = resized_w / 2
    Cy = resized_h / 2
    
    points_2d = []
    
    for pt in points_3d:
        # For X: perspective projection
        # Ray from camera center to 3D point
        ray_dir = np.array(pt) - cam_center
        
        # Transform ray to camera coordinates
        ray_dir_cam = R @ ray_dir
        cam_center_cam = R @ cam_center + T
        
        # Find intersection with detector plane (z = effective_dsd)
        if abs(ray_dir_cam[2]) > 1e-6:
            t = (effective_dsd - cam_center_cam[2]) / ray_dir_cam[2]
            image_point_cam = cam_center_cam + t * ray_dir_cam
            
            # X coordinate (perspective)
            u = image_point_cam[0] + Cx
            
            # Y coordinate (orthographic - direct mapping)
            # In 3D: positive Y is up
            # In 2D image: positive Y is down
            v = Cy - pt[1]  # Direct Y mapping (orthographic)
            
            # Scale back to original image size
            u = u / scale_factor
            v = v / scale_factor
            
            points_2d.append([u, v])
        else:
            # Ray parallel to detector plane (shouldn't happen)
            points_2d.append([0, 0])
    
    return np.array(points_2d)


def get_2d_bbox_from_3d_bbox(points_3d, fanbeam, image_shape, visual_dsd=1100):
    """
    Calculate 2D bounding box from 3D bounding box using orthographic Y projection.
    
    Args:
        points_3d: 8x3 array of 3D bbox corners
        fanbeam: Camera calibration parameters
        image_shape: (height, width) of the image
        visual_dsd: Visual DSD parameter
    
    Returns:
        list or None: [x1, y1, x2, y2] bounding box or None if out of bounds
    """
    H, W = image_shape
    
    # Project all points
    points_2d = project_3d_to_2d_with_y(points_3d, fanbeam, image_shape, visual_dsd)
    
    # Get X bounds from projection (only consider X values within image)
    valid_x_indices = (points_2d[:, 0] >= 0) & (points_2d[:, 0] <= W)
    if not np.any(valid_x_indices):
        return None
    
    valid_x_coords = points_2d[valid_x_indices, 0]
    x_min = int(np.min(valid_x_coords))
    x_max = int(np.max(valid_x_coords))
    
    # Get Y bounds directly from 3D bbox (orthographic)
    # Bottom face Y (points 0-3) and top face Y (points 4-7)
    y_3d_bottom = points_3d[0][1]  # Y of bottom face
    y_3d_top = points_3d[4][1]     # Y of top face
    
    # Convert to image coordinates
    # Scale factor for Y
    scale_factor = (fanbeam[3] + visual_dsd) / fanbeam[3]
    resized_h = int(H * scale_factor)
    Cy = resized_h / 2
    
    # Y mapping (orthographic)
    y_min = (Cy - y_3d_top) / scale_factor    # Top of 3D box -> smaller Y in image
    y_max = (Cy - y_3d_bottom) / scale_factor  # Bottom of 3D box -> larger Y in image
    
    y_min = int(y_min)
    y_max = int(y_max)
    
    # Check if bbox is within image bounds
    if x_max < 0 or x_min > W or y_max < 0 or y_min > H:
        return None
    
    # Clip to image bounds
    x_min = max(0, x_min)
    x_max = min(W, x_max)
    y_min = max(0, y_min)
    y_max = min(H, y_max)
    
    return [x_min, y_min, x_max, y_max]


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bbox [x1, y1, x2, y2]
        box2: Second bbox [x1, y1, x2, y2]
    
    Returns:
        float: IoU value between 0 and 1
    """
    if box1 is None or box2 is None:
        return 0.0
    
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


# ============================================================================
# Visualization Functions
# ============================================================================

def draw_bbox(image, bbox, label, color, thickness=2):
    """
    Draw bounding box with label on image.
    
    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        label: Text label for the box
        color: BGR color tuple
        thickness: Line thickness
    """
    if bbox is None:
        return
    
    x1, y1, x2, y2 = bbox
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label with background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    
    # Background rectangle for text
    cv2.rectangle(image, 
                  (x1, y1 - text_size[1] - 4), 
                  (x1 + text_size[0], y1), 
                  color, -1)
    
    # Draw text
    cv2.putText(image, label, (x1, y1 - 2), font, font_scale, 
                (0, 0, 0), font_thickness)


# ============================================================================
# File Processing Utilities
# ============================================================================

def extract_numeric_key(filename):
    """
    Extract numeric view index from filename.
    
    Args:
        filename: Name of the file
    
    Returns:
        int or None: Extracted view index
    """
    try:
        # Handle different naming conventions
        if '_he.json' in filename:
            return int(filename.split('_he.json')[0][-1])
        elif '_H.json' in filename:
            return int(filename.split('_H.json')[0][-1])
        elif '.json' in filename:
            # Try pattern like "00035671_0.json"
            base = filename.replace('.json', '')
            parts = base.split('_')
            if len(parts) > 1 and parts[-1].isdigit():
                return int(parts[-1])
        
        # Fallback: find any number in filename
        numbers = re.findall(r'\d+', filename)
        if numbers:
            # Return the last number found
            return int(numbers[-1])
            
    except Exception as e:
        print(f"Error extracting numeric key from {filename}: {e}")
    
    return None


def extract_image_view_index(filename):
    """
    Extract view index from image filename.
    
    Args:
        filename: Image filename
    
    Returns:
        int or None: View index
    """
    try:
        # Pattern like "00035671_0_he.png"
        if '_he.png' in filename:
            parts = filename.replace('_he.png', '').split('_')
            if len(parts) > 1 and parts[-1].isdigit():
                return int(parts[-1])
        
        # Pattern like "00035671_0_H.png"
        elif '_H.png' in filename:
            parts = filename.replace('_H.png', '').split('_')
            if len(parts) > 1 and parts[-1].isdigit():
                return int(parts[-1])
        
        # General pattern with underscore and number
        parts = filename.split('_')
        for i in range(len(parts)-1, -1, -1):
            if parts[i].split('.')[0].isdigit():
                return int(parts[i].split('.')[0])
                
    except Exception as e:
        print(f"Error extracting view index from image {filename}: {e}")
    
    return None


def create_sorted_file_mapping(directory, pattern, extract_func):
    """
    Create a sorted mapping of files by their view index.
    
    Args:
        directory: Directory containing files
        pattern: Glob pattern for files
        extract_func: Function to extract view index from filename
    
    Returns:
        dict: Mapping from view index to file path
    """
    file_mapping = {}
    files = glob.glob(os.path.join(directory, pattern))
    
    for file_path in files:
        filename = os.path.basename(file_path)
        view_idx = extract_func(filename)
        if view_idx is not None:
            file_mapping[view_idx] = file_path
    
    return file_mapping


# ============================================================================
# Main Evaluation Function
# ============================================================================

def main(args):
    """
    Main evaluation function that orchestrates the entire evaluation process.
    
    Args:
        args: Command line arguments
    """
    # Set default paths if not provided
    if not args.json_path:
        args.json_path = os.path.join('data/inference_results', args.id)
    if not args.bbox3d_path:
        args.bbox3d_path = os.path.join('data/bbox3d', args.id)
    
    # Load camera calibration
    print(f"Loading calibration data from: {args.calibration_path}")
    fanbeams = np.load(args.calibration_path)
    num_views = len(fanbeams)
    print(f"Found {num_views} camera views")
    
    # Load 3D bounding boxes
    bboxes_3d = load_3d_bboxes(args.bbox3d_path)
    print(f"Found {len(bboxes_3d)} 3D bounding boxes")
    
    # Create output directory for visualization
    if args.save_images:
        output_dir = os.path.join(args.output_dir, args.id)
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize evaluation metrics
    all_ious = []
    results_per_view = []
    
    # Get image dimensions from first image
    image_files = sorted(glob.glob(os.path.join('data/raw_image', args.id, '*.png')))
    if not image_files:
        print("Error: No images found")
        return
    
    first_image = cv2.imread(image_files[0])
    image_shape = first_image.shape[:2]  # (H, W)
    print(f"Image shape: {image_shape}")
    
    # Create sorted mappings for all file types
    print("\nCreating sorted file mappings...")
    json_mapping = create_sorted_file_mapping(args.json_path, '*.json', extract_numeric_key)
    image_mapping = create_sorted_file_mapping('data/raw_image/' + args.id, '*.png', extract_image_view_index)
    
    print(f"Found {len(json_mapping)} JSON files")
    print(f"Found {len(image_mapping)} image files")
    
    # Process views in order
    for view_idx in range(num_views):
        # Check if we have JSON file for this view
        if view_idx not in json_mapping:
            print(f"\nSkipping view {view_idx}: No JSON file found")
            continue
            
        json_file = json_mapping[view_idx]
        print(f"\nProcessing view {view_idx}: {os.path.basename(json_file)}")
        
        # Load 2D detections for this view
        detections_2d = load_2d_detections(json_file)
        print(f"  Found {len(detections_2d)} 2D detections")
        
        # Get calibration for this specific view
        fanbeam = fanbeams[view_idx]
        print(f"  Using calibration: Tx={fanbeam[0]:.2f}, Tz={fanbeam[1]:.2f}, "
              f"theta={fanbeam[2]*0.01:.3f}, DSD={fanbeam[3]}")
        
        view_results = {
            'view_idx': view_idx,
            'filename': os.path.basename(json_file),
            'matches': []
        }
        
        # Load corresponding image for visualization
        if args.save_images:
            if view_idx in image_mapping:
                image_path = image_mapping[view_idx]
                image = cv2.imread(image_path)
                print(f"  Loaded image: {os.path.basename(image_path)}")
            else:
                print(f"  Warning: No image found for view {view_idx}, using blank")
                image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
        
        # Process each 3D bounding box
        for class_name, points_3d in bboxes_3d.items():
            # Get 2D bbox using orthographic Y projection (same as visualization)
            bbox_3d_proj = get_2d_bbox_from_3d_bbox(points_3d, fanbeam, image_shape, args.visual_dsd)
            
            if bbox_3d_proj is None:
                print(f"  {class_name}: Outside image bounds")
                continue
            
            # Find best matching 2D detection
            best_iou = 0.0
            best_match = None
            
            for det_2d in detections_2d:
                iou = compute_iou(bbox_3d_proj, det_2d['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = det_2d
            
            if best_match and best_iou > 0.1:  # Threshold for valid match
                print(f"  {class_name}: Best match: {best_match['label']} with IoU = {best_iou:.3f}")
                print(f"    3D projection bbox: {bbox_3d_proj}")
                print(f"    2D ground truth bbox: {best_match['bbox']}")
                print(f"    3D height: {bbox_3d_proj[3] - bbox_3d_proj[1]}")
                print(f"    2D height: {best_match['bbox'][3] - best_match['bbox'][1]}")
                
                view_results['matches'].append({
                    'class_3d': class_name,
                    'class_2d': best_match['label'],
                    'iou': best_iou,
                    'bbox_3d_proj': bbox_3d_proj,
                    'bbox_2d': best_match['bbox']
                })
                all_ious.append(best_iou)
                
                # Draw 3D projection in MAGENTA (same as visualization)
                if args.save_images:
                    draw_bbox(image, bbox_3d_proj, f"3D-{class_name}", (255, 0, 255), 2)
            else:
                print(f"  {class_name}: No matching 2D detection found (best IoU = {best_iou:.3f})")
        
        # Draw all 2D detections in GREEN
        if args.save_images:
            for det_2d in detections_2d:
                draw_bbox(image, det_2d['bbox'], f"2D-{det_2d['label']}", (0, 255, 0), 1)
            
            # Save visualization image
            output_filename = f"eval_view_{view_idx}.png"
            cv2.imwrite(os.path.join(output_dir, output_filename), image)
            print(f"  Saved visualization to {output_filename}")
        
        results_per_view.append(view_results)
    
    # Print evaluation summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    if all_ious:
        mean_iou = np.mean(all_ious)
        print(f"3D ORTHOGRAPHIC Y PROJECTION METHOD:")
        print(f"  Overall mean IoU: {mean_iou:.3f}")
        print(f"  Total matches: {len(all_ious)}")
        print(f"  IoU range: [{min(all_ious):.3f}, {max(all_ious):.3f}]")
    else:
        print("No matches found!")
    
    # Save detailed results to JSON
    results_file = os.path.join(args.output_dir, f"eval_results_{args.id}.json")
    results_data = {
        'summary': {
            '3d_orthographic_y_method': {
                'mean_iou': float(np.mean(all_ious)) if all_ious else 0.0,
                'total_matches': len(all_ious)
            },
            'num_views': len(results_per_view),
            'num_3d_bboxes': len(bboxes_3d)
        },
        'per_view_results': results_per_view
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    if args.save_images:
        print(f"Visualization images saved to: {os.path.join(args.output_dir, args.id)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate IoU between 2D detections and reprojected 3D bounding boxes')
    parser.add_argument('--id', required=True, 
                       help='Patient ID to evaluate')
    parser.add_argument('--calibration_path', type=str, 
                       default="./data/calibration/calibration_results.npy", 
                       help='Calibration npy file path')
    parser.add_argument('--json_path', type=str, 
                       help='Path to 2D detection JSONs (default: data/inference_results/{id})')
    parser.add_argument('--bbox3d_path', type=str, 
                       help='Path to 3D bboxes (default: data/bbox3d/{id})')
    parser.add_argument('--output_dir', type=str, default='./data/eval_results', 
                       help='Output directory for evaluation results')
    parser.add_argument('--save_images', action='store_true', 
                       help='Save visualization images')
    parser.add_argument('--visual_dsd', type=float, default=1100, 
                       help='Visual DSD parameter')
    
    args = parser.parse_args()
    main(args)