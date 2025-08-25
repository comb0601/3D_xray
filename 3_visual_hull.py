import os
import re
import cv2
import json
import argparse
import numpy as np
import open3d as o3d
import time

# Check for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy detected: GPU acceleration enabled.")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not found: running on CPU.")

# Global configuration variables
VOXEL_SIZE = None          # 3D voxel space dimensions [x, y, z]
VISUAL_DSD = 1100         # Visual source-to-detector distance offset
VISUAL_HULL_RESULT = []   # Storage for visual hull computations
IMAGE_SIZE = None         # Original image dimensions [width, height]
W = None                  # Working width (may differ from original)
H = None                  # Working height (may differ from original)


# ============================================================================
# Visualization Helper Functions
# ============================================================================

def draw_cuboid(cube_center: np.array, cube_size: np.array):
    """
    Create a wireframe cuboid for visualization.
    
    Args:
        cube_center: 3D center position of the cuboid
        cube_size: Size of the cuboid in each dimension [x, y, z]
    
    Returns:
        o3d.geometry.LineSet: Combined wireframe geometry
    """
    # Create coordinate axes at origin
    colorlines = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    origin = np.array([[0], [0], [0], [1]])
    unit = 0.
    axes = np.array([[unit, 0, 0],
                    [0, unit, 0],
                    [0, 0, unit],
                    [1, 1, 1]]) * np.vstack([np.hstack([cube_size, cube_size, cube_size]), np.ones((1, 3))])
    points = np.vstack([np.transpose(origin), np.transpose(axes)])[:, :-1]
    points += cube_center.squeeze()
    lines = [[0, 1], [0, 2], [0, 3]]
    worldframe = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    worldframe.colors = o3d.utility.Vector3dVector(colorlines)
    
    # Create cuboid wireframe
    xyz_min = cube_center.squeeze() + np.array([-0.5, -0.5, -0.5]) * cube_size.squeeze()
    xyz_max = cube_center.squeeze() + np.array([0.5, 0.5, 0.5]) * cube_size.squeeze()
    points = [[xyz_min[0], xyz_min[1], xyz_min[2]],
            [xyz_max[0], xyz_min[1], xyz_min[2]],
            [xyz_min[0], xyz_max[1], xyz_min[2]],
            [xyz_max[0], xyz_max[1], xyz_min[2]],
            [xyz_min[0], xyz_min[1], xyz_max[2]],
            [xyz_max[0], xyz_min[1], xyz_max[2]],
            [xyz_min[0], xyz_max[1], xyz_max[2]],
            [xyz_max[0], xyz_max[1], xyz_max[2]]]
    
    lines = [
        [0, 1], [0, 2], [1, 3], [2, 3],  # Bottom face
        [4, 5], [4, 6], [5, 7], [6, 7],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set_bbox = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set_bbox.colors = o3d.utility.Vector3dVector(colors)
    return line_set_bbox + worldframe


def draw_camera_with_rays_to_3d_bbox(fanbeam, bbox_3d_points, camera_index, color=[1, 0, 0]):
    """
    Create camera visualization with rays projecting through 3D bounding box corners to image plane.
    
    Args:
        fanbeam: [Tx, Tz, theta, DSD] calibration parameters
        bbox_3d_points: 8x3 array of 3D bounding box corners
        camera_index: Index of the camera for labeling
        color: RGB color for the camera
    
    Returns:
        Tuple of (geometries list, camera center position)
    """
    global W, H, VISUAL_DSD
    
    # Extract camera parameters
    Tx, Tz, theta, DSD = fanbeam
    theta = theta * 0.01  # Convert to radians
    
    # Camera extrinsics
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])
    T = np.array([Tx, 0, Tz])
    
    # Camera center in world coordinates
    cam_center = -R.T @ T
    cam_center[1] = 0
    
    # Create pyramid representation for camera
    pyramid_height = 50
    pyramid_base = 30
    
    vertices_cam = np.array([
        [0, 0, 0],  # Apex (camera center)
        [-pyramid_base/2, -pyramid_base/2, -pyramid_height],
        [pyramid_base/2, -pyramid_base/2, -pyramid_height],
        [pyramid_base/2, pyramid_base/2, -pyramid_height],
        [-pyramid_base/2, pyramid_base/2, -pyramid_height],
    ])
    
    # Transform to world coordinates
    vertices_world = []
    for v in vertices_cam:
        v_world = R.T @ v + cam_center
        vertices_world.append(v_world)
    vertices_world = np.array(vertices_world)
    
    # Define pyramid faces
    triangles = [
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
        [1, 3, 2], [1, 4, 3],
    ]
    
    # Create camera mesh
    camera_mesh = o3d.geometry.TriangleMesh()
    camera_mesh.vertices = o3d.utility.Vector3dVector(vertices_world)
    camera_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    camera_mesh.compute_vertex_normals()
    camera_mesh.paint_uniform_color(color)
    
    # Create coordinate frame at camera location
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=40, origin=cam_center
    )
    camera_frame.rotate(R.T, center=cam_center)
    
    # Add camera index label
    label_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
    label_sphere.translate(cam_center + np.array([0, pyramid_base, 0]))
    label_sphere.paint_uniform_color(color)
    
    rays = []
    
    # Create rays extending through 3D bbox corners to image plane
    if bbox_3d_points is not None and len(bbox_3d_points) >= 8:
        # Only shoot rays to corners that define the width (diagonal corners)
        width_corners = [0, 2]  # Two diagonal corners of bottom face
        
        effective_dsd = DSD + VISUAL_DSD
        
        for i in width_corners:
            corner_world = bbox_3d_points[i].copy()
            
            # Transform corner to camera coordinates
            corner_cam = R @ corner_world + T
            
            # Extend ray to image plane (where z = effective_dsd)
            if corner_cam[2] > 0:  # Make sure point is in front of camera
                # Calculate scale factor to extend to image plane
                scale = effective_dsd / corner_cam[2]
                corner_on_image_cam = corner_cam * scale
                corner_on_image_cam[2] = effective_dsd
                
                # Transform back to world coordinates
                corner_on_image_world = R.T @ (corner_on_image_cam - T)
                corner_on_image_world[1] = corner_on_image_cam[1]  # Preserve Y
                
                # Create ray from camera through 3D point to image plane
                points = np.vstack([cam_center, corner_world, corner_on_image_world])
                lines = [[0, 1], [1, 2]]  # Two segments: camera to 3D point, 3D point to image
                ray = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(points),
                    lines=o3d.utility.Vector2iVector(lines),
                )
                
                # Different color intensity for the two segments
                colors_ray = [color, [c * 0.5 for c in color]]  # Second segment is dimmer
                ray.colors = o3d.utility.Vector3dVector(colors_ray)
                rays.append(ray)
    
    # Combine meshes
    combined_mesh = camera_mesh + camera_frame + label_sphere
    geometries = [combined_mesh] + rays
    
    return geometries, cam_center


def project_3d_bbox_to_image(fanbeam, bbox_3d_points):
    """
    Project 3D bounding box corners to 2D image coordinates.
    
    Args:
        fanbeam: Camera calibration parameters
        bbox_3d_points: 8x3 array of 3D bounding box corners
    
    Returns:
        2D projected points array
    """
    global W, H, VISUAL_DSD
    
    # Extract camera parameters
    Tx, Tz, theta, DSD = fanbeam
    theta = theta * 0.01  # Convert to radians
    
    # Apply scale factor for visualization
    scale_factor = (DSD + VISUAL_DSD) / DSD
    resized_w = int(W * scale_factor)
    resized_h = int(H * scale_factor)
    Cx = resized_w / 2
    Cy = resized_h / 2
    
    effective_dsd = DSD + VISUAL_DSD
    
    # Camera extrinsics
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])
    T = np.array([Tx, 0, Tz])
    
    # Project each 3D point to 2D
    points_2d = []
    for point_3d in bbox_3d_points:
        # Transform to camera coordinates
        point_cam = R @ point_3d + T
        
        # Project to 2D
        if point_cam[2] > 0:  # Point is in front of camera
            u = (point_cam[0] * effective_dsd / point_cam[2]) + Cx
            v = Cy - point_cam[1]  # Y is same in 2D and 3D space
            points_2d.append([u, v])
        else:
            points_2d.append([np.nan, np.nan])
    
    return np.array(points_2d)


def draw_image_with_3d_bbox_projection(fanbeam, image_path, bbox_3d_points, original_2d_bbox=None):
    """
    Create textured mesh for X-ray image with 3D bbox projection overlay.
    
    Args:
        fanbeam: Camera calibration parameters
        image_path: Path to the image file
        bbox_3d_points: 8x3 array of 3D bounding box corners
        original_2d_bbox: Original 2D bounding box [x1, y1, x2, y2] for this view (not used anymore)
    
    Returns:
        o3d.geometry.TriangleMesh: Textured mesh representing the image with overlay
    """
    global W, H, VISUAL_DSD, VOXEL_SIZE
    
    # Load and resize image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    scale_factor = (fanbeam[3] + VISUAL_DSD) / fanbeam[3]
    resized_w = int(W * scale_factor)
    resized_h = int(H * scale_factor)
    image = cv2.resize(image, (resized_w, resized_h))
    
    # Project 3D bbox to 2D if provided
    if bbox_3d_points is not None:
        points_2d = project_3d_bbox_to_image(fanbeam, bbox_3d_points)
        
        # Use 3D Y coordinates directly (orthographic projection in Y)
        bbox_color = (255, 0, 255)  # Magenta
        thickness = 3
        
        # Get X coordinates from 3D projection (for width)
        x_coords = [pt[0] for pt in points_2d[:4] if not np.isnan(pt[0])]
        if len(x_coords) >= 2:
            x_min = min(x_coords)
            x_max = max(x_coords)
            
            # Get Y coordinates from 3D bbox directly (orthographic in Y)
            # Bottom face Y (points 0-3) and top face Y (points 4-7)
            y_3d_bottom = bbox_3d_points[0][1]  # Y of bottom face
            y_3d_top = bbox_3d_points[4][1]     # Y of top face
            
            # Convert 3D Y to 2D image Y (Y-axis is independent/orthographic)
            # In 3D: positive Y is up
            # In 2D image: positive Y is down
            # The conversion is: y_2d = Cy - y_3d
            Cy = resized_h / 2
            y_min = Cy - y_3d_top     # Top of 3D box -> smaller Y in image
            y_max = Cy - y_3d_bottom  # Bottom of 3D box -> larger Y in image
            
            # Draw rectangle with projected X and 3D-derived Y
            cv2.rectangle(image, 
                         (int(x_min), int(y_min)), 
                         (int(x_max), int(y_max)), 
                         bbox_color, thickness)
            
            # Draw corner markers
            corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
            for corner in corners:
                cv2.circle(image, (int(corner[0]), int(corner[1])), 5, bbox_color, -1)
    
    Cx = resized_w / 2
    Cy = resized_h / 2
    
    # Camera parameters
    DSD = fanbeam[3] + VISUAL_DSD
    T = np.array([fanbeam[0], 0, fanbeam[1]])
    theta_y = fanbeam[2] * 0.01
    R = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)]])
    
    # Create image corners in world coordinates
    pts1 = R.T @ (np.array([-Cx, Cy, DSD]) - T)
    pts1[1] = Cy
    
    pts2 = R.T @ (np.array([Cx, Cy, DSD]) - T)
    pts2[1] = Cy
    
    pts3 = R.T @ (np.array([-Cx, -Cy, DSD]) - T)
    pts3[1] = -Cy
    
    pts4 = R.T @ (np.array([Cx, -Cy, DSD]) - T)
    pts4[1] = -Cy
    
    # Create front-facing mesh
    vertices = np.array([pts3, pts4, pts2, pts1])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    uv = np.array([[0, 1], [1, 1], [1, 0], 
                  [0, 1], [1, 0], [0, 0]])
    mesh.textures = [o3d.geometry.Image(image)]
    mesh.triangle_uvs = o3d.utility.Vector2dVector(uv)
    mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(triangles))
    
    # Create back-facing mesh (flipped)
    vertices_ = np.array([pts4, pts3, pts1, pts2])
    triangles_ = np.array([[0, 1, 2], [0, 2, 3]])
    mesh_ = o3d.geometry.TriangleMesh()
    mesh_.vertices = o3d.utility.Vector3dVector(vertices_)
    mesh_.triangles = o3d.utility.Vector3iVector(triangles_)
    
    image_flip = cv2.flip(image, 1)
    mesh_.textures = [o3d.geometry.Image(image_flip)]
    mesh_.triangle_uvs = o3d.utility.Vector2dVector(uv)
    mesh_.triangle_material_ids = o3d.utility.IntVector([0] * len(triangles))
    
    mesh += mesh_
    return mesh


def get_raw_3d_optimized(path):
    """
    Load and transform raw 3D voxel data from .npy file.
    
    Args:
        path: Directory containing the .npy file
    
    Returns:
        o3d.geometry.PointCloud: Transformed 3D point cloud
    """
    global VOXEL_SIZE
    
    # Find .npy file
    npy_files = [f for f in os.listdir(path) if f.endswith('.npy')]
    if not npy_files:
        raise FileNotFoundError("No .npy files found in the specified directory.")
    raw_file_path = os.path.join(path, npy_files[-1])
    
    # Extract voxel dimensions from filename
    voxel_size_match = re.search(r'(\d+)x(\d+)x(\d+)', raw_file_path)
    if voxel_size_match:
        VOXEL_SIZE = [int(voxel_size_match.group(1)), 
                    int(voxel_size_match.group(3)), 
                    int(voxel_size_match.group(2))]
        print(f"VOXEL_SIZE : {VOXEL_SIZE}")
    else:
        VOXEL_SIZE = [512, 630, 512]
        print("No voxel size found in filename. Using default dimensions:", VOXEL_SIZE)

    # Load data
    raw_cpu = np.load(raw_file_path)
    
    # Apply coordinate transformations
    if GPU_AVAILABLE:
        # GPU processing
        raw_gpu = cp.asarray(raw_cpu)
        raw_gpu[:, 1], raw_gpu[:, 2] = raw_gpu[:, 2].copy(), raw_gpu[:, 1].copy()
        theta_z = -90 * np.pi / 180
        rot_z = cp.array([[cp.cos(theta_z), -cp.sin(theta_z), 0],
                        [cp.sin(theta_z),  cp.cos(theta_z), 0],
                        [0,                0,               1]])
        raw_gpu = cp.dot(raw_gpu, rot_z.T)
        raw = cp.asnumpy(raw_gpu)
        raw[:, 2] *= -1
        print("************Using GPU************")
    else:
        # CPU processing
        raw_cpu[:, 1], raw_cpu[:, 2] = raw_cpu[:, 2].copy(), raw_cpu[:, 1].copy()
        theta_z = -90 * np.pi / 180
        rot_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                        [np.sin(theta_z),  np.cos(theta_z), 0],
                        [0,                0,               1]])
        raw = np.dot(raw_cpu, rot_z.T)
        raw[:, 2] *= -1
        print("************Using CPU************")
    
    # Create point cloud
    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(raw))
    point_cloud.paint_uniform_color([0, 0, 1])
    return point_cloud


def get_image_size(image_dir):
    """
    Get dimensions of images in the directory.
    
    Args:
        image_dir: Directory containing images
    
    Returns:
        [width, height] of the images
    """
    image_ls = os.listdir(image_dir)
    image_path = [os.path.join(image_dir, img) for img in image_ls if img.endswith('.png')]
    image = cv2.cvtColor(cv2.imread(image_path[0]), cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape  
    print(f"IMAGE_SIZE: [{width}, {height}]")
    return [width, height]


def extract_numeric_key(filename):
    """
    Extract numeric view index from filename.
    
    Args:
        filename: Name of the file
    
    Returns:
        Numeric key or None if not found
    """
    try:
        if '_he.json' in filename:
            return int(filename.split('_he.json')[0][-1])
        elif '_H.json' in filename:
            return int(filename.split('_H.json')[0][-1])
        else:
            match = re.search(r'(\d+)\.json$', filename)
            if match:
                return int(match.group(1))
            else:
                match = re.search(r'(\d+)[^0-9]*\.json$', filename)
                if match:
                    return int(match.group(1))
                else:
                    print(f"Warning: Could not extract numeric key from {filename}")
                    return None
    except Exception as e:
        print(f"Error extracting numeric key from {filename}: {e}")
        return None


# ============================================================================
# Visual Hull Computation Functions
# ============================================================================

def get_visual_hull(fanbeams, bboxes, args):
    """
    Compute visual hull from multiple 2D bounding boxes.
    
    Args:
        fanbeams: Camera calibration parameters for all views
        bboxes: 2D bounding boxes for each view
        args: Command line arguments
    
    Returns:
        o3d.geometry.LineSet: Visual hull as thick lines
    """
    global VISUAL_HULL_RESULT, VOXEL_SIZE, W, H
    
    # Initialize voxel plane for intersection
    voxel_plane_size = (VOXEL_SIZE[0], VOXEL_SIZE[2]) 
    voxel_plane = np.ones(voxel_plane_size)

    line_set_box = None
    
    # Project each bbox constraint onto the voxel plane
    for i in range(len(fanbeams)):
        fanbeam = fanbeams[i]
        bbox = bboxes[i]
        if sum(bbox) == 0:  # Skip empty bboxes
            continue

        # Apply margin to bbox
        x_margin = (bbox[2] - bbox[0]) * args.margin[0]
        Cx = W / 2 
        Cy = H / 2
        dsd = fanbeam[3]

        # Camera intrinsics
        k = np.array([[dsd, 0, Cx],
                    [0, dsd, Cy],
                    [0, 0, 1]])
        
        # Camera extrinsics
        theta_y = fanbeam[2] * 0.01
        r = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])
        t = np.array([fanbeam[0], 0, fanbeam[1]])
        rt = np.hstack((r, t[:, None]))
        
        # Create grid of 3D points on the ground plane
        x = np.arange(voxel_plane_size[0]) - voxel_plane_size[0] // 2
        z = np.arange(voxel_plane_size[1]) - voxel_plane_size[1] // 2 
        xz = np.meshgrid(x, z)  
        
        # Project 3D points to 2D
        p3d = np.array([xz[0], np.zeros_like(xz[0]), xz[1], np.ones_like(xz[0])])
        p2d = np.dot(k, np.dot(rt, p3d.reshape(4, -1)))
        p2d = p2d / p2d[2]
        u = p2d[0].reshape(voxel_plane_size)
        
        # Mark points outside bbox as invalid
        unsatisfied_mask = (u < (bbox[0] - x_margin)) | (u > (bbox[2] + x_margin))
        voxel_plane[unsatisfied_mask] = 0
        
    if voxel_plane.sum() == 0:
        print("There is no visual hull!")
    else:
        # Find contour of valid region
        contours, _ = cv2.findContours(voxel_plane.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        box = box - voxel_plane_size[0] // 2

        VISUAL_HULL_RESULT.append(-box)
        
        # Calculate Y bounds from 2D bboxes
        non_zero_bboxes = bboxes[np.all(bboxes != 0, axis=1)]
        y_min_img = min(non_zero_bboxes[:,1])  # Top in image
        y_max_img = max(non_zero_bboxes[:,3])  # Bottom in image
        
        # Convert to 3D Y coordinates
        y_3d_top = VOXEL_SIZE[1] // 2 - y_min_img
        y_3d_bottom = VOXEL_SIZE[1] // 2 - y_max_img
        
        # Apply margin
        y_margin = abs(y_3d_top - y_3d_bottom) * args.margin[1]
        y_3d = [y_3d_bottom - y_margin, y_3d_top + y_margin]

        # Create 3D box
        box_3d = []
        # Bottom face
        for x, z in box:
            box_3d.append([x, y_3d[0], -z])
        # Top face
        for x, z in box:
            box_3d.append([x, y_3d[1], -z])

        points = np.array(box_3d, dtype=float)
        points[:, 2] *= -1  # Flip Z axis
        points = points.astype(int)
        
        # Define edges
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
        ]
    
        # Create thick cylindrical lines
        line_sets = []
        line_thickness = args.line_thickness
        
        for line in lines:
            start = np.array(points[line[0]])
            end = np.array(points[line[1]])
            direction = end - start
            length = np.linalg.norm(direction)
            
            if length < 1e-6:
                continue
                
            direction = direction / length
            
            # Calculate perpendicular directions
            if abs(direction[1]) > 0.9:  # Nearly vertical
                perp1 = np.cross(direction, np.array([1, 0, 0]))
                perp2 = np.cross(direction, perp1)
            else:
                perp1 = np.cross(direction, np.array([0, 1, 0]))
                perp2 = np.cross(direction, perp1)
            
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = perp2 / np.linalg.norm(perp2)
            
            # Create cylindrical representation
            num_points = max(16, int(line_thickness * 8))
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                offset = line_thickness * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                line_points = np.array([start + offset, end + offset])
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(line_points),
                    lines=o3d.utility.Vector2iVector([[0, 1]]),
                )
                line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
                line_sets.append(line_set)
        
        # Combine all line sets
        if line_sets:
            line_set_box = line_sets[0]
            for line_set in line_sets[1:]:
                line_set_box += line_set

    return line_set_box


def check_ray_intersection(fanbeam1, bbox1, fanbeam2, bbox2):
    """
    Check if rays from two bboxes intersect in the x-z plane.
    
    Args:
        fanbeam1, bbox1: First camera and bbox
        fanbeam2, bbox2: Second camera and bbox
    
    Returns:
        Boolean indicating if rays intersect
    """
    global W, H, VISUAL_DSD
    
    # Process first bbox
    scale_factor1 = (fanbeam1[3] + VISUAL_DSD) / fanbeam1[3]
    resized_w1 = int(W * scale_factor1)
    bbox1 = bbox1.copy()
    bbox1[0] *= scale_factor1
    bbox1[2] *= scale_factor1
    Cx1 = resized_w1 / 2
    Cy1 = H / 2
    
    dsd1 = fanbeam1[3] + VISUAL_DSD
    t1 = np.array([fanbeam1[0], 0, fanbeam1[1]])
    theta_y1 = fanbeam1[2] * 0.01
    r1 = np.array([[np.cos(theta_y1), 0, np.sin(theta_y1)],
                  [0, 1, 0],
                  [-np.sin(theta_y1), 0, np.cos(theta_y1)]])
    cam_center1 = -np.dot(r1.T, t1)
    
    # Get ray endpoints for first bbox
    left_x1 = bbox1[0]
    right_x1 = bbox1[2]
    top_y1 = bbox1[1]
    
    # Create rays from top corners
    top_left_x1 = np.array([left_x1 - Cx1, 0, dsd1])
    leftx_world1 = r1.T @ (top_left_x1 - t1)
    leftx_world1[1] = -top_y1 + Cy1
    
    top_right_x1 = np.array([right_x1 - Cx1, 0, dsd1])
    rightx_world1 = r1.T @ (top_right_x1 - t1)
    rightx_world1[1] = -top_y1 + Cy1
    
    cam_center1[1] = -top_y1 + Cy1
    
    # Process second bbox
    scale_factor2 = (fanbeam2[3] + VISUAL_DSD) / fanbeam2[3]
    resized_w2 = int(W * scale_factor2)
    bbox2 = bbox2.copy()
    bbox2[0] *= scale_factor2
    bbox2[2] *= scale_factor2
    Cx2 = resized_w2 / 2
    Cy2 = H / 2
    
    dsd2 = fanbeam2[3] + VISUAL_DSD
    t2 = np.array([fanbeam2[0], 0, fanbeam2[1]])
    theta_y2 = fanbeam2[2] * 0.01
    r2 = np.array([[np.cos(theta_y2), 0, np.sin(theta_y2)],
                  [0, 1, 0],
                  [-np.sin(theta_y2), 0, np.cos(theta_y2)]])
    cam_center2 = -np.dot(r2.T, t2)
    
    left_x2 = bbox2[0]
    right_x2 = bbox2[2]
    top_y2 = bbox2[1]
    
    # Create rays from top corners
    top_left_x2 = np.array([left_x2 - Cx2, 0, dsd2])
    leftx_world2 = r2.T @ (top_left_x2 - t2)
    leftx_world2[1] = -top_y2 + Cy2
    
    top_right_x2 = np.array([right_x2 - Cx2, 0, dsd2])
    rightx_world2 = r2.T @ (top_right_x2 - t2)
    rightx_world2[1] = -top_y2 + Cy2
    
    cam_center2[1] = -top_y2 + Cy2
    
    # Define rays
    rays1 = [
        (cam_center1, leftx_world1),
        (cam_center1, rightx_world1)
    ]
    
    rays2 = [
        (cam_center2, leftx_world2),
        (cam_center2, rightx_world2)
    ]
    
    # Check intersection in 2D (x-z plane)
    for ray1 in rays1:
        for ray2 in rays2:
            p1, p2 = ray1
            p3, p4 = ray2
            
            # Convert to 2D by ignoring y
            p1_2d = np.array([p1[0], p1[2]])
            p2_2d = np.array([p2[0], p2[2]])
            p3_2d = np.array([p3[0], p3[2]])
            p4_2d = np.array([p4[0], p4[2]])
            
            # Calculate intersection
            d1 = p2_2d - p1_2d
            d2 = p4_2d - p3_2d
            
            cross = np.cross(d1, d2)
            
            if np.abs(cross) > 1e-6:
                t = np.cross(p3_2d - p1_2d, d2) / cross
                s = np.cross(p3_2d - p1_2d, d1) / cross
                
                # Check if intersection is within ray segments (with margin)
                if -0.5 <= t <= 1.5 and -0.5 <= s <= 1.5:
                    return True
    
    return False


# ============================================================================
# Bounding Box Extraction and Grouping Functions
# ============================================================================

def extract_bboxes_from_json(directory_path, args):
    """
    Extract 2D bounding boxes from JSON detection files and group them into 3D candidates.
    
    Args:
        directory_path: Directory containing JSON detection files
        args: Command line arguments
    
    Returns:
        Tuple of (candidates_3d array, class_counts dictionary)
    """
    # Initialize data structures
    candidates_3d = np.array([])
    class_counts = {}
    used_bboxes = set()
    fanbeams = np.load(args.calibration_path)
    
    # Process each JSON file
    for filename in sorted(os.listdir(directory_path)):
        if not filename.endswith('.json'):
            continue
            
        try:
            numeric_key = extract_numeric_key(filename)
            if numeric_key is None:
                continue
                
            file_path = os.path.join(directory_path, filename)
            
            with open(file_path, 'r') as file:
                data = json.load(file)
                
                # Process each detection in the file
                for bbox_idx, item in enumerate(data['data']):
                    bbox = item['bbox']
                    bbox_key = (numeric_key, bbox_idx)
                    
                    if bbox_key in used_bboxes:
                        continue
                        
                    found = False
                    
                    # Try to match with existing candidates
                    for candidate_idx in range(len(candidates_3d)):
                        non_zero_indices = np.where(np.any(candidates_3d[candidate_idx] != 0, axis=1))[0]
                        if not non_zero_indices.size:
                            continue
                            
                        # Skip if view already exists in candidate
                        if numeric_key in non_zero_indices:
                            continue
                        
                        # Check height similarity
                        existing_heights = candidates_3d[candidate_idx][non_zero_indices][:, [1, 3]]
                        avg_y1 = np.mean(existing_heights[:, 0])
                        avg_y2 = np.mean(existing_heights[:, 1])
                        
                        if not (abs(bbox[1] - avg_y1) <= 50 and abs(bbox[3] - avg_y2) <= 50):
                            continue
                        
                        # Check ray intersection
                        has_intersection = False
                        for existing_idx in non_zero_indices:
                            existing_bbox = candidates_3d[candidate_idx][existing_idx]
                            if check_ray_intersection(fanbeams[numeric_key], bbox, 
                                                    fanbeams[existing_idx], existing_bbox):
                                has_intersection = True
                                break
                        
                        if has_intersection:
                            # Add to existing candidate
                            candidates_3d[candidate_idx][numeric_key] = bbox
                            used_bboxes.add(bbox_key)
                            class_name = item['label']
                            if candidate_idx not in class_counts:
                                class_counts[candidate_idx] = {}
                            class_counts[candidate_idx][class_name] = class_counts[candidate_idx].get(class_name, 0) + 1
                            found = True
                            break
                    
                    # Create new candidate if no match found
                    if not found:
                        new_candidate = np.zeros((1, 9, 4))  # 9 views, 4 bbox coords
                        new_candidate[0][numeric_key] = bbox
                        candidates_3d = np.vstack((candidates_3d, new_candidate)) if len(candidates_3d) else new_candidate
                        used_bboxes.add(bbox_key)
                        class_name = item['label']
                        class_counts[len(candidates_3d)-1] = {class_name: 1}
                        
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue

    # Print most common class for each candidate
    for i in range(len(candidates_3d)):
        if i in class_counts:
            most_common_class = max(class_counts[i].items(), key=lambda x: x[1])[0]
            print(f"Candidate {i} most common class: {most_common_class}")

    return candidates_3d, class_counts


def save_bounding_boxes(visual_hull_results, bboxes_candidates, class_counts, candidates, args):
    """
    Save 3D bounding boxes to JSON files.
    
    Args:
        visual_hull_results: Computed visual hull regions
        bboxes_candidates: 2D bbox candidates
        class_counts: Class counts for each candidate
        candidates: Boolean array indicating valid candidates
        args: Command line arguments
    """
    global VOXEL_SIZE
    
    # Create output directory
    output_dir = os.path.join("data/bbox3d", args.name)
    os.makedirs(output_dir, exist_ok=True)
    
    count = -1
    for i in range(len(candidates)):
        if not candidates[i]:
            continue
            
        count += 1
        # Get most common class name
        class_name = max(class_counts[i].items(), key=lambda x: x[1])[0]
        
        # Format output filename
        output_file = os.path.join(output_dir, f"{class_name}.json")
        
        # Generate 3D bounding box
        if count < len(visual_hull_results):
            cuboid_coords = visual_hull_results[count]
            obj_bboxes = bboxes_candidates[i]
            non_zero_bboxes = obj_bboxes[np.all(obj_bboxes != 0, axis=1)]
            
            # Calculate Y bounds
            y_min_img = min(non_zero_bboxes[:,1])
            y_max_img = max(non_zero_bboxes[:,3])

            y_3d_top = VOXEL_SIZE[1] // 2 - y_min_img
            y_3d_bottom = VOXEL_SIZE[1] // 2 - y_max_img

            y_margin = abs(y_3d_top - y_3d_bottom) * args.margin[1]
            y_3d = [y_3d_bottom - y_margin, y_3d_top + y_margin]

            # Apply coordinate transformations
            cuboid_coords_flipped = np.copy(cuboid_coords)
            cuboid_coords_flipped[:, 0] *= -1  # Y flip
            cuboid_coords_flipped[:, 1] *= -1  # Z flip

            # Create 3D box vertices
            box_3d = []
            # Bottom face
            for x, z in cuboid_coords_flipped:
                box_3d.append([x, y_3d[0], -z])
            # Top face
            for x, z in cuboid_coords_flipped:
                box_3d.append([x, y_3d[1], -z])

            points = np.array(box_3d, dtype=float)
            points[:, 2] *= -1
            coords_list = points.astype(int).tolist()

            # Save to JSON
            with open(output_file, 'w') as f:
                json.dump(coords_list, f)
            print(f"Saved 3D bounding box for {class_name} to {output_file}")


# ============================================================================
# Visualization Helper Functions
# ============================================================================

def draw_all_cameras_with_rays_to_3d_bbox(fanbeams, bbox_3d_points):
    """
    Create camera visualizations with rays to 3D bbox corners for all views.
    
    Args:
        fanbeams: Array of calibration parameters for all cameras
        bbox_3d_points: 8x3 array of 3D bounding box corners
    
    Returns:
        Tuple of (individual camera geometries, camera centers)
    """
    individual_camera_geoms = []
    camera_centers = []
    
    # Define colors for different cameras
    colors = [
        [1, 0, 0],    # Red
        [0, 1, 0],    # Green
        [0, 0, 1],    # Blue
        [1, 1, 0],    # Yellow
        [1, 0, 1],    # Magenta
        [0, 1, 1],    # Cyan
        [0.5, 0.5, 0],  # Olive
        [0.5, 0, 0.5],  # Purple
        [0, 0.5, 0.5],  # Teal
    ]
    
    for i, fanbeam in enumerate(fanbeams):
        color = colors[i % len(colors)]
        
        camera_geoms, cam_center = draw_camera_with_rays_to_3d_bbox(fanbeam, bbox_3d_points, i, color)
        individual_camera_geoms.append(camera_geoms)
        camera_centers.append(cam_center)
    
    return individual_camera_geoms, camera_centers


def toggle_camera_visibility(vis, image, camera_geoms, cameras_visible, index):
    """
    Toggle visibility of camera, rays, and image.
    
    Args:
        vis: Open3D visualizer
        image: Image geometry
        camera_geoms: Camera geometries
        cameras_visible: Visibility state array
        index: Camera index
    """
    if cameras_visible[index]:
        # Turn off
        vis.remove_geometry(image)
        for geom in camera_geoms:
            vis.remove_geometry(geom)
    else:
        # Turn on
        vis.add_geometry(image)
        for geom in camera_geoms:
            vis.add_geometry(geom)

    cameras_visible[index] = not cameras_visible[index]
    vis.poll_events()
    vis.update_renderer()


# ============================================================================
# Main Processing Function
# ============================================================================

def main(args):
    """
    Main processing function for 3D visual hull generation.
    
    Args:
        args: Command line arguments
    """
    from license.license_check import check_license
    check_license()
    
    global H, W, VISUAL_HULL_RESULT, VOXEL_SIZE, IMAGE_SIZE
    start = time.perf_counter()

    # Load raw 3D data
    raw_3d = get_raw_3d_optimized(args.raw3d_path)

    # Create voxel space visualization
    voxel_size = np.array([[VOXEL_SIZE[0]], [VOXEL_SIZE[1]], [VOXEL_SIZE[2]]])
    empty_cuboid = draw_cuboid(np.zeros((3,1)), voxel_size)
    
    # Get image dimensions
    IMAGE_SIZE = get_image_size(args.images_path)
    W = IMAGE_SIZE[0]
    H = IMAGE_SIZE[1]
    VISUAL_HULL_RESULT = []

    # Adjust height if needed
    if H != VOXEL_SIZE[1]:
        H = VOXEL_SIZE[1]

    height_ratio = H / IMAGE_SIZE[1]
    
    # Load camera calibration
    fanbeams = np.load(args.calibration_path)
    
    # Extract bounding boxes from JSON files
    bboxes_candidates, class_counts = extract_bboxes_from_json(args.json_path, args)
    print('\n############################################################\n')
    print('Candidates count ', len(bboxes_candidates))
    print(bboxes_candidates)

    if len(bboxes_candidates) == 0:
        print('There is no candidates')
        return
    
    # Scale bounding boxes to match voxel space
    bboxes_candidates = bboxes_candidates.astype(np.float32)
    bboxes_candidates[:, :, 1] *= height_ratio   
    bboxes_candidates[:, :, 3] *= height_ratio  
    bboxes_candidates = bboxes_candidates.astype(np.int32)

    # Generate visual hulls for each candidate
    objects = []
    candidates = []
    used_bboxes = set()

    for bboxes in bboxes_candidates:
        non_zero_indices = np.where(np.any(bboxes != 0, axis=1))[0]
        if non_zero_indices.size >= args.min_detection:
            # Check for duplicate bboxes
            bbox_keys = [(i, tuple(bboxes[i])) for i in non_zero_indices]
            if any(key in used_bboxes for key in bbox_keys):
                candidates.append(False)
                continue
                
            # Compute visual hull
            visual_hull = get_visual_hull(fanbeams, bboxes, args)
            objects.append(visual_hull)
            candidates.append(True)
            used_bboxes.update(bbox_keys)
        else:
            candidates.append(False)
    
    print('\n')
    end = time.perf_counter()
    print(f"Run time : {end - start:.6f} seconds")
    
    # Save 3D bounding boxes
    save_bounding_boxes(VISUAL_HULL_RESULT, bboxes_candidates, class_counts, candidates, args)

    # Visualization mode
    if args.visualization is True:
        while True:
            print("Select 3D object index you want to visualize, you have 0 ~", len(objects)-1, "objects ")
            print("If you want to quit, press 'q'")
            index = input("Index : ")
            if index == 'q':
                break
                
            try:
                index = int(index)
                if index < 0 or index >= len(objects):
                    print(f"Invalid index. Please enter a number between 0 and {len(objects)-1}")
                    continue
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
                continue
                
            # Find the corresponding candidate
            count = -1
            for i in range(len(candidates)):
                if candidates[i]:
                    count += 1
                if count == index:
                    break
            
            # Get bboxes for this object
            obj_bboxes = bboxes_candidates[i]
            non_zero_indices = np.where(np.any(obj_bboxes != 0, axis=1))[0]
            
            # Generate 3D bbox corners for ray visualization
            bbox_3d_points = None
            original_2d_heights = None
            if index < len(VISUAL_HULL_RESULT):
                cuboid_coords = VISUAL_HULL_RESULT[index]
                non_zero_bboxes = obj_bboxes[np.all(obj_bboxes != 0, axis=1)]
                
                # Store original 2D heights for each view
                original_2d_heights = non_zero_bboxes[:, [1, 3]]  # y1, y2 for each view
                
                # Calculate Y bounds
                y_min_img = min(non_zero_bboxes[:,1])
                y_max_img = max(non_zero_bboxes[:,3])

                y_3d_top = VOXEL_SIZE[1] // 2 - y_min_img
                y_3d_bottom = VOXEL_SIZE[1] // 2 - y_max_img

                y_margin = abs(y_3d_top - y_3d_bottom) * args.margin[1]
                y_3d = [y_3d_bottom - y_margin, y_3d_top + y_margin]

                # Apply transformations
                cuboid_coords_flipped = np.copy(cuboid_coords)
                cuboid_coords_flipped[:, 0] *= -1
                cuboid_coords_flipped[:, 1] *= -1

                box_3d = []
                # Bottom face corners
                for x, z in cuboid_coords_flipped:
                    box_3d.append([x, y_3d[0], -z])
                # Top face corners
                for x, z in cuboid_coords_flipped:
                    box_3d.append([x, y_3d[1], -z])
                points = np.array(box_3d, dtype=float)
                points[:, 2] *= -1
                bbox_3d_points = points.astype(int)

            # Create camera visualizations with rays to 3D bbox
            individual_camera_geoms, camera_centers = draw_all_cameras_with_rays_to_3d_bbox(fanbeams, bbox_3d_points)
            
            # Load images
            image_dir = args.images_path
            image_ls = sorted(os.listdir(image_dir))
            image_path = sorted([os.path.join(image_dir, img) for img in image_ls if img.endswith('.png')]) 
            
            # Create image visualizations with 3D bbox projections
            images = []
            for j in range(len(fanbeams)):
                # Get original 2D bbox for this view if it exists
                original_bbox = None
                if j in non_zero_indices:
                    original_bbox = obj_bboxes[j]
                images.append(draw_image_with_3d_bbox_projection(fanbeams[j], image_path[j], bbox_3d_points, original_bbox))

            # Create visualization window
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(window_name="3D Visualization", width=1980, height=1080)
            vis.add_geometry(empty_cuboid)
            vis.add_geometry(raw_3d)
            vis.add_geometry(objects[index])

            # Add all visualizations
            print("\nShowing cameras with rays to 3D bounding box bottom corners")
            print("3D bounding box is projected onto images in magenta color")
            for camera_geoms in individual_camera_geoms:
                for geom in camera_geoms:
                    vis.add_geometry(geom)

            print("\nYou can turn on and off camera, rays, and image with key number 0 ~", len(images)-1, "\n\n")
            cameras_visible = [True] * len(images)
            for k in range(len(images)):
                vis.add_geometry(images[k])
                vis.register_key_callback(
                    ord(str(k)), 
                    lambda vis, k=k: toggle_camera_visibility(
                        vis, images[k], individual_camera_geoms[k], cameras_visible, k
                    )
                )
            vis.run()
            vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Visual Hull Generation from Multi-View X-ray Images")
    parser.add_argument('--name', type=str, default='35671', help='Base name for data files')
    parser.add_argument('--calibration_path', type=str, default="./data/calibration/calibration_results.npy", 
                       help='Calibration npy file path')
    parser.add_argument('--margin', type=float, nargs=2, default=[0.1, 0.1], 
                       help='3D margin percentage [x margin, y margin]')
    parser.add_argument('--min_detection', type=int, default=4, 
                       help='Minimum 2D detection count to draw visual hull')
    parser.add_argument('--visualization', type=str, default="False", 
                       help='Enable 3D visualization of images and rays')
    parser.add_argument('--line_thickness', type=float, default=5.0, 
                       help='Thickness of the lines in the visual hull')
    
    args = parser.parse_args()
    
    # Set data paths based on name
    args.json_path = './data/inference_results/' + args.name
    args.raw3d_path = './data/raw_voxel/' + args.name
    args.images_path = './data/inference_results/' + args.name
    
    # Convert visualization argument to boolean
    args.visualization = args.visualization.lower() == "true"

    main(args)