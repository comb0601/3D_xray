import os
import re
import cv2
import json
import argparse
import numpy as np
import open3d as o3d
import time


try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy detected: GPU acceleration enabled.")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not found: running on CPU.")


VOXEL_SIZE = None
VISUAL_DSD = None
Y_SHIFT = None
Z_SHIFT = 0 
VISUAL_HULL_RESULT = []
IMAGE_SIZE = None
W = None
H = None

def draw_cuboid(cube_center: np.array, cube_size: np.array):
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
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set_bbox = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set_bbox.colors = o3d.utility.Vector3dVector(colors)
    return line_set_bbox + worldframe

def draw_image(fanbeam, image_path): 
    global W, H, VISUAL_DSD, Y_SHIFT, Z_SHIFT
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) 
    scale_factor =  (fanbeam[3] + VISUAL_DSD) / fanbeam[3] 
    resized_w = int(W * scale_factor)
    image = cv2.resize(image, (resized_w, H))  
    Cx = resized_w / 2 
    Cy = H / 2

    DSD = fanbeam[3] + VISUAL_DSD
    T = np.array([fanbeam[0], 0, fanbeam[1]])
    theta_y = fanbeam[2] * 0.01
    R = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)]])
    
    pts1 = np.array([-Cx, Cy, DSD])
    pts1 = R.T @ (pts1 - T)
    pts1[1] = Cy + Y_SHIFT
    pts1[2] += Z_SHIFT
    pts2 = np.array([Cx, Cy, DSD])
    pts2 = R.T @ (pts2 - T)
    pts2[1] = Cy + Y_SHIFT
    pts2[2] += Z_SHIFT
    pts3 = np.array([-Cx, -Cy, DSD])
    pts3 = R.T @ (pts3 - T)
    pts3[1] = -Cy + Y_SHIFT
    pts3[2] += Z_SHIFT
    pts4 = np.array([Cx, -Cy, DSD])
    pts4 = R.T @ (pts4 - T)
    pts4[1] = -Cy + Y_SHIFT
    pts4[2] += Z_SHIFT

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
    
    vertices_ = np.array([pts4, pts3, pts1, pts2])
    triangles_ = np.array([[0, 1, 2], [0, 2, 3]])
    mesh_ = o3d.geometry.TriangleMesh()
    mesh_.vertices = o3d.utility.Vector3dVector(vertices_)
    mesh_.triangles = o3d.utility.Vector3iVector(triangles_)
    

    v_uv = np.array([[0, 1], [1, 1], [1, 0], 
                    [0, 1], [1, 0], [0, 0]])
    image_flip = cv2.flip(image, 1) 
    mesh_.textures = [o3d.geometry.Image(image_flip)]
    mesh_.triangle_uvs = o3d.utility.Vector2dVector(v_uv)
    mesh_.triangle_material_ids = o3d.utility.IntVector([0] * len(triangles))
    
    mesh += mesh_
    return mesh

def draw_ray(fanbeam, bbox, index):
    global Z_SHIFT, VISUAL_HULL_RESULT, W, H, Y_SHIFT
    scale_factor =  (fanbeam[3] + VISUAL_DSD) / fanbeam[3] 
    resized_w = int(W * scale_factor)
    bbox[0] *= scale_factor 
    bbox[2] *= scale_factor 
    Cx = resized_w / 2 
    Cy = H / 2

    dsd = fanbeam[3] + VISUAL_DSD
    t = np.array([fanbeam[0], 0, fanbeam[1]])
    theta_y = fanbeam[2] * 0.01
    r = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)]])
    cam_center = - np.dot(r.T, t)

    left_x = bbox[0]
    right_x = bbox[2]
    top_y = bbox[1]
    bottom_y = bbox[3]
    
    top_left_x = np.array([left_x - Cx, 0, dsd])
    leftx_world = r.T @ (top_left_x - t)
    leftx_world[1] = -top_y + Cy + Y_SHIFT
    leftx_world[2] += Z_SHIFT 

    top_right_x = np.array([right_x - Cx, 0, dsd])
    rightx_world = r.T @ (top_right_x - t)
    rightx_world[1] = -top_y + Cy + Y_SHIFT
    rightx_world[2] += Z_SHIFT 

    cam_center += [0, (-top_y + Cy + Y_SHIFT), Z_SHIFT]

    if Z_SHIFT == 0:
        lower_x_world = leftx_world if leftx_world[2] >= rightx_world[2] else rightx_world
        slope = (cam_center[2] - lower_x_world[2]) / (cam_center[0] - lower_x_world[0])
        y_intercepts = cam_center[2] - (slope * cam_center[0])
        lower_x, lower_z = VISUAL_HULL_RESULT[index][np.argmax(VISUAL_HULL_RESULT[index][:, 1])]
        higher_x, higher_z = VISUAL_HULL_RESULT[index][np.argmin(VISUAL_HULL_RESULT[index][:, 1])]
        cal_z = (slope * -lower_x) + y_intercepts 
        Z_SHIFT = lower_z - cal_z + ((lower_z - higher_z) / 4)
        leftx_world[2] += Z_SHIFT 
        rightx_world[2] += Z_SHIFT
        cam_center += [0, 0, Z_SHIFT]

    bbox[0] /= scale_factor 
    bbox[2] /= scale_factor

    line1 = draw_line(leftx_world, cam_center)
    line2 = draw_line(rightx_world, cam_center)

    return line1 + line2

def draw_line(point1, point2): 
    lines = [[0, 1]]
    points = np.vstack([point1, point2])
    line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    return line

def get_raw_3d_optimized(path):

    global VOXEL_SIZE
    npy_files = [f for f in os.listdir(path) if f.endswith('.npy')]
    if not npy_files:
        raise FileNotFoundError("No .npy files found in the specified directory.")
    raw_file_path = os.path.join(path, npy_files[-1])
    

    voxel_size_match = re.search(r'(\d+)x(\d+)x(\d+)', raw_file_path)
    if voxel_size_match:
        VOXEL_SIZE = [int(voxel_size_match.group(1)), 
                    int(voxel_size_match.group(3)), 
                    int(voxel_size_match.group(2))]
        print(f"VOXEL_SIZE : {VOXEL_SIZE}")
    else:
        VOXEL_SIZE = [512, 630, 512]
        print("No voxel size found in filename. Using default dimensions:", VOXEL_SIZE)

    raw_cpu = np.load(raw_file_path)
    
    if GPU_AVAILABLE:
        raw_gpu = cp.asarray(raw_cpu)

        raw_gpu[:, 1], raw_gpu[:, 2] = raw_gpu[:, 2].copy(), raw_gpu[:, 1].copy()
        theta_z = -90 * np.pi / 180
        rot_z = cp.array([[cp.cos(theta_z), -cp.sin(theta_z), 0],
                        [cp.sin(theta_z),  cp.cos(theta_z), 0],
                        [0,                0,               1]])
        raw_gpu = cp.dot(raw_gpu, rot_z.T)
        raw = cp.asnumpy(raw_gpu)
    else:
        raw_cpu[:, 1], raw_cpu[:, 2] = raw_cpu[:, 2].copy(), raw_cpu[:, 1].copy()
        theta_z = -90 * np.pi / 180
        rot_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                        [np.sin(theta_z),  np.cos(theta_z), 0],
                        [0,                0,               1]])
        raw = np.dot(raw_cpu, rot_z.T)
    
    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(raw))
    point_cloud.paint_uniform_color([0, 0, 1])
    return point_cloud

def get_visual_hull(fanbeams, bboxes):
    global VISUAL_HULL_RESULT, VOXEL_SIZE
    voxel_plane_size = (VOXEL_SIZE[0], VOXEL_SIZE[2]) 
    voxel_plane = np.ones(voxel_plane_size)

    line_set_box = None
    for i in range(len(fanbeams)):
        fanbeam = fanbeams[i]
        bbox = bboxes[i]
        if sum(bbox) == 0:
            continue

        x_margin = (bbox[2] - bbox[0]) * args.margin[0]
        Cx = W / 2 
        Cy = H / 2
        dsd = fanbeam[3]

        k = np.array([[dsd, 0, Cx],
                    [0, dsd, Cy],
                    [0, 0, 1]])
        theta_y = fanbeam[2] * 0.01
        r = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])
        t = np.array([fanbeam[0], 0, fanbeam[1]])
        rt = np.hstack((r, t[:, None]))
        
        x = np.arange(voxel_plane_size[0]) - voxel_plane_size[0] // 2
        z = np.arange(voxel_plane_size[1]) - voxel_plane_size[1] // 2 
        xz = np.meshgrid(x, z)  
        
        p3d = np.array([xz[0], np.zeros_like(xz[0]), xz[1], np.ones_like(xz[0])])
        p2d = np.dot(k, np.dot(rt, p3d.reshape(4, -1)))
        p2d = p2d / p2d[2]
        u = p2d[0].reshape(voxel_plane_size)
        unsatisfied_mask = (u < (bbox[0] - x_margin)) | (u > (bbox[2] + x_margin))
        voxel_plane[unsatisfied_mask] = 0
        
    if voxel_plane.sum() == 0:
        print("There is no visual hull!")
    else:
        contours, _ = cv2.findContours(voxel_plane.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        box = box - voxel_plane_size[0] // 2

        VISUAL_HULL_RESULT.append(-box)
        
        non_zero_bboxes = bboxes[np.all(bboxes != 0, axis=1)]
        y_3d = [min(non_zero_bboxes[:,1]) - Y_SHIFT, max(non_zero_bboxes[:,3]) - Y_SHIFT]
        y_margin = (y_3d[1] - y_3d[0]) * args.margin[1]
        y_3d[0] -= y_margin
        y_3d[1] += y_margin

        box_3d = [[x, -(y_3d[0] - VOXEL_SIZE[1] // 2), -z] for x, z in box]
        box_3d += [[x, -(y_3d[1] - VOXEL_SIZE[1] // 2), -z] for x, z in box]

        points = np.array(box_3d).astype(int)
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
        ]
    
        # Create cylindrical lines with higher density
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
            
            # Calculate two perpendicular directions for cylindrical expansion
            if direction[1] != 0:  # Vertical line
                perp1 = np.cross(direction, np.array([1, 0, 0]))
                perp2 = np.cross(direction, perp1)
            else:  # Horizontal line
                perp1 = np.cross(direction, np.array([0, 1, 0]))
                perp2 = np.cross(direction, perp1)
            
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = perp2 / np.linalg.norm(perp2)
            
            # Create more points for denser cylindrical lines
            num_points = max(16, int(line_thickness * 8))  # More points for denser appearance
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
        line_set_box = line_sets[0]
        for line_set in line_sets[1:]:
            line_set_box += line_set

    return line_set_box

def check_ray_intersection(fanbeam1, bbox1, fanbeam2, bbox2):
    global W, H, Y_SHIFT, Z_SHIFT
    
    # Create rays for first bbox (candidate)
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
    
    left_x1 = bbox1[0]
    right_x1 = bbox1[2]
    top_y1 = bbox1[1]
    bottom_y1 = bbox1[3]
    
    # Create rays from top corners
    top_left_x1 = np.array([left_x1 - Cx1, 0, dsd1])
    leftx_world1 = r1.T @ (top_left_x1 - t1)
    leftx_world1[1] = -top_y1 + Cy1 + Y_SHIFT
    leftx_world1[2] += Z_SHIFT
    
    top_right_x1 = np.array([right_x1 - Cx1, 0, dsd1])
    rightx_world1 = r1.T @ (top_right_x1 - t1)
    rightx_world1[1] = -top_y1 + Cy1 + Y_SHIFT
    rightx_world1[2] += Z_SHIFT
    
    cam_center1 += [0, (-top_y1 + Cy1 + Y_SHIFT), Z_SHIFT]
    
    # Create rays for second bbox (new bbox)
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
    bottom_y2 = bbox2[3]
    
    # Create rays from top corners
    top_left_x2 = np.array([left_x2 - Cx2, 0, dsd2])
    leftx_world2 = r2.T @ (top_left_x2 - t2)
    leftx_world2[1] = -top_y2 + Cy2 + Y_SHIFT
    leftx_world2[2] += Z_SHIFT
    
    top_right_x2 = np.array([right_x2 - Cx2, 0, dsd2])
    rightx_world2 = r2.T @ (top_right_x2 - t2)
    rightx_world2[1] = -top_y2 + Cy2 + Y_SHIFT
    rightx_world2[2] += Z_SHIFT
    
    cam_center2 += [0, (-top_y2 + Cy2 + Y_SHIFT), Z_SHIFT]
    
    # Get all rays for both bboxes
    rays1 = [
        (cam_center1, leftx_world1),
        (cam_center1, rightx_world1)
    ]
    
    rays2 = [
        (cam_center2, leftx_world2),
        (cam_center2, rightx_world2)
    ]
    
    # Check if any pair of rays intersect in 2D (x-z plane)
    for ray1 in rays1:
        for ray2 in rays2:
            p1, p2 = ray1
            p3, p4 = ray2
            
            # Convert to 2D (x-z plane) by ignoring y coordinate
            p1_2d = np.array([p1[0], p1[2]])
            p2_2d = np.array([p2[0], p2[2]])
            p3_2d = np.array([p3[0], p3[2]])
            p4_2d = np.array([p4[0], p4[2]])
            
            # Calculate intersection in 2D
            d1 = p2_2d - p1_2d
            d2 = p4_2d - p3_2d
            
            cross = np.cross(d1, d2)
            
            if np.abs(cross) > 1e-6:
                t = np.cross(p3_2d - p1_2d, d2) / cross
                s = np.cross(p3_2d - p1_2d, d1) / cross
                
                # Allow intersection points slightly outside the ray segments
                if -0.5 <= t <= 1.5 and -0.5 <= s <= 1.5:
                    return True
    
    return False

def extract_numeric_key(filename):
    try:
        # '_he.json' 
        if '_he.json' in filename:
            return int(filename.split('_he.json')[0][-1])
        # '_H.json'
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
                    # If all else fails, return None
                    print(f"Warning: Could not extract numeric key from {filename}")
                    return None
    except Exception as e:
        print(f"Error extracting numeric key from {filename}: {e}")
        return None
        
def extract_bboxes_from_json(directory_path, args):
    # Initialize data structures
    candidates_3d = np.array([])
    class_counts = {}
    used_bboxes = set()
    fanbeams = np.load(args.calibration_path)
    
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
                
                for bbox_idx, item in enumerate(data['data']):
                    bbox = item['bbox']
                    bbox_key = (numeric_key, bbox_idx)
                    
                    if bbox_key in used_bboxes:
                        continue
                        
                    found = False
                    
                    # Try to find matching candidate
                    for candidate_idx in range(len(candidates_3d)):
                        non_zero_indices = np.where(np.any(candidates_3d[candidate_idx] != 0, axis=1))[0]
                        if not non_zero_indices.size:
                            continue
                            
                        # Skip if view already exists in candidate
                        if numeric_key in non_zero_indices:
                            continue
                        
                        # Check height condition
                        existing_heights = candidates_3d[candidate_idx][non_zero_indices][:, [1, 3]]
                        avg_y1 = np.mean(existing_heights[:, 0])  # Average of y1 coordinates
                        avg_y2 = np.mean(existing_heights[:, 1])  # Average of y2 coordinates
                        
                        # Compare y1 and y2 separately
                        if not (abs(bbox[1] - avg_y1) <= 50 and abs(bbox[3] - avg_y2) <= 50):
                            continue
                        
                        # Check x-coordinate using ray intersection
                        has_intersection = False
                        for existing_idx in non_zero_indices:
                            existing_bbox = candidates_3d[candidate_idx][existing_idx]
                            if check_ray_intersection(fanbeams[numeric_key], bbox, 
                                                    fanbeams[existing_idx], existing_bbox):
                                has_intersection = True
                                break
                        
                        if has_intersection:
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
                        new_candidate = np.zeros((1, 9, 4))
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

def toggle_image_visibility(vis, image, ray, images_visible, index):
    if images_visible[index]:
        vis.remove_geometry(image)
        vis.remove_geometry(ray)
    else:
        vis.add_geometry(image)
        vis.add_geometry(ray)

    images_visible[index] = not images_visible[index]
    vis.update_geometry(image)
    vis.update_geometry(ray)
    vis.poll_events()
    vis.update_renderer()

def get_image_size(image_dir):
    image_ls = os.listdir(image_dir)
    image_path = [os.path.join(image_dir, img) for img in image_ls if img.endswith('.png')]
    image = cv2.cvtColor(cv2.imread(image_path[0]), cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape  
    print(f"IMAGE_SIZE: [{width}, {height}]")
    return [width, height]

def save_bounding_boxes(visual_hull_results, bboxes_candidates, class_counts, candidates, args):
    """
    Save 3D bounding boxes to JSON files in the specified output directory.
    """
    global Y_SHIFT
    
    # Create output directory
    output_dir = os.path.join("data/bbox3d", args.name)
    os.makedirs(output_dir, exist_ok=True)
    
    count = -1
    for i in range(len(candidates)):
        if not candidates[i]:
            continue
            
        count += 1
        # Get class name
        class_name = max(class_counts[i].items(), key=lambda x: x[1])[0]
        
        # Format output filename
        output_file = os.path.join(output_dir, f"{class_name}.json")
        
        # Get bounding box data
        if count < len(visual_hull_results):
            # Get the top face coordinates
            cuboid_coords = visual_hull_results[count]
            
            # Get y coordinates from the bboxes
            obj_bboxes = bboxes_candidates[i]
            non_zero_bboxes = obj_bboxes[np.all(obj_bboxes != 0, axis=1)]
            y_min = min(non_zero_bboxes[:,1]) - Y_SHIFT
            y_max = max(non_zero_bboxes[:,3]) - Y_SHIFT
            y_margin = (y_max - y_min) * args.margin[1]
            y_min -= y_margin
            y_max += y_margin
            
            # Calculate 3D coordinates for all 8 points of the cuboid in simple format
            coords_list = []
            # Bottom face points (y = y_min)
            for coord in cuboid_coords:
                coords_list.append([int(coord[0]), int(y_min), int(coord[1])])
            # Top face points (y = y_max)
            for coord in cuboid_coords:
                coords_list.append([int(coord[0]), int(y_max), int(coord[1])])
            
            # Write to file - just the coordinates line by line
            with open(output_file, 'w') as f:
                json.dump(coords_list, f)
                
            print(f"Saved 3D bounding box for {class_name} to {output_file}")

def main(args):
    global H, W, VISUAL_DSD, Y_SHIFT, Z_SHIFT, VISUAL_HULL_RESULT, VOXEL_SIZE, IMAGE_SIZE
    start = time.perf_counter()

    VISUAL_DSD = 1100
    Y_SHIFT = -25

    raw_3d = get_raw_3d_optimized(args.raw3d_path)

    voxel_size = np.array([[VOXEL_SIZE[0]], [VOXEL_SIZE[1]], [VOXEL_SIZE[2]]])
    empty_cuboid = draw_cuboid(np.zeros((3,1)), voxel_size)
    
    IMAGE_SIZE = get_image_size(args.images_path)
    W = IMAGE_SIZE[0]
    H = IMAGE_SIZE[1]
    VISUAL_HULL_RESULT = []

    if H != VOXEL_SIZE[1]:
        H = VOXEL_SIZE[1]

    height_ratio = H / IMAGE_SIZE[1]
    
    fanbeams = np.load(args.calibration_path)  # (camera count, 4)
    
    bboxes_candidates, class_counts = extract_bboxes_from_json(args.json_path, args)
    print('\n############################################################\n')
    print('Candidates count ', len(bboxes_candidates))
    print(bboxes_candidates)

    if len(bboxes_candidates) == 0:
        print('There is no candidates')
        return
    
    bboxes_candidates = bboxes_candidates.astype(np.float32)
    bboxes_candidates[:, :, 1] *= height_ratio   
    bboxes_candidates[:, :, 3] *= height_ratio  
    bboxes_candidates = bboxes_candidates.astype(np.int32)

    objects = []
    candidates = []
    used_bboxes = set()  # Track which bboxes have been used

    for bboxes in bboxes_candidates:
        non_zero_indices = np.where(np.any(bboxes != 0, axis=1))[0]
        if non_zero_indices.size >= args.min_detection:
            # Check if any bbox in this candidate has already been used
            bbox_keys = [(i, tuple(bboxes[i])) for i in non_zero_indices]
            if any(key in used_bboxes for key in bbox_keys):
                candidates.append(False)
                continue
                
            visual_hull = get_visual_hull(fanbeams, bboxes)
            objects.append(visual_hull)
            candidates.append(True)
            # Mark these bboxes as used
            used_bboxes.update(bbox_keys)
        else:
            candidates.append(False)
    
    print('\n')
    end = time.perf_counter()
    print(f"Run time : {end - start:.6f} seconds")
    
    # Save 3D bounding boxes
    save_bounding_boxes(VISUAL_HULL_RESULT, bboxes_candidates, class_counts, candidates, args)

    # Only show visualization if explicitly enabled and True as a boolean, not string
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
                
            count = -1
            for i in range(len(candidates)):
                if candidates[i]:
                    count += 1
                if count == index:
                    break
            
            # Get the corresponding bboxes for this object
            obj_bboxes = bboxes_candidates[i]
            non_zero_indices = np.where(np.any(obj_bboxes != 0, axis=1))[0]
            
            # Print class name and 3D coordinates
            print(f"\nObject {index} Information:")
            print("\nClass name:", max(class_counts[i].items(), key=lambda x: x[1])[0])
            print("\n3D Cuboid Coordinates:")
            if index < len(VISUAL_HULL_RESULT):
                # Get the top face coordinates
                cuboid_coords = VISUAL_HULL_RESULT[index]
                
                # Get y coordinates from the bboxes
                non_zero_bboxes = obj_bboxes[np.all(obj_bboxes != 0, axis=1)]
                y_min = min(non_zero_bboxes[:,1]) - Y_SHIFT
                y_max = max(non_zero_bboxes[:,3]) - Y_SHIFT
                y_margin = (y_max - y_min) * args.margin[1]
                y_min -= y_margin
                y_max += y_margin
                
                # Calculate 3D coordinates for all 8 points of the cuboid
                print("\n3D Points:")
                # Bottom face points (y = y_min)
                for j, coord in enumerate(cuboid_coords):
                    print(f"Point {j}: ({coord[0]}, {y_min}, {coord[1]})")
                # Top face points (y = y_max)
                for j, coord in enumerate(cuboid_coords):
                    print(f"Point {j+4}: ({coord[0]}, {y_max}, {coord[1]})")
            else:
                print("\nNo 3D cuboid coordinates available for this object")
            
            image_dir = args.images_path
            image_ls = os.listdir(image_dir)
            image_path = [os.path.join(image_dir, img) for img in image_ls if img.endswith('.png')]
            
            rays = []
            images = []
            Z_SHIFT = 0
            for j in range(len(fanbeams)):
                if sum(bboxes_candidates[i][j]) == 0:
                    continue
                rays.append(draw_ray(fanbeams[j], bboxes_candidates[i][j], index))
                images.append(draw_image(fanbeams[j], image_path[j]))
            
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(window_name="3D Visualization", width=1980, height=1080)
            vis.add_geometry(empty_cuboid)
            vis.add_geometry(raw_3d)
            vis.add_geometry(objects[index])
            print("\nYou can turn on and off image with key number 0 ~", len(images)-1, "\n\n")
            images_visible = [True] * len(images)
            for i in range(len(images)):                                                                                                              
                vis.add_geometry(images[i])
                vis.add_geometry(rays[i])
                vis.register_key_callback(ord(str(i)), lambda vis, i=i: toggle_image_visibility(vis, images[i], rays[i], images_visible, i))
            vis.run()
            vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 35671, 35672, 35823
    parser.add_argument('--name', type=str, default='35672', help='Base name')
    parser.add_argument('--calibration_path', type=str, default="./data/calibration/calibration_results.npy", help='calibration npy file path')
    parser.add_argument('--margin', type=list, default=[0.1, 0.1], help='3d margin percentage [x margin, y margin]')
    parser.add_argument('--min_detection', type=int, default=4, help='minimum 2d detection count to draw visual_hull')
    parser.add_argument('--visualization', type=str, default="True", help='images and rays visualization in 3d')
    parser.add_argument('--line_thickness', type=float, default=5.0, help='thickness of the lines in the visual hull')
    args = parser.parse_args()
    
    args.json_path = './data/inference_results/' + args.name
    args.raw3d_path = './data/raw_voxel/' + args.name
    args.images_path = './data/inference_results/' + args.name
    0
    # Convert visualization argument from string to boolean
    args.visualization = args.visualization.lower() == "true"

    main(args)