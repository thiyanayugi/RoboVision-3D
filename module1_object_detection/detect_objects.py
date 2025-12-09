#!/usr/bin/env python3
"""
Challenge 1: Object Detection and Localization

This script detects objects in indoor environments using YOLO and localizes them
in 3D space using LiDAR point clouds. It processes synchronized sensor data to
create accurate bounding boxes with dimensions and poses.

Main steps:
1. Load synchronized RGB images, point clouds, and odometry data
2. Extract 3D points within YOLO detection bounding boxes
3. Transform points to world coordinates using robot odometry
4. Cluster detections of the same object across multiple frames
5. Fit oriented bounding boxes to get accurate dimensions and orientation
6. Save results as JSON with pose, dimensions, and class for each object

Usage:
    python detect_objects.py

Output:
    results/challenge1/bathroom_detections.json
    results/challenge1/office_detections.json
"""

import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.cluster import DBSCAN

# Define SynchronizedFrame class for pickle compatibility
class SynchronizedFrame:
    """Container for synchronized sensor data from a single timestamp"""
    def __init__(self, timestamp, rgb_path, depth_path, pointcloud_path):
        self.timestamp = timestamp
        self.rgb_path = rgb_path
        self.depth_path = depth_path
        self.pointcloud_path = pointcloud_path

# Compatibility fix for old pickle files that reference old module names
class CompatUnpickler(pickle.Unpickler):
    """Custom unpickler to handle old module names in pickle files"""
    def find_class(self, module, name):
        # Redirect old module names to avoid import errors
        if module == 'data_synchronizer':
            module = '__main__'
        return super().find_class(module, name)


# ZED camera intrinsic parameters
CAMERA_FX = 527.2972398956961
CAMERA_FY = 527.2972398956961
CAMERA_CX = 658.8206787109375
CAMERA_CY = 372.25787353515625


def load_survey_data(survey_name):
    """
    Load all required data for a survey.
    
    Args:
        survey_name: Name of survey ('bathroom' or 'office')
        
    Returns:
        frames: List of synchronized sensor frames
        odometry: Dictionary mapping timestamps to robot poses
        detections: List of YOLO detections per frame
    """
    print(f"\n{'='*70}")
    print(f"LOADING DATA: {survey_name.upper()}")
    print(f"{'='*70}")
    
    # Load synchronized sensor frames
    frames_file = Path(f'../synchronized_data/{survey_name}_frames.pkl')
    with open(frames_file, 'rb') as f:
        frames = CompatUnpickler(f).load()
    print(f"✅ Loaded {len(frames)} synchronized frames")

    # Load robot odometry (from scan-to-map matching)
    odom_file = Path(f'../scan_to_map_odom/{survey_name}_scan_to_map_odom.pkl')
    with open(odom_file, 'rb') as f:
        odometry = CompatUnpickler(f).load()
    print(f"✅ Loaded {len(odometry)} odometry poses")
    
    # Load YOLO object detections
    det_file = Path(f'detections/{survey_name}/detections.json')
    with open(det_file, 'r') as f:
        detections = json.load(f)
    print(f"✅ Loaded {len(detections)} YOLO detections")
    
    return frames, odometry, detections


def extract_bbox_points(frame, bbox):
    """
    Extract LiDAR points that fall within a 2D bounding box.
    
    This function projects 3D LiDAR points onto the camera image plane
    and selects points that fall within the given bounding box.
    
    Args:
        frame: Synchronized frame containing point cloud path
        bbox: Bounding box [x_min, y_min, x_max, y_max] in pixels
        
    Returns:
        Nx3 array of points in LiDAR frame, or None if no points found
    """
    if not frame.pointcloud_path or not Path(frame.pointcloud_path).exists():
        return None
    
    # Load point cloud (LiDAR frame: X=forward, Y=left, Z=up)
    points_lidar = np.load(str(frame.pointcloud_path))
    
    if len(points_lidar) == 0:
        return None
    
    points_in_bbox = []
    
    for point in points_lidar:
        lidar_x, lidar_y, lidar_z = point
        
        # Transform from LiDAR frame to camera frame
        # Camera frame: X=right, Y=down, Z=forward
        x_cam = -lidar_y  # Camera right = -LiDAR left
        y_cam = -lidar_z  # Camera down = -LiDAR up
        z_cam = lidar_x   # Camera forward = LiDAR forward
        
        # Skip points behind camera
        if z_cam <= 0.1:
            continue
        
        # Project 3D point to 2D image coordinates
        u = int(CAMERA_FX * x_cam / z_cam + CAMERA_CX)
        v = int(CAMERA_FY * y_cam / z_cam + CAMERA_CY)
        
        # Check if point falls within bounding box
        if bbox[0] <= u <= bbox[2] and bbox[1] <= v <= bbox[3]:
            points_in_bbox.append(point)
    
    return np.array(points_in_bbox) if len(points_in_bbox) > 0 else None


def transform_to_world_frame(points_lidar, robot_x, robot_y, robot_yaw):
    """
    Transform points from robot LiDAR frame to world frame.

    Args:
        points_lidar: Nx3 array of points in LiDAR frame
        robot_x: Robot X position in world frame (meters)
        robot_y: Robot Y position in world frame (meters)
        robot_yaw: Robot orientation in world frame (radians)

    Returns:
        Nx3 array of points in world frame
    """
    cos_yaw = np.cos(robot_yaw)
    sin_yaw = np.sin(robot_yaw)

    points_world = np.zeros_like(points_lidar)

    for i, point in enumerate(points_lidar):
        lidar_x, lidar_y, lidar_z = point

        # Apply 2D rotation and translation to XY coordinates
        world_x = robot_x + lidar_x * cos_yaw - lidar_y * sin_yaw
        world_y = robot_y + lidar_x * sin_yaw + lidar_y * cos_yaw
        world_z = lidar_z

        points_world[i] = [world_x, world_y, world_z]

    return points_world


def fit_oriented_bounding_box(points):
    """
    Fit an oriented bounding box to 3D points using PCA.

    This gives more accurate dimensions than axis-aligned boxes,
    especially for rotated objects like chairs and tables.

    Args:
        points: Nx3 array of 3D points in world frame

    Returns:
        center: [x, y, z] center position
        dimensions: (width, depth, height) in meters
        orientation: Rotation angle in radians
    """
    if len(points) < 3:
        # Not enough points for PCA
        center = np.mean(points, axis=0) if len(points) > 0 else np.array([0, 0, 0])
        return center, (0.5, 0.5, 0.5), 0.0

    # Calculate center point
    center = np.mean(points, axis=0)

    # Center the points around origin
    centered_points = points - center

    # Use PCA on XY plane to find principal orientation
    points_2d = centered_points[:, :2]

    # Compute covariance matrix
    cov_matrix = np.cov(points_2d.T)

    # Get eigenvectors (principal directions)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Principal direction is the eigenvector with largest eigenvalue
    principal_idx = np.argmax(eigenvalues)
    principal_vec = eigenvectors[:, principal_idx]

    # Calculate orientation angle
    orientation = np.arctan2(principal_vec[1], principal_vec[0])

    # Rotate points to align with principal axes
    cos_theta = np.cos(-orientation)
    sin_theta = np.sin(-orientation)

    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])

    # Rotate XY coordinates
    rotated_xy = points_2d @ rotation_matrix.T

    # Calculate dimensions in the rotated (aligned) frame
    width = np.max(rotated_xy[:, 0]) - np.min(rotated_xy[:, 0])
    depth = np.max(rotated_xy[:, 1]) - np.min(rotated_xy[:, 1])
    height = np.max(centered_points[:, 2]) - np.min(centered_points[:, 2])

    # Apply minimum dimension threshold
    width = max(width, 0.1)
    depth = max(depth, 0.1)
    height = max(height, 0.1)

    return center, (width, depth, height), orientation


def apply_size_constraints(class_name, width, depth, height):
    """
    Apply realistic size constraints based on object class.

    This prevents unrealistic dimensions from noisy point clouds.

    Args:
        class_name: Object class ('chair', 'table', etc.)
        width, depth, height: Unconstrained dimensions in meters

    Returns:
        Constrained (width, depth, height) tuple
    """
    # Realistic size ranges for common furniture (in meters)
    constraints = {
        'chair': {'width': (0.3, 0.8), 'depth': (0.3, 0.8), 'height': (0.4, 1.2)},
        'couch': {'width': (1.2, 2.5), 'depth': (0.6, 1.2), 'height': (0.4, 1.0)},
        'table': {'width': (0.5, 2.0), 'depth': (0.5, 1.5), 'height': (0.3, 1.0)},
        'shelf': {'width': (0.4, 1.5), 'depth': (0.2, 0.6), 'height': (0.5, 2.5)},
        'toilet': {'width': (0.4, 0.8), 'depth': (0.4, 0.7), 'height': (0.4, 0.8)},
        'bathtub': {'width': (1.0, 1.8), 'depth': (0.5, 0.9), 'height': (0.3, 0.6)},
    }

    # Get constraints for this class (or use defaults)
    c = constraints.get(class_name, {
        'width': (0.2, 3.0),
        'depth': (0.2, 3.0),
        'height': (0.2, 2.5)
    })

    # Clip dimensions to valid range
    width = np.clip(width, c['width'][0], c['width'][1])
    depth = np.clip(depth, c['depth'][0], c['depth'][1])
    height = np.clip(height, c['height'][0], c['height'][1])

    return width, depth, height


def collect_all_detections(frames, odometry, detections_per_frame):
    """
    Collect all detections with their 3D points in world coordinates.

    This processes every frame and extracts 3D points for each detection,
    transforming them to world coordinates using robot odometry.

    Args:
        frames: List of synchronized sensor frames
        odometry: Dictionary mapping timestamps to robot poses
        detections_per_frame: List of YOLO detections for each frame

    Returns:
        List of detections with 3D points in world frame
    """
    all_detections = []

    print(f"\n🔧 Step 1: Collecting 3D points for all detections...")
    print(f"   Processing {len(detections_per_frame)} frames...")

    for frame_idx, frame_det_data in enumerate(detections_per_frame):
        # Show progress every 200 frames
        if frame_idx % 200 == 0 and frame_idx > 0:
            print(f"   Progress: {frame_idx}/{len(detections_per_frame)} frames, "
                  f"{len(all_detections)} detections")

        if frame_idx >= len(frames):
            continue

        frame = frames[frame_idx]

        # Get robot pose for this frame
        odom = odometry.get(frame.timestamp)
        if odom is None:
            continue

        robot_x = odom['position']['x']
        robot_y = odom['position']['y']
        robot_yaw = 2 * np.arctan2(odom['orientation']['z'], odom['orientation']['w'])

        # Process each detection in this frame
        for det in frame_det_data['detections']:
            bbox = det['bbox']

            # Extract LiDAR points within bounding box
            points_lidar = extract_bbox_points(frame, bbox)

            if points_lidar is None or len(points_lidar) < 10:
                continue

            # Transform points to world frame
            points_world = transform_to_world_frame(points_lidar, robot_x, robot_y, robot_yaw)

            all_detections.append({
                'class': det['class'],
                'confidence': det['confidence'],
                'points_world': points_world,
                'frame_idx': frame_idx
            })

    print(f"   ✅ Collected {len(all_detections)} detections from "
          f"{len(detections_per_frame)} frames")
    return all_detections


def merge_detections_by_clustering(detections_with_points):
    """
    Merge multiple detections of the same object using spatial clustering.

    Since the robot sees the same object from multiple viewpoints, we get
    many detections of the same object. This function clusters nearby detections
    of the same class and merges them into a single object.

    Args:
        detections_with_points: List of detections with 3D points

    Returns:
        List of merged object detections with pose and dimensions
    """
    print(f"\n🔧 Step 2: Clustering and merging detections...")

    # Group detections by object class
    by_class = {}
    for det in detections_with_points:
        cls = det['class']
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(det)

    merged_objects = []

    for cls, dets in by_class.items():
        print(f"   Processing {cls}: {len(dets)} detections...")

        # Calculate center position of each detection
        centers = []
        for det in dets:
            center = np.median(det['points_world'], axis=0)
            centers.append(center[:2])  # Use XY only for clustering

        if len(centers) == 0:
            continue

        centers = np.array(centers)

        # Cluster detection centers using DBSCAN
        # eps=0.3 means detections within 30cm are considered the same object
        clustering = DBSCAN(eps=0.3, min_samples=1).fit(centers)

        # Merge all detections in each cluster
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Skip noise points
                continue

            # Get all detections in this cluster
            cluster_indices = np.where(clustering.labels_ == cluster_id)[0]

            # Combine all points from these detections
            all_points = []
            all_confidences = []

            for idx in cluster_indices:
                all_points.extend(dets[idx]['points_world'])
                all_confidences.append(dets[idx]['confidence'])

            if len(all_points) < 10:
                continue

            all_points = np.array(all_points)

            # Fit oriented bounding box to get accurate dimensions
            center, dimensions, orientation = fit_oriented_bounding_box(all_points)
            width, depth, height = dimensions
            width, depth, height = apply_size_constraints(cls, width, depth, height)

            merged_objects.append({
                'class': cls,
                'confidence': float(np.mean(all_confidences)),
                'pose': {
                    'x': float(center[0]),
                    'y': float(center[1]),
                    'orientation': float(orientation)
                },
                'dimensions': {
                    'width': float(width),
                    'depth': float(depth),
                    'height': float(height)
                },
                'num_points': len(all_points),
                'num_observations': len(cluster_indices)
            })

    print(f"   ✅ Merged to {len(merged_objects)} unique objects")
    return merged_objects


def process_survey(survey_name):
    """
    Process one survey to detect and localize objects.

    Args:
        survey_name: Name of survey ('bathroom' or 'office')
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING SURVEY: {survey_name.upper()}")
    print(f"{'='*70}")

    # Load all required data
    frames, odometry, detections_per_frame = load_survey_data(survey_name)

    # Collect 3D points for all detections
    detections_with_points = collect_all_detections(frames, odometry, detections_per_frame)

    # Merge detections using clustering
    final_detections = merge_detections_by_clustering(detections_with_points)

    # Print summary statistics
    print(f"\n📊 Final detections by class:")
    by_class = {}
    for det in final_detections:
        cls = det['class']
        by_class[cls] = by_class.get(cls, 0) + 1
    for cls, count in sorted(by_class.items()):
        print(f"   • {cls}: {count}")

    # Save results to JSON file
    output_dir = Path('../results/challenge1')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'{survey_name}_detections.json'

    result = {
        'survey': survey_name,
        'num_detections': len(final_detections),
        'odometry_source': 'scan_to_map',
        'processing': {
            'method': 'DBSCAN_clustering_OBB_fitting',
            'description': 'Oriented Bounding Box fitting with PCA-aligned dimensions',
            'frames_processed': len(detections_per_frame)
        },
        'detections': final_detections
    }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n💾 Saved: {output_file}")


if __name__ == "__main__":
    print("="*70)
    print("CHALLENGE 1: OBJECT DETECTION AND LOCALIZATION")
    print("="*70)

    # Process both surveys
    for survey in ['bathroom', 'office']:
        process_survey(survey)

    print(f"\n{'='*70}")
    print("✅ OBJECT DETECTION COMPLETE!")
    print(f"{'='*70}")

