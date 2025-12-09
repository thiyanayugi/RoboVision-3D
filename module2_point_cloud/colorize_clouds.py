#!/usr/bin/env python3
"""
RoboVision-3D Module 2: Point Cloud Colorization

This script creates colorized point clouds by projecting RGB camera images
onto LiDAR point clouds and concatenating all frames into a single cloud
per survey.

Main steps:
1. Load synchronized RGB images and point clouds
2. Transform points from LiDAR frame to camera frame
3. Project 3D points to 2D image coordinates to sample colors
4. Transform colored points to world frame using robot odometry
5. Concatenate all colored point clouds
6. Save as PLY file

Usage:
    python colorize_clouds.py

Input:
    synchronized_data/bathroom_frames.pkl
    synchronized_data/office_frames.pkl
    scan_to_map_odom/bathroom_scan_to_map_odom.pkl
    scan_to_map_odom/office_scan_to_map_odom.pkl

Output:
    results/module2/bathroom_colorized.ply
    results/module2/office_colorized.ply
"""

import sys
import pickle
import numpy as np
from pathlib import Path
import open3d as o3d
import time
import cv2

# Add challenge1 directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "module1_object_detection"))


# ZED camera intrinsic parameters
CAMERA_FX = 527.2972398956961
CAMERA_FY = 527.2972398956961
CAMERA_CX = 658.8206787109375
CAMERA_CY = 372.25787353515625


class PointCloudColorizer:
    """
    Colorizes LiDAR point clouds using RGB camera images.
    
    The colorization process involves:
    1. Transforming LiDAR points to camera frame
    2. Projecting points onto the image plane
    3. Sampling RGB colors from the image
    4. Transforming colored points to world frame
    5. Concatenating all frames into a single cloud
    """
    
    def __init__(self, workspace_dir):
        """
        Initialize the colorizer.
        
        Args:
            workspace_dir: Path to workspace root
        """
        self.workspace_dir = Path(workspace_dir)
        self.sync_data_dir = self.workspace_dir / "synchronized_data"
        self.odom_dir = self.workspace_dir / "scan_to_map_odom"
        self.output_dir = self.workspace_dir / "results" / "module2"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_survey_data(self, survey_name):
        """
        Load synchronized frames and odometry for a survey.
        
        Args:
            survey_name: Name of survey ('bathroom' or 'office')
            
        Returns:
            frames: List of synchronized sensor frames
            odometry: Dictionary mapping timestamps to robot poses
        """
        print(f"\n📂 Loading data for {survey_name}...")
        
        # Load synchronized frames
        frames_file = self.sync_data_dir / f"{survey_name}_frames.pkl"
        with open(frames_file, 'rb') as f:
            frames = pickle.load(f)
        print(f"   ✅ Loaded {len(frames)} frames")
        
        # Load odometry
        odom_file = self.odom_dir / f"{survey_name}_scan_to_map_odom.pkl"
        with open(odom_file, 'rb') as f:
            odometry = pickle.load(f)
        print(f"   ✅ Loaded {len(odometry)} odometry poses")
        
        return frames, odometry
    
    def transform_lidar_to_camera(self, points_lidar):
        """
        Transform points from LiDAR frame to camera frame.
        
        LiDAR frame: X=forward, Y=left, Z=up
        Camera frame: X=right, Y=down, Z=forward
        
        Args:
            points_lidar: Nx3 array of points in LiDAR frame
            
        Returns:
            Nx3 array of points in camera frame
        """
        points_camera = np.zeros_like(points_lidar)
        points_camera[:, 0] = -points_lidar[:, 1]  # X_cam = -Y_lidar
        points_camera[:, 1] = -points_lidar[:, 2]  # Y_cam = -Z_lidar
        points_camera[:, 2] = points_lidar[:, 0]   # Z_cam = X_lidar
        return points_camera
    
    def project_and_color(self, points_lidar, rgb_image):
        """
        Project LiDAR points onto RGB image and sample colors.
        
        Args:
            points_lidar: Nx3 array of points in LiDAR frame
            rgb_image: RGB image (H x W x 3)
            
        Returns:
            colored_points: Mx3 array of points with valid projections
            colors: Mx3 array of RGB colors [0-1]
        """
        # Transform to camera frame
        points_camera = self.transform_lidar_to_camera(points_lidar)
        
        # Filter points behind camera
        valid_depth = points_camera[:, 2] > 0.1
        points_camera = points_camera[valid_depth]
        
        if len(points_camera) == 0:
            return np.array([]), np.array([])
        
        # Project to image coordinates
        u = (CAMERA_FX * points_camera[:, 0] / points_camera[:, 2] + CAMERA_CX).astype(int)
        v = (CAMERA_FY * points_camera[:, 1] / points_camera[:, 2] + CAMERA_CY).astype(int)
        
        # Filter points within image bounds
        h, w = rgb_image.shape[:2]
        valid_proj = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        
        u = u[valid_proj]
        v = v[valid_proj]
        points_valid = points_lidar[valid_depth][valid_proj]
        
        if len(u) == 0:
            return np.array([]), np.array([])
        
        # Sample colors from image (convert BGR to RGB and normalize to [0-1])
        colors = rgb_image[v, u, ::-1] / 255.0

        return points_valid, colors

    def transform_to_world_frame(self, points_lidar, robot_x, robot_y, robot_yaw):
        """
        Transform points from LiDAR frame to world frame.

        Args:
            points_lidar: Nx3 array of points in LiDAR frame
            robot_x, robot_y: Robot position in world frame (meters)
            robot_yaw: Robot orientation in world frame (radians)

        Returns:
            Nx3 array of points in world frame
        """
        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)

        points_world = np.zeros_like(points_lidar)

        # Apply 2D rotation and translation
        points_world[:, 0] = robot_x + points_lidar[:, 0] * cos_yaw - points_lidar[:, 1] * sin_yaw
        points_world[:, 1] = robot_y + points_lidar[:, 0] * sin_yaw + points_lidar[:, 1] * cos_yaw
        points_world[:, 2] = points_lidar[:, 2]

        return points_world

    def process_survey(self, survey_name):
        """
        Process one survey to create a colorized point cloud.

        Args:
            survey_name: Name of survey ('bathroom' or 'office')

        Returns:
            Path to output PLY file
        """
        print(f"\n{'='*70}")
        print(f"PROCESSING SURVEY: {survey_name.upper()}")
        print(f"{'='*70}")

        start_time = time.time()

        # Load data
        frames, odometry = self.load_survey_data(survey_name)

        # Collect all colored points
        all_points = []
        all_colors = []

        print(f"\n🎨 Colorizing point clouds...")

        for i, frame in enumerate(frames):
            # Show progress every 100 frames
            if i % 100 == 0:
                print(f"   Progress: {i}/{len(frames)} frames")

            # Get odometry for this frame
            odom = odometry.get(frame.timestamp)
            if odom is None:
                continue

            # Load point cloud
            if not frame.pointcloud_path or not Path(frame.pointcloud_path).exists():
                continue

            points_lidar = np.load(str(frame.pointcloud_path))

            if len(points_lidar) == 0:
                continue

            # Load RGB image
            rgb_image = cv2.imread(str(frame.rgb_path))
            if rgb_image is None:
                continue

            # Project and color points
            colored_points_lidar, colors = self.project_and_color(points_lidar, rgb_image)

            if len(colored_points_lidar) == 0:
                continue

            # Transform to world frame
            robot_x = odom['position']['x']
            robot_y = odom['position']['y']
            robot_yaw = 2 * np.arctan2(odom['orientation']['z'], odom['orientation']['w'])

            colored_points_world = self.transform_to_world_frame(
                colored_points_lidar, robot_x, robot_y, robot_yaw
            )

            all_points.append(colored_points_world)
            all_colors.append(colors)

        print(f"   Progress: {len(frames)}/{len(frames)} frames (100%)")

        # Concatenate all points
        print(f"\n🔗 Concatenating point clouds...")
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)

        print(f"   Total points: {len(all_points):,}")

        # Create Open3D point cloud
        print(f"\n💾 Saving point cloud...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)

        # Apply voxel downsampling to reduce file size
        print(f"   Downsampling (voxel size = 0.01m)...")
        pcd = pcd.voxel_down_sample(voxel_size=0.01)

        print(f"   Points after downsampling: {len(pcd.points):,}")

        # Save as PLY
        output_file = self.output_dir / f"{survey_name}_colorized.ply"
        o3d.io.write_point_cloud(str(output_file), pcd)

        elapsed_time = time.time() - start_time

        print(f"\n✅ Saved: {output_file}")
        print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"   Processing time: {elapsed_time:.1f} seconds")

        return output_file


def main():
    """Main entry point for the colorization script."""
    print("\n" + "="*70)
    print("MODULE 2: POINT CLOUD COLORIZATION")
    print("="*70)

    # Initialize colorizer (use script location to find workspace)
    workspace_dir = Path(__file__).parent.parent.resolve()
    colorizer = PointCloudColorizer(workspace_dir)

    # Process both surveys
    output_paths = {}

    output_paths['bathroom'] = colorizer.process_survey('bathroom')
    output_paths['office'] = colorizer.process_survey('office')

    print("\n" + "="*70)
    print("✅ ALL SURVEYS COMPLETE!")
    print("="*70)
    print(f"\nOutput files:")
    for survey, path in output_paths.items():
        print(f"  {survey}: {path}")


if __name__ == "__main__":
    main()

