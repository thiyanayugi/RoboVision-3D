#!/usr/bin/env python3
"""
Compute scan-to-map odometry from laser scans and occupancy grid maps.

This script performs scan matching to estimate the robot's pose at each timestamp
by aligning laser scans with the pre-built occupancy grid map. This provides more
accurate localization than wheel odometry alone.

Method: Correlative Scan Matching
- For each laser scan, search for the best pose that aligns the scan with the map
- Uses a coarse-to-fine search strategy for efficiency
- Scores poses based on how many scan points hit occupied cells in the map

Usage:
    python utils/compute_scan_to_map_odom.py
    python utils/compute_scan_to_map_odom.py bathroom
    python utils/compute_scan_to_map_odom.py office
"""

import numpy as np
import cv2
import json
import pickle
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import yaml

# ROS2 message types
try:
    from sensor_msgs.msg import LaserScan
    from rclpy.serialization import deserialize_message
except ImportError:
    print("Warning: ROS2 not available, using fallback deserialization")
    LaserScan = None
    deserialize_message = None


@dataclass
class Pose2D:
    """2D pose (x, y, theta)"""
    x: float
    y: float
    theta: float
    timestamp: int


@dataclass
class OccupancyMap:
    """Occupancy grid map"""
    data: np.ndarray  # 2D array, 0=free, 255=occupied, 127=unknown
    resolution: float  # meters per pixel
    origin_x: float    # map origin in meters
    origin_y: float
    width: int
    height: int


class ScanToMapMatcher:
    """Scan-to-map matching for pose estimation"""
    
    def __init__(self, occupancy_map: OccupancyMap):
        self.map = occupancy_map
        
        # Create distance transform for scoring
        # Distance from each pixel to nearest occupied cell
        occupied = (self.map.data > 200).astype(np.uint8) * 255
        self.dist_transform = cv2.distanceTransform(255 - occupied, cv2.DIST_L2, 5)
        
    def world_to_map(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to map pixel coordinates"""
        mx = int((x - self.map.origin_x) / self.map.resolution)
        my = int((y - self.map.origin_y) / self.map.resolution)
        return mx, my
    
    def scan_to_points(self, scan_data: dict, pose: Pose2D) -> np.ndarray:
        """Convert laser scan to world coordinates given robot pose"""
        ranges = np.array(scan_data['ranges'])
        angle_min = scan_data['angle_min']
        angle_increment = scan_data['angle_increment']
        range_max = scan_data['range_max']
        
        # Filter invalid ranges
        valid = (ranges > 0.1) & (ranges < range_max)
        ranges = ranges[valid]
        
        # Compute angles
        angles = angle_min + np.arange(len(scan_data['ranges'])) * angle_increment
        angles = angles[valid]
        
        # Convert to cartesian in robot frame
        x_robot = ranges * np.cos(angles)
        y_robot = ranges * np.sin(angles)
        
        # Transform to world frame
        cos_theta = np.cos(pose.theta)
        sin_theta = np.sin(pose.theta)
        
        x_world = pose.x + x_robot * cos_theta - y_robot * sin_theta
        y_world = pose.y + x_robot * sin_theta + y_robot * cos_theta
        
        return np.column_stack([x_world, y_world])
    
    def score_pose(self, scan_data: dict, pose: Pose2D) -> float:
        """Score how well a scan matches the map at given pose"""
        points = self.scan_to_points(scan_data, pose)
        
        score = 0.0
        valid_points = 0
        
        for x, y in points:
            mx, my = self.world_to_map(x, y)
            
            # Check if point is within map bounds
            if 0 <= mx < self.map.width and 0 <= my < self.map.height:
                # Score based on distance to occupied cells
                # Lower distance = better match
                dist = self.dist_transform[my, mx]
                score += np.exp(-dist / 2.0)  # Gaussian falloff
                valid_points += 1
        
        if valid_points == 0:
            return 0.0
        
        return score / valid_points
    
    def match_scan(self, scan_data: dict, initial_pose: Pose2D,
                   search_window: float = 0.5, angle_window: float = 0.3,
                   resolution: float = 0.05) -> Pose2D:
        """
        Find best pose for scan using grid search around initial pose.
        
        Args:
            scan_data: Laser scan data
            initial_pose: Initial pose estimate (e.g., from wheel odometry)
            search_window: Search radius in meters
            angle_window: Search radius in radians
            resolution: Search resolution in meters
        
        Returns:
            Best pose found
        """
        best_score = -1
        best_pose = initial_pose
        
        # Grid search
        x_range = np.arange(-search_window, search_window + resolution, resolution)
        y_range = np.arange(-search_window, search_window + resolution, resolution)
        theta_range = np.arange(-angle_window, angle_window + 0.1, 0.1)

        for dx in x_range:
            for dy in y_range:
                for dtheta in theta_range:
                    test_pose = Pose2D(
                        x=initial_pose.x + dx,
                        y=initial_pose.y + dy,
                        theta=initial_pose.theta + dtheta,
                        timestamp=initial_pose.timestamp
                    )

                    score = self.score_pose(scan_data, test_pose)

                    if score > best_score:
                        best_score = score
                        best_pose = test_pose

        return best_pose


def load_occupancy_map(map_dir: Path) -> OccupancyMap:
    """Load occupancy grid map from PGM and YAML files"""
    pgm_path = map_dir / "room.pgm"
    yaml_path = map_dir / "room.yaml"

    # Load YAML metadata
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load PGM image
    map_image = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)

    return OccupancyMap(
        data=map_image,
        resolution=config['resolution'],
        origin_x=config['origin'][0],
        origin_y=config['origin'][1],
        width=map_image.shape[1],
        height=map_image.shape[0]
    )


def load_wheel_odometry(workspace_dir: Path, survey_name: str) -> Dict[int, Pose2D]:
    """Load wheel odometry from extracted data"""
    odom_dir = workspace_dir / "extracted_data" / survey_name / "odometry"

    if not odom_dir.exists():
        print(f"⚠️  Wheel odometry not found at {odom_dir}")
        print("   Run: python utils/extract_rosbag_data.py --topics odometry")
        return {}

    odometry = {}

    for odom_file in sorted(odom_dir.glob("*.json")):
        with open(odom_file, 'r') as f:
            data = json.load(f)

        timestamp = data['timestamp']

        # Convert quaternion to yaw angle
        qx = data['orientation']['x']
        qy = data['orientation']['y']
        qz = data['orientation']['z']
        qw = data['orientation']['w']

        # Quaternion to Euler (yaw only)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        odometry[timestamp] = Pose2D(
            x=data['position']['x'],
            y=data['position']['y'],
            theta=yaw,
            timestamp=timestamp
        )

    return odometry


def load_laser_scans(workspace_dir: Path, survey_name: str) -> Dict[int, dict]:
    """Load laser scans from extracted data"""
    scan_dir = workspace_dir / "extracted_data" / survey_name / "laser_scan"

    if not scan_dir.exists():
        print(f"⚠️  Laser scans not found at {scan_dir}")
        print("   Run: python utils/extract_rosbag_data.py --topics laser")
        return {}

    scans = {}

    for scan_file in sorted(scan_dir.glob("*.json")):
        with open(scan_file, 'r') as f:
            data = json.load(f)
        scans[data['timestamp']] = data

    return scans


def compute_scan_to_map_odometry(survey_name: str, workspace_dir: Path = None):
    """
    Compute scan-to-map odometry for a survey.

    Args:
        survey_name: 'bathroom' or 'office'
        workspace_dir: Workspace directory (default: parent of this script)
    """
    if workspace_dir is None:
        workspace_dir = Path(__file__).parent.parent

    print(f"\n{'='*70}")
    print(f"Computing scan-to-map odometry for: {survey_name.upper()}")
    print(f"{'='*70}")

    # Load occupancy map
    print("\n📂 Loading occupancy map...")
    map_dir = workspace_dir / "Challenge_Data" / survey_name
    occ_map = load_occupancy_map(map_dir)
    print(f"   ✅ Map loaded: {occ_map.width}x{occ_map.height} @ {occ_map.resolution}m/px")

    # Load wheel odometry (initial estimates)
    print("\n📂 Loading wheel odometry...")
    wheel_odom = load_wheel_odometry(workspace_dir, survey_name)
    print(f"   ✅ Loaded {len(wheel_odom)} odometry messages")

    # Load laser scans
    print("\n📂 Loading laser scans...")
    laser_scans = load_laser_scans(workspace_dir, survey_name)
    print(f"   ✅ Loaded {len(laser_scans)} laser scans")

    if not wheel_odom or not laser_scans:
        print("\n❌ Missing required data. Please extract odometry and laser scans first.")
        return

    # Initialize matcher
    print("\n🔧 Initializing scan matcher...")
    matcher = ScanToMapMatcher(occ_map)
    print("   ✅ Matcher ready")

    # Match scans to map
    print(f"\n🔧 Matching {len(laser_scans)} scans to map...")
    scan_to_map_odom = {}

    timestamps = sorted(laser_scans.keys())
    for i, timestamp in enumerate(timestamps):
        if i % 100 == 0:
            print(f"   Progress: {i}/{len(timestamps)} scans")

        scan_data = laser_scans[timestamp]

        # Find closest wheel odometry
        closest_odom_ts = min(wheel_odom.keys(), key=lambda t: abs(t - timestamp))
        initial_pose = wheel_odom[closest_odom_ts]
        initial_pose.timestamp = timestamp

        # Match scan to map
        corrected_pose = matcher.match_scan(scan_data, initial_pose)
        scan_to_map_odom[timestamp] = corrected_pose

    print(f"   ✅ Matched all {len(scan_to_map_odom)} scans")

    # Save results
    output_dir = workspace_dir / "scan_to_map_odom"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{survey_name}_scan_to_map_odom.pkl"

    print(f"\n💾 Saving results to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(scan_to_map_odom, f)

    print(f"   ✅ Saved {len(scan_to_map_odom)} poses")
    print(f"\n{'='*70}")
    print("✅ SCAN-TO-MAP ODOMETRY COMPUTATION COMPLETE")
    print(f"{'='*70}\n")


def main():
    """Main entry point"""
    import sys

    if len(sys.argv) > 1:
        survey = sys.argv[1]
        if survey not in ['bathroom', 'office']:
            print("Usage: python compute_scan_to_map_odom.py [bathroom|office]")
            sys.exit(1)
        surveys = [survey]
    else:
        surveys = ['bathroom', 'office']

    for survey in surveys:
        compute_scan_to_map_odometry(survey)


if __name__ == "__main__":
    main()


