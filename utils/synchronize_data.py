#!/usr/bin/env python3
"""
Synchronize sensor data from multiple sources by timestamp.

This script matches RGB images, depth images, point clouds, and odometry
based on timestamps to create synchronized frames for object detection.

The synchronization process:
1. Load all sensor data with timestamps
2. For each RGB image, find the closest point cloud (within time threshold)
3. Create SynchronizedFrame objects containing paths to all sensor data
4. Save synchronized frames as pickle files

Usage:
    python synchronize_data.py
    python synchronize_data.py bathroom
    python synchronize_data.py office
"""

import numpy as np
import json
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import sys


@dataclass
class SynchronizedFrame:
    """Container for synchronized sensor data from a single timestamp"""
    timestamp: int
    rgb_path: str
    depth_path: Optional[str]
    pointcloud_path: str


def load_timestamps_from_directory(directory: Path, extension: str) -> Dict[int, Path]:
    """
    Load all files from a directory and extract timestamps from filenames.
    
    Args:
        directory: Directory containing timestamped files
        extension: File extension to filter (e.g., '.jpg', '.npy')
    
    Returns:
        Dictionary mapping timestamp to file path
    """
    files = {}
    
    if not directory.exists():
        return files
    
    for file_path in directory.glob(f"*{extension}"):
        # Extract timestamp from filename (e.g., "1234567890.jpg" -> 1234567890)
        try:
            timestamp = int(file_path.stem)
            files[timestamp] = file_path
        except ValueError:
            continue
    
    return files


def find_closest_timestamp(target_ts: int, available_ts: List[int], 
                           max_diff: int = 50000000) -> Optional[int]:
    """
    Find the closest timestamp within a maximum time difference.
    
    Args:
        target_ts: Target timestamp
        available_ts: List of available timestamps
        max_diff: Maximum allowed time difference in nanoseconds (default: 50ms)
                  This threshold ensures sensors are temporally aligned
    
    Returns:
        Closest timestamp or None if no match within threshold
    """
    if not available_ts:
        return None
    
    # Find closest timestamp
    closest = min(available_ts, key=lambda ts: abs(ts - target_ts))
    
    # Check if within threshold
    if abs(closest - target_ts) <= max_diff:
        return closest
    
    return None


def synchronize_survey_data(survey_name: str, workspace_dir: Path = None) -> List[SynchronizedFrame]:
    """
    Synchronize all sensor data for a survey.
    
    Args:
        survey_name: Name of survey ('bathroom' or 'office')
        workspace_dir: Workspace directory (default: parent of this script)
    
    Returns:
        List of synchronized frames
    """
    if workspace_dir is None:
        workspace_dir = Path(__file__).parent.parent
    
    print(f"\n{'='*70}")
    print(f"Synchronizing data for: {survey_name.upper()}")
    print(f"{'='*70}")
    
    # Define data directories
    extracted_dir = workspace_dir / "extracted_data" / survey_name
    rgb_dir = extracted_dir / "rgb"
    depth_dir = extracted_dir / "depth"
    pointcloud_dir = extracted_dir / "pointcloud"
    
    # Load all timestamps
    print("\n📂 Loading sensor data...")
    rgb_files = load_timestamps_from_directory(rgb_dir, ".jpg")
    depth_files = load_timestamps_from_directory(depth_dir, ".png")
    pointcloud_files = load_timestamps_from_directory(pointcloud_dir, ".npy")
    
    print(f"   ✅ RGB images: {len(rgb_files)}")
    print(f"   ✅ Depth images: {len(depth_files)}")
    print(f"   ✅ Point clouds: {len(pointcloud_files)}")
    
    if not rgb_files or not pointcloud_files:
        print("\n❌ Missing required data (RGB or point clouds)")
        print("   Run: python utils/extract_rosbag_data.py --topics rgb pointcloud")
        return []
    
    # Synchronize frames
    print("\n🔧 Synchronizing frames...")
    synchronized_frames = []
    
    pointcloud_timestamps = list(pointcloud_files.keys())
    depth_timestamps = list(depth_files.keys()) if depth_files else []
    
    for rgb_ts, rgb_path in sorted(rgb_files.items()):
        # Find closest point cloud
        pc_ts = find_closest_timestamp(rgb_ts, pointcloud_timestamps)
        if pc_ts is None:
            continue
        
        # Find closest depth image (optional)
        depth_ts = find_closest_timestamp(rgb_ts, depth_timestamps) if depth_timestamps else None
        depth_path = str(depth_files[depth_ts]) if depth_ts else None
        
        # Create synchronized frame
        frame = SynchronizedFrame(
            timestamp=rgb_ts,
            rgb_path=str(rgb_path),
            depth_path=depth_path,
            pointcloud_path=str(pointcloud_files[pc_ts])
        )
        
        synchronized_frames.append(frame)
    
    print(f"   ✅ Synchronized {len(synchronized_frames)} frames")

    return synchronized_frames


def save_synchronized_data(survey_name: str, frames: List[SynchronizedFrame],
                           workspace_dir: Path = None):
    """
    Save synchronized frames to pickle file.

    Args:
        survey_name: Name of survey
        frames: List of synchronized frames
        workspace_dir: Workspace directory
    """
    if workspace_dir is None:
        workspace_dir = Path(__file__).parent.parent

    output_dir = workspace_dir / "synchronized_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save frames
    frames_file = output_dir / f"{survey_name}_frames.pkl"
    print(f"\n💾 Saving synchronized frames to {frames_file}...")
    with open(frames_file, 'wb') as f:
        pickle.dump(frames, f)
    print(f"   ✅ Saved {len(frames)} frames")

    # Save metadata
    metadata = {
        'survey': survey_name,
        'num_frames': len(frames),
        'timestamp_range': {
            'min': min(f.timestamp for f in frames),
            'max': max(f.timestamp for f in frames)
        },
        'has_depth': any(f.depth_path is not None for f in frames)
    }

    metadata_file = output_dir / f"{survey_name}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Saved metadata to {metadata_file}")


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        survey = sys.argv[1]
        if survey not in ['bathroom', 'office']:
            print("Usage: python synchronize_data.py [bathroom|office]")
            sys.exit(1)
        surveys = [survey]
    else:
        surveys = ['bathroom', 'office']

    workspace_dir = Path(__file__).parent.parent

    for survey in surveys:
        frames = synchronize_survey_data(survey, workspace_dir)
        if frames:
            save_synchronized_data(survey, frames, workspace_dir)

    print(f"\n{'='*70}")
    print("✅ DATA SYNCHRONIZATION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()


