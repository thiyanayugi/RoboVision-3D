#!/usr/bin/env python3
"""
Extract all sensor data from ROS2 bags and save to extracted_data folder.

This unified script extracts:
- RGB images (compressed) - /zed/zed_node/rgb/image_rect_color/compressed
- Depth images (compressed) - /zed/zed_node/depth/depth_registered/compressedDepth
- Point clouds (PointCloud2) - /livox/lidar
- Laser scans (LaserScan) - /scan
- Odometry (wheel odometry) - /odom
- ZED odometry - /zed/zed_node/odom
- IMU data - /imu, /livox/imu
- Camera info - /zed/zed_node/rgb/camera_info
- TF transforms - /tf, /tf_static

Usage:
    python extract_rosbag_data.py [survey_name] [--topics topic1 topic2 ...]
    
    If no survey name is provided, extracts from both bathroom and office.
    If no topics are specified, extracts all topics.

Examples:
    python extract_rosbag_data.py bathroom
    python extract_rosbag_data.py office --topics rgb depth pointcloud
    python extract_rosbag_data.py  # Extract all from both surveys
"""

import sqlite3
import cv2
import numpy as np
import json
import struct
import argparse
from pathlib import Path
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage, PointCloud2, LaserScan, Imu, CameraInfo
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage


class RosbagExtractor:
    """Unified extractor for all ROS2 bag topics."""
    
    def __init__(self, workspace_dir=None):
        """Initialize the extractor."""
        if workspace_dir is None:
            workspace_dir = Path(__file__).parent.parent
        else:
            workspace_dir = Path(workspace_dir)
        
        self.workspace_dir = workspace_dir
        self.data_dir = workspace_dir / "Challenge_Data"
        self.output_base = workspace_dir / "extracted_data"
        
    def get_rosbag_path(self, survey_name):
        """Get the rosbag directory for a survey."""
        survey_path = self.data_dir / survey_name
        rosbag_dirs = list(survey_path.glob("rosbag2_*"))
        
        if not rosbag_dirs:
            raise FileNotFoundError(f"No rosbag directory found in {survey_path}")
        
        return rosbag_dirs[0]
    
    def get_db_files(self, rosbag_path):
        """Get all database files in a rosbag directory."""
        return sorted(rosbag_path.glob("*.db3"))
    
    def extract_rgb_images(self, survey_name):
        """Extract RGB images from compressed image topic."""
        print(f"\n{'='*70}")
        print(f"Extracting RGB images for: {survey_name.upper()}")
        print(f"{'='*70}")
        
        rosbag_path = self.get_rosbag_path(survey_name)
        output_dir = self.output_base / survey_name / "rgb"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        db_paths = self.get_db_files(rosbag_path)
        print(f"Found {len(db_paths)} database files")
        
        topic = '/zed/zed_node/rgb/image_rect_color/compressed'
        total_extracted = 0
        
        for db_idx, db_path in enumerate(db_paths):
            print(f"\nProcessing database {db_idx + 1}/{len(db_paths)}: {db_path.name}")
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            cursor.execute("SELECT id FROM topics WHERE name = ?", (topic,))
            result = cursor.fetchone()
            if not result:
                print(f"  ⚠️  Topic not found in this database")
                conn.close()
                continue
            topic_id = result[0]
            
            cursor.execute(
                "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
                (topic_id,)
            )
            
            count = 0
            for timestamp, data in cursor:
                try:
                    msg = deserialize_message(data, CompressedImage)
                    
                    # Decode compressed image
                    np_arr = np.frombuffer(msg.data, np.uint8)
                    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        output_file = output_dir / f"{timestamp}.jpg"
                        cv2.imwrite(str(output_file), image)
                        count += 1
                        total_extracted += 1
                except Exception as e:
                    print(f"  ⚠️  Error processing message: {e}")
                    continue
            
            print(f"  ✅ Extracted {count} images from this database")
            conn.close()
        
        print(f"\n✅ Total RGB images extracted: {total_extracted}")
        return total_extracted

    def extract_depth_images(self, survey_name):
        """Extract depth images from compressed depth topic."""
        print(f"\n{'='*70}")
        print(f"Extracting depth images for: {survey_name.upper()}")
        print(f"{'='*70}")

        rosbag_path = self.get_rosbag_path(survey_name)
        output_dir = self.output_base / survey_name / "depth"
        output_dir.mkdir(parents=True, exist_ok=True)

        db_paths = self.get_db_files(rosbag_path)
        print(f"Found {len(db_paths)} database files")

        topic = '/zed/zed_node/depth/depth_registered/compressedDepth'
        total_extracted = 0

        for db_idx, db_path in enumerate(db_paths):
            print(f"\nProcessing database {db_idx + 1}/{len(db_paths)}: {db_path.name}")

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM topics WHERE name = ?", (topic,))
            result = cursor.fetchone()
            if not result:
                print(f"  ⚠️  Topic not found in this database")
                conn.close()
                continue
            topic_id = result[0]

            cursor.execute(
                "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
                (topic_id,)
            )

            count = 0
            for timestamp, data in cursor:
                try:
                    msg = deserialize_message(data, CompressedImage)

                    # Decode compressed depth (16-bit PNG)
                    np_arr = np.frombuffer(msg.data, np.uint8)
                    depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

                    if depth_image is not None:
                        output_file = output_dir / f"{timestamp}.png"
                        cv2.imwrite(str(output_file), depth_image)
                        count += 1
                        total_extracted += 1
                except Exception as e:
                    print(f"  ⚠️  Error processing message: {e}")
                    continue

            print(f"  ✅ Extracted {count} depth images from this database")
            conn.close()

        print(f"\n✅ Total depth images extracted: {total_extracted}")
        return total_extracted

    def parse_pointcloud2(self, msg):
        """Parse PointCloud2 message to extract XYZ coordinates."""
        # Find x, y, z field offsets
        x_offset = y_offset = z_offset = None
        for field in msg.fields:
            if field.name == 'x':
                x_offset = field.offset
            elif field.name == 'y':
                y_offset = field.offset
            elif field.name == 'z':
                z_offset = field.offset

        if x_offset is None or y_offset is None or z_offset is None:
            return None

        # Parse binary data
        points = []
        point_step = msg.point_step

        for i in range(0, len(msg.data), point_step):
            if i + point_step > len(msg.data):
                break

            # Extract x, y, z (assuming float32)
            x = struct.unpack_from('f', msg.data, i + x_offset)[0]
            y = struct.unpack_from('f', msg.data, i + y_offset)[0]
            z = struct.unpack_from('f', msg.data, i + z_offset)[0]

            # Filter out invalid points
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z) or
                    np.isinf(x) or np.isinf(y) or np.isinf(z)):
                points.append([x, y, z])

        return np.array(points, dtype=np.float32) if points else None

    def extract_pointclouds(self, survey_name):
        """Extract point clouds from PointCloud2 topic."""
        print(f"\n{'='*70}")
        print(f"Extracting point clouds for: {survey_name.upper()}")
        print(f"{'='*70}")

        rosbag_path = self.get_rosbag_path(survey_name)
        output_dir = self.output_base / survey_name / "pointcloud"
        output_dir.mkdir(parents=True, exist_ok=True)

        db_paths = self.get_db_files(rosbag_path)
        print(f"Found {len(db_paths)} database files")

        topic = '/livox/lidar'  # Livox LiDAR point cloud
        total_extracted = 0

        for db_idx, db_path in enumerate(db_paths):
            print(f"\nProcessing database {db_idx + 1}/{len(db_paths)}: {db_path.name}")

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM topics WHERE name = ?", (topic,))
            result = cursor.fetchone()
            if not result:
                print(f"  ⚠️  Topic not found in this database")
                conn.close()
                continue
            topic_id = result[0]

            cursor.execute(
                "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
                (topic_id,)
            )

            count = 0
            for timestamp, data in cursor:
                try:
                    msg = deserialize_message(data, PointCloud2)
                    points = self.parse_pointcloud2(msg)

                    if points is not None and len(points) > 0:
                        output_file = output_dir / f"{timestamp}.npy"
                        np.save(str(output_file), points)
                        count += 1
                        total_extracted += 1
                except Exception as e:
                    print(f"  ⚠️  Error processing message: {e}")
                    continue

            print(f"  ✅ Extracted {count} point clouds from this database")
            conn.close()

        print(f"\n✅ Total point clouds extracted: {total_extracted}")
        return total_extracted

    def extract_laser_scans(self, survey_name):
        """Extract laser scan data."""
        print(f"\n{'='*70}")
        print(f"Extracting laser scans for: {survey_name.upper()}")
        print(f"{'='*70}")

        rosbag_path = self.get_rosbag_path(survey_name)
        output_dir = self.output_base / survey_name / "laser_scan"
        output_dir.mkdir(parents=True, exist_ok=True)

        db_paths = self.get_db_files(rosbag_path)
        print(f"Found {len(db_paths)} database files")

        topic = '/scan'
        total_extracted = 0

        for db_idx, db_path in enumerate(db_paths):
            print(f"\nProcessing database {db_idx + 1}/{len(db_paths)}: {db_path.name}")

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM topics WHERE name = ?", (topic,))
            result = cursor.fetchone()
            if not result:
                print(f"  ⚠️  Topic not found in this database")
                conn.close()
                continue
            topic_id = result[0]

            cursor.execute(
                "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
                (topic_id,)
            )

            count = 0
            for timestamp, data in cursor:
                try:
                    msg = deserialize_message(data, LaserScan)

                    scan_data = {
                        'timestamp': timestamp,
                        'angle_min': msg.angle_min,
                        'angle_max': msg.angle_max,
                        'angle_increment': msg.angle_increment,
                        'range_min': msg.range_min,
                        'range_max': msg.range_max,
                        'ranges': list(msg.ranges)
                    }

                    output_file = output_dir / f"{timestamp}.json"
                    with open(output_file, 'w') as f:
                        json.dump(scan_data, f)

                    count += 1
                    total_extracted += 1
                except Exception as e:
                    print(f"  ⚠️  Error processing message: {e}")
                    continue

            print(f"  ✅ Extracted {count} laser scans from this database")
            conn.close()

        print(f"\n✅ Total laser scans extracted: {total_extracted}")
        return total_extracted

    def extract_odometry(self, survey_name):
        """Extract wheel odometry data."""
        print(f"\n{'='*70}")
        print(f"Extracting odometry for: {survey_name.upper()}")
        print(f"{'='*70}")

        rosbag_path = self.get_rosbag_path(survey_name)
        output_dir = self.output_base / survey_name / "odometry"
        output_dir.mkdir(parents=True, exist_ok=True)

        db_paths = self.get_db_files(rosbag_path)
        print(f"Found {len(db_paths)} database files")

        topic = '/odom'
        total_extracted = 0

        for db_idx, db_path in enumerate(db_paths):
            print(f"\nProcessing database {db_idx + 1}/{len(db_paths)}: {db_path.name}")

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM topics WHERE name = ?", (topic,))
            result = cursor.fetchone()
            if not result:
                print(f"  ⚠️  Topic not found in this database")
                conn.close()
                continue
            topic_id = result[0]

            cursor.execute(
                "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
                (topic_id,)
            )

            count = 0
            for timestamp, data in cursor:
                try:
                    msg = deserialize_message(data, Odometry)

                    odom_data = {
                        'timestamp': timestamp,
                        'position': {
                            'x': msg.pose.pose.position.x,
                            'y': msg.pose.pose.position.y,
                            'z': msg.pose.pose.position.z
                        },
                        'orientation': {
                            'x': msg.pose.pose.orientation.x,
                            'y': msg.pose.pose.orientation.y,
                            'z': msg.pose.pose.orientation.z,
                            'w': msg.pose.pose.orientation.w
                        }
                    }

                    output_file = output_dir / f"{timestamp}.json"
                    with open(output_file, 'w') as f:
                        json.dump(odom_data, f)

                    count += 1
                    total_extracted += 1
                except Exception as e:
                    print(f"  ⚠️  Error processing message: {e}")
                    continue

            print(f"  ✅ Extracted {count} odometry messages from this database")
            conn.close()

        print(f"\n✅ Total odometry messages extracted: {total_extracted}")
        return total_extracted

    def extract_imu(self, survey_name):
        """Extract IMU data from both /imu and /livox/imu topics."""
        print(f"\n{'='*70}")
        print(f"Extracting IMU data for: {survey_name.upper()}")
        print(f"{'='*70}")

        rosbag_path = self.get_rosbag_path(survey_name)
        output_dir = self.output_base / survey_name / "imu"
        output_dir.mkdir(parents=True, exist_ok=True)

        db_paths = self.get_db_files(rosbag_path)
        print(f"Found {len(db_paths)} database files")

        topics = ['/imu', '/livox/imu']
        total_extracted = 0

        for topic in topics:
            topic_name = topic.replace('/', '_')

            for db_idx, db_path in enumerate(db_paths):
                if db_idx == 0:
                    print(f"\nExtracting {topic}...")

                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()

                cursor.execute("SELECT id FROM topics WHERE name = ?", (topic,))
                result = cursor.fetchone()
                if not result:
                    conn.close()
                    continue
                topic_id = result[0]

                cursor.execute(
                    "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
                    (topic_id,)
                )

                count = 0
                for timestamp, data in cursor:
                    try:
                        msg = deserialize_message(data, Imu)

                        imu_data = {
                            'timestamp': timestamp,
                            'orientation': {
                                'x': msg.orientation.x,
                                'y': msg.orientation.y,
                                'z': msg.orientation.z,
                                'w': msg.orientation.w
                            },
                            'angular_velocity': {
                                'x': msg.angular_velocity.x,
                                'y': msg.angular_velocity.y,
                                'z': msg.angular_velocity.z
                            },
                            'linear_acceleration': {
                                'x': msg.linear_acceleration.x,
                                'y': msg.linear_acceleration.y,
                                'z': msg.linear_acceleration.z
                            }
                        }

                        output_file = output_dir / f"{topic_name}_{timestamp}.json"
                        with open(output_file, 'w') as f:
                            json.dump(imu_data, f)

                        count += 1
                        total_extracted += 1
                    except Exception as e:
                        continue

                conn.close()

            print(f"  ✅ Extracted {count} messages from {topic}")

        print(f"\n✅ Total IMU messages extracted: {total_extracted}")
        return total_extracted

    def extract_camera_info(self, survey_name):
        """Extract camera info from ZED camera."""
        print(f"\n{'='*70}")
        print(f"Extracting camera info for: {survey_name.upper()}")
        print(f"{'='*70}")

        rosbag_path = self.get_rosbag_path(survey_name)
        output_dir = self.output_base / survey_name / "camera_info"
        output_dir.mkdir(parents=True, exist_ok=True)

        db_paths = self.get_db_files(rosbag_path)
        print(f"Found {len(db_paths)} database files")

        topic = '/zed/zed_node/rgb/camera_info'
        total_extracted = 0

        for db_idx, db_path in enumerate(db_paths):
            print(f"\nProcessing database {db_idx + 1}/{len(db_paths)}: {db_path.name}")

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM topics WHERE name = ?", (topic,))
            result = cursor.fetchone()
            if not result:
                print(f"  ⚠️  Topic not found in this database")
                conn.close()
                continue
            topic_id = result[0]

            cursor.execute(
                "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
                (topic_id,)
            )

            count = 0
            for timestamp, data in cursor:
                try:
                    msg = deserialize_message(data, CameraInfo)

                    camera_data = {
                        'timestamp': timestamp,
                        'width': msg.width,
                        'height': msg.height,
                        'distortion_model': msg.distortion_model,
                        'D': list(msg.d),
                        'K': list(msg.k),
                        'R': list(msg.r),
                        'P': list(msg.p)
                    }

                    output_file = output_dir / f"{timestamp}.json"
                    with open(output_file, 'w') as f:
                        json.dump(camera_data, f)

                    count += 1
                    total_extracted += 1
                except Exception as e:
                    continue

            print(f"  ✅ Extracted {count} camera info messages from this database")
            conn.close()

        print(f"\n✅ Total camera info messages extracted: {total_extracted}")
        return total_extracted

    def extract_all(self, survey_name, topics=None):
        """
        Extract all topics for a survey.

        Args:
            survey_name: 'bathroom' or 'office'
            topics: List of topics to extract (None = all)
        """
        print(f"\n{'#'*70}")
        print(f"# EXTRACTING DATA FOR: {survey_name.upper()}")
        print(f"{'#'*70}")

        available_topics = {
            'rgb': self.extract_rgb_images,
            'depth': self.extract_depth_images,
            'pointcloud': self.extract_pointclouds,
            'laser': self.extract_laser_scans,
            'odometry': self.extract_odometry,
            'imu': self.extract_imu,
            'camera_info': self.extract_camera_info
        }

        if topics is None:
            topics = list(available_topics.keys())

        results = {}
        for topic in topics:
            if topic in available_topics:
                try:
                    results[topic] = available_topics[topic](survey_name)
                except Exception as e:
                    print(f"\n❌ Error extracting {topic}: {e}")
                    results[topic] = 0
            else:
                print(f"\n⚠️  Unknown topic: {topic}")

        return results


def main():
    """Main entry point for the extraction script."""
    parser = argparse.ArgumentParser(
        description='Extract sensor data from ROS2 bags',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s bathroom                    # Extract all topics from bathroom
  %(prog)s office --topics rgb depth   # Extract only RGB and depth from office
  %(prog)s                             # Extract all topics from both surveys
        """
    )

    parser.add_argument(
        'survey',
        nargs='?',
        choices=['bathroom', 'office'],
        help='Survey to extract (if not specified, extracts both)'
    )

    parser.add_argument(
        '--topics',
        nargs='+',
        choices=['rgb', 'depth', 'pointcloud', 'laser', 'odometry', 'imu', 'camera_info'],
        help='Topics to extract (if not specified, extracts all)'
    )

    args = parser.parse_args()

    # Initialize extractor
    extractor = RosbagExtractor()

    # Determine which surveys to process
    surveys = [args.survey] if args.survey else ['bathroom', 'office']

    # Extract data
    print("\n" + "="*70)
    print("ROS2 BAG DATA EXTRACTION")
    print("="*70)

    for survey in surveys:
        extractor.extract_all(survey, args.topics)

    print("\n" + "="*70)
    print("✅ EXTRACTION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()

