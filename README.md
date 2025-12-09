# RoboVision-3D

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)

**A comprehensive computer vision and robotics system for indoor environment mapping, object detection, and 3D reconstruction.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Modules](#project-modules)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Results](#results)
- [Technical Stack](#technical-stack)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**RoboVision-3D** is an advanced robotics perception system designed for autonomous indoor navigation and environment understanding. The system integrates multiple sensor modalities including stereo vision, LiDAR, and laser scanning to create comprehensive 3D representations of indoor spaces with semantic object understanding.

### Sensor Suite

The system utilizes data from a MESA robot equipped with:

| Sensor                | Purpose                        | Output                     |
| --------------------- | ------------------------------ | -------------------------- |
| **ZED Stereo Camera** | RGB imaging & depth estimation | RGB images, depth maps     |
| **Livox LiDAR**       | High-density 3D scanning       | Point clouds               |
| **2D Laser Scanner**  | Mapping & localization         | Laser scans                |
| **IMU**               | Motion tracking                | Inertial measurements      |
| **Wheel Odometry**    | Pose estimation                | Robot position/orientation |

### Test Environments

- **Bathroom**: Compact indoor space with fixtures
- **Office**: Large workspace with furniture

---

## ✨ Features

### 🎯 3D Object Detection & Localization

- Deep learning-based object detection (YOLOv8)
- 6-DOF pose estimation with oriented bounding boxes
- Multi-sensor fusion (RGB + LiDAR)
- Robust multi-frame clustering

### 🌈 Point Cloud Colorization

- RGB-LiDAR sensor fusion
- Camera-to-LiDAR projection
- Full environment reconstruction
- Efficient voxel downsampling

### 🗺️ Multi-Survey Map Alignment

- Feature-based map registration
- RANSAC-based robust alignment
- Global optimization with differential evolution
- Sub-pixel accuracy alignment

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RoboVision-3D System                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   RGB-D      │  │    LiDAR     │  │  Laser Scan  │      │
│  │   Camera     │  │   Sensor     │  │   + IMU      │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │               │
│         └──────────────────┴──────────────────┘              │
│                            │                                  │
│                   ┌────────▼────────┐                        │
│                   │  Data Fusion &  │                        │
│                   │ Synchronization │                        │
│                   └────────┬────────┘                        │
│                            │                                  │
│         ┌──────────────────┼──────────────────┐             │
│         │                  │                  │              │
│  ┌──────▼──────┐  ┌────────▼────────┐  ┌─────▼──────┐     │
│  │   Object    │  │  Point Cloud    │  │    Map     │     │
│  │  Detection  │  │  Colorization   │  │ Alignment  │     │
│  │  Module     │  │     Module      │  │   Module   │     │
│  └──────┬──────┘  └────────┬────────┘  └─────┬──────┘     │
│         │                  │                  │              │
│         └──────────────────┴──────────────────┘             │
│                            │                                  │
│                   ┌────────▼────────┐                        │
│                   │  3D Environment │                        │
│                   │  Representation │                        │
│                   └─────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Project Modules

### Module 1: Object Detection and Localization

Detect and localize furniture objects in 3D space with accurate oriented bounding boxes.

**Target Objects**: Bathtub, Chair, Couch, Shelf, Table, Toilet

**Key Capabilities:**

- YOLOv8-based 2D detection in RGB images
- LiDAR-based 3D localization
- PCA-based oriented bounding box fitting
- Multi-frame clustering for robustness

**📖 [Detailed Documentation](module1_object_detection.md)**

---

### Module 2: Point Cloud Colorization

Create photorealistic colored point clouds by fusing RGB camera data with LiDAR scans.

**Key Capabilities:**

- Camera-LiDAR calibration and projection
- Multi-frame point cloud aggregation
- Voxel-based downsampling
- PLY format export

**📖 [Detailed Documentation](module2_point_cloud.md)**

---

### Module 3: Map Alignment

Align occupancy grid maps from multiple surveys of the same environment.

**Key Capabilities:**

- ORB feature detection and matching
- RANSAC-based robust transformation estimation
- Differential evolution optimization
- Multi-metric alignment scoring

**📖 [Detailed Documentation](module3_map_alignment.md)**

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.10 or higher
- **Git LFS**: For large point cloud files
- **Operating System**: Linux (Ubuntu 20.04+ recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/yugimariraj01/RoboVision-3D.git
cd RoboVision-3D
```

### 2. Install Git LFS

Point cloud files (`.ply`) are stored using Git LFS:

```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# macOS
brew install git-lfs

# Initialize and pull large files
git lfs install
git lfs pull
```

**Large files managed by LFS:**

- `results/module2/bathroom_colorized.ply` (141 MB, 5.5M points)
- `results/module2/office_colorized.ply` (510 MB, 19.8M points)

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**

| Package         | Version | Purpose                   |
| --------------- | ------- | ------------------------- |
| `numpy`         | 1.24+   | Numerical computing       |
| `opencv-python` | 4.8+    | Computer vision           |
| `pyyaml`        | -       | Configuration parsing     |
| `scikit-learn`  | -       | Clustering algorithms     |
| `scipy`         | -       | Optimization              |
| `open3d`        | 0.17+   | 3D point cloud processing |
| `ultralytics`   | -       | YOLOv8 detection          |

---

## 🎮 Quick Start

### Data Preparation (Optional)

The repository includes pre-processed data. To regenerate from ROS bags:

```bash
# Extract sensor data from ROS2 bags
python utils/extract_rosbag_data.py

# Synchronize multi-sensor streams
python utils/synchronize_data.py

# Compute accurate robot poses
python utils/compute_scan_to_map_odom.py

# Run YOLO object detection
python utils/run_yolo_detection.py
```

### Run Module 1: Object Detection

```bash
cd module1_object_detection
python detect_objects.py
python filter_detections.py
python visualize_detections.py
```

**Output:**

- `results/module1/bathroom_detections.json`
- `results/module1/bathroom_detections.png`
- `results/module1/office_detections.json`
- `results/module1/office_detections.png`

### Run Module 2: Point Cloud Colorization

```bash
cd module2_point_cloud
python colorize_clouds.py
```

**Output:**

- `results/module2/bathroom_colorized.ply`
- `results/module2/office_colorized.ply`

### Run Module 3: Map Alignment

```bash
cd module3_map_alignment
python main.py
```

**Output:**

- `results/module3/alignment_transform.yaml`
- `results/module3/aligned_overlay.png`

---

## 📊 Results

### Object Detection Performance

| Environment | Toilets | Chairs | Couches | Tables | Total  |
| ----------- | ------- | ------ | ------- | ------ | ------ |
| Bathroom    | 1       | 0      | 0       | 0      | **1**  |
| Office      | 0       | 9      | 7       | 3      | **19** |

**Detection Quality:**

- All objects have oriented 3D bounding boxes
- Average confidence: 0.82
- Multi-frame validation (50+ observations per object)

### Point Cloud Statistics

| Environment | Points | File Size | Processing Time |
| ----------- | ------ | --------- | --------------- |
| Bathroom    | 5.5M   | 141 MB    | ~2-3 min        |
| Office      | 19.8M  | 510 MB    | ~10 min         |

### Map Alignment Accuracy

```yaml
Translation: (16.30m, -5.48m)
Rotation: 80.27° (1.401 rad)
Scale: 0.9887
Reprojection Error: 1.08 pixels
Edge Alignment: 55.8% within 3px
```

---

## 🛠️ Technical Stack

### Computer Vision

- **Object Detection**: YOLOv8 (Ultralytics)
- **Feature Detection**: ORB (OpenCV)
- **Image Processing**: OpenCV 4.8+

### 3D Processing

- **Point Cloud Library**: Open3D
- **Coordinate Transformations**: NumPy + SciPy
- **Voxel Processing**: Open3D VoxelGrid

### Machine Learning

- **Clustering**: DBSCAN (scikit-learn)
- **Optimization**: Differential Evolution (SciPy)
- **Robust Estimation**: RANSAC (OpenCV)

### Robotics

- **Data Format**: ROS2 bags
- **Localization**: Scan-to-map matching
- **Sensor Fusion**: Multi-modal data synchronization

---

## 📁 Repository Structure

```
RoboVision-3D/
├── README.md                          # This file
├── module1_object_detection.md        # Module 1 documentation
├── module2_point_cloud.md             # Module 2 documentation
├── module3_map_alignment.md           # Module 3 documentation
├── requirements.txt                   # Python dependencies
│
├── module1_object_detection/          # Object detection module
│   ├── detect_objects.py              # Detection & localization
│   ├── filter_detections.py           # False positive filtering
│   ├── visualize_detections.py        # Map visualization
│   ├── config.yaml                    # Configuration
│   ├── detections/                    # YOLO results
│   └── models/                        # Model weights
│
├── module2_point_cloud/               # Point cloud module
│   ├── colorize_clouds.py             # RGB-LiDAR fusion
│   └── config.yaml                    # Configuration
│
├── module3_map_alignment/             # Map alignment module
│   ├── main.py                        # Main pipeline
│   ├── map_loader.py                  # Map I/O
│   ├── feature_matcher.py             # Feature processing
│   ├── aligner.py                     # RANSAC alignment
│   ├── optimizer.py                   # Optimization
│   ├── visualizer.py                  # Visualization
│   └── config.yaml                    # Configuration
│
├── utils/                             # Utility scripts
│   ├── extract_rosbag_data.py         # ROS2 bag extraction
│   ├── synchronize_data.py            # Sensor synchronization
│   ├── run_yolo_detection.py          # Batch YOLO inference
│   └── compute_scan_to_map_odom.py    # Pose estimation
│
├── synchronized_data/                 # Preprocessed data
│   ├── bathroom_frames.pkl
│   ├── bathroom_metadata.json
│   ├── office_frames.pkl
│   └── office_metadata.json
│
├── scan_to_map_odom/                  # Robot poses
│   ├── bathroom_scan_to_map_odom.pkl
│   └── office_scan_to_map_odom.pkl
│
├── Challenge_Data/                    # Raw input data
│   ├── bathroom/
│   │   ├── room.pgm                   # Occupancy map
│   │   ├── room.yaml                  # Map metadata
│   │   └── *.db3                      # ROS2 bags
│   └── office/
│       ├── room.pgm
│       ├── room.yaml
│       └── *.db3
│
└── results/                           # Output results
    ├── module1/                       # Detection results
    ├── module2/                       # Point clouds
    └── module3/                       # Aligned maps
```

---

## 🎨 Visualization

### View Point Clouds

```bash
# Using Open3D (Python)
python3 -c "import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('results/module2/bathroom_colorized.ply')])"

# Using CloudCompare (recommended)
cloudcompare results/module2/bathroom_colorized.ply
```

### View Detection Results

```bash
# Any image viewer
eog results/module1/bathroom_detections.png
```

### View Map Alignment

```bash
# View aligned maps
eog results/module3/aligned_overlay.png
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yugimariraj01/RoboVision-3D.git
cd RoboVision-3D

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Thiyanayugi Mariraj**

- GitHub: [@yugimariraj01](https://github.com/yugimariraj01)
- Email: thiyanayugi@example.com

---

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics team for the object detection framework
- **Open3D**: Intel ISL for the 3D processing library
- **OpenCV**: OpenCV team for computer vision tools
- **ROS2**: Open Robotics for the robotics middleware

---

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{robovision3d2025,
  author = {Mariraj, Thiyanayugi},
  title = {RoboVision-3D: Computer Vision and Robotics for Indoor Navigation},
  year = {2025},
  url = {https://github.com/yugimariraj01/RoboVision-3D}
}
```

---

<div align="center">
  <strong>Built with ❤️ for autonomous robotics</strong>
</div>
