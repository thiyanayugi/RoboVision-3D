#!/usr/bin/env python3
"""
Visualize Object Detections on Map

This script visualizes detected objects as oriented bounding boxes
overlaid on the occupancy grid map.

Usage:
    python visualize_detections.py

Input:
    results/challenge1/bathroom_detections_filtered.json
    results/challenge1/office_detections_filtered.json
    Challenge_Data/bathroom/room.pgm and room.yaml
    Challenge_Data/office/room.pgm and room.yaml

Output:
    results/challenge1/bathroom_detections.png
    results/challenge1/office_detections.png
"""

import json
import cv2
import numpy as np
import yaml
from pathlib import Path


def world_to_map(x: float, y: float, origin: list, resolution: float, map_height: int) -> tuple:
    """
    Convert world coordinates to map pixel coordinates.
    
    Args:
        x, y: World coordinates in meters
        origin: Map origin [x, y, theta] from YAML
        resolution: Map resolution (meters per pixel)
        map_height: Height of map image in pixels
        
    Returns:
        (map_x, map_y) pixel coordinates
    """
    map_x = int((x - origin[0]) / resolution)
    map_y = int((y - origin[1]) / resolution)
    # Flip y-axis (map origin is bottom-left, image origin is top-left)
    map_y = map_height - map_y
    return map_x, map_y


def draw_oriented_bbox(img, center_world, width, depth, orientation, 
                       origin, resolution, color, thickness=2):
    """
    Draw an oriented bounding box on the map image.
    
    Args:
        img: Map image (BGR)
        center_world: (x, y) center position in world frame
        width, depth: Box dimensions in meters
        orientation: Rotation angle in radians
        origin: Map origin from YAML
        resolution: Map resolution (meters per pixel)
        color: BGR color tuple
        thickness: Line thickness in pixels
    """
    # Convert center to map coordinates
    cx_map, cy_map = world_to_map(center_world[0], center_world[1], 
                                   origin, resolution, img.shape[0])
    
    # Convert dimensions to pixels
    width_px = int(width / resolution)
    depth_px = int(depth / resolution)
    
    # Create rotated rectangle
    rect = ((cx_map, cy_map), (width_px, depth_px), np.degrees(orientation))
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    # Draw bounding box
    cv2.drawContours(img, [box], 0, color, thickness)
    
    # Draw center point
    cv2.circle(img, (cx_map, cy_map), 3, color, -1)
    
    # Draw orientation arrow
    arrow_length = max(width_px, depth_px) // 2
    end_x = int(cx_map + arrow_length * np.cos(orientation))
    end_y = int(cy_map + arrow_length * np.sin(orientation))
    cv2.arrowedLine(img, (cx_map, cy_map), (end_x, end_y), color, thickness)


def visualize_detections(survey_name, detections_json, map_pgm, map_yaml, output_png):
    """
    Visualize all detections on the occupancy grid map.
    
    Args:
        survey_name: Name of survey ('bathroom' or 'office')
        detections_json: Path to detections JSON file
        map_pgm: Path to map PGM file
        map_yaml: Path to map YAML configuration
        output_png: Path to output visualization image
    """
    # Load map image
    map_img = cv2.imread(str(map_pgm), cv2.IMREAD_GRAYSCALE)
    map_img = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
    
    # Load map metadata
    with open(map_yaml, 'r') as f:
        map_data = yaml.safe_load(f)
    
    resolution = map_data['resolution']
    origin = map_data['origin']
    
    # Load detections
    with open(detections_json, 'r') as f:
        data = json.load(f)
    
    detections = data['detections']
    
    # Color map for different object classes
    colors = {
        'chair': (255, 0, 0),      # Blue
        'couch': (0, 255, 0),      # Green
        'table': (0, 0, 255),      # Red
        'shelf': (255, 255, 0),    # Cyan
        'toilet': (255, 0, 255),   # Magenta
        'bathtub': (0, 255, 255),  # Yellow
    }
    
    # Draw each detection
    for det in detections:
        cls = det['class']
        color = colors.get(cls, (128, 128, 128))  # Gray for unknown classes
        
        center = (det['pose']['x'], det['pose']['y'])
        width = det['dimensions']['width']
        depth = det['dimensions']['depth']
        orientation = det['pose']['orientation']
        
        draw_oriented_bbox(map_img, center, width, depth, orientation, 
                          origin, resolution, color, thickness=2)
    
    # Save visualization
    cv2.imwrite(str(output_png), map_img)
    
    print(f"📦 Detections: {len(detections)}")
    print(f"💾 Saved: {output_png}")
    
    # Print summary by class
    by_class = {}
    for det in detections:
        cls = det['class']
        by_class[cls] = by_class.get(cls, 0) + 1
    
    print(f"\n📊 Detections by class:")
    for cls, count in sorted(by_class.items()):
        print(f"   • {cls}: {count}")


if __name__ == '__main__':
    print("="*70)
    print("VISUALIZING BATHROOM DETECTIONS")
    print("="*70)
    visualize_detections(
        'bathroom',
        '../results/challenge1/bathroom_detections_filtered.json',
        '../Challenge_Data/bathroom/room.pgm',
        '../Challenge_Data/bathroom/room.yaml',
        '../results/challenge1/bathroom_detections.png'
    )
    
    print("\n" + "="*70)
    print("VISUALIZING OFFICE DETECTIONS")
    print("="*70)
    visualize_detections(
        'office',
        '../results/challenge1/office_detections_filtered.json',
        '../Challenge_Data/office/room.pgm',
        '../Challenge_Data/office/room.yaml',
        '../results/challenge1/office_detections.png'
    )
    
    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*70)

