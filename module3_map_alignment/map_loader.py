#!/usr/bin/env python3
"""
Map Loader Module for RoboVision-3D Module 3
Loads PGM occupancy grid maps and their YAML configuration files
"""

import cv2
import numpy as np
import yaml
import os
from typing import Dict, Tuple, Optional


class MapData:
    """Container for map data and metadata"""
    
    def __init__(self, image: np.ndarray, resolution: float, origin: list, 
                 occupied_thresh: float, free_thresh: float, filepath: str):
        self.image = image
        self.resolution = resolution  # meters per pixel
        self.origin = origin  # [x, y, theta] in meters
        self.occupied_thresh = occupied_thresh
        self.free_thresh = free_thresh
        self.filepath = filepath
        self.height, self.width = image.shape[:2]
        
    def get_size_meters(self) -> Tuple[float, float]:
        """Get map size in meters"""
        width_m = self.width * self.resolution
        height_m = self.height * self.resolution
        return width_m, height_m
    
    def __repr__(self):
        width_m, height_m = self.get_size_meters()
        return (f"MapData({self.width}×{self.height} pixels, "
                f"{width_m:.2f}×{height_m:.2f} meters, "
                f"resolution={self.resolution} m/pixel)")


class MapLoader:
    """Loads occupancy grid maps from PGM files and YAML configurations"""
    
    def __init__(self, map_directory: str):
        """
        Initialize map loader
        
        Args:
            map_directory: Path to directory containing room.pgm and room.yaml
        """
        self.map_directory = map_directory
        self.pgm_path = os.path.join(map_directory, "room.pgm")
        self.yaml_path = os.path.join(map_directory, "room.yaml")
        
    def load(self) -> MapData:
        """
        Load map image and configuration
        
        Returns:
            MapData object containing image and metadata
        """
        # Load YAML configuration
        if not os.path.exists(self.yaml_path):
            raise FileNotFoundError(f"YAML file not found: {self.yaml_path}")
            
        with open(self.yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load PGM image
        if not os.path.exists(self.pgm_path):
            raise FileNotFoundError(f"PGM file not found: {self.pgm_path}")
            
        image = cv2.imread(self.pgm_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Failed to load image: {self.pgm_path}")
        
        # Create MapData object
        map_data = MapData(
            image=image,
            resolution=config.get('resolution', 0.015),
            origin=config.get('origin', [0, 0, 0]),
            occupied_thresh=config.get('occupied_thresh', 0.65),
            free_thresh=config.get('free_thresh', 0.25),
            filepath=self.pgm_path
        )
        
        return map_data
    
    @staticmethod
    def load_map(map_directory: str) -> MapData:
        """
        Convenience static method to load a map
        
        Args:
            map_directory: Path to directory containing room.pgm and room.yaml
            
        Returns:
            MapData object
        """
        loader = MapLoader(map_directory)
        return loader.load()


def load_both_maps(bathroom_dir: str, office_dir: str) -> Tuple[MapData, MapData]:
    """
    Load both bathroom and office maps
    
    Args:
        bathroom_dir: Path to bathroom map directory
        office_dir: Path to office map directory
        
    Returns:
        Tuple of (bathroom_map, office_map)
    """
    bathroom_map = MapLoader.load_map(bathroom_dir)
    office_map = MapLoader.load_map(office_dir)
    
    return bathroom_map, office_map


if __name__ == "__main__":
    # Test the loader
    import sys
    
    if len(sys.argv) > 1:
        map_dir = sys.argv[1]
        loader = MapLoader(map_dir)
        map_data = loader.load()
        print(f"Loaded: {map_data}")
        print(f"Origin: {map_data.origin}")
        print(f"Size: {map_data.get_size_meters()} meters")
    else:
        print("Usage: python map_loader.py <map_directory>")

