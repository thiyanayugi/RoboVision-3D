#!/usr/bin/env python3
"""
Visualizer Module for RoboVision-3D Module 3
Creates visualizations of map alignment results
"""

import cv2
import numpy as np
import yaml
import os
from typing import Tuple, Optional
from map_loader import MapData
from aligner import AlignmentResult


class Visualizer:
    """Creates visualizations for map alignment"""
    
    def __init__(self, output_dir: str = "results/challenge3"):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_alignment_visualization(self, 
                                       map1: MapData, 
                                       map2: MapData,
                                       alignment: AlignmentResult,
                                       map1_name: str = "Map 1",
                                       map2_name: str = "Map 2") -> np.ndarray:
        """
        Create comprehensive alignment visualization
        
        Args:
            map1: First map (target)
            map2: Second map (source)
            alignment: Alignment result
            map1_name: Name for first map
            map2_name: Name for second map
            
        Returns:
            Visualization image
        """
        # Transform map2 to align with map1
        aligned_map2 = cv2.warpAffine(
            map2.image,
            alignment.transform_matrix,
            (map1.width, map1.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=205  # Gray for unknown areas
        )
        
        # Create colored versions for overlay
        # Map1 in red channel, Map2 in cyan (green + blue)
        overlay = np.zeros((map1.height, map1.width, 3), dtype=np.uint8)
        
        # Red channel: map1 (inverted so walls are bright)
        overlay[:, :, 2] = 255 - map1.image
        
        # Green and Blue channels: aligned map2 (inverted)
        aligned_inverted = 255 - aligned_map2
        overlay[:, :, 0] = aligned_inverted  # Blue
        overlay[:, :, 1] = aligned_inverted  # Green
        
        # Where both maps overlap (walls), color will be white
        # Where only map1 has walls, color will be red
        # Where only map2 has walls, color will be cyan
        
        # Create side-by-side comparison
        map1_color = cv2.cvtColor(map1.image, cv2.COLOR_GRAY2BGR)
        map2_color = cv2.cvtColor(map2.image, cv2.COLOR_GRAY2BGR)
        aligned_map2_color = cv2.cvtColor(aligned_map2, cv2.COLOR_GRAY2BGR)

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (0, 255, 0)  # Green text

        cv2.putText(map1_color, map1_name, (10, 30), font, font_scale, color, thickness)
        cv2.putText(map2_color, map2_name, (10, 30), font, font_scale, color, thickness)
        cv2.putText(aligned_map2_color, f"{map2_name} (Aligned)", (10, 30),
                   font, font_scale, color, thickness)
        cv2.putText(overlay, "Overlay (Red: Map1, Cyan: Map2)", (10, 30),
                   font, font_scale, (255, 255, 255), thickness)

        # Resize all images to same height for display
        max_height = max(map1_color.shape[0], map2_color.shape[0])

        def resize_to_height(img, target_height):
            if img.shape[0] == target_height:
                return img
            aspect_ratio = img.shape[1] / img.shape[0]
            target_width = int(target_height * aspect_ratio)
            return cv2.resize(img, (target_width, target_height))

        map1_resized = resize_to_height(map1_color, max_height)
        map2_resized = resize_to_height(map2_color, max_height)
        aligned_resized = resize_to_height(aligned_map2_color, max_height)
        overlay_resized = resize_to_height(overlay, max_height)

        # Create 2x2 grid
        top_row = np.hstack([map1_resized, map2_resized])
        bottom_row = np.hstack([aligned_resized, overlay_resized])

        # Ensure same width
        if top_row.shape[1] != bottom_row.shape[1]:
            max_width = max(top_row.shape[1], bottom_row.shape[1])
            if top_row.shape[1] < max_width:
                padding = np.zeros((top_row.shape[0], max_width - top_row.shape[1], 3), dtype=np.uint8)
                top_row = np.hstack([top_row, padding])
            if bottom_row.shape[1] < max_width:
                padding = np.zeros((bottom_row.shape[0], max_width - bottom_row.shape[1], 3), dtype=np.uint8)
                bottom_row = np.hstack([bottom_row, padding])

        result = np.vstack([top_row, bottom_row])
        
        # Add title and info at the top
        info_height = 120
        info_panel = np.zeros((info_height, result.shape[1], 3), dtype=np.uint8)
        
        # Add text information
        y_offset = 25
        line_height = 25
        
        cv2.putText(info_panel, "RoboVision-3D Module 3: Map Registration Results", 
                   (10, y_offset), font, 0.8, (255, 255, 255), 2)
        y_offset += line_height
        
        cv2.putText(info_panel, 
                   f"Translation: ({alignment.translation_meters[0]:.3f}m, {alignment.translation_meters[1]:.3f}m)", 
                   (10, y_offset), font, 0.6, (0, 255, 255), 1)
        y_offset += line_height
        
        rotation_deg = np.degrees(alignment.rotation_radians)
        cv2.putText(info_panel, 
                   f"Rotation: {alignment.rotation_radians:.3f} rad ({rotation_deg:.1f} deg)", 
                   (10, y_offset), font, 0.6, (0, 255, 255), 1)
        y_offset += line_height
        
        cv2.putText(info_panel, 
                   f"Matches: {alignment.num_inliers}/{alignment.num_matches} inliers | "
                   f"Error: {alignment.reprojection_error:.2f} px", 
                   (10, y_offset), font, 0.6, (0, 255, 255), 1)
        
        # Combine info panel with result
        final_result = np.vstack([info_panel, result])
        
        return final_result
    
    def create_aligned_overlay(self,
                               map1: MapData,
                               map2: MapData,
                               alignment: AlignmentResult,
                               map1_name: str = "Map 1",
                               map2_name: str = "Map 2") -> np.ndarray:
        """
        Create a clean overlay image of aligned maps

        Args:
            map1: First map (target)
            map2: Second map (source)
            alignment: Alignment result
            map1_name: Name for first map
            map2_name: Name for second map

        Returns:
            Overlay image
        """
        # Transform map2 to align with map1
        aligned_map2 = cv2.warpAffine(
            map2.image,
            alignment.transform_matrix,
            (map1.width, map1.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=205  # Gray for unknown areas
        )

        # Create colored overlay
        # Map1 in red channel, Map2 in cyan (green + blue)
        overlay = np.zeros((map1.height, map1.width, 3), dtype=np.uint8)

        # Red channel: map1 (inverted so walls are bright)
        overlay[:, :, 2] = 255 - map1.image

        # Green and Blue channels: aligned map2 (inverted)
        aligned_inverted = 255 - aligned_map2
        overlay[:, :, 0] = aligned_inverted  # Blue
        overlay[:, :, 1] = aligned_inverted  # Green

        # Add title and legend
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Add title at top
        cv2.putText(overlay, f"Aligned Maps: {map1_name} (Red) + {map2_name} (Cyan)",
                   (10, 30), font, 0.7, (255, 255, 255), 2)

        # Add transform info at bottom
        info_y = map1.height - 60
        cv2.putText(overlay,
                   f"Translation: ({alignment.translation_meters[0]:.2f}m, {alignment.translation_meters[1]:.2f}m)",
                   (10, info_y), font, 0.5, (255, 255, 255), 1)

        rotation_deg = np.degrees(alignment.rotation_radians)
        cv2.putText(overlay,
                   f"Rotation: {rotation_deg:.1f} deg | Inliers: {alignment.num_inliers}/{alignment.num_matches}",
                   (10, info_y + 25), font, 0.5, (255, 255, 255), 1)

        return overlay

    def create_whole_aligned_map(self,
                                  map1: MapData,
                                  map2: MapData,
                                  alignment: AlignmentResult,
                                  map1_name: str = "Map 1",
                                  map2_name: str = "Map 2",
                                  padding: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a complete aligned map showing both maps in a single large canvas

        Args:
            map1: First map (target, e.g., bathroom)
            map2: Second map (source, e.g., office)
            alignment: Alignment result
            map1_name: Name for first map
            map2_name: Name for second map
            padding: Padding around the combined map in pixels

        Returns:
            Tuple of (combined_map_grayscale, combined_map_color, metadata_image)
        """
        # Get the transform matrix
        M = alignment.transform_matrix

        # Transform the corners of map2 to find the bounding box
        h2, w2 = map2.image.shape
        corners_map2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        corners_transformed = cv2.transform(corners_map2, M).reshape(-1, 2)

        # Get corners of map1
        h1, w1 = map1.image.shape
        corners_map1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])

        # Combine all corners to find the bounding box
        all_corners = np.vstack([corners_map1, corners_transformed])

        # Find min and max coordinates
        min_x = np.floor(np.min(all_corners[:, 0])).astype(int)
        max_x = np.ceil(np.max(all_corners[:, 0])).astype(int)
        min_y = np.floor(np.min(all_corners[:, 1])).astype(int)
        max_y = np.ceil(np.max(all_corners[:, 1])).astype(int)

        # Calculate canvas size with padding
        canvas_width = max_x - min_x + 2 * padding
        canvas_height = max_y - min_y + 2 * padding

        # Create offset to shift everything into positive coordinates
        offset_x = padding - min_x
        offset_y = padding - min_y

        # Create translation matrix to shift map1
        M_map1 = np.float32([[1, 0, offset_x], [0, 1, offset_y]])

        # Create translation matrix for map2 (combine alignment + offset)
        M_map2 = M.copy()
        M_map2[0, 2] += offset_x
        M_map2[1, 2] += offset_y

        # Create canvas for grayscale combined map
        combined_gray = np.full((canvas_height, canvas_width), 205, dtype=np.uint8)  # Unknown = gray

        # Warp both maps onto the canvas
        map1_warped = cv2.warpAffine(
            map1.image, M_map1, (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=205
        )

        map2_warped = cv2.warpAffine(
            map2.image, M_map2, (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=205
        )

        # Combine maps: take the minimum value (darker = more occupied)
        # Where both maps have data, show the occupied areas
        mask1 = map1_warped < 205
        mask2 = map2_warped < 205

        combined_gray[mask1] = map1_warped[mask1]
        combined_gray[mask2] = np.minimum(combined_gray[mask2], map2_warped[mask2])

        # Create colored version showing which map contributed what
        combined_color = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        combined_color[:, :] = [205, 205, 205]  # Gray background

        # Map1 in red, Map2 in cyan, overlap in white
        map1_inverted = 255 - map1_warped
        map2_inverted = 255 - map2_warped

        # Red channel: map1
        combined_color[:, :, 2] = map1_inverted

        # Green and Blue channels: map2
        combined_color[:, :, 0] = map2_inverted  # Blue
        combined_color[:, :, 1] = map2_inverted  # Green

        # Create metadata image with info
        info_height = 150
        metadata_img = np.zeros((info_height, canvas_width, 3), dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        line_height = 30

        cv2.putText(metadata_img, "Complete Aligned Map",
                   (10, y_offset), font, 1.0, (255, 255, 255), 2)
        y_offset += line_height

        cv2.putText(metadata_img,
                   f"{map1_name} (Red) + {map2_name} (Cyan) | White = Overlap",
                   (10, y_offset), font, 0.6, (0, 255, 255), 1)
        y_offset += line_height

        # Calculate size in meters
        width_m = canvas_width * map1.resolution
        height_m = canvas_height * map1.resolution

        cv2.putText(metadata_img,
                   f"Size: {canvas_width}×{canvas_height} px ({width_m:.2f}×{height_m:.2f} m)",
                   (10, y_offset), font, 0.6, (200, 200, 200), 1)
        y_offset += line_height

        rotation_deg = np.degrees(alignment.rotation_radians)
        cv2.putText(metadata_img,
                   f"Transform: ({alignment.translation_meters[0]:.2f}m, {alignment.translation_meters[1]:.2f}m), {rotation_deg:.1f}°",
                   (10, y_offset), font, 0.6, (200, 200, 200), 1)

        # Combine metadata with colored map
        combined_with_info = np.vstack([metadata_img, combined_color])

        return combined_gray, combined_with_info, (offset_x, offset_y, canvas_width, canvas_height)

    def save_results(self,
                     visualization: np.ndarray,
                     alignment: AlignmentResult,
                     overlay_image: Optional[np.ndarray] = None,
                     whole_map_gray: Optional[np.ndarray] = None,
                     whole_map_color: Optional[np.ndarray] = None,
                     image_filename: str = "aligned_maps.png",
                     overlay_filename: str = "aligned_overlay.png",
                     whole_map_filename: str = "whole_aligned_map.png",
                     whole_map_gray_filename: str = "whole_aligned_map_gray.pgm",
                     yaml_filename: str = "alignment_transform.yaml"):
        """
        Save visualization and alignment data

        Args:
            visualization: Visualization image (4-panel grid)
            alignment: Alignment result
            overlay_image: Optional standalone overlay image
            whole_map_gray: Optional whole aligned map (grayscale)
            whole_map_color: Optional whole aligned map (color with metadata)
            image_filename: Output image filename for 4-panel grid
            overlay_filename: Output filename for standalone overlay
            whole_map_filename: Output filename for whole aligned map (color)
            whole_map_gray_filename: Output filename for whole aligned map (grayscale PGM)
            yaml_filename: Output YAML filename
        """
        # Save 4-panel visualization
        image_path = os.path.join(self.output_dir, image_filename)
        cv2.imwrite(image_path, visualization)
        print(f"   ✅ Saved: {image_path}")

        # Save standalone overlay if provided
        if overlay_image is not None:
            overlay_path = os.path.join(self.output_dir, overlay_filename)
            cv2.imwrite(overlay_path, overlay_image)
            print(f"   ✅ Saved: {overlay_path}")

        # Save whole aligned map (color) if provided
        if whole_map_color is not None:
            whole_color_path = os.path.join(self.output_dir, whole_map_filename)
            cv2.imwrite(whole_color_path, whole_map_color)
            print(f"   ✅ Saved: {whole_color_path}")

        # Save whole aligned map (grayscale PGM) if provided
        if whole_map_gray is not None:
            whole_gray_path = os.path.join(self.output_dir, whole_map_gray_filename)
            cv2.imwrite(whole_gray_path, whole_map_gray)
            print(f"   ✅ Saved: {whole_gray_path}")

        # Save YAML
        yaml_path = os.path.join(self.output_dir, yaml_filename)
        with open(yaml_path, 'w') as f:
            yaml.dump(alignment.to_dict(), f, default_flow_style=False, sort_keys=False)
        print(f"   ✅ Saved: {yaml_path}")

    def save_whole_map_yaml(self, map1: MapData, canvas_info: tuple,
                            yaml_filename: str = "whole_aligned_map.yaml"):
        """
        Save YAML configuration for the whole aligned map

        Args:
            map1: First map (for resolution and origin reference)
            canvas_info: Tuple of (offset_x, offset_y, canvas_width, canvas_height)
            yaml_filename: Output YAML filename
        """
        offset_x, offset_y, canvas_width, canvas_height = canvas_info

        # Calculate the new origin
        # The origin is at the bottom-left corner in ROS convention
        # Convert numpy types to Python native types
        import numpy as np

        origin_x_val = map1.origin[0]
        origin_y_val = map1.origin[1]
        origin_theta_val = map1.origin[2]

        # Convert to Python float if numpy
        if isinstance(origin_x_val, (np.integer, np.floating)):
            origin_x_val = float(origin_x_val)
        if isinstance(origin_y_val, (np.integer, np.floating)):
            origin_y_val = float(origin_y_val)
        if isinstance(origin_theta_val, (np.integer, np.floating)):
            origin_theta_val = float(origin_theta_val)

        origin_x = origin_x_val - offset_x * map1.resolution
        origin_y = origin_y_val - offset_y * map1.resolution
        origin_theta = origin_theta_val

        config = {
            'image': 'whole_aligned_map_gray.pgm',
            'mode': 'trinary',
            'resolution': float(map1.resolution),
            'origin': [origin_x, origin_y, origin_theta],
            'negate': 0,
            'occupied_thresh': 0.65,
            'free_thresh': 0.25
        }

        yaml_path = os.path.join(self.output_dir, yaml_filename)

        # Use safe_dump to avoid numpy object serialization
        with open(yaml_path, 'w') as f:
            # Manually format to match ROS convention
            f.write(f"image: {config['image']}\n")
            f.write(f"mode: {config['mode']}\n")
            f.write(f"resolution: {config['resolution']}\n")
            f.write(f"origin: [{config['origin'][0]}, {config['origin'][1]}, {config['origin'][2]}]\n")
            f.write(f"negate: {config['negate']}\n")
            f.write(f"occupied_thresh: {config['occupied_thresh']}\n")
            f.write(f"free_thresh: {config['free_thresh']}\n")

        print(f"   ✅ Saved: {yaml_path}")

