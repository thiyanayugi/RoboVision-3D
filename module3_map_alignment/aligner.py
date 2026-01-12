#!/usr/bin/env python3
"""
Map Aligner Module for RoboVision-3D Module 3
Estimates transformation between two maps using RANSAC
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional
from feature_matcher import FeatureMatchResult
from map_loader import MapData


class AlignmentResult:
    """Container for alignment results"""
    
    def __init__(self, transform_matrix: np.ndarray, 
                 translation_pixels: Tuple[float, float],
                 translation_meters: Tuple[float, float],
                 rotation_radians: float,
                 scale: float,
                 num_matches: int,
                 num_inliers: int,
                 inlier_mask: np.ndarray,
                 reprojection_error: float,
                 method: str):
        self.transform_matrix = transform_matrix
        self.translation_pixels = translation_pixels
        self.translation_meters = translation_meters
        self.rotation_radians = rotation_radians
        self.scale = scale
        self.num_matches = num_matches
        self.num_inliers = num_inliers
        self.inlier_mask = inlier_mask
        self.reprojection_error = reprojection_error
        self.method = method
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for YAML export"""
        return {
            'transform': {
                'translation': {
                    'x': float(self.translation_meters[0]),
                    'y': float(self.translation_meters[1])
                },
                'rotation': float(self.rotation_radians),
                'scale': float(self.scale)
            },
            'method': self.method,
            'num_matches': int(self.num_matches),
            'inliers': int(self.num_inliers),
            'reprojection_error': float(self.reprojection_error),
            'notes': "Transform from map2 (office) to map1 (bathroom) coordinate frame"
        }
    
    def __repr__(self):
        return (f"AlignmentResult(translation={self.translation_meters}, "
                f"rotation={self.rotation_radians:.3f} rad, "
                f"inliers={self.num_inliers}/{self.num_matches})")


class MapAligner:
    """Estimates transformation between two maps"""
    
    def __init__(self, ransac_threshold: float = 5.0, 
                 confidence: float = 0.99,
                 max_iters: int = 2000):
        """
        Initialize map aligner
        
        Args:
            ransac_threshold: RANSAC reprojection threshold in pixels
            confidence: RANSAC confidence level
            max_iters: Maximum RANSAC iterations
        """
        self.ransac_threshold = ransac_threshold
        self.confidence = confidence
        self.max_iters = max_iters
    
    def align(self, match_result: FeatureMatchResult, 
              map1: MapData, map2: MapData) -> AlignmentResult:
        """
        Estimate transformation from map2 to map1
        
        Args:
            match_result: Feature matching results
            map1: First map (target, e.g., bathroom)
            map2: Second map (source, e.g., office)
            
        Returns:
            AlignmentResult object
        """
        # Get matched point coordinates
        points1, points2 = match_result.get_matched_points()
        
        # Estimate affine transformation using RANSAC for robust outlier rejection
        # RANSAC iteratively finds the best transformation that maximizes inliers
        # This finds transform that maps points2 -> points1
        transform_matrix, inlier_mask = cv2.estimateAffinePartial2D(
            points2, points1,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            maxIters=self.max_iters,
            confidence=self.confidence
        )
        
        if transform_matrix is None:
            raise ValueError("Failed to estimate transformation")
        
        # Count inliers
        inlier_mask = inlier_mask.ravel()
        num_inliers = np.sum(inlier_mask)
        num_matches = len(points1)
        
        # Extract transformation parameters from the 2x3 affine matrix
        # Transform matrix structure: [[cos(θ)*s, -sin(θ)*s, tx],
        #                               [sin(θ)*s,  cos(θ)*s, ty]]
        # where s=scale, θ=rotation angle, (tx,ty)=translation
        tx_pixels = transform_matrix[0, 2]
        ty_pixels = transform_matrix[1, 2]
        
        # Extract rotation and scale from the affine matrix elements
        # Scale is the magnitude of the first column vector
        # Rotation is the angle of the first column vector
        a = transform_matrix[0, 0]
        b = transform_matrix[0, 1]
        scale = np.sqrt(a**2 + b**2)
        rotation_radians = np.arctan2(b, a)
        
        # Convert translation to meters
        resolution = map1.resolution  # Both maps should have same resolution
        tx_meters = tx_pixels * resolution
        ty_meters = ty_pixels * resolution
        
        # Calculate reprojection error for inliers only
        # This measures how well the transformation aligns the matched points
        points2_transformed = cv2.transform(
            points2.reshape(-1, 1, 2), 
            transform_matrix
        ).reshape(-1, 2)
        
        errors = np.linalg.norm(points1 - points2_transformed, axis=1)
        inlier_errors = errors[inlier_mask.astype(bool)]
        reprojection_error = np.mean(inlier_errors) if len(inlier_errors) > 0 else 0.0
        
        return AlignmentResult(
            transform_matrix=transform_matrix,
            translation_pixels=(tx_pixels, ty_pixels),
            translation_meters=(tx_meters, ty_meters),
            rotation_radians=rotation_radians,
            scale=scale,
            num_matches=num_matches,
            num_inliers=num_inliers,
            inlier_mask=inlier_mask,
            reprojection_error=reprojection_error,
            method="Affine Partial 2D + RANSAC"
        )

