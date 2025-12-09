#!/usr/bin/env python3
"""
Feature Matcher Module for RoboVision-3D Module 3
Detects and matches features between two occupancy grid maps
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


class FeatureMatchResult:
    """Container for feature matching results"""
    
    def __init__(self, keypoints1, keypoints2, descriptors1, descriptors2, 
                 matches, good_matches):
        self.keypoints1 = keypoints1
        self.keypoints2 = keypoints2
        self.descriptors1 = descriptors1
        self.descriptors2 = descriptors2
        self.matches = matches
        self.good_matches = good_matches
        
    def get_matched_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get matched point coordinates
        
        Returns:
            Tuple of (points1, points2) as Nx2 arrays
        """
        points1 = np.float32([self.keypoints1[m.queryIdx].pt 
                              for m in self.good_matches])
        points2 = np.float32([self.keypoints2[m.trainIdx].pt 
                              for m in self.good_matches])
        return points1, points2
    
    def __repr__(self):
        return (f"FeatureMatchResult(kp1={len(self.keypoints1)}, "
                f"kp2={len(self.keypoints2)}, "
                f"matches={len(self.good_matches)})")


class FeatureMatcher:
    """Detects and matches features between two images"""
    
    def __init__(self, detector_type: str = 'ORB', 
                 match_ratio: float = 0.75,
                 min_matches: int = 10):
        """
        Initialize feature matcher
        
        Args:
            detector_type: Type of feature detector ('ORB', 'SIFT', 'AKAZE')
            match_ratio: Ratio test threshold for good matches (Lowe's ratio)
            min_matches: Minimum number of matches required
        """
        self.detector_type = detector_type
        self.match_ratio = match_ratio
        self.min_matches = min_matches
        self.detector = self._create_detector()
        self.matcher = self._create_matcher()
        
    def _create_detector(self):
        """Create feature detector based on type"""
        if self.detector_type == 'ORB':
            return cv2.ORB_create(nfeatures=2000)
        elif self.detector_type == 'SIFT':
            return cv2.SIFT_create()
        elif self.detector_type == 'AKAZE':
            return cv2.AKAZE_create()
        else:
            raise ValueError(f"Unknown detector type: {self.detector_type}")
    
    def _create_matcher(self):
        """Create feature matcher"""
        if self.detector_type == 'ORB':
            # Use Hamming distance for binary descriptors
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            # Use L2 distance for float descriptors
            return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    def detect_and_match(self, image1: np.ndarray, image2: np.ndarray) -> FeatureMatchResult:
        """
        Detect features and match between two images
        
        Args:
            image1: First image (grayscale)
            image2: Second image (grayscale)
            
        Returns:
            FeatureMatchResult object
        """
        # Detect keypoints and compute descriptors
        keypoints1, descriptors1 = self.detector.detectAndCompute(image1, None)
        keypoints2, descriptors2 = self.detector.detectAndCompute(image2, None)
        
        if descriptors1 is None or descriptors2 is None:
            raise ValueError("Failed to detect features in one or both images")
        
        # Match descriptors using KNN (k=2 for ratio test)
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        
        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < self.min_matches:
            raise ValueError(f"Insufficient matches: {len(good_matches)} < {self.min_matches}")
        
        return FeatureMatchResult(
            keypoints1=keypoints1,
            keypoints2=keypoints2,
            descriptors1=descriptors1,
            descriptors2=descriptors2,
            matches=matches,
            good_matches=good_matches
        )
    
    def draw_matches(self, image1: np.ndarray, image2: np.ndarray, 
                     match_result: FeatureMatchResult, 
                     max_matches: int = 50) -> np.ndarray:
        """
        Draw matches between two images
        
        Args:
            image1: First image
            image2: Second image
            match_result: FeatureMatchResult object
            max_matches: Maximum number of matches to draw
            
        Returns:
            Image with drawn matches
        """
        # Convert to color if grayscale
        if len(image1.shape) == 2:
            image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        if len(image2.shape) == 2:
            image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
        
        # Draw only top matches
        matches_to_draw = match_result.good_matches[:max_matches]
        
        match_image = cv2.drawMatches(
            image1, match_result.keypoints1,
            image2, match_result.keypoints2,
            matches_to_draw, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return match_image

