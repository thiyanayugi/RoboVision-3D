#!/usr/bin/env python3
"""
Filter Object Detections

This script filters object detections to remove false positives and duplicates.
It applies multiple filtering strategies:
- Confidence threshold filtering
- Minimum observation count filtering
- Dimension validation
- Duplicate merging based on spatial proximity
- Non-maximum suppression to remove overlapping boxes

Usage:
    python filter_detections.py

Input:
    results/challenge1/bathroom_detections.json
    results/challenge1/office_detections.json

Output:
    results/challenge1/bathroom_detections_filtered.json
    results/challenge1/office_detections_filtered.json
"""

import json
import numpy as np


def calculate_distance(det1: dict, det2: dict) -> float:
    """Calculate Euclidean distance between two detection centers."""
    x1, y1 = det1['pose']['x'], det1['pose']['y']
    x2, y2 = det2['pose']['x'], det2['pose']['y']
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def calculate_iou(box1: dict, box2: dict) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.
    
    Uses axis-aligned approximation for simplicity.
    """
    x1_min = box1['x'] - box1['width'] / 2
    x1_max = box1['x'] + box1['width'] / 2
    y1_min = box1['y'] - box1['depth'] / 2
    y1_max = box1['y'] + box1['depth'] / 2
    
    x2_min = box2['x'] - box2['width'] / 2
    x2_max = box2['x'] + box2['width'] / 2
    y2_min = box2['y'] - box2['depth'] / 2
    y2_max = box2['y'] + box2['depth'] / 2
    
    # Calculate intersection
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection = x_overlap * y_overlap
    
    # Calculate union
    area1 = box1['width'] * box1['depth']
    area2 = box2['width'] * box2['depth']
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def calculate_quality_score(det: dict) -> float:
    """
    Calculate quality score for a detection.
    
    Higher scores indicate more reliable detections based on:
    - Detection confidence
    - Number of observations (frames where object was seen)
    - Number of 3D points
    """
    confidence = det['confidence']
    num_points = det.get('num_points', 0)
    num_observations = det.get('num_observations', 1)
    
    score = (
        0.4 * confidence +
        0.5 * min(num_observations / 50.0, 1.0) +
        0.1 * min(num_points / 1000.0, 1.0)
    )
    return score


def merge_nearby_detections(detections: list, distance_threshold: float = 0.5) -> list:
    """
    Merge detections that are very close to each other.
    
    Keeps the detection with the highest quality score from each cluster.
    """
    if not detections:
        return []
    
    merged = []
    used = set()
    
    for i, det1 in enumerate(detections):
        if i in used:
            continue
        
        # Find all detections close to this one
        cluster = [det1]
        cluster_indices = [i]
        
        for j, det2 in enumerate(detections):
            if j <= i or j in used:
                continue
            
            dist = calculate_distance(det1, det2)
            if dist < distance_threshold:
                cluster.append(det2)
                cluster_indices.append(j)
                used.add(j)
        
        # Keep the detection with highest quality
        best_det = max(cluster, key=calculate_quality_score)
        merged.append(best_det)
        used.add(i)
    
    return merged


def is_realistic_dimensions(det: dict) -> bool:
    """
    Check if detection dimensions are realistic for the object class.
    
    This helps filter out false positives with unrealistic sizes.
    """
    cls = det['class']
    w = det['dimensions']['width']
    d = det['dimensions']['depth']
    h = det['dimensions']['height']

    # Realistic size ranges for common furniture (in meters)
    constraints = {
        'chair': {'width': (0.30, 0.85), 'depth': (0.30, 0.85), 'height': (0.40, 1.25)},
        'couch': {'width': (1.20, 2.60), 'depth': (0.60, 1.25), 'height': (0.40, 1.00)},
        'table': {'width': (0.50, 2.00), 'depth': (0.50, 1.50), 'height': (0.30, 1.00)},
        'shelf': {'width': (0.40, 1.50), 'depth': (0.20, 0.60), 'height': (0.50, 2.50)},
        'toilet': {'width': (0.35, 0.90), 'depth': (0.35, 0.85), 'height': (0.35, 0.90)},
        'bathtub': {'width': (1.00, 1.80), 'depth': (0.50, 0.90), 'height': (0.30, 0.70)},
    }

    if cls not in constraints:
        return True

    c = constraints[cls]
    return (c['width'][0] <= w <= c['width'][1] and
            c['depth'][0] <= d <= c['depth'][1] and
            c['height'][0] <= h <= c['height'][1])


def filter_detections(input_json, output_json, min_confidence=0.50,
                      min_observations=10, class_specific_thresholds=None):
    """
    Apply filtering to remove false positives and duplicates.

    Args:
        input_json: Path to input detections JSON file
        output_json: Path to output filtered JSON file
        min_confidence: Minimum detection confidence (0-1)
        min_observations: Minimum number of frames object was observed
        class_specific_thresholds: Optional dict of per-class thresholds

    Returns:
        List of filtered detections
    """
    with open(input_json, 'r') as f:
        data = json.load(f)

    detections = data['detections']
    print(f"Initial detections: {len(detections)}")

    # Apply class-specific or global thresholds
    if class_specific_thresholds:
        filtered = []
        for d in detections:
            cls = d['class']
            if cls in class_specific_thresholds:
                min_conf = class_specific_thresholds[cls]['confidence']
                min_obs = class_specific_thresholds[cls]['observations']
                if d['confidence'] >= min_conf and d.get('num_observations', 0) >= min_obs:
                    filtered.append(d)
            else:
                if d['confidence'] >= min_confidence and d.get('num_observations', 0) >= min_observations:
                    filtered.append(d)
        detections = filtered
    else:
        # Filter by confidence
        detections = [d for d in detections if d['confidence'] >= min_confidence]
        print(f"After confidence filter (>={min_confidence}): {len(detections)}")

        # Filter by observation count
        detections = [d for d in detections if d.get('num_observations', 0) >= min_observations]
        print(f"After observation filter (>={min_observations}): {len(detections)}")

    print(f"After threshold filters: {len(detections)}")

    # Filter by realistic dimensions
    detections = [d for d in detections if is_realistic_dimensions(d)]
    print(f"After dimension filter: {len(detections)}")

    # Merge nearby duplicates by class
    by_class = {}
    for det in detections:
        cls = det['class']
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(det)

    filtered_detections = []
    for cls, dets in by_class.items():
        print(f"\n{cls}: {len(dets)} detections")
        merged = merge_nearby_detections(dets, distance_threshold=0.5)
        print(f"  After merging: {len(merged)}")
        filtered_detections.extend(merged)

    # Apply non-maximum suppression to remove overlapping boxes
    final_detections = []
    by_class = {}
    for det in filtered_detections:
        cls = det['class']
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(det)

    for cls, dets in by_class.items():
        # Sort by quality score (highest first)
        dets_sorted = sorted(dets, key=calculate_quality_score, reverse=True)

        keep = []
        for det in dets_sorted:
            # Check overlap with already kept detections
            overlap = False
            for kept_det in keep:
                box1 = {'x': det['pose']['x'], 'y': det['pose']['y'],
                       'width': det['dimensions']['width'], 'depth': det['dimensions']['depth']}
                box2 = {'x': kept_det['pose']['x'], 'y': kept_det['pose']['y'],
                       'width': kept_det['dimensions']['width'], 'depth': kept_det['dimensions']['depth']}

                iou = calculate_iou(box1, box2)
                if iou > 0.2:  # Overlapping boxes
                    overlap = True
                    break

            if not overlap:
                keep.append(det)

        final_detections.extend(keep)

    # Update data and save
    data['detections'] = final_detections
    data['num_detections'] = len(final_detections)

    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Final detections: {len(final_detections)}")

    # Print summary by class
    by_class = {}
    for det in final_detections:
        cls = det['class']
        by_class[cls] = by_class.get(cls, 0) + 1

    print("\nDetections by class:")
    for cls, count in sorted(by_class.items()):
        print(f"  • {cls}: {count}")

    return final_detections


if __name__ == '__main__':
    print("="*70)
    print("FILTERING DETECTIONS - BATHROOM")
    print("="*70)

    filter_detections(
        '../results/challenge1/bathroom_detections.json',
        '../results/challenge1/bathroom_detections_filtered.json',
        min_confidence=0.70,
        min_observations=50
    )

    print("\n" + "="*70)
    print("FILTERING DETECTIONS - OFFICE")
    print("="*70)

    # Use class-specific thresholds for office
    office_thresholds = {
        'chair': {'confidence': 0.50, 'observations': 15},
        'couch': {'confidence': 0.35, 'observations': 2},
        'table': {'confidence': 0.40, 'observations': 5},
        'shelf': {'confidence': 0.99, 'observations': 9999},  # Exclude shelves
    }

    filter_detections(
        '../results/challenge1/office_detections.json',
        '../results/challenge1/office_detections_filtered.json',
        min_confidence=0.35,
        min_observations=2,
        class_specific_thresholds=office_thresholds
    )

    print("\n" + "="*70)
    print("✅ FILTERING COMPLETE!")
    print("="*70)

