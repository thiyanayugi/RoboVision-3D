#!/usr/bin/env python3
"""
Advanced optimization with rotation, scale, and translation search
Focuses on perfect wall alignment
"""

import cv2
import numpy as np
import os
import yaml
from scipy.optimize import differential_evolution

from map_loader import MapLoader
from feature_matcher import FeatureMatcher
from aligner import MapAligner, AlignmentResult
from visualizer import Visualizer


def compute_wall_alignment_score(map1_img, map2_img, transform_matrix):
    """
    Compute score focused on wall alignment with robust error handling

    Returns:
        score (higher is better)
    """
    h, w = map1_img.shape
    warped2 = cv2.warpAffine(map2_img, transform_matrix, (w, h), borderValue=255)

    # Extract edges (walls)
    edges1 = cv2.Canny(map1_img, 50, 150)
    edges2 = cv2.Canny(warped2, 50, 150)

    # Distance transforms - for each pixel, distance to nearest edge
    dist1 = cv2.distanceTransform(255 - edges1, cv2.DIST_L2, 5)
    dist2 = cv2.distanceTransform(255 - edges2, cv2.DIST_L2, 5)

    # Score 1: Average distance from edges1 to nearest edge in edges2
    edge_pixels1 = edges1 > 0
    edge_pixels2 = edges2 > 0

    num_edges1 = np.sum(edge_pixels1)
    num_edges2 = np.sum(edge_pixels2)

    # Need at least some edges to score
    if num_edges1 < 10 or num_edges2 < 10:
        return 0.0

    distances_1to2 = dist2[edge_pixels1]
    avg_dist_1to2 = np.mean(distances_1to2)
    close_edges_1to2 = np.sum(distances_1to2 <= 3.0)
    ratio_close_1to2 = close_edges_1to2 / max(num_edges1, 1)

    # Score 2: Average distance from edges2 to nearest edge in edges1
    distances_2to1 = dist1[edge_pixels2]
    avg_dist_2to1 = np.mean(distances_2to1)
    close_edges_2to1 = np.sum(distances_2to1 <= 3.0)
    ratio_close_2to1 = close_edges_2to1 / max(num_edges2, 1)

    # Combined distance score - lower is better, so invert
    avg_distance = (avg_dist_1to2 + avg_dist_2to1) / 2.0
    distance_score = 1.0 / (1.0 + avg_distance)

    # Ratio of edges within 3 pixels
    close_ratio = (ratio_close_1to2 + ratio_close_2to1) / 2.0

    # Score 3: Direct edge overlap (white areas)
    edge_overlap = np.sum(edge_pixels1 & edge_pixels2)
    total_edges = num_edges1 + num_edges2
    overlap_ratio = (2.0 * edge_overlap) / max(total_edges, 1)

    # Score 4: Occupied pixel IoU
    occupied1 = (map1_img < 50)
    occupied2 = (warped2 < 50)
    intersection = np.sum(occupied1 & occupied2)
    union = np.sum(occupied1 | occupied2)
    iou = intersection / max(union, 1)

    # Score 5: Spatial balance (penalize scale mismatch)
    # Split into multiple vertical strips for better scale detection
    num_strips = 5
    strip_width = w // num_strips
    strip_distances = []

    for i in range(num_strips):
        x_start = i * strip_width
        x_end = (i + 1) * strip_width if i < num_strips - 1 else w

        strip_edges1 = edges1[:, x_start:x_end] > 0
        strip_edges2 = edges2[:, x_start:x_end] > 0

        if np.sum(strip_edges1) > 10:  # Only consider strips with enough edges
            strip_dist2 = dist2[:, x_start:x_end]
            avg_dist = np.mean(strip_dist2[strip_edges1])
            strip_distances.append(avg_dist)

    # Penalize variance in distances (want consistent alignment across all strips)
    if len(strip_distances) >= 2:
        std_dev = np.std(strip_distances)
        if np.isnan(std_dev) or np.isinf(std_dev):
            balance_score = 0.0
        else:
            balance_score = 1.0 / (1.0 + std_dev)
    else:
        balance_score = 0.5

    # Combined score emphasizing TRUE alignment (low distances, high overlap)
    total_score = (close_ratio * 2000.0 +      # MOST IMPORTANT: edges within 3px
                   distance_score * 1500.0 +   # Average distance (inverted)
                   overlap_ratio * 1000.0 +    # Direct overlap ratio
                   iou * 500.0 +                # Occupied IoU
                   balance_score * 1000.0)     # Spatial balance

    # Final NaN check
    if np.isnan(total_score) or np.isinf(total_score):
        return 0.0

    return total_score


def exhaustive_search(map1_img, map2_img, initial_transform):
    """
    Exhaustive search over translation, rotation, and scale
    """
    print("\n🔍 Exhaustive parameter search...")
    
    # Extract initial parameters
    initial_scale = np.sqrt(initial_transform[0, 0]**2 + initial_transform[0, 1]**2)
    initial_rotation = np.arctan2(initial_transform[1, 0], initial_transform[0, 0])
    initial_tx = initial_transform[0, 2]
    initial_ty = initial_transform[1, 2]
    
    print(f"   Initial: tx={initial_tx:.1f}, ty={initial_ty:.1f}, rot={np.degrees(initial_rotation):.2f}°, scale={initial_scale:.4f}")
    
    best_score = -1
    best_params = None
    best_transform = initial_transform.copy()
    
    # Search ranges
    tx_range = range(-30, 31, 2)  # ±30px, step 2
    ty_range = range(-30, 31, 2)
    rot_range = np.arange(-3, 3.1, 0.2)  # ±3°, step 0.2°
    scale_range = np.arange(0.92, 1.09, 0.005)  # 0.92-1.08, step 0.005
    
    total_iterations = len(tx_range) * len(ty_range) * len(rot_range) * len(scale_range)
    print(f"   Searching {total_iterations:,} combinations...")
    
    iteration = 0
    for scale_factor in scale_range:
        new_scale = initial_scale * scale_factor
        
        for rot_deg in rot_range:
            rot_rad = initial_rotation + np.radians(rot_deg)
            
            cos_r = np.cos(rot_rad) * new_scale
            sin_r = np.sin(rot_rad) * new_scale
            
            for dx in tx_range:
                for dy in ty_range:
                    tx = initial_tx + dx
                    ty = initial_ty + dy
                    
                    transform = np.array([
                        [cos_r, -sin_r, tx],
                        [sin_r, cos_r, ty]
                    ], dtype=np.float32)
                    
                    score = compute_wall_alignment_score(map1_img, map2_img, transform)
                    
                    if score > best_score:
                        best_score = score
                        best_params = (dx, dy, rot_deg, scale_factor)
                        best_transform = transform.copy()
                    
                    iteration += 1
                    if iteration % 50000 == 0:
                        print(f"     Progress: {iteration:,}/{total_iterations:,} ({100*iteration/total_iterations:.1f}%)")
    
    dx, dy, rot_deg, scale_factor = best_params
    print(f"\n   ✅ Best parameters found:")
    print(f"      Δtx: {dx:+d}px")
    print(f"      Δty: {dy:+d}px")
    print(f"      Δrotation: {rot_deg:+.2f}°")
    print(f"      Scale factor: {scale_factor:.4f} ({(scale_factor-1)*100:+.1f}%)")
    print(f"      Score: {best_score:.2f}")
    
    return best_transform, best_score


def differential_evolution_optimize(map1_img, map2_img, initial_transform):
    """
    Use differential evolution for global optimization
    """
    print("\n🧬 Differential evolution optimization...")
    
    # Extract initial parameters
    initial_scale = np.sqrt(initial_transform[0, 0]**2 + initial_transform[0, 1]**2)
    initial_rotation = np.arctan2(initial_transform[1, 0], initial_transform[0, 0])
    initial_tx = initial_transform[0, 2]
    initial_ty = initial_transform[1, 2]
    
    # Objective function (minimize negative score)
    def objective(params):
        dx, dy, rot_deg, scale_factor = params
        
        tx = initial_tx + dx
        ty = initial_ty + dy
        rot_rad = initial_rotation + np.radians(rot_deg)
        new_scale = initial_scale * scale_factor
        
        cos_r = np.cos(rot_rad) * new_scale
        sin_r = np.sin(rot_rad) * new_scale
        
        transform = np.array([
            [cos_r, -sin_r, tx],
            [sin_r, cos_r, ty]
        ], dtype=np.float32)
        
        score = compute_wall_alignment_score(map1_img, map2_img, transform)
        return -score  # Minimize negative score
    
    # Bounds - expanded for better exploration
    bounds = [
        (-40, 40),      # dx (expanded from ±30)
        (-40, 40),      # dy (expanded from ±30)
        (-4, 4),        # rotation (degrees) (expanded from ±3)
        (0.90, 1.10)    # scale factor (expanded from 0.92-1.08)
    ]

    # Run optimization with larger population for better exploration
    result = differential_evolution(
        objective,
        bounds,
        maxiter=150,      # More iterations
        popsize=20,       # Larger population
        tol=0.001,        # Tighter tolerance
        seed=42,
        workers=1,
        disp=True,
        atol=0.0001,      # Absolute tolerance
        updating='deferred'  # Better for parallel evaluation
    )
    
    # Extract best parameters
    dx, dy, rot_deg, scale_factor = result.x
    
    tx = initial_tx + dx
    ty = initial_ty + dy
    rot_rad = initial_rotation + np.radians(rot_deg)
    new_scale = initial_scale * scale_factor
    
    cos_r = np.cos(rot_rad) * new_scale
    sin_r = np.sin(rot_rad) * new_scale
    
    best_transform = np.array([
        [cos_r, -sin_r, tx],
        [sin_r, cos_r, ty]
    ], dtype=np.float32)
    
    print(f"\n   ✅ Optimized parameters:")
    print(f"      Δtx: {dx:+.2f}px")
    print(f"      Δty: {dy:+.2f}px")
    print(f"      Δrotation: {rot_deg:+.3f}°")
    print(f"      Scale factor: {scale_factor:.4f} ({(scale_factor-1)*100:+.2f}%)")
    print(f"      Score: {-result.fun:.2f}")
    
    return best_transform, -result.fun


def main():
    """Run advanced optimization"""
    print("="*70)
    print("ADVANCED ALIGNMENT OPTIMIZATION")
    print("Focus on perfect wall alignment")
    print("="*70)

    # Load maps
    print("\n📂 Loading maps...")
    bathroom_map = MapLoader("../Challenge_Data/bathroom").load()
    office_map = MapLoader("../Challenge_Data/office").load()
    print(f"   ✅ Bathroom: {bathroom_map.width}×{bathroom_map.height}")
    print(f"   ✅ Office: {office_map.width}×{office_map.height}")

    # Get original alignment
    print("\n🔍 Computing original alignment...")
    matcher = FeatureMatcher(detector_type='ORB')
    matches = matcher.detect_and_match(bathroom_map.image, office_map.image)
    aligner = MapAligner()
    original_alignment = aligner.align(matches, bathroom_map, office_map)

    print(f"   ✅ Inliers: {original_alignment.num_inliers}/{original_alignment.num_matches}")
    print(f"   ✅ Reprojection error: {original_alignment.reprojection_error:.2f}px")

    # Compute original score
    orig_score = compute_wall_alignment_score(
        bathroom_map.image, office_map.image, original_alignment.transform_matrix
    )
    print(f"   Original wall alignment score: {orig_score:.2f}")

    # Use differential evolution (faster than exhaustive)
    optimized_transform, optimized_score = differential_evolution_optimize(
        bathroom_map.image, office_map.image,
        original_alignment.transform_matrix
    )

    # Create final alignment result
    scale = np.sqrt(optimized_transform[0, 0]**2 + optimized_transform[0, 1]**2)
    rotation = np.arctan2(optimized_transform[1, 0], optimized_transform[0, 0])
    translation_pixels = (optimized_transform[0, 2], optimized_transform[1, 2])
    translation_meters = (translation_pixels[0] * bathroom_map.resolution,
                         translation_pixels[1] * bathroom_map.resolution)

    optimized_alignment = AlignmentResult(
        transform_matrix=optimized_transform,
        translation_pixels=translation_pixels,
        translation_meters=translation_meters,
        rotation_radians=rotation,
        scale=scale,
        num_matches=original_alignment.num_matches,
        num_inliers=original_alignment.num_inliers,
        inlier_mask=original_alignment.inlier_mask,
        reprojection_error=original_alignment.reprojection_error,
        method="ORB + Differential Evolution (Wall-Aligned)"
    )

    # Save results
    print("\n📊 Saving optimized results...")
    output_dir = "../results/challenge3/advanced_optimized"
    os.makedirs(output_dir, exist_ok=True)

    visualizer = Visualizer(output_dir=output_dir)

    # Create visualizations
    overlay = visualizer.create_aligned_overlay(bathroom_map, office_map, optimized_alignment)
    comparison = visualizer.create_alignment_visualization(bathroom_map, office_map, optimized_alignment,
                                                          "Bathroom", "Office")
    whole_gray, whole_color, canvas_info = visualizer.create_whole_aligned_map(
        bathroom_map, office_map, optimized_alignment
    )

    # Save all visualizations
    cv2.imwrite(f"{output_dir}/aligned_overlay.png", overlay)
    cv2.imwrite(f"{output_dir}/aligned_maps.png", comparison)
    cv2.imwrite(f"{output_dir}/whole_aligned_map.png", whole_color)
    cv2.imwrite(f"{output_dir}/whole_aligned_map_gray.pgm", whole_gray)

    # Save YAML
    with open(f"{output_dir}/alignment_transform.yaml", 'w') as f:
        yaml.dump(optimized_alignment.to_dict(), f, default_flow_style=False)

    yaml_data = {
        'image': 'whole_aligned_map_gray.pgm',
        'resolution': bathroom_map.resolution,
        'origin': [canvas_info[0] * bathroom_map.resolution,
                  canvas_info[1] * bathroom_map.resolution, 0.0],
        'negate': 0,
        'occupied_thresh': bathroom_map.occupied_thresh,
        'free_thresh': bathroom_map.free_thresh
    }
    with open(f"{output_dir}/whole_aligned_map.yaml", 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    # Create detailed comparison
    print("\n📊 Creating detailed comparison...")
    h, w = bathroom_map.image.shape

    # Original
    orig_warped = cv2.warpAffine(office_map.image, original_alignment.transform_matrix,
                                 (w, h), borderValue=255)
    orig_edges1 = cv2.Canny(bathroom_map.image, 50, 150)
    orig_edges2 = cv2.Canny(orig_warped, 50, 150)
    orig_vis = np.zeros((h, w, 3), dtype=np.uint8)
    orig_vis[:, :, 2] = orig_edges1  # Red
    orig_vis[:, :, 1] = orig_edges2  # Cyan

    # Optimized
    opt_warped = cv2.warpAffine(office_map.image, optimized_transform,
                                (w, h), borderValue=255)
    opt_edges1 = cv2.Canny(bathroom_map.image, 50, 150)
    opt_edges2 = cv2.Canny(opt_warped, 50, 150)
    opt_vis = np.zeros((h, w, 3), dtype=np.uint8)
    opt_vis[:, :, 2] = opt_edges1  # Red
    opt_vis[:, :, 1] = opt_edges2  # Cyan

    # Side by side
    comparison_edges = np.hstack([orig_vis, opt_vis])
    cv2.putText(comparison_edges, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison_edges, "Optimized", (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imwrite(f"{output_dir}/edge_comparison.png", comparison_edges)

    print(f"   ✅ Saved to: {output_dir}")

    print("\n" + "="*70)
    print("✅ Advanced Optimization Complete!")
    print("="*70)
    print(f"\nFinal parameters:")
    print(f"  Translation: ({translation_meters[0]:.3f}m, {translation_meters[1]:.3f}m)")
    print(f"  Rotation: {np.degrees(rotation):.3f}°")
    print(f"  Scale: {scale:.4f}")
    print(f"\nWall alignment score: {orig_score:.2f} → {optimized_score:.2f} ({100*(optimized_score-orig_score)/orig_score:+.1f}%)")
    print(f"\nView edge comparison: {output_dir}/edge_comparison.png")


if __name__ == "__main__":
    main()


