#!/usr/bin/env python3
"""
RoboVision-3D Module 3: Map Registration
Main entry point for map alignment pipeline
"""

import os
import sys
import numpy as np
from map_loader import MapLoader
from feature_matcher import FeatureMatcher
from aligner import MapAligner
from visualizer import Visualizer


def print_header():
    """Print header"""
    print("=" * 70)
    print("RoboVision-3D Module 3: Map Registration")
    print("=" * 70)
    print()


def print_section(title):
    """Print section header"""
    print(f"{title}")


def main():
    """Main pipeline for map registration"""
    
    print_header()
    
    # Define paths
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bathroom_dir = os.path.join(workspace_dir, "Challenge_Data", "bathroom")
    office_dir = os.path.join(workspace_dir, "Challenge_Data", "office")
    output_dir = os.path.join(workspace_dir, "results", "challenge3")
    
    try:
        # Step 1: Load maps
        print_section("📂 Loading maps...")
        
        bathroom_loader = MapLoader(bathroom_dir)
        bathroom_map = bathroom_loader.load()
        print(f"   ✅ Bathroom map: {bathroom_map.width}×{bathroom_map.height} pixels")
        
        office_loader = MapLoader(office_dir)
        office_map = office_loader.load()
        print(f"   ✅ Office map: {office_map.width}×{office_map.height} pixels")
        print()
        
        # Step 2: Detect and match features
        print_section("🔍 Detecting features...")
        
        # Try ORB first, fall back to AKAZE if needed
        detector_type = 'ORB'
        try:
            matcher = FeatureMatcher(detector_type=detector_type, match_ratio=0.75, min_matches=10)
            match_result = matcher.detect_and_match(bathroom_map.image, office_map.image)
        except ValueError as e:
            print(f"   ⚠️  ORB failed: {e}")
            print(f"   🔄 Trying AKAZE...")
            detector_type = 'AKAZE'
            matcher = FeatureMatcher(detector_type=detector_type, match_ratio=0.75, min_matches=10)
            match_result = matcher.detect_and_match(bathroom_map.image, office_map.image)
        
        print(f"   ✅ Bathroom: {len(match_result.keypoints1)} keypoints")
        print(f"   ✅ Office: {len(match_result.keypoints2)} keypoints")
        print()
        
        # Step 3: Match features
        print_section("🔗 Matching features...")
        print(f"   ✅ Found {len(match_result.good_matches)} matches")
        print()
        
        # Step 4: Estimate transformation
        print_section("📐 Computing transform...")

        aligner = MapAligner(ransac_threshold=5.0, confidence=0.99, max_iters=2000)
        alignment = aligner.align(match_result, bathroom_map, office_map)

        print(f"   ✅ After RANSAC: {alignment.num_inliers} inliers")
        print(f"   ✅ Translation: ({alignment.translation_meters[0]:.3f}m, "
              f"{alignment.translation_meters[1]:.3f}m)")

        rotation_deg = np.degrees(alignment.rotation_radians)
        print(f"   ✅ Rotation: {alignment.rotation_radians:.3f} radians ({rotation_deg:.1f}°)")
        print(f"   ✅ Scale: {alignment.scale:.3f}")
        print(f"   ✅ Reprojection error: {alignment.reprojection_error:.2f} pixels")
        print()

        # Step 4.5: Optimize alignment for perfect wall alignment
        print_section("🔧 Optimizing alignment (wall-focused)...")
        from optimizer import differential_evolution_optimize, compute_wall_alignment_score

        orig_score = compute_wall_alignment_score(
            bathroom_map.image, office_map.image, alignment.transform_matrix
        )
        print(f"   Original wall alignment score: {orig_score:.2f}")

        optimized_transform, opt_score = differential_evolution_optimize(
            bathroom_map.image, office_map.image,
            alignment.transform_matrix
        )

        # Update alignment with optimized transform
        scale = np.sqrt(optimized_transform[0, 0]**2 + optimized_transform[0, 1]**2)
        rotation = np.arctan2(optimized_transform[1, 0], optimized_transform[0, 0])
        translation_pixels = (optimized_transform[0, 2], optimized_transform[1, 2])
        translation_meters = (translation_pixels[0] * bathroom_map.resolution,
                             translation_pixels[1] * bathroom_map.resolution)

        from aligner import AlignmentResult
        alignment = AlignmentResult(
            transform_matrix=optimized_transform,
            translation_pixels=translation_pixels,
            translation_meters=translation_meters,
            rotation_radians=rotation,
            scale=scale,
            num_matches=alignment.num_matches,
            num_inliers=alignment.num_inliers,
            inlier_mask=alignment.inlier_mask,
            reprojection_error=alignment.reprojection_error,
            method="ORB + RANSAC + Differential Evolution (Wall-Aligned)"
        )

        print(f"   ✅ Optimized translation: ({translation_meters[0]:.3f}m, {translation_meters[1]:.3f}m)")
        print(f"   ✅ Optimized rotation: {np.degrees(rotation):.3f}°")
        print(f"   ✅ Optimized scale: {scale:.4f}")
        print(f"   ✅ Wall alignment score: {orig_score:.2f} → {opt_score:.2f} ({100*(opt_score-orig_score)/orig_score:+.1f}%)")
        print()
        
        # Step 5: Create visualization
        print_section("🎨 Creating visualization...")

        visualizer = Visualizer(output_dir=output_dir)

        # Create 4-panel grid visualization
        visualization = visualizer.create_alignment_visualization(
            bathroom_map, office_map, alignment,
            map1_name="Bathroom", map2_name="Office"
        )

        # Create standalone aligned overlay
        overlay_image = visualizer.create_aligned_overlay(
            bathroom_map, office_map, alignment,
            map1_name="Bathroom", map2_name="Office"
        )

        # Create whole aligned map
        print_section("🗺️  Creating whole aligned map...")
        whole_map_gray, whole_map_color, canvas_info = visualizer.create_whole_aligned_map(
            bathroom_map, office_map, alignment,
            map1_name="Bathroom", map2_name="Office"
        )

        offset_x, offset_y, canvas_width, canvas_height = canvas_info
        width_m = canvas_width * bathroom_map.resolution
        height_m = canvas_height * bathroom_map.resolution

        print(f"   ✅ Canvas size: {canvas_width}×{canvas_height} pixels ({width_m:.2f}×{height_m:.2f} m)")
        print()

        # Save results
        visualizer.save_results(
            visualization, alignment, overlay_image,
            whole_map_gray, whole_map_color
        )

        # Save YAML for whole map
        visualizer.save_whole_map_yaml(bathroom_map, canvas_info)
        print()
        
        # Print summary
        print("=" * 70)
        print("✅ Challenge 3 Complete!")
        print("=" * 70)
        print()
        print(f"Results saved to: {output_dir}")
        print(f"  - alignment_transform.yaml (transform parameters)")
        print(f"  - aligned_maps.png (4-panel comparison)")
        print(f"  - aligned_overlay.png (clean overlay)")
        print(f"  - whole_aligned_map.png (complete map - color)")
        print(f"  - whole_aligned_map_gray.pgm (complete map - grayscale)")
        print(f"  - whole_aligned_map.yaml (whole map configuration)")
        print()
        
        # Print interpretation
        print("📊 Interpretation:")
        if alignment.num_inliers >= 20:
            print("   ✅ Good alignment with sufficient inliers")
        elif alignment.num_inliers >= 10:
            print("   ⚠️  Moderate alignment - maps may have limited overlap")
        else:
            print("   ⚠️  Poor alignment - maps may not overlap significantly")
        
        if alignment.reprojection_error < 3.0:
            print("   ✅ Low reprojection error - high confidence")
        elif alignment.reprojection_error < 10.0:
            print("   ⚠️  Moderate reprojection error")
        else:
            print("   ⚠️  High reprojection error - alignment may be unreliable")
        print()
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ Error occurred:")
        print("=" * 70)
        print(f"{type(e).__name__}: {e}")
        print()
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

