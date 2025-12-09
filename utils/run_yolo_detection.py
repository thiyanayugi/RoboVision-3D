#!/usr/bin/env python3
"""
Run YOLO object detection on RGB images.

This script processes all RGB images from a survey and runs YOLOv8 object detection
to identify furniture objects (bathtub, chair, couch, shelf, table, toilet).

The detection process:
1. Load YOLOv8 model (downloads automatically if not present)
2. Process each RGB image
3. Filter detections for target object classes
4. Save detection results as JSON files

Usage:
    python run_yolo_detection.py
    python run_yolo_detection.py bathroom
    python run_yolo_detection.py office
"""

import json
import pickle
from pathlib import Path
from ultralytics import YOLO
import sys
from tqdm import tqdm


# Target object classes (COCO dataset class names)
TARGET_CLASSES = {
    'bathtub': 65,
    'chair': 56,
    'couch': 57,
    'potted plant': 58,  # Sometimes used for shelves
    'dining table': 60,
    'toilet': 61,
    'tv': 62,  # Sometimes on shelves
}

# Simplified mapping for our challenge
CLASS_MAPPING = {
    65: 'bathtub',
    56: 'chair',
    57: 'couch',
    60: 'table',
    61: 'toilet',
}


def load_yolo_model(model_path: Path = None):
    """
    Load YOLOv8 model.
    
    Args:
        model_path: Path to model weights (default: models/yolov8x.pt)
    
    Returns:
        YOLO model
    """
    if model_path is None:
        model_path = Path(__file__).parent / "models" / "yolov8x.pt"
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🤖 Loading YOLO model from {model_path}...")
    
    if not model_path.exists():
        print("   Model not found. Downloading YOLOv8x...")
        model = YOLO('yolov8x.pt')  # This will download the model
        # Save to our models directory
        import shutil
        shutil.copy('yolov8x.pt', model_path)
    else:
        model = YOLO(str(model_path))
    
    print("   ✅ Model loaded")
    return model


def run_detection_on_image(model, image_path: Path, confidence_threshold: float = 0.3):
    """
    Run YOLO detection on a single image.
    
    Args:
        model: YOLO model
        image_path: Path to image
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        List of detections
    """
    # Run inference
    results = model(str(image_path), verbose=False)
    
    detections = []
    
    for result in results:
        boxes = result.boxes
        
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            confidence = float(boxes.conf[i])
            
            # Filter by class and confidence
            if cls_id in CLASS_MAPPING and confidence >= confidence_threshold:
                bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                
                detections.append({
                    'class': CLASS_MAPPING[cls_id],
                    'class_id': cls_id,
                    'confidence': confidence,
                    'bbox': {
                        'x1': float(bbox[0]),
                        'y1': float(bbox[1]),
                        'x2': float(bbox[2]),
                        'y2': float(bbox[3])
                    }
                })
    
    return detections


def process_survey(survey_name: str, model, workspace_dir: Path = None):
    """
    Process all images from a survey.
    
    Args:
        survey_name: Name of survey ('bathroom' or 'office')
        model: YOLO model
        workspace_dir: Workspace directory
    """
    if workspace_dir is None:
        workspace_dir = Path(__file__).parent.parent
    
    print(f"\n{'='*70}")
    print(f"Processing survey: {survey_name.upper()}")
    print(f"{'='*70}")
    
    # Load synchronized frames to get RGB image paths
    sync_file = workspace_dir / "synchronized_data" / f"{survey_name}_frames.pkl"

    if not sync_file.exists():
        print(f"\n❌ Synchronized data not found: {sync_file}")
        print("   Run: python utils/synchronize_data.py")
        return
    
    with open(sync_file, 'rb') as f:
        frames = pickle.load(f)
    
    print(f"\n📂 Loaded {len(frames)} frames")
    
    # Create output directory
    output_dir = workspace_dir / "challenge1_object_detection" / "detections" / survey_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each frame
    print(f"\n🔍 Running YOLO detection on {len(frames)} images...")

    total_detections = 0
    frames_with_detections = 0

    for frame in tqdm(frames, desc="Processing frames"):
        image_path = Path(frame.rgb_path)

        if not image_path.exists():
            continue

        # Run detection
        detections = run_detection_on_image(model, image_path)

        if detections:
            total_detections += len(detections)
            frames_with_detections += 1

        # Save detections for this frame
        output_file = output_dir / f"{frame.timestamp}.json"
        detection_data = {
            'timestamp': frame.timestamp,
            'image_path': str(image_path),
            'num_detections': len(detections),
            'detections': detections
        }

        with open(output_file, 'w') as f:
            json.dump(detection_data, f, indent=2)

    print(f"\n✅ Detection complete:")
    print(f"   Total frames processed: {len(frames)}")
    print(f"   Frames with detections: {frames_with_detections}")
    print(f"   Total detections: {total_detections}")
    print(f"   Average detections per frame: {total_detections / len(frames):.2f}")
    print(f"   Output directory: {output_dir}")


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        survey = sys.argv[1]
        if survey not in ['bathroom', 'office']:
            print("Usage: python run_yolo_detection.py [bathroom|office]")
            sys.exit(1)
        surveys = [survey]
    else:
        surveys = ['bathroom', 'office']

    # Load YOLO model once
    model = load_yolo_model()

    workspace_dir = Path(__file__).parent.parent

    # Process each survey
    for survey in surveys:
        process_survey(survey, model, workspace_dir)

    print(f"\n{'='*70}")
    print("✅ YOLO DETECTION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()


