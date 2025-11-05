"""Debug door detection."""

import cv2
import numpy as np
from floor_plan_analyzer.preprocessing import preprocess_floor_plan
from floor_plan_analyzer.wall_extraction import (
    extract_walls_morphological,
    detect_doors_from_walls,
    visualize_doors,
)
from floor_plan_analyzer.scale_estimation import estimate_scale_from_doors

# Load and preprocess image
image_path = "assets/floorplan-scanned-apartment-room-names.png"
print(f"Loading: {image_path}")

processed_img, bbox, skew = preprocess_floor_plan(
    image_path,
    crop_to_yellow=True,
    deskew=True
)

print(f"Image shape: {processed_img.shape}")

# Extract walls
print("\nExtracting walls...")
wall_mask = extract_walls_morphological(processed_img)
print(f"Wall pixels: {np.sum(wall_mask > 0)}")

# Save wall mask for inspection
cv2.imwrite("outputs/debug_wall_mask.png", wall_mask)
print("Saved: outputs/debug_wall_mask.png")

# Detect doors
print("\nDetecting doors...")
doors = detect_doors_from_walls(
    wall_mask,
    min_gap_width=15,
    max_gap_width=100,
    min_wall_length=80
)

print(f"Found {len(doors)} doors:")
for i, (x, y, w, h, orientation, gap_width) in enumerate(doors):
    print(f"  Door {i+1}: {orientation}, gap={gap_width:.1f}px at ({x},{y})")

# Try with different parameters
print("\n\nTrying with relaxed parameters...")
doors2 = detect_doors_from_walls(
    wall_mask,
    min_gap_width=10,
    max_gap_width=150,
    min_wall_length=50
)
print(f"Found {len(doors2)} doors with relaxed parameters")

# Visualize
vis = visualize_doors(processed_img, doors2, wall_mask)
cv2.imwrite("outputs/debug_doors_relaxed.png", vis)
print("Saved: outputs/debug_doors_relaxed.png")

# Try scale estimation
print("\n\nTrying scale estimation from doors...")
scale_info = estimate_scale_from_doors(processed_img)

if scale_info:
    print(f"Scale: {scale_info.mm_per_pixel:.4f} mm/pixel")
    print(f"Confidence: {scale_info.confidence:.2f}")
    if 'door_count' in scale_info.metadata:
        print(f"Door count: {scale_info.metadata['door_count']}")
        print(f"Most common width: {scale_info.metadata['most_common_width_px']:.1f} px")
        print(f"Clusters: {scale_info.metadata.get('all_clusters', [])}")
else:
    print("Scale estimation failed - no doors detected")
