"""Analyze detected door widths to calibrate scale."""

import numpy as np
from floor_plan_analyzer.preprocessing import preprocess_floor_plan
from floor_plan_analyzer.improved_door_detection import detect_doors_simple
from floor_plan_analyzer.wall_extraction import cluster_door_widths

# Load image
image_path = "assets/floorplan-scanned-apartment-room-names.png"
processed_img, _, _ = preprocess_floor_plan(image_path, crop_to_yellow=True, deskew=True)

# Detect doors
doors = detect_doors_simple(processed_img)

print(f"Total doors detected: {len(doors)}")
print(f"\nDoor details:")

# Extract widths
widths = []
for i, (x, y, w, h, orient, gap_w) in enumerate(doors):
    print(f"  Door {i+1}: {orient:10s} at ({x:4d},{y:4d}) size=({w:3d}x{h:3d}) gap={gap_w:.1f}px")
    widths.append(gap_w)

# Cluster widths
clusters = cluster_door_widths(widths, tolerance=5.0)

print(f"\n\nDoor width clusters:")
for i, (center, count) in enumerate(clusters):
    print(f"  Cluster {i+1}: {center:.1f} px ({count} doors, {count/len(doors)*100:.1f}%)")

# Try different calibration assumptions
print(f"\n\nCalibration scenarios:")
print(f"{'Door Width (px)':<20} {'Assumed (mm)':<15} {'Scale (mm/px)':<15} {'Total Area (m²)':<15}")
print("-" * 70)

# Known pixel areas from analysis
room_pixel_areas = {
    'OH': 115.48,
    'TYÖH': 79.21,
    'K': 72.55,
    'MH': 69.03,
    'ET': 62.92,
    'LASPI': 50.36,
    'KPH': 29.54,
    'WC': 18.91,
}
total_pixel_area = sum(room_pixel_areas.values())

# Current scale assumption (most common door width = 800mm)
most_common_width = clusters[0][0]
scale_current = 800 / most_common_width
area_current = total_pixel_area * (scale_current / 1000) ** 2
print(f"{most_common_width:<20.1f} {800:<15.0f} {scale_current:<15.2f} {area_current:<15.1f}")

# Try assuming largest cluster is 900mm (larger interior doors)
scale_900 = 900 / most_common_width
area_900 = total_pixel_area * (scale_900 / 1000) ** 2
print(f"{most_common_width:<20.1f} {900:<15.0f} {scale_900:<15.2f} {area_900:<15.1f}")

# Try assuming second cluster is 800mm doors
if len(clusters) > 1:
    second_width = clusters[1][0]
    scale_second = 800 / second_width
    area_second = total_pixel_area * (scale_second / 1000) ** 2
    print(f"{second_width:<20.1f} {800:<15.0f} {scale_second:<15.2f} {area_second:<15.1f}")

# Work backwards: what door width gives us 80 m²?
target_area = 80  # Middle of 70-90 m² range
target_scale = np.sqrt(target_area / total_pixel_area) * 1000
target_door_px = 800 / target_scale
print(f"\nTo achieve {target_area} m² total:")
print(f"  Need scale: {target_scale:.2f} mm/px")
print(f"  800mm doors would be: {target_door_px:.1f} px")

# Find which cluster is closest
print(f"\n  Closest cluster to {target_door_px:.1f} px:")
closest_cluster = min(clusters, key=lambda c: abs(c[0] - target_door_px))
print(f"    {closest_cluster[0]:.1f} px ({closest_cluster[1]} doors)")
