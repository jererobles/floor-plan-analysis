#!/usr/bin/env python3
"""Test precise door detection method."""

import cv2
import numpy as np

from floor_plan_analyzer.preprocessing import preprocess_floor_plan
from floor_plan_analyzer.models import ProcessingParams
from floor_plan_analyzer.precise_door_detection import (
    detect_precise_door_openings,
    filter_door_candidates,
)


def main():
    """Test precise door detection."""
    print("="*70)
    print("PRECISE DOOR OPENING DETECTION")
    print("="*70)

    # Test on colored image
    image_path = "assets/floorplan-scanned-apartment-room-names.png"

    print("\nPreprocessing...")
    processed_img, bbox, _ = preprocess_floor_plan(
        image_path,
        crop_to_yellow=True,
        deskew=True,
        params=ProcessingParams(),
    )

    print("\nDetecting door openings...")

    # Try different parameter ranges
    configs = [
        (10, 40, "Narrow (10-40px)"),
        (15, 50, "Medium (15-50px)"),
        (20, 60, "Wide (20-60px)"),
    ]

    best_result = None
    best_score = 0

    for min_w, max_w, desc in configs:
        gaps = detect_precise_door_openings(processed_img, min_w, max_w)
        filtered = filter_door_candidates(gaps)

        if not filtered:
            print(f"\n{desc}: No doors detected")
            continue

        widths = [g.width_px for g in filtered]
        median_width = np.median(widths)
        std_dev = np.std(widths)
        consistency = 1.0 - min(1.0, std_dev / median_width)

        # Score: prefer 7-9 doors with high consistency
        count_score = 1.0 - abs(len(filtered) - 8) / 8.0
        count_score = max(0, count_score)
        score = count_score * 0.6 + consistency * 0.4

        print(f"\n{desc}:")
        print(f"  Detected: {len(filtered)} doors")
        print(f"  Median width: {median_width:.1f}px")
        print(f"  Std dev: {std_dev:.1f}px")
        print(f"  Consistency: {consistency:.1%}")
        print(f"  Score: {score:.3f}")

        if score > best_score:
            best_score = score
            best_result = (min_w, max_w, desc, filtered, median_width)

    if best_result is None:
        print("\n❌ No suitable doors detected!")
        return

    min_w, max_w, desc, doors, median_width = best_result

    print("\n" + "="*70)
    print(f"BEST: {desc}")
    print("="*70)
    print(f"Doors: {len(doors)}")
    print(f"Median width: {median_width:.1f}px")

    # List all doors
    print("\nDetected doors:")
    for i, door in enumerate(doors, 1):
        print(f"  {i:2d}. {door.width_px:3d}px @ ({door.x:4d},{door.y:4d}) "
              f"{door.orientation:10s} wall_thick={door.wall_thickness}px")

    # Calculate scales for different door size assumptions
    print("\n" + "="*70)
    print("SCALE CALIBRATION")
    print("="*70)

    door_sizes = [
        ("Standard (850mm)", 850),
        ("Small (800mm)", 800),
        ("Large (900mm)", 900),
        ("Bathroom (750mm)", 750),
    ]

    # Reference area
    ref_area_at_19_3mm = 785.23  # From earlier runs

    print(f"\nWith median door width = {median_width:.1f}px:\n")

    for name, door_mm in door_sizes:
        mm_per_px = door_mm / median_width
        area_ratio = (mm_per_px / 19.3182) ** 2
        estimated_area = ref_area_at_19_3mm * area_ratio

        marker = "  ✓✓✓" if 70 <= estimated_area <= 90 else ""
        print(f"  {name:20s}: {mm_per_px:6.3f} mm/px → {estimated_area:6.1f} m²{marker}")

    # Create visualization
    print("\nCreating visualization...")
    vis = processed_img.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for i, door in enumerate(doors, 1):
        color = (0, 255, 0)  # Green

        # Draw center
        cv2.circle(vis, (door.x, door.y), 7, color, -1)
        cv2.circle(vis, (door.x, door.y), 10, (255, 255, 255), 2)

        # Draw door opening box
        half_width = door.width_px // 2
        if door.orientation == "vertical":
            x1, y1 = door.x - half_width, door.y - 20
            x2, y2 = door.x + half_width, door.y + 20
        else:
            x1, y1 = door.x - 20, door.y - half_width
            x2, y2 = door.x + 20, door.y + half_width

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

        # Label
        label = f"D{i}: {door.width_px}px"
        cv2.putText(vis, label, (door.x + 15, door.y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Add summary
    cv2.putText(vis, f"{len(doors)} doors detected", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(vis, f"Median: {median_width:.1f}px", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    output_path = "outputs/precise_door_detection.png"
    cv2.imwrite(output_path, vis)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
