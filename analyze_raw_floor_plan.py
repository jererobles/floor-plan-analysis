#!/usr/bin/env python3
"""Analyze floor plan using raw image without color overlays."""

import cv2
import numpy as np
from pathlib import Path

from floor_plan_analyzer.door_detection import detect_doors, analyze_door_consistency
from floor_plan_analyzer.preprocessing import preprocess_floor_plan
from floor_plan_analyzer.models import ProcessingParams

def main():
    """Analyze the raw floor plan image."""
    # Use the image without room color labels
    image_path = "assets/floorplan-scanned-apartment.png"

    print("="*70)
    print("ANALYZING RAW FLOOR PLAN (without color overlays)")
    print("="*70)

    # Preprocess
    print("\nPreprocessing image...")
    processed_img, bbox, _ = preprocess_floor_plan(
        image_path,
        crop_to_yellow=True,
        deskew=True,
        params=ProcessingParams(),
    )

    # Save processed version
    cv2.imwrite("outputs/raw_processed.png", processed_img)
    print(f"Saved processed image to: outputs/raw_processed.png")

    # Detect doors with various parameter settings
    print("\n" + "="*70)
    print("TESTING DIFFERENT DETECTION PARAMETERS")
    print("="*70)

    test_configs = [
        (5, 25, "Very narrow features (5-25px)"),
        (10, 35, "Narrow features (10-35px)"),
        (15, 45, "Medium features (15-45px)"),
        (20, 60, "Wide features (20-60px)"),
    ]

    best_config = None
    best_score = 0

    for min_w, max_w, description in test_configs:
        print(f"\n--- {description} ---")
        doors = detect_doors(processed_img, min_width_px=min_w, max_width_px=max_w, debug=False)

        if not doors:
            print("  No doors detected")
            continue

        median_width, std_dev, widths = analyze_door_consistency(doors)
        consistency = 1.0 - min(1.0, std_dev / median_width) if median_width > 0 else 0

        # Score based on number of doors (expect 7-9) and consistency
        door_count_score = 1.0 - abs(len(doors) - 8) / 8.0
        door_count_score = max(0, door_count_score)

        score = door_count_score * 0.6 + consistency * 0.4

        print(f"  Doors: {len(doors)}")
        print(f"  Median width: {median_width:.1f}px")
        print(f"  Std dev: {std_dev:.1f}px")
        print(f"  Consistency: {consistency:.1%}")
        print(f"  Score: {score:.3f}")

        if score > best_score:
            best_score = score
            best_config = (min_w, max_w, description, doors, median_width)

    if best_config is None:
        print("\n❌ No suitable configuration found!")
        return

    min_w, max_w, description, doors, median_width = best_config

    print("\n" + "="*70)
    print(f"BEST CONFIGURATION: {description}")
    print("="*70)
    print(f"Parameters: min={min_w}px, max={max_w}px")
    print(f"Doors detected: {len(doors)}")
    print(f"Median width: {median_width:.1f}px")

    # Calculate what door width would give us the target area
    print("\n" + "="*70)
    print("REVERSE CALCULATION")
    print("="*70)
    print("Working backwards from expected area range...")

    # Current area with default 850mm door assumption would be X
    # We want area to be 70-90 m²
    # If median_width pixels should represent Y mm, what should Y be?

    target_areas = [70, 75, 80, 85, 90]
    current_area_at_850mm = 785.23  # From previous runs
    current_median_at_850mm = 44.0  # From previous runs with 850mm assumption

    print(f"\nCurrent state (from previous analysis with color overlay image):")
    print(f"  Assumed 850mm doors → {current_area_at_850mm:.1f} m² total area")
    print(f"  Detected median width: {current_median_at_850mm:.1f}px")

    print(f"\nIf we want target area, what door width assumption do we need?")
    print(f"Current detection: median width = {median_width:.1f}px\n")

    for target_area in target_areas:
        # Area scales with (scale)^2
        # target_area / current_area = (target_scale / current_scale)^2
        scale_ratio = np.sqrt(target_area / current_area_at_850mm)

        # current_scale = 850mm / current_median_px
        # target_scale = target_door_mm / current_median_px
        # scale_ratio = target_door_mm / 850mm

        required_door_mm = 850 * scale_ratio

        # With our current median width
        required_scale = required_door_mm / median_width

        print(f"  Target {target_area}m²: door={required_door_mm:.0f}mm, scale={required_scale:.3f}mm/px")

    # Create visualization
    print("\nCreating visualization...")
    vis = processed_img.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for i, door in enumerate(doors, 1):
        color = (0, 255, 0) if door.gap_detected else (0, 255, 255)
        cv2.circle(vis, (door.x, door.y), 5, color, -1)

        half_width = door.width_px // 2
        if door.orientation == "vertical":
            x1, y1 = door.x - half_width, door.y - 15
            x2, y2 = door.x + half_width, door.y + 15
        else:
            x1, y1 = door.x - 15, door.y - half_width
            x2, y2 = door.x + 15, door.y + half_width

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"D{i}:{door.width_px}px"
        cv2.putText(vis, label, (door.x + 10, door.y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite("outputs/raw_door_detection.png", vis)
    print(f"Saved to: outputs/raw_door_detection.png")

if __name__ == "__main__":
    main()
