#!/usr/bin/env python3
"""Debug script to visualize door detection."""

import cv2
import numpy as np
from pathlib import Path

from floor_plan_analyzer.door_detection import detect_doors, analyze_door_consistency
from floor_plan_analyzer.preprocessing import preprocess_floor_plan
from floor_plan_analyzer.models import ProcessingParams

def main():
    """Run door detection with debug visualization."""
    image_path = "assets/floorplan-scanned-apartment-room-names.png"

    # Preprocess image
    print("Preprocessing image...")
    processed_img, bbox, _ = preprocess_floor_plan(
        image_path,
        crop_to_yellow=True,
        deskew=True,
        params=ProcessingParams(),
    )

    # Detect doors with debug visualization
    print("\nDetecting doors...")
    # Try different ranges to find the right features
    print("\n--- Testing with range 10-50px (actual door openings) ---")
    doors = detect_doors(processed_img, min_width_px=10, max_width_px=50, debug=False)

    print(f"\n{'='*60}")
    print(f"DOOR DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"Total doors detected: {len(doors)}")
    print()

    # Analyze consistency
    median_width, std_dev, widths = analyze_door_consistency(doors)
    print(f"Width statistics:")
    print(f"  Median: {median_width:.1f} px")
    print(f"  Std dev: {std_dev:.1f} px")
    print(f"  Range: {min(widths):.0f} - {max(widths):.0f} px")
    print(f"  Coefficient of variation: {(std_dev/median_width)*100:.1f}%")
    print()

    # List individual doors
    print(f"Individual doors:")
    for i, door in enumerate(doors, 1):
        features = []
        if door.arc_detected:
            features.append("arc")
        if door.gap_detected:
            features.append("gap")
        feature_str = "+".join(features)

        print(f"  Door {i:2d}: {door.width_px:3d}px @ ({door.x:4d}, {door.y:4d}) "
              f"{door.orientation:10s} [{feature_str:8s}] conf={door.confidence:.2f}")
    print()

    # Create visualization
    print("Creating visualization...")
    vis = processed_img.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # Draw each door
    for i, door in enumerate(doors, 1):
        # Color based on confidence
        if door.confidence > 0.8:
            color = (0, 255, 0)  # Green
        elif door.confidence > 0.5:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red

        # Draw center point
        cv2.circle(vis, (door.x, door.y), 5, color, -1)

        # Draw bounding box for door opening
        half_width = door.width_px // 2
        if door.orientation == "vertical":
            x1, y1 = door.x - half_width, door.y - 15
            x2, y2 = door.x + half_width, door.y + 15
        else:
            x1, y1 = door.x - 15, door.y - half_width
            x2, y2 = door.x + 15, door.y + half_width

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Add label with door number and width
        label = f"D{i}: {door.width_px}px"
        cv2.putText(
            vis,
            label,
            (door.x + 15, door.y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    # Add summary text
    summary_y = 30
    cv2.putText(
        vis,
        f"Detected: {len(doors)} doors",
        (10, summary_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        vis,
        f"Median width: {median_width:.1f}px",
        (10, summary_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Save visualization
    output_path = "outputs/door_detection_debug.png"
    cv2.imwrite(output_path, vis)
    print(f"Saved visualization to: {output_path}")

    # Calculate different scale scenarios
    print(f"\n{'='*60}")
    print("SCALE CALIBRATION SCENARIOS")
    print(f"{'='*60}")

    scenarios = [
        ("Standard door (850mm)", 850),
        ("Small door (800mm)", 800),
        ("Large door (900mm)", 900),
        ("Bathroom door (750mm)", 750),
        ("Very small door (700mm)", 700),
        ("Extra small (600mm)", 600),
        ("Tiny (500mm)", 500),
        ("Very tiny (400mm)", 400),
        ("Minimal (300mm)", 300),
    ]

    # Get a reference area (OH room is largest, should be ~23% of total)
    # If total is 80m², OH should be ~18-19m²

    for name, door_mm in scenarios:
        mm_per_px = door_mm / median_width
        # Calculate what total area would be
        # Current area is 785m² with current scale
        # Area scales with (mm_per_px)^2
        current_mm_per_px = 19.3182
        area_ratio = (mm_per_px / current_mm_per_px) ** 2
        estimated_total = 785.23 * area_ratio

        marker = "  ✓✓✓ TARGET RANGE!" if 70 <= estimated_total <= 90 else ""
        print(f"{name:25s}: {mm_per_px:6.3f} mm/px → {estimated_total:6.1f} m²{marker}")

if __name__ == "__main__":
    main()
