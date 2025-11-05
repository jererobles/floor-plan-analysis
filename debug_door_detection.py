#!/usr/bin/env python3
"""Debug script for door detection."""

import cv2
import numpy as np
from pathlib import Path

from floor_plan_analyzer.preprocessing import preprocess_floor_plan
from floor_plan_analyzer.scale_estimation import (
    detect_wall_gaps,
    detect_arcs,
    detect_door_openings,
    visualize_detected_doors,
)


def main():
    """Debug door detection."""
    # Try the raw floor plan without room colors
    image_path = "assets/floorplan-scanned-apartment.png"

    print(f"Loading and preprocessing image: {image_path}")
    processed_img, bounding_box, skew_angle = preprocess_floor_plan(
        image_path,
        crop_to_yellow=True,
        deskew=True,
    )

    print(f"Image shape: {processed_img.shape}")
    print(f"Bounding box: {bounding_box}")

    # Convert to grayscale for detection
    if len(processed_img.shape) == 3:
        gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = processed_img.copy()

    # Save grayscale
    cv2.imwrite("outputs/debug_gray.png", gray)
    print("Saved: outputs/debug_gray.png")

    # Detect wall gaps
    print("\nDetecting wall gaps...")
    gaps = detect_wall_gaps(gray, min_gap_px=10, max_gap_px=120)
    print(f"Found {len(gaps)} wall gaps")

    # Visualize gaps
    gap_vis = processed_img.copy() if len(processed_img.shape) == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y, w, h, orientation, gap_size in gaps:
        cv2.rectangle(gap_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
        label = f"{gap_size:.0f}px"
        cv2.putText(gap_vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    cv2.imwrite("outputs/debug_gaps.png", gap_vis)
    print("Saved: outputs/debug_gaps.png")

    # Detect arcs
    print("\nDetecting arcs...")
    arc_mask = detect_arcs(gray)
    cv2.imwrite("outputs/debug_arcs.png", arc_mask)
    print("Saved: outputs/debug_arcs.png")

    # Detect doors
    print("\nDetecting door openings...")
    doors = detect_door_openings(gray, min_width_px=10, max_width_px=120)
    print(f"Found {len(doors)} door candidates")

    if doors:
        door_widths = []
        for x, y, w, h, orientation in doors:
            width = w if orientation == "vertical" else h
            door_widths.append(float(width))
            print(f"  Door at ({x}, {y}), size={width:.1f}px, orientation={orientation}")

        # Visualize doors
        door_vis = visualize_detected_doors(processed_img, doors, door_widths)
        cv2.imwrite("outputs/debug_doors.png", door_vis)
        print("Saved: outputs/debug_doors.png")
    else:
        print("No doors detected!")


if __name__ == "__main__":
    main()
