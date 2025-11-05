#!/usr/bin/env python3
"""Improved door detection using arc-based approach."""

import cv2
import numpy as np
from pathlib import Path

from floor_plan_analyzer.preprocessing import preprocess_floor_plan


def detect_door_arcs_advanced(image):
    """Detect door swing arcs using multiple methods."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # Detect edges with Canny
    edges = cv2.Canny(blurred, 30, 100)

    # Try to detect circles/arcs using Hough Circle Transform with various parameters
    door_candidates = []

    # Try different parameter sets to catch different door types
    param_sets = [
        # (dp, minDist, param1, param2, minRadius, maxRadius)
        (1, 30, 50, 30, 15, 50),
        (1, 25, 40, 25, 20, 60),
        (1, 30, 50, 35, 25, 70),
    ]

    all_circles = []
    for dp, minDist, param1, param2, minRadius, maxRadius in param_sets:
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius,
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                # Check if this is a new circle (not too close to existing ones)
                is_new = True
                for existing_x, existing_y, existing_r in all_circles:
                    dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                    if dist < 30:  # Too close to existing circle
                        is_new = False
                        break
                if is_new:
                    all_circles.append((x, y, r))

    return all_circles


def measure_door_widths_from_arcs(image, arcs):
    """Measure the door width from detected arcs."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    door_widths = []

    for x, y, r in arcs:
        # The radius of the arc should correspond to the door width
        # Most door arcs are quarter circles, so the width is approximately the radius
        # But let's validate by checking the local structure

        # Extract region around the arc
        margin = 20
        x1 = max(0, x - r - margin)
        y1 = max(0, y - r - margin)
        x2 = min(gray.shape[1], x + r + margin)
        y2 = min(gray.shape[0], y + r + margin)

        if x2 - x1 < 10 or y2 - y1 < 10:
            continue

        region = gray[y1:y2, x1:x2]

        # The door width is approximately 2*r (diameter) for most quarter-circle arcs
        # But in architectural drawings, it's typically the radius that matches the opening
        door_width_px = float(r)
        door_widths.append((x, y, r, door_width_px))

    return door_widths


def main():
    """Debug door detection with arc-based approach."""
    image_path = "assets/floorplan-scanned-apartment.png"

    print(f"Loading and preprocessing image: {image_path}")
    processed_img, bounding_box, skew_angle = preprocess_floor_plan(
        image_path,
        crop_to_yellow=True,
        deskew=True,
    )

    print(f"Image shape: {processed_img.shape}")

    # Detect door arcs
    print("\nDetecting door arcs...")
    arcs = detect_door_arcs_advanced(processed_img)
    print(f"Found {len(arcs)} door arc candidates")

    # Measure door widths
    door_widths = measure_door_widths_from_arcs(processed_img, arcs)
    print(f"\nDoor width measurements:")

    # Collect just the widths for analysis
    widths_only = [w for _, _, _, w in door_widths]

    if widths_only:
        widths_array = np.array(widths_only)
        print(f"  Found {len(widths_only)} doors")
        print(f"  Min: {np.min(widths_array):.1f}px")
        print(f"  Max: {np.max(widths_array):.1f}px")
        print(f"  Mean: {np.mean(widths_array):.1f}px")
        print(f"  Median: {np.median(widths_array):.1f}px")
        print(f"  Std: {np.std(widths_array):.1f}px")

        # Filter outliers
        q1 = np.percentile(widths_array, 25)
        q3 = np.percentile(widths_array, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        filtered = widths_array[(widths_array >= lower) & (widths_array <= upper)]
        print(f"\n  After outlier removal: {len(filtered)} doors")
        print(f"  Median: {np.median(filtered):.1f}px")
        print(f"  Mean: {np.mean(filtered):.1f}px ± {np.std(filtered):.1f}px")
        print(f"  CV: {(np.std(filtered) / np.mean(filtered) * 100):.1f}%")

        # Calculate scale (assuming 825mm standard Finnish door)
        median_width_px = np.median(filtered)
        mm_per_pixel = 825.0 / median_width_px
        print(f"\n  Calculated scale: {mm_per_pixel:.4f} mm/pixel")
        print(f"  This gives approximately {mm_per_pixel * np.sqrt(processed_img.shape[0] * processed_img.shape[1]) / 1000:.0f}m perimeter")

        # Visualize
        vis = processed_img.copy() if len(processed_img.shape) == 3 else cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)

        for x, y, r, w in door_widths:
            # Draw the arc
            cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
            cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)

            # Label
            label = f"{w:.0f}px"
            cv2.putText(vis, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imwrite("outputs/debug_door_arcs.png", vis)
        print("\nSaved: outputs/debug_door_arcs.png")


if __name__ == "__main__":
    main()
