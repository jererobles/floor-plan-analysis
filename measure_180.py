#!/usr/bin/env python3
"""Measure the '180' dimension to determine scale."""

import cv2
import numpy as np


def analyze_hallway_dimension(image_path: str) -> float:
    """Analyze the hallway dimension marked as '180'.

    By visual inspection of the floor plan:
    - The '180' text is located at the bottom of the apartment
    - It appears to mark a vertical dimension (height of a section)
    - Looking at the hallway (ET - green/yellow), the dimension seems to mark
      the distance between two horizontal wall lines

    Args:
        image_path: Path to the processed floor plan

    Returns:
        Estimated mm_per_pixel scale
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load: {image_path}")

    h, w, _ = img.shape

    print("🔍 Analyzing the '180' dimension marking...")
    print(f"   Image size: {w}x{h} pixels")
    print()

    # The "180" text is in the bottom portion of the image
    # Let's analyze the bottom 15% where the hallway (yellow/green) is

    # Extract the hallway color (green/yellow) to find its extent
    # Hallway color from our segmentation: (169, 210, 103) RGB = (103, 210, 169) BGR
    hallway_bgr = np.array([103, 210, 169])

    # Create mask for hallway
    lower = hallway_bgr - 30
    upper = hallway_bgr + 30
    hallway_mask = cv2.inRange(img, lower, upper)

    print("   Searching for hallway (ET) region...")

    # Find hallway contours
    contours, _ = cv2.findContours(hallway_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("   ❌ Could not find hallway region")
        return 0.0

    # Get the largest hallway region
    hallway_contour = max(contours, key=cv2.contourArea)
    x, y, hc_w, hc_h = cv2.boundingRect(hallway_contour)

    print(f"   Hallway bounding box: x={x}, y={y}, w={hc_w}, h={hc_h}")
    print()

    # The "180" marking appears to refer to a horizontal distance
    # Looking at the floor plan, it seems to mark the width of a corridor section
    # Let's analyze the left side of the hallway

    # Looking at the image, the "180" appears to be marking the vertical height
    # of the hallway in the bottom-left section

    # Measure the vertical extent of the hallway in its leftmost section
    # (where the "180" text is located)

    left_section = hallway_mask[:, x : x + int(hc_w * 0.3)]  # Left 30% of hallway

    # Find the vertical extent in the left section
    vertical_profile = np.sum(left_section, axis=1)
    hallway_rows = np.where(vertical_profile > 0)[0]

    if len(hallway_rows) > 0:
        min_row = hallway_rows[0]
        max_row = hallway_rows[-1]
        vertical_span = max_row - min_row

        print(f"   Hallway vertical span (left section): {vertical_span} pixels")
        print(f"   From row {min_row} to row {max_row}")
        print()

        # The "180" likely refers to 1800mm (standard corridor width)
        assumed_distance_mm = 1800.0

        mm_per_pixel = assumed_distance_mm / vertical_span

        print(f"   If this {vertical_span} pixels = {assumed_distance_mm}mm (180cm)")
        print(f"   Then scale = {mm_per_pixel:.4f} mm/pixel")
        print()

        # Validate this makes sense
        # Total image width should be a reasonable apartment width
        total_width_mm = w * mm_per_pixel
        total_height_mm = h * mm_per_pixel

        print(f"   Validation:")
        print(f"   - Total apartment width: {total_width_mm/1000:.2f} m")
        print(f"   - Total apartment height: {total_height_mm/1000:.2f} m")
        print()

        # Estimate total area
        # We know the pixel area sum is ~2,111,600 pixels (from 498m² at 15.38mm/px)
        current_scale = 15.3846
        current_total_px = 498.0 / ((current_scale / 1000) ** 2)

        new_total_area = current_total_px * ((mm_per_pixel / 1000) ** 2)

        print(f"   Estimated total area: {new_total_area:.2f} m²")
        print()

        # Check if this is reasonable (70-90 m² expected)
        if 60 < new_total_area < 120:
            print("   ✅ This scale seems reasonable!")
        else:
            print(f"   ⚠️  This scale gives {new_total_area:.0f}m², which may be outside expected range")

        return mm_per_pixel

    return 0.0


def try_wall_thickness_method(image_path: str) -> float:
    """Measure wall thickness as a reference.

    Finnish walls are typically 150-200mm thick.

    Args:
        image_path: Path to the image

    Returns:
        Estimated mm_per_pixel scale
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("🧱 Analyzing wall thickness...")
    print()

    # Detect walls (dark lines)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Clean up
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find vertical lines (walls)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Find horizontal lines (walls)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Measure thickness of a few walls
    # Sample from middle section to avoid edges
    h, w = gray.shape
    mid_h = h // 2
    mid_w = w // 2

    # Look at a horizontal slice to measure vertical wall thickness
    slice_h = vertical_lines[mid_h, :]

    # Find transitions (edges of walls)
    transitions = np.diff(slice_h)
    wall_starts = np.where(transitions > 200)[0]
    wall_ends = np.where(transitions < -200)[0]

    if len(wall_starts) > 0 and len(wall_ends) > 0:
        thicknesses = []
        for start in wall_starts[:5]:  # Check first 5 walls
            # Find corresponding end
            matching_ends = wall_ends[wall_ends > start]
            if len(matching_ends) > 0:
                thickness = matching_ends[0] - start
                if 3 < thickness < 30:  # Reasonable wall thickness in pixels
                    thicknesses.append(thickness)

        if thicknesses:
            avg_thickness = np.mean(thicknesses)
            print(f"   Average wall thickness: {avg_thickness:.1f} pixels")

            # Assume 150mm wall thickness (conservative estimate)
            assumed_thickness_mm = 150.0

            mm_per_pixel = assumed_thickness_mm / avg_thickness

            print(f"   If {avg_thickness:.1f} pixels = {assumed_thickness_mm}mm")
            print(f"   Then scale = {mm_per_pixel:.4f} mm/pixel")
            print()

            return mm_per_pixel

    print("   ⚠️  Could not reliably measure wall thickness")
    return 0.0


def main():
    """Main function."""
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/floorplan-scanned-apartment-room-names_processed.png"

    print("\n" + "=" * 70)
    print("  MEASURING '180' DIMENSION FOR SCALE CALIBRATION")
    print("=" * 70)
    print()

    # Method 1: Analyze hallway dimension
    scale_hallway = analyze_hallway_dimension(image_path)

    print("\n" + "-" * 70)
    print()

    # Method 2: Analyze wall thickness
    scale_wall = try_wall_thickness_method(image_path)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()

    if scale_hallway > 0:
        print(f"✓ Hallway method: {scale_hallway:.4f} mm/pixel")

    if scale_wall > 0:
        print(f"✓ Wall thickness method: {scale_wall:.4f} mm/pixel")

    if scale_hallway > 0 and scale_wall > 0:
        avg_scale = (scale_hallway + scale_wall) / 2
        print()
        print(f"📊 Average: {avg_scale:.4f} mm/pixel")

    print()


if __name__ == "__main__":
    main()
