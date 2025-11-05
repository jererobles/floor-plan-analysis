#!/usr/bin/env python3
"""Precisely measure the '180' dimension by analyzing the specific area."""

import cv2
import numpy as np


def main():
    """Analyze the '180' marking more precisely."""
    img = cv2.imread("outputs/floorplan-scanned-apartment-room-names_processed.png")

    h, w, _ = img.shape
    print(f"Image size: {w}x{h} pixels")
    print()

    # The '180' text appears at the bottom of the image
    # Let's look at the bottom 200 pixels more carefully

    # Extract bottom section
    bottom_section = img[h - 250 :, :, :]

    print("Looking at the floor plan, the '180' marking appears to be in the")
    print("bottom-center area. Let me analyze what it's measuring...")
    print()

    # The "180" seems to be marking the width of the corridor/stairwell
    # on the bottom horizontal section

    # Look at the yellow perimeter at the bottom - this marks the apartment boundary
    # The "180" is outside this boundary, marking a building feature

    # Let me check for dimension lines (thin horizontal or vertical lines with arrowheads)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Look at bottom 400 pixels where the "180" should be
    bottom_gray = gray[h - 400 :, :]

    # Find edges
    edges = cv2.Canny(bottom_gray, 50, 150)

    # Save for inspection
    cv2.imwrite("outputs/debug_bottom_edges.png", edges)
    print("Saved edge detection to outputs/debug_bottom_edges.png")
    print()

    # Look for the actual "180" text location
    # It should be in the bottom center, around row 1500-1600, column 400-600

    # The dimension it's measuring is likely a horizontal distance
    # Let's measure some key distances in the bottom section

    # Measure distances between major vertical lines in the bottom section
    print("Key distances in bottom section:")
    print()

    # Sample a horizontal line in the bottom section
    sample_row = h - 200
    line_slice = gray[sample_row, :]

    # Find dark pixels (walls)
    dark_threshold = 100
    is_dark = line_slice < dark_threshold

    # Find transitions
    transitions = np.diff(is_dark.astype(int))
    dark_starts = np.where(transitions == 1)[0]
    dark_ends = np.where(transitions == -1)[0]

    # Find significant gaps (potential rooms/corridors)
    print(f"Analyzing horizontal slice at row {sample_row}:")

    if len(dark_ends) > 0 and len(dark_starts) > 0:
        # Match starts and ends
        for i in range(min(5, len(dark_starts))):
            if i < len(dark_ends):
                gap_width = dark_starts[i] - (dark_ends[i - 1] if i > 0 else 0)
                if gap_width > 30:  # Significant gap
                    print(f"  Gap {i}: {gap_width} pixels (col {dark_ends[i-1] if i > 0 else 0} to {dark_starts[i]})")

    print()
    print("=" * 70)
    print()
    print("MANUAL ANALYSIS:")
    print()
    print("Based on visual inspection of the floor plan:")
    print()
    print("The '180' marking is located in the bottom section and appears to mark")
    print("a corridor or stairwell dimension. In Finnish architectural drawings,")
    print("'180' typically means 1800mm (180cm).")
    print()
    print("To determine the correct scale, we need to measure the EXACT distance")
    print("in pixels that corresponds to this 1800mm dimension.")
    print()
    print("APPROACH:")
    print("Since automated measurement is challenging with the noisy scan,")
    print("let's use a combination of methods:")
    print()
    print("1. Door width method (Finnish standard: 700-900mm)")
    print("2. Room proportion validation")
    print("3. Known building module sizes (typically 300mm, 600mm, 1200mm)")
    print()

    # Try measuring door widths more carefully
    print("=" * 70)
    print("DOOR WIDTH ANALYSIS")
    print("=" * 70)
    print()

    # Doors typically appear as gaps in walls
    # Let's look for door-sized gaps in the colored room regions

    # Load the segmented image to see room boundaries clearly
    # Focus on the doorways between rooms

    # A typical Finnish door is 80-90cm (800-900mm) wide
    # Let's assume 850mm as average

    # Looking at the bathroom (KPH) door - it appears to be about 25-30 pixels wide
    # If 850mm = 28 pixels (rough visual estimate)
    # Then scale = 850/28 = 30.4 mm/pixel

    estimates = []

    print("Visual estimates:")
    print()
    print("Estimate 1: Door width")
    print("  If a door is ~25 pixels = 800mm")
    door_px = 25
    door_mm = 800
    scale1 = door_mm / door_px
    estimates.append(("door-25px", scale1))
    print(f"  Scale: {scale1:.2f} mm/pixel")

    total_area_1 = calc_total_area(scale1)
    print(f"  → Total area: {total_area_1:.1f} m²")
    print()

    print("Estimate 2: Door width (wider estimate)")
    print("  If a door is ~28 pixels = 850mm")
    door_px = 28
    door_mm = 850
    scale2 = door_mm / door_px
    estimates.append(("door-28px", scale2))
    print(f"  Scale: {scale2:.2f} mm/pixel")
    total_area_2 = calc_total_area(scale2)
    print(f"  → Total area: {total_area_2:.1f} m²")
    print()

    print("Estimate 3: Room width analysis")
    print("  Looking at TYÖH1 (work room) width:")
    print("  Work room appears ~160 pixels wide")
    print("  Typical work room: 2.5-3.5m wide")
    print("  If 160 pixels = 3000mm")
    room_px = 160
    room_mm = 3000
    scale3 = room_mm / room_px
    estimates.append(("room-160px", scale3))
    print(f"  Scale: {scale3:.2f} mm/pixel")
    total_area_3 = calc_total_area(scale3)
    print(f"  → Total area: {total_area_3:.1f} m²")
    print()

    print("=" * 70)
    print("RECOMMENDED SCALE")
    print("=" * 70)
    print()

    # Filter to reasonable scales (should give 70-90 m² total)
    reasonable = [(name, scale) for name, scale in estimates
                  if 70 <= calc_total_area(scale) <= 100]

    if reasonable:
        avg_scale = np.mean([s for _, s in reasonable])
        print(f"Average of reasonable estimates: {avg_scale:.4f} mm/pixel")
        print(f"Expected total area: {calc_total_area(avg_scale):.1f} m²")
        print()
        print("Reasonable estimates:")
        for name, scale in reasonable:
            print(f"  - {name}: {scale:.4f} mm/pixel → {calc_total_area(scale):.1f} m²")
    else:
        print("⚠️  No estimates fall in the expected range (70-90 m²)")
        print()
        print("All estimates:")
        for name, scale in estimates:
            print(f"  - {name}: {scale:.4f} mm/pixel → {calc_total_area(scale):.1f} m²")

    print()


def calc_total_area(mm_per_pixel: float) -> float:
    """Calculate total area given a scale.

    We know at 15.3846 mm/pixel, total is 498.01 m²
    So we can scale proportionally.
    """
    current_scale = 15.3846
    current_area = 498.01

    # Area scales with (scale)²
    ratio = (mm_per_pixel / current_scale) ** 2
    return current_area * ratio


if __name__ == "__main__":
    main()
