#!/usr/bin/env python3
"""Validate scale calculation by comparing with known dimensions."""

import cv2
import numpy as np

from floor_plan_analyzer.preprocessing import preprocess_floor_plan


def main():
    """Validate scale against floor plan dimensions."""
    image_path = "assets/floorplan-scanned-apartment.png"

    print("Loading floor plan...")
    processed_img, bounding_box, skew_angle = preprocess_floor_plan(
        image_path,
        crop_to_yellow=True,
        deskew=True,
    )

    h, w = processed_img.shape[:2]
    print(f"Cropped image size: {w} x {h} pixels")
    print()

    # The plan shows dimensions like "9760" and "10340" which are likely in mm
    # Let's estimate what the scale should be for a typical apartment

    print("Expected apartment size analysis:")
    print("-" * 50)

    # User mentioned 70-90m² expected
    expected_areas = [70, 80, 90]

    for area_m2 in expected_areas:
        # Assume roughly square proportions (adjust ratio as needed)
        # For a rectangular space, estimate dimensions
        width_m = np.sqrt(area_m2 * (w / h))
        height_m = np.sqrt(area_m2 * (h / w))

        width_mm = width_m * 1000
        height_mm = height_m * 1000

        mm_per_pixel_w = width_mm / w
        mm_per_pixel_h = height_mm / h

        print(f"\nFor {area_m2}m² apartment:")
        print(f"  Estimated dimensions: {width_m:.1f}m × {height_m:.1f}m")
        print(f"  Required scale: ~{mm_per_pixel_w:.2f} mm/pixel (horizontal)")
        print(f"  Required scale: ~{mm_per_pixel_h:.2f} mm/pixel (vertical)")
        print(f"  Average scale: ~{(mm_per_pixel_w + mm_per_pixel_h)/2:.2f} mm/pixel")

        # What would door widths be at this scale?
        door_widths_mm = [700, 800, 900]
        print(f"  At this scale, door widths would be:")
        for door_mm in door_widths_mm:
            door_px = door_mm / mm_per_pixel_w
            print(f"    {door_mm}mm door → {door_px:.1f} pixels")

    print("\n" + "=" * 50)
    print("Current detection results:")
    print(f"  Detected door median: 33 pixels")
    print(f"  Current scale (assuming 825mm door): 25.0 mm/pixel")
    print(f"  This gives apartment dimensions: {w*25/1000:.1f}m × {h*25/1000:.1f}m")
    print(f"  Total area at this scale: {(w*h*25*25)/(1000*1000):.1f}m²")
    print()
    print("⚠️  This suggests we may be detecting circles that are NOT doors,")
    print("   or our assumption of 825mm door width may be incorrect.")


if __name__ == "__main__":
    main()
