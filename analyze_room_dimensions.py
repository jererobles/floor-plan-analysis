#!/usr/bin/env python3
"""Analyze room dimensions using Finnish building standards."""

import cv2
import numpy as np


def analyze_bathroom_dimensions():
    """Analyze bathroom dimensions - these are quite standardized.

    Finnish bathroom standards:
    - Minimum bathroom: 2.0 m² (very small)
    - Typical bathroom: 3.5-5.5 m²
    - Toilet: 1.0-2.0 m²

    KPH shown in our results: 29.54 m² (way too large!)
    WC shown in our results: 18.91 m² (way too large!)

    Expected:
    - KPH: ~4-5 m²
    - WC: ~1.5 m²
    """
    img = cv2.imread("outputs/floorplan-scanned-apartment-room-names_processed.png")

    # Load color image to find bathroom
    # KPH color: (76, 135, 251) RGB = (251, 135, 76) BGR
    kph_bgr = np.array([251, 135, 76])

    # Create mask
    lower = kph_bgr - 30
    upper = kph_bgr + 30
    kph_mask = cv2.inRange(img, lower, upper)

    # Find contours
    contours, _ = cv2.findContours(kph_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        kph_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(kph_contour)
        pixel_area = cv2.contourArea(kph_contour)

        print("🚿 BATHROOM (KPH) ANALYSIS")
        print("=" * 70)
        print()
        print(f"Bounding box: {w}x{h} pixels")
        print(f"Pixel area: {pixel_area:.0f} pixels")
        print()

        print("Finnish bathroom standards:")
        print("  - Typical: 3.5-5.5 m²")
        print("  - Let's assume 4.5 m² (middle estimate)")
        print()

        # If bathroom is 4.5 m², what's the scale?
        expected_area_m2 = 4.5
        mm_per_pixel = np.sqrt((expected_area_m2 * 1_000_000) / pixel_area)

        print(f"If KPH area = 4.5 m² and {pixel_area:.0f} pixels")
        print(f"Then scale = {mm_per_pixel:.4f} mm/pixel")
        print()

        total_area = calc_total_area(mm_per_pixel)
        print(f"→ Total apartment area: {total_area:.1f} m²")
        print()

        return mm_per_pixel

    return None


def analyze_toilet_dimensions():
    """Analyze toilet (WC) dimensions."""
    img = cv2.imread("outputs/floorplan-scanned-apartment-room-names_processed.png")

    # WC color: (143, 253, 255) RGB = (255, 253, 143) BGR
    wc_bgr = np.array([255, 253, 143])

    # Create mask
    lower = wc_bgr - 30
    upper = wc_bgr + 30
    wc_mask = cv2.inRange(img, lower, upper)

    # Find contours
    contours, _ = cv2.findContours(wc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        wc_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(wc_contour)
        pixel_area = cv2.contourArea(wc_contour)

        print("🚽 TOILET (WC) ANALYSIS")
        print("=" * 70)
        print()
        print(f"Bounding box: {w}x{h} pixels")
        print(f"Pixel area: {pixel_area:.0f} pixels")
        print()

        print("Finnish toilet standards:")
        print("  - Typical: 1.0-2.0 m²")
        print("  - Let's assume 1.5 m²")
        print()

        expected_area_m2 = 1.5
        mm_per_pixel = np.sqrt((expected_area_m2 * 1_000_000) / pixel_area)

        print(f"If WC area = 1.5 m² and {pixel_area:.0f} pixels")
        print(f"Then scale = {mm_per_pixel:.4f} mm/pixel")
        print()

        total_area = calc_total_area(mm_per_pixel)
        print(f"→ Total apartment area: {total_area:.1f} m²")
        print()

        return mm_per_pixel

    return None


def analyze_kitchen_dimensions():
    """Analyze kitchen dimensions.

    Finnish building code (Suomen rakentamismääräyskokoelma):
    - Minimum kitchen in apartment: 6 m²
    - Typical: 8-12 m²
    """
    img = cv2.imread("outputs/floorplan-scanned-apartment-room-names_processed.png")

    # K color: (220, 43, 3) RGB = (3, 43, 220) BGR
    k_bgr = np.array([3, 43, 220])

    # Create mask
    lower = k_bgr - 30
    upper = k_bgr + 30
    k_mask = cv2.inRange(img, lower, upper)

    # Find contours
    contours, _ = cv2.findContours(k_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        k_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(k_contour)
        pixel_area = cv2.contourArea(k_contour)

        print("🍳 KITCHEN (K) ANALYSIS")
        print("=" * 70)
        print()
        print(f"Bounding box: {w}x{h} pixels")
        print(f"Pixel area: {pixel_area:.0f} pixels")
        print()

        print("Finnish kitchen standards:")
        print("  - Typical: 8-12 m²")
        print("  - Let's assume 10 m²")
        print()

        expected_area_m2 = 10.0
        mm_per_pixel = np.sqrt((expected_area_m2 * 1_000_000) / pixel_area)

        print(f"If K area = 10 m² and {pixel_area:.0f} pixels")
        print(f"Then scale = {mm_per_pixel:.4f} mm/pixel")
        print()

        total_area = calc_total_area(mm_per_pixel)
        print(f"→ Total apartment area: {total_area:.1f} m²")
        print()

        return mm_per_pixel

    return None


def calc_total_area(mm_per_pixel: float) -> float:
    """Calculate total area given a scale."""
    current_scale = 15.3846
    current_area = 498.01
    ratio = (mm_per_pixel / current_scale) ** 2
    return current_area * ratio


def main():
    """Main function."""
    print()
    print("=" * 70)
    print("  SCALE ESTIMATION USING FINNISH BUILDING STANDARDS")
    print("=" * 70)
    print()

    scales = []

    # Bathroom
    scale_kph = analyze_bathroom_dimensions()
    if scale_kph:
        scales.append(("Bathroom (4.5m²)", scale_kph))

    # Toilet
    scale_wc = analyze_toilet_dimensions()
    if scale_wc:
        scales.append(("Toilet (1.5m²)", scale_wc))

    # Kitchen
    scale_k = analyze_kitchen_dimensions()
    if scale_k:
        scales.append(("Kitchen (10m²)", scale_k))

    print()
    print("=" * 70)
    print("  RECOMMENDED SCALE")
    print("=" * 70)
    print()

    if scales:
        # Average of all methods
        avg_scale = np.mean([s for _, s in scales])
        total = calc_total_area(avg_scale)

        print(f"Average scale: {avg_scale:.4f} mm/pixel")
        print(f"Expected total area: {total:.1f} m²")
        print()

        print("Individual estimates:")
        for name, scale in scales:
            area = calc_total_area(scale)
            print(f"  • {name:20s}: {scale:.4f} mm/px → {area:.1f} m² total")

        print()

        # Check reasonableness
        if 60 < total < 120:
            print("✅ This scale produces a reasonable total area for an apartment!")
        else:
            print(f"⚠️  Total of {total:.0f}m² may be outside typical range")

        print()
        print(f"To use this scale, update scale_estimation.py with: {avg_scale:.4f} mm/pixel")

    print()


if __name__ == "__main__":
    main()
