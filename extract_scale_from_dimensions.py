#!/usr/bin/env python3
"""Extract scale by measuring known dimensions on the floor plan."""

import cv2
import numpy as np
from pathlib import Path


def find_180_dimension(image_path: str) -> tuple[float, str]:
    """Find and measure the '180' dimension marking.

    The '180' marking is visible at the bottom of the apartment unit,
    likely indicating 1800mm (180cm).

    Args:
        image_path: Path to the processed floor plan image

    Returns:
        Tuple of (mm_per_pixel, description)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = img.shape[:2]

    print(f"📐 Image dimensions: {w}x{h} pixels")
    print()

    # The "180" text is visible at the bottom of the image
    # It appears to be marking a horizontal distance
    # Let's look at the bottom area of the image

    # Method 1: Measure the corridor width at the bottom
    # Looking at the image, the "180" seems to mark the width of the corridor/hallway
    # between the ventilation shaft wall and the apartment wall

    print("🔍 Analysis of '180' marking:")
    print("  Location: Bottom center of the apartment unit")
    print("  Likely meaning: 1800mm (180cm) - standard corridor/hallway width")
    print()

    # The hallway (ET - yellow/green color) extends across the bottom
    # Let's measure its approximate width

    # Looking at the processed image, the hallway section appears to be
    # roughly in the bottom 20% of the image
    # The "180" seems to mark the vertical height of a section

    # Manual measurement approach:
    # Looking at the yellow corridor section at the bottom left where "180" is marked
    # This appears to span from approximately y=850 to y=950 (about 100 pixels)

    # But let me be more systematic - let's measure wall-to-wall distances
    # in the area where "180" is marked

    # For now, let's use a manual measurement based on visual inspection
    # The corridor height where "180" is marked appears to be approximately 100-110 pixels

    # If 180cm = ~105 pixels (estimated from visual inspection)
    # Then 1800mm / 105px = 17.14 mm/pixel

    # Let's verify this makes sense:
    # Total apartment width appears to be ~1440 pixels
    # At 17.14 mm/pixel: 1440 * 17.14 = 24,682mm = 24.7m (too wide!)

    # Let me reconsider - maybe the scale is different
    # Looking more carefully at typical apartment dimensions:
    # - Living room (OH) should be around 15-25 m²
    # - Kitchen (K) should be around 8-15 m²
    # - Bedroom (MH) should be around 10-15 m²

    # Current results show OH = 115 m², which is about 5-7x too large
    # So our scale is off by a factor of 5-7
    # Current: 15.38 mm/pixel
    # Should be: ~2.5-3.0 mm/pixel

    # If 180 measurement is 1800mm, and current scale is 15.38 mm/pixel
    # That means 180cm should be: 1800 / 15.38 = 117 pixels

    # Let me try a different approach: measure actual features
    print("📊 Let me measure some features manually...")
    print()

    # Strategy: Find prominent features we can measure
    # 1. Find wall thickness (should be ~150-200mm)
    # 2. Find door width (should be 700-900mm)
    # 3. Measure the distance marked as "180"

    return 0.0, "Manual measurement needed"


def interactive_measurement(image_path: str) -> None:
    """Interactive tool to measure distances on the image.

    Shows the image with coordinates for manual measurement.

    Args:
        image_path: Path to the image
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return

    h, w = img.shape[:2]

    print("\n🖱️  INTERACTIVE MEASUREMENT TOOL")
    print("=" * 60)
    print("Move your mouse over the image to see coordinates.")
    print("Click two points to measure distance.")
    print("Press 'q' to quit, 'r' to reset measurement.")
    print("=" * 60)
    print()

    # State for measurement
    points = []
    measuring = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal points, measuring

        if event == cv2.EVENT_MOUSEMOVE:
            # Show coordinates
            temp_img = img.copy()

            # Draw crosshair
            cv2.line(temp_img, (x, 0), (x, h), (0, 255, 0), 1)
            cv2.line(temp_img, (0, y), (w, y), (0, 255, 0), 1)

            # Show coordinates
            cv2.putText(
                temp_img,
                f"({x}, {y})",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Draw existing points
            for i, pt in enumerate(points):
                cv2.circle(temp_img, pt, 5, (255, 0, 0), -1)
                cv2.putText(
                    temp_img,
                    f"P{i+1}",
                    (pt[0] + 10, pt[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

            # Draw line if measuring
            if len(points) == 1:
                cv2.line(temp_img, points[0], (x, y), (255, 0, 0), 2)

            # Draw measurement if complete
            if len(points) == 2:
                cv2.line(temp_img, points[0], points[1], (0, 0, 255), 2)
                dist = np.sqrt((points[1][0] - points[0][0]) ** 2 + (points[1][1] - points[0][1]) ** 2)
                mid_x = (points[0][0] + points[1][0]) // 2
                mid_y = (points[0][1] + points[1][1]) // 2
                cv2.putText(
                    temp_img,
                    f"{dist:.1f} px",
                    (mid_x, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Measurement Tool", temp_img)

        elif event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                points.append((x, y))
                print(f"  Point {len(points)}: ({x}, {y})")

                if len(points) == 2:
                    dist = np.sqrt((points[1][0] - points[0][0]) ** 2 + (points[1][1] - points[0][1]) ** 2)
                    print(f"  Distance: {dist:.2f} pixels")
                    print()
                    print("  Enter the real-world distance in mm (or 'r' to reset):")

    cv2.namedWindow("Measurement Tool")
    cv2.setMouseCallback("Measurement Tool", mouse_callback)
    cv2.imshow("Measurement Tool", img)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            points = []
            measuring = False
            print("  Reset measurement")

    cv2.destroyAllWindows()


def calculate_scale_from_measurement(
    pixel_distance: float, real_distance_mm: float
) -> tuple[float, float]:
    """Calculate scale from a measurement.

    Args:
        pixel_distance: Distance in pixels
        real_distance_mm: Real distance in millimeters

    Returns:
        Tuple of (mm_per_pixel, resulting_total_area_m2_estimate)
    """
    mm_per_pixel = real_distance_mm / pixel_distance

    print(f"\n📏 Scale Calculation:")
    print(f"  Pixel distance: {pixel_distance:.2f} px")
    print(f"  Real distance: {real_distance_mm:.0f} mm ({real_distance_mm/10:.1f} cm)")
    print(f"  Scale: {mm_per_pixel:.4f} mm/pixel")
    print()

    # Estimate what the total area would be
    # Current pixel area sum: ~498 m² at 15.38 mm/pixel
    # At new scale:
    current_scale = 15.3846
    current_total_pixels = 498.0 / ((current_scale / 1000) ** 2)

    new_total_area = current_total_pixels * ((mm_per_pixel / 1000) ** 2)

    print(f"  Estimated total apartment area: {new_total_area:.2f} m²")

    return mm_per_pixel, new_total_area


def main():
    """Main function."""
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "outputs/floorplan-scanned-apartment-room-names_processed.png"

    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return

    print("🔍 FLOOR PLAN SCALE EXTRACTION")
    print("=" * 60)
    print()

    # Try automated extraction
    find_180_dimension(image_path)

    print()
    print("📝 Manual Measurement Instructions:")
    print()
    print("1. Identify a dimension marking on the plan (e.g., '180')")
    print("2. Measure the corresponding distance in pixels")
    print("3. Calculate: scale = real_distance_mm / pixel_distance")
    print()
    print("Example measurements to try:")
    print("  • The '180' marking (likely 1800mm)")
    print("  • Wall thickness (typically 150-200mm)")
    print("  • Door width (typically 800mm)")
    print()

    # Launch interactive tool
    response = input("Launch interactive measurement tool? (y/n): ")
    if response.lower() == "y":
        interactive_measurement(image_path)
    else:
        print("\nManual measurement examples:")
        print("\nExample 1: If '180' marking spans 60 pixels")
        calculate_scale_from_measurement(60, 1800)

        print("\nExample 2: If '180' marking spans 80 pixels")
        calculate_scale_from_measurement(80, 1800)

        print("\nExample 3: If '180' marking spans 100 pixels")
        calculate_scale_from_measurement(100, 1800)


if __name__ == "__main__":
    main()
