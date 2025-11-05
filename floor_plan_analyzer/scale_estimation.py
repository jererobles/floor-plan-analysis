"""Scale estimation using standard building elements."""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

from .models import ProcessingParams, ScaleInfo


def visualize_detected_doors(
    image: np.ndarray,
    door_candidates: List[Tuple[int, int, int, int, str]],
    door_widths: Optional[List[float]] = None,
) -> np.ndarray:
    """Visualize detected door openings on the image.

    Args:
        image: Input image
        door_candidates: List of detected doors
        door_widths: Optional list of door widths for labeling

    Returns:
        Image with doors highlighted
    """
    # Create a color copy
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    # Draw each door
    for idx, (x, y, w, h, orientation) in enumerate(door_candidates):
        # Draw rectangle
        color = (0, 255, 0)  # Green
        thickness = 2
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)

        # Add label
        if door_widths and idx < len(door_widths):
            width_px = door_widths[idx]
            label = f"{width_px:.0f}px"
        else:
            label = f"D{idx+1}"

        # Position label
        label_x = x + w // 2 - 20
        label_y = y - 5 if orientation == "vertical" else y + h // 2

        cv2.putText(
            vis,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return vis


def detect_lines(image: np.ndarray, threshold: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Detect horizontal and vertical lines in floor plan.

    Args:
        image: Input image
        threshold: Threshold for line detection

    Returns:
        Tuple of (horizontal lines image, vertical lines image)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    return horizontal_lines, vertical_lines


def detect_arcs(image: np.ndarray) -> List[Tuple[int, int, int]]:
    """Detect arc patterns in the image (door swing indicators).

    Args:
        image: Input image

    Returns:
        List of (x, y, radius) tuples for detected arcs
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # Try multiple parameter sets to detect doors of different sizes
    all_arcs = []

    param_sets = [
        # (dp, minDist, param1, param2, minRadius, maxRadius)
        # For a 70-90m² apartment at ~5mm/pixel, doors should be ~140-175 pixels
        # Using slightly wider range to account for variations
        (1, 80, 50, 30, 100, 200),  # Standard interior doors (primary)
        (1, 60, 40, 25, 80, 180),   # Slightly smaller range for edge cases
    ]

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
            circles = np.round(circles[0, :]).astype(int)
            for x, y, r in circles:
                # Check if this is a new arc (not too close to existing ones)
                is_new = True
                for existing_x, existing_y, existing_r in all_arcs:
                    dx = float(x) - float(existing_x)
                    dy = float(y) - float(existing_y)
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < 30:  # Too close to existing arc
                        is_new = False
                        break
                if is_new:
                    all_arcs.append((int(x), int(y), int(r)))

    return all_arcs


def detect_wall_gaps(
    image: np.ndarray,
    min_gap_px: int = 15,
    max_gap_px: int = 100,
) -> List[Tuple[int, int, int, int, str, float]]:
    """Detect gaps in walls that could be door openings.

    Uses edge-based approach to find door-sized openings in walls.

    Args:
        image: Input image
        min_gap_px: Minimum gap width in pixels
        max_gap_px: Maximum gap width in pixels

    Returns:
        List of (x, y, width, height, orientation, gap_size) tuples
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply threshold to isolate lines
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Invert so walls are white
    binary_inv = cv2.bitwise_not(binary)

    # Use morphological operations with smaller kernels that work better on this plan
    # Horizontal structures (walls running left-right)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, h_kernel, iterations=1)

    # Vertical structures (walls running up-down)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, v_kernel, iterations=1)

    gaps = []

    # Detect lines using Hough Line Transform for more robust wall detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        # Group lines by orientation
        h_lines = []  # horizontal lines
        v_lines = []  # vertical lines

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Calculate angle
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            # Classify as horizontal or vertical
            if (angle < 15 or angle > 165):  # Horizontal
                h_lines.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), (y1 + y2) // 2))
            elif 75 < angle < 105:  # Vertical
                v_lines.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), (x1 + x2) // 2))

        # Group parallel horizontal lines and look for gaps
        h_lines.sort(key=lambda x: x[4])  # Sort by y-coordinate
        for i in range(len(h_lines) - 1):
            x1a, y1a, x2a, y2a, _ = h_lines[i]
            x1b, y1b, x2b, y2b, _ = h_lines[i + 1]

            # Check if lines are close (part of same wall)
            if abs(y1a - y1b) < 20:  # Within 20 pixels (wall thickness)
                # Check for gaps in x-direction
                if x1b > x2a:  # There's a gap
                    gap_width = x1b - x2a
                    if min_gap_px <= gap_width <= max_gap_px:
                        # This is a potential door
                        mid_y = (y1a + y1b) // 2
                        gaps.append((x2a, mid_y - 5, gap_width, 10, "vertical", float(gap_width)))

        # Group parallel vertical lines and look for gaps
        v_lines.sort(key=lambda x: x[4])  # Sort by x-coordinate
        for i in range(len(v_lines) - 1):
            x1a, y1a, x2a, y2a, _ = v_lines[i]
            x1b, y1b, x2b, y2b, _ = v_lines[i + 1]

            # Check if lines are close (part of same wall)
            if abs(x1a - x1b) < 20:  # Within 20 pixels (wall thickness)
                # Check for gaps in y-direction
                if y1b > y2a:  # There's a gap
                    gap_height = y1b - y2a
                    if min_gap_px <= gap_height <= max_gap_px:
                        # This is a potential door
                        mid_x = (x1a + x1b) // 2
                        gaps.append((mid_x - 5, y2a, 10, gap_height, "horizontal", float(gap_height)))

    return gaps


def detect_door_openings(
    image: np.ndarray,
    min_width_px: int = 80,
    max_width_px: int = 200,
) -> List[Tuple[int, int, int, int, str]]:
    """Detect potential door openings in the floor plan.

    Uses door swing arcs as primary indicators since they are most reliable.
    The radius of the arc corresponds to the door width.

    Args:
        image: Input image
        min_width_px: Minimum door width in pixels (default 80 for ~400-600mm at 5-7mm/px)
        max_width_px: Maximum door width in pixels (default 200 for ~1000-1400mm at 5-7mm/px)

    Returns:
        List of (x, y, width, height, orientation) tuples where width/height is the door opening size
    """
    # Detect arcs (primary method)
    arcs = detect_arcs(image)

    # Convert arcs to door candidates
    # The radius of the arc represents the door width
    door_candidates = []

    for x, y, r in arcs:
        # Filter by size - only keep arc radii that match typical door widths
        if min_width_px <= r <= max_width_px:
            # Create a door representation
            # We don't know the exact orientation without more analysis,
            # so we'll use the radius as the "width" and mark as vertical by default
            door_candidates.append((x - r//2, y - r//2, r, r, "vertical"))

    return door_candidates


def find_gaps_in_line(
    line_image: np.ndarray,
    horizontal: bool = True,
) -> List[Tuple[int, int]]:
    """Find gaps in a line image.

    Args:
        line_image: Binary image of a line
        horizontal: True if scanning horizontally, False if vertically

    Returns:
        List of (position, length) tuples for gaps
    """
    # Project the line to 1D
    if horizontal:
        projection = np.sum(line_image, axis=0)
    else:
        projection = np.sum(line_image, axis=1)

    # Find gaps (where projection is zero or very low)
    threshold = np.max(projection) * 0.1
    is_gap = projection < threshold

    # Find continuous gap regions
    gaps = []
    in_gap = False
    gap_start = 0

    for i, gap_val in enumerate(is_gap):
        if gap_val and not in_gap:
            gap_start = i
            in_gap = True
        elif not gap_val and in_gap:
            gaps.append((gap_start, i - gap_start))
            in_gap = False

    if in_gap:
        gaps.append((gap_start, len(is_gap) - gap_start))

    return gaps


def estimate_scale_from_doors(
    image: np.ndarray,
    params: Optional[ProcessingParams] = None,
    verbose: bool = True,
    save_visualization: Optional[str] = None,
) -> Optional[ScaleInfo]:
    """Estimate scale by detecting door openings.

    Finnish standard door widths:
    - Interior doors: 800-900mm (most common: 800-850mm)
    - Bathroom doors: 700-800mm
    - Entrance doors: 900-1000mm

    Args:
        image: Input floor plan image
        params: Processing parameters
        verbose: If True, print detailed information about detected doors
        save_visualization: If provided, save door visualization to this path

    Returns:
        ScaleInfo object or None if scale cannot be estimated
    """
    if params is None:
        params = ProcessingParams()

    # Detect door candidates - use default range optimized for typical apartment floor plans
    door_candidates = detect_door_openings(image)

    if not door_candidates:
        if verbose:
            print("   ⚠️  No door openings detected")
        return None

    # Collect door widths (in pixels)
    door_widths = []
    for x, y, w, h, orientation in door_candidates:
        if orientation == "vertical":
            door_widths.append(float(w))
        else:
            door_widths.append(float(h))

    if not door_widths:
        return None

    door_widths_array = np.array(door_widths)

    # Calculate statistics
    median_width = np.median(door_widths_array)
    mean_width = np.mean(door_widths_array)
    std_width = np.std(door_widths_array)
    min_width = np.min(door_widths_array)
    max_width = np.max(door_widths_array)

    if verbose:
        print(f"\n   🚪 Detected {len(door_candidates)} door opening candidates")
        print(f"   Door widths (pixels): min={min_width:.1f}, max={max_width:.1f}, "
              f"mean={mean_width:.1f}±{std_width:.1f}, median={median_width:.1f}")

    # Filter outliers using IQR method for better consistency
    q1 = np.percentile(door_widths_array, 25)
    q3 = np.percentile(door_widths_array, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    filtered_widths = door_widths_array[
        (door_widths_array >= lower_bound) & (door_widths_array <= upper_bound)
    ]

    if len(filtered_widths) < 3:
        # Not enough doors after filtering
        if verbose:
            print(f"   ⚠️  Only {len(filtered_widths)} doors after outlier removal (need at least 3)")
        filtered_widths = door_widths_array  # Use all doors

    # Calculate coefficient of variation to assess consistency
    cv = (np.std(filtered_widths) / np.mean(filtered_widths)) * 100 if len(filtered_widths) > 0 else 100

    if verbose:
        print(f"   After outlier removal: {len(filtered_widths)} doors, "
              f"mean={np.mean(filtered_widths):.1f}±{np.std(filtered_widths):.1f}px")
        print(f"   Coefficient of variation: {cv:.1f}% " +
              ("(Good consistency)" if cv < 15 else "(Moderate consistency)" if cv < 25 else "(Poor consistency)"))

    # Use median of filtered widths for robustness
    median_door_width_px = float(np.median(filtered_widths))

    # Assume standard Finnish interior door (most common size: 825mm)
    # This is a reasonable average for Finnish apartments
    assumed_door_width_mm = 825.0

    if verbose:
        print(f"   Using median door width: {median_door_width_px:.1f} pixels")
        print(f"   Assuming standard Finnish door: {assumed_door_width_mm:.0f}mm")

    # Calculate scale
    mm_per_pixel = assumed_door_width_mm / median_door_width_px
    pixels_per_mm = 1.0 / mm_per_pixel

    # Calculate confidence based on:
    # 1. Number of doors detected (more is better)
    # 2. Consistency of measurements (lower CV is better)
    num_doors_score = min(1.0, len(filtered_widths) / 8.0)  # Max score at 8+ doors
    consistency_score = max(0.3, 1.0 - (cv / 100.0))  # Lower CV = higher score
    confidence = (num_doors_score + consistency_score) / 2.0

    if verbose:
        print(f"   Calculated scale: {mm_per_pixel:.4f} mm/pixel")
        print(f"   Confidence: {confidence:.2f} (doors={num_doors_score:.2f}, consistency={consistency_score:.2f})")

    # Save visualization if requested
    if save_visualization:
        vis_img = visualize_detected_doors(image, door_candidates, door_widths)
        cv2.imwrite(save_visualization, vis_img)
        if verbose:
            print(f"   💾 Door visualization saved to: {save_visualization}")

    return ScaleInfo(
        mm_per_pixel=mm_per_pixel,
        pixels_per_mm=pixels_per_mm,
        detected_features=[f"door_{i}" for i in range(len(door_candidates))],
        confidence=confidence,
        metadata={
            "num_doors": len(door_candidates),
            "num_doors_filtered": len(filtered_widths),
            "median_door_width_px": median_door_width_px,
            "door_width_cv": cv,
            "assumed_door_width_mm": assumed_door_width_mm,
            "door_candidates": door_candidates,
            "door_widths": door_widths,
        }
    )


def estimate_scale_from_grid(
    image: np.ndarray,
    expected_grid_size_mm: float = 500.0,
) -> Optional[ScaleInfo]:
    """Estimate scale from grid lines if present.

    Args:
        image: Input floor plan image
        expected_grid_size_mm: Expected grid size in millimeters

    Returns:
        ScaleInfo object or None if no grid detected
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Detect lines
    horizontal_lines, vertical_lines = detect_lines(image, threshold=50)

    # Find spacing between parallel lines
    h_spacing = detect_line_spacing(horizontal_lines, horizontal=True)
    v_spacing = detect_line_spacing(vertical_lines, horizontal=False)

    if h_spacing is None and v_spacing is None:
        return None

    # Use average spacing
    spacings = [s for s in [h_spacing, v_spacing] if s is not None]
    avg_spacing_px = float(np.mean(spacings))

    # Calculate scale
    mm_per_pixel = expected_grid_size_mm / avg_spacing_px
    pixels_per_mm = 1.0 / mm_per_pixel

    return ScaleInfo(
        mm_per_pixel=mm_per_pixel,
        pixels_per_mm=pixels_per_mm,
        detected_features=["grid"],
        confidence=0.7,  # Medium confidence as grid size is assumed
    )


def detect_line_spacing(
    line_image: np.ndarray,
    horizontal: bool = True,
) -> Optional[float]:
    """Detect spacing between parallel lines.

    Args:
        line_image: Binary image with lines
        horizontal: True for horizontal lines, False for vertical

    Returns:
        Average spacing in pixels or None
    """
    # Project to 1D
    if horizontal:
        projection = np.sum(line_image, axis=1)
    else:
        projection = np.sum(line_image, axis=0)

    # Find peaks (line positions)
    threshold = np.max(projection) * 0.5
    peaks = np.where(projection > threshold)[0]

    if len(peaks) < 2:
        return None

    # Calculate spacing between consecutive peaks
    spacings = np.diff(peaks)

    # Filter out very small spacings (noise)
    spacings = spacings[spacings > 10]

    if len(spacings) == 0:
        return None

    return float(np.median(spacings))


def estimate_scale_multi_method(
    image: np.ndarray,
    params: Optional[ProcessingParams] = None,
    door_visualization_path: Optional[str] = None,
) -> ScaleInfo:
    """Estimate scale using multiple methods and combine results.

    Args:
        image: Input floor plan image
        params: Processing parameters
        door_visualization_path: If provided, save door detection visualization

    Returns:
        Best ScaleInfo estimate
    """
    methods = [
        ("doors", estimate_scale_from_doors(
            image,
            params,
            verbose=True,
            save_visualization=door_visualization_path
        )),
        ("grid", estimate_scale_from_grid(image)),
    ]

    # Filter out None results
    valid_methods = [(name, scale) for name, scale in methods if scale is not None]

    if not valid_methods:
        # Fallback: assume a reasonable default scale
        # Typical apartment floor plans at ~1:100 scale scanned at 150 DPI
        # This gives roughly 0.16 mm/pixel
        print("   ⚠️  No scale estimation methods succeeded, using default")
        return ScaleInfo(
            mm_per_pixel=0.16,
            pixels_per_mm=6.25,
            detected_features=["default"],
            confidence=0.3,
        )

    # Use the method with highest confidence
    best_method, best_scale = max(valid_methods, key=lambda x: x[1].confidence)

    return best_scale
