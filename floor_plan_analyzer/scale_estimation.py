"""Scale estimation using standard building elements."""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

from .models import ProcessingParams, ScaleInfo


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


def detect_door_openings(
    image: np.ndarray,
    min_width_px: int = 10,
    max_width_px: int = 100,
) -> List[Tuple[int, int, int, int, str]]:
    """Detect potential door openings in the floor plan.

    Door openings typically appear as gaps in walls (parallel lines).

    Args:
        image: Input image
        min_width_px: Minimum door width in pixels
        max_width_px: Maximum door width in pixels

    Returns:
        List of (x, y, width, height, orientation) tuples
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines
    horizontal_lines, vertical_lines = detect_lines(image)

    door_candidates = []

    # Look for gaps in horizontal walls (vertical doors)
    h_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in h_contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Look for horizontal lines that are long and thin
        if w > 50 and h < 10:
            # Check for gaps (potential doors) along this line
            line_segment = horizontal_lines[y : y + h, x : x + w]
            # Scan for gaps
            gaps = find_gaps_in_line(line_segment, horizontal=True)
            for gap_x, gap_w in gaps:
                if min_width_px <= gap_w <= max_width_px:
                    door_candidates.append((x + gap_x, y, gap_w, h, "vertical"))

    # Look for gaps in vertical walls (horizontal doors)
    v_contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in v_contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Look for vertical lines that are long and thin
        if h > 50 and w < 10:
            # Check for gaps (potential doors) along this line
            line_segment = vertical_lines[y : y + h, x : x + w]
            # Scan for gaps
            gaps = find_gaps_in_line(line_segment, horizontal=False)
            for gap_y, gap_h in gaps:
                if min_width_px <= gap_h <= max_width_px:
                    door_candidates.append((x, y + gap_y, w, gap_h, "horizontal"))

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
) -> Optional[ScaleInfo]:
    """Estimate scale by detecting door openings.

    Finnish standard door widths:
    - Interior doors: 800-900mm
    - Bathroom doors: 700-800mm

    Args:
        image: Input floor plan image
        params: Processing parameters

    Returns:
        ScaleInfo object or None if scale cannot be estimated
    """
    if params is None:
        params = ProcessingParams()

    # Detect door candidates
    door_candidates = detect_door_openings(image, min_width_px=15, max_width_px=80)

    if not door_candidates:
        return None

    # Collect door widths (in pixels)
    door_widths = []
    for x, y, w, h, orientation in door_candidates:
        if orientation == "vertical":
            door_widths.append(w)
        else:
            door_widths.append(h)

    if not door_widths:
        return None

    # Use median door width for robustness
    median_door_width_px = float(np.median(door_widths))

    # Assume standard Finnish interior door (800mm)
    assumed_door_width_mm = (params.door_width_mm_range[0] + params.door_width_mm_range[1]) / 2

    # Calculate scale
    mm_per_pixel = assumed_door_width_mm / median_door_width_px
    pixels_per_mm = 1.0 / mm_per_pixel

    return ScaleInfo(
        mm_per_pixel=mm_per_pixel,
        pixels_per_mm=pixels_per_mm,
        detected_features=[f"door_{i}" for i in range(len(door_candidates))],
        confidence=min(1.0, len(door_candidates) / 5.0),  # More doors = higher confidence
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


def estimate_scale_from_room_standards(
    image: np.ndarray,
    rooms: dict,
) -> Optional[ScaleInfo]:
    """Estimate scale using Finnish building standards for room sizes.

    Finnish building code regulations specify minimum/typical sizes for various rooms.
    This provides a reliable method when rooms have been segmented.

    Standards used:
    - Bathroom (KPH): 3.5-5.5 m² (typical: 4.5 m²)
    - Toilet (WC): 1.0-2.0 m² (typical: 1.5 m²)
    - Kitchen (K): 8-12 m² (typical: 10 m²)

    Args:
        image: Input image
        rooms: Dictionary of segmented rooms with pixel areas

    Returns:
        ScaleInfo based on room size standards or None
    """
    # Room size standards (in m²)
    room_standards = {
        "KPH": (4.5, 0.9),  # (typical size, confidence)
        "WC": (1.5, 0.85),   # Lower confidence due to high variability
        "K": (10.0, 0.85),
    }

    estimates = []

    for room_name, (expected_m2, confidence) in room_standards.items():
        if room_name in rooms:
            pixel_area = rooms[room_name].pixel_area

            # Calculate required scale
            # area_m2 = pixel_area * (mm_per_pixel / 1000)^2
            # mm_per_pixel = sqrt(area_m2 * 1_000_000 / pixel_area)
            mm_per_pixel = np.sqrt((expected_m2 * 1_000_000) / pixel_area)

            estimates.append({
                "room": room_name,
                "scale": mm_per_pixel,
                "confidence": confidence,
            })

    if not estimates:
        return None

    # Use weighted average based on confidence
    total_weight = sum(e["confidence"] for e in estimates)
    weighted_scale = sum(e["scale"] * e["confidence"] for e in estimates) / total_weight

    # Overall confidence is average of individual confidences
    avg_confidence = np.mean([e["confidence"] for e in estimates])

    # Detect outliers (estimates more than 30% different from median)
    scales = [e["scale"] for e in estimates]
    median_scale = np.median(scales)
    valid_estimates = [e for e in estimates if abs(e["scale"] - median_scale) / median_scale < 0.3]

    if valid_estimates:
        # Recalculate without outliers
        total_weight = sum(e["confidence"] for e in valid_estimates)
        weighted_scale = sum(e["scale"] * e["confidence"] for e in valid_estimates) / total_weight
        detected_features = [f"room_{e['room']}" for e in valid_estimates]
    else:
        detected_features = [f"room_{e['room']}" for e in estimates]

    return ScaleInfo(
        mm_per_pixel=weighted_scale,
        pixels_per_mm=1.0 / weighted_scale,
        detected_features=detected_features,
        confidence=min(0.95, avg_confidence),  # High confidence for standards-based method
    )


def estimate_scale_multi_method(
    image: np.ndarray,
    params: Optional[ProcessingParams] = None,
    rooms: Optional[dict] = None,
) -> ScaleInfo:
    """Estimate scale using multiple methods and combine results.

    Args:
        image: Input floor plan image
        params: Processing parameters
        rooms: Optional dictionary of segmented rooms (for standards-based estimation)

    Returns:
        Best ScaleInfo estimate
    """
    methods = []

    # Method 1: Room standards (highest priority if rooms available)
    if rooms:
        room_scale = estimate_scale_from_room_standards(image, rooms)
        if room_scale:
            methods.append(("room_standards", room_scale))

    # Method 2: Door detection
    door_scale = estimate_scale_from_doors(image, params)
    if door_scale:
        methods.append(("doors", door_scale))

    # Method 3: Grid detection
    grid_scale = estimate_scale_from_grid(image)
    if grid_scale:
        methods.append(("grid", grid_scale))

    if not methods:
        # Fallback: assume a reasonable default scale
        # Typical apartment floor plans at ~1:100 scale scanned at 150 DPI
        # This gives roughly 0.16 mm/pixel
        return ScaleInfo(
            mm_per_pixel=0.16,
            pixels_per_mm=6.25,
            detected_features=["default"],
            confidence=0.3,
        )

    # Use the method with highest confidence
    best_method, best_scale = max(methods, key=lambda x: x[1].confidence)

    return best_scale
