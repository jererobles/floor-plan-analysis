"""Advanced door detection for floor plan analysis.

This module implements robust door detection by identifying:
1. Door arcs (quarter/half circles representing door swing)
2. Gaps in wall lines (door openings)
3. Combined validation to ensure accuracy
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import distance


@dataclass
class Door:
    """Represents a detected door in the floor plan."""

    x: int  # Center X coordinate
    y: int  # Center Y coordinate
    width_px: int  # Door opening width in pixels
    orientation: str  # 'horizontal' or 'vertical'
    arc_detected: bool  # Whether door arc was found
    gap_detected: bool  # Whether wall gap was found
    confidence: float  # Detection confidence (0-1)


def detect_doors(
    image: np.ndarray,
    min_width_px: int = 10,
    max_width_px: int = 50,
    debug: bool = False,
) -> List[Door]:
    """Detect doors in a floor plan using multiple methods.

    Note: We prioritize detecting the actual door OPENING (gap in wall),
    not the door swing arc radius. The opening width is what matters for
    calibration with standard door sizes (700-900mm).

    Args:
        image: Input floor plan image (color or grayscale)
        min_width_px: Minimum expected door opening width in pixels (default: 10)
        max_width_px: Maximum expected door opening width in pixels (default: 50)
        debug: If True, save debug visualizations

    Returns:
        List of detected doors with confidence scores
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Step 1: Detect wall gaps (primary method for measuring door opening)
    gap_candidates = detect_wall_gaps(gray, min_width_px, max_width_px)

    # Step 2: Detect door arcs (for validation and location)
    arc_candidates = detect_door_arcs(gray, min_width_px * 2, max_width_px * 2)

    # Step 3: Combine and validate detections
    # Prioritize gap measurements over arc measurements
    doors = combine_detections(gap_candidates, arc_candidates, max_distance=50)

    # Step 4: Filter and rank by confidence
    doors = filter_and_rank_doors(doors, min_confidence=0.3)

    if debug:
        visualize_detections(image, doors, "door_detection_debug.png")

    return doors


def detect_door_arcs(
    gray: np.ndarray,
    min_width: int,
    max_width: int,
) -> List[Tuple[int, int, int, str]]:
    """Detect door arcs (quarter/half circles representing door swing).

    Args:
        gray: Grayscale image
        min_width: Minimum door width
        max_width: Maximum door width

    Returns:
        List of (x, y, width, orientation) tuples
    """
    arc_candidates = []

    # Apply edge detection
    edges = cv2.Canny(gray, 30, 100)

    # Use morphological operations to clean up arcs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Detect circles using Hough Transform
    # Door arcs typically have radius = door width
    min_radius = min_width // 2
    max_radius = max_width

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_width,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, radius = circle
            # Door width is approximately 2 * radius for a quarter circle arc
            # but the opening itself is closer to the radius
            width = int(radius * 1.4)  # Empirical factor

            # Determine orientation by analyzing nearby pixels
            # This is a simplified approach - check if more edges vertically or horizontally
            roi_size = radius * 2
            roi_x1 = max(0, x - roi_size)
            roi_x2 = min(gray.shape[1], x + roi_size)
            roi_y1 = max(0, y - roi_size)
            roi_y2 = min(gray.shape[0], y + roi_size)

            roi_edges = edges[roi_y1:roi_y2, roi_x1:roi_x2]

            # Count edges in horizontal and vertical directions
            if roi_edges.size > 0:
                h_sum = np.sum(roi_edges, axis=0)
                v_sum = np.sum(roi_edges, axis=1)
                h_edges = h_sum.max() if h_sum.size > 0 else 0
                v_edges = v_sum.max() if v_sum.size > 0 else 0
            else:
                h_edges = v_edges = 0

            orientation = "vertical" if v_edges > h_edges else "horizontal"

            if min_width <= width <= max_width:
                arc_candidates.append((int(x), int(y), width, orientation))

    return arc_candidates


def detect_wall_gaps(
    gray: np.ndarray,
    min_width: int,
    max_width: int,
) -> List[Tuple[int, int, int, str]]:
    """Detect gaps in walls (door openings).

    Args:
        gray: Grayscale image
        min_width: Minimum gap width
        max_width: Maximum gap width

    Returns:
        List of (x, y, width, orientation) tuples
    """
    gap_candidates = []

    # Apply adaptive thresholding to detect walls
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )

    # Detect horizontal and vertical walls
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))

    horizontal_walls = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)
    vertical_walls = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)

    # Find gaps in horizontal walls (vertical door openings)
    gap_candidates.extend(
        find_gaps_in_walls(horizontal_walls, min_width, max_width, horizontal=True)
    )

    # Find gaps in vertical walls (horizontal door openings)
    gap_candidates.extend(
        find_gaps_in_walls(vertical_walls, min_width, max_width, horizontal=False)
    )

    return gap_candidates


def find_gaps_in_walls(
    wall_image: np.ndarray,
    min_width: int,
    max_width: int,
    horizontal: bool,
) -> List[Tuple[int, int, int, str]]:
    """Find gaps in wall lines.

    Args:
        wall_image: Binary image with walls
        min_width: Minimum gap width
        max_width: Maximum gap width
        horizontal: True if walls are horizontal (vertical doors)

    Returns:
        List of (x, y, width, orientation) tuples
    """
    gaps = []

    # Find contours of walls
    contours, _ = cv2.findContours(wall_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Check if this is a long, thin wall
        if horizontal and w > 100 and h < 15:
            # This is a horizontal wall, look for vertical gaps
            wall_slice = wall_image[y : y + h, x : x + w]

            # Project to 1D
            projection = np.sum(wall_slice, axis=0)
            threshold = np.max(projection) * 0.3

            # Find gaps
            in_gap = False
            gap_start = 0

            for i, val in enumerate(projection):
                if val < threshold and not in_gap:
                    gap_start = i
                    in_gap = True
                elif val >= threshold and in_gap:
                    gap_width = i - gap_start
                    if min_width <= gap_width <= max_width:
                        gap_x = x + gap_start + gap_width // 2
                        gap_y = y + h // 2
                        gaps.append((gap_x, gap_y, gap_width, "vertical"))
                    in_gap = False

        elif not horizontal and h > 100 and w < 15:
            # This is a vertical wall, look for horizontal gaps
            wall_slice = wall_image[y : y + h, x : x + w]

            # Project to 1D
            projection = np.sum(wall_slice, axis=1)
            threshold = np.max(projection) * 0.3

            # Find gaps
            in_gap = False
            gap_start = 0

            for i, val in enumerate(projection):
                if val < threshold and not in_gap:
                    gap_start = i
                    in_gap = True
                elif val >= threshold and in_gap:
                    gap_width = i - gap_start
                    if min_width <= gap_width <= max_width:
                        gap_x = x + w // 2
                        gap_y = y + gap_start + gap_width // 2
                        gaps.append((gap_x, gap_y, gap_width, "horizontal"))
                    in_gap = False

    return gaps


def combine_detections(
    gap_candidates: List[Tuple[int, int, int, str]],
    arc_candidates: List[Tuple[int, int, int, str]],
    max_distance: int = 50,
) -> List[Door]:
    """Combine gap and arc detections to create validated door list.

    Args:
        gap_candidates: List of gap detections (PRIORITIZED - actual opening width)
        arc_candidates: List of arc detections (for validation)
        max_distance: Maximum distance to consider matches

    Returns:
        List of Door objects with confidence scores
    """
    doors = []
    matched_gaps = set()
    matched_arcs = set()

    # Match gaps with arcs (prioritize gap measurements)
    for i, (gx, gy, gw, g_orient) in enumerate(gap_candidates):
        best_match = None
        best_dist = float("inf")

        for j, (ax, ay, aw, a_orient) in enumerate(arc_candidates):
            if j in matched_arcs:
                continue

            # Check if orientations are compatible
            if g_orient != a_orient:
                continue

            # Calculate distance
            dist = np.sqrt((gx - ax) ** 2 + (gy - ay) ** 2)

            if dist < max_distance and dist < best_dist:
                best_dist = dist
                best_match = j

        if best_match is not None:
            ax, ay, aw, a_orient = arc_candidates[best_match]
            # Use gap width (actual opening), not arc radius
            # Average positions for location
            door = Door(
                x=(gx + ax) // 2,
                y=(gy + ay) // 2,
                width_px=gw,  # Use GAP width, not arc width!
                orientation=g_orient,
                arc_detected=True,
                gap_detected=True,
                confidence=1.0,  # High confidence - both features detected
            )
            doors.append(door)
            matched_gaps.add(i)
            matched_arcs.add(best_match)

    # Add unmatched gaps with high confidence (gaps are reliable)
    for i, (gx, gy, gw, g_orient) in enumerate(gap_candidates):
        if i not in matched_gaps:
            door = Door(
                x=gx,
                y=gy,
                width_px=gw,
                orientation=g_orient,
                arc_detected=False,
                gap_detected=True,
                confidence=0.7,  # Good confidence - gap measurement is reliable
            )
            doors.append(door)

    # Add unmatched arcs with lower confidence (arc radius is not opening width)
    for j, (ax, ay, aw, a_orient) in enumerate(arc_candidates):
        if j not in matched_arcs:
            # Arc width is typically ~2x the actual opening width
            # Adjust accordingly
            estimated_width = int(aw * 0.5)
            door = Door(
                x=ax,
                y=ay,
                width_px=estimated_width,
                orientation=a_orient,
                arc_detected=True,
                gap_detected=False,
                confidence=0.4,  # Low confidence - estimated from arc
            )
            doors.append(door)

    return doors


def filter_and_rank_doors(
    doors: List[Door],
    min_confidence: float = 0.6,
) -> List[Door]:
    """Filter doors by confidence and remove duplicates.

    Args:
        doors: List of detected doors
        min_confidence: Minimum confidence threshold (default: 0.6)

    Returns:
        Filtered and ranked list of doors
    """
    # Filter by confidence
    doors = [d for d in doors if d.confidence >= min_confidence]

    # Remove duplicates (doors too close together)
    filtered = []
    for door in sorted(doors, key=lambda d: d.confidence, reverse=True):
        is_duplicate = False
        for existing in filtered:
            dist = np.sqrt((door.x - existing.x) ** 2 + (door.y - existing.y) ** 2)
            if dist < 40:  # Within 40 pixels
                is_duplicate = True
                break

        if not is_duplicate:
            filtered.append(door)

    # Filter outliers based on width consistency
    if len(filtered) > 3:
        widths = [d.width_px for d in filtered]
        median_width = np.median(widths)
        std_dev = np.std(widths)

        # Remove doors that are more than 2 standard deviations from median
        filtered = [
            d
            for d in filtered
            if abs(d.width_px - median_width) < 2 * std_dev
        ]

    return filtered


def analyze_door_consistency(doors: List[Door]) -> Tuple[float, float, List[float]]:
    """Analyze consistency of door widths.

    Args:
        doors: List of detected doors

    Returns:
        Tuple of (median_width, std_dev, all_widths)
    """
    if not doors:
        return 0.0, 0.0, []

    widths = [d.width_px for d in doors]
    median_width = float(np.median(widths))
    std_dev = float(np.std(widths))

    return median_width, std_dev, widths


def calculate_scale_from_doors(
    doors: List[Door],
    standard_door_width_mm: float = 850.0,
    max_std_dev_ratio: float = 0.15,
) -> Optional[float]:
    """Calculate mm-to-px ratio from door measurements.

    Args:
        doors: List of detected doors
        standard_door_width_mm: Standard door width in millimeters
        max_std_dev_ratio: Maximum acceptable std dev / median ratio

    Returns:
        mm_per_pixel ratio, or None if doors are inconsistent
    """
    if len(doors) < 5:  # Need at least 5 doors for reliable calibration
        return None

    median_width, std_dev, widths = analyze_door_consistency(doors)

    # Check consistency
    if median_width == 0:
        return None

    consistency_ratio = std_dev / median_width

    if consistency_ratio > max_std_dev_ratio:
        # Doors are too inconsistent
        return None

    # Calculate scale
    mm_per_pixel = standard_door_width_mm / median_width

    return mm_per_pixel


def visualize_detections(
    image: np.ndarray,
    doors: List[Door],
    output_path: str,
) -> None:
    """Create visualization of detected doors.

    Args:
        image: Input image
        doors: List of detected doors
        output_path: Path to save visualization
    """
    # Convert to color if needed
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    # Draw doors with different colors based on confidence
    for i, door in enumerate(doors):
        # Color based on confidence: green (high) to yellow (medium) to red (low)
        if door.confidence > 0.8:
            color = (0, 255, 0)  # Green
        elif door.confidence > 0.5:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red

        # Draw circle at door location
        cv2.circle(vis, (door.x, door.y), 5, color, -1)

        # Draw bounding box
        half_width = door.width_px // 2
        if door.orientation == "vertical":
            x1, y1 = door.x - half_width, door.y - 10
            x2, y2 = door.x + half_width, door.y + 10
        else:
            x1, y1 = door.x - 10, door.y - half_width
            x2, y2 = door.x + 10, door.y + half_width

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Add label
        label = f"D{i+1}: {door.width_px}px"
        cv2.putText(
            vis,
            label,
            (door.x + 10, door.y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )

    # Save visualization
    cv2.imwrite(output_path, vis)
