"""Wall extraction and structural element detection."""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage


def extract_walls_morphological(
    image: np.ndarray,
    min_wall_thickness: int = 3,
    max_wall_thickness: int = 15,
) -> np.ndarray:
    """Extract walls using multi-scale morphological filtering.

    This separates thick lines (walls) from thin lines (annotations, ventilation, etc.)

    Args:
        image: Input image (BGR or grayscale)
        min_wall_thickness: Minimum wall thickness in pixels
        max_wall_thickness: Maximum wall thickness in pixels

    Returns:
        Binary mask with walls (255 = wall, 0 = background)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply adaptive threshold to get binary image
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        5
    )

    # Use morphological operations to extract thick lines (walls)
    # We'll use different kernel sizes and combine results

    wall_mask = np.zeros_like(binary)

    # Extract horizontal walls
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max_wall_thickness * 3, min_wall_thickness)
    )
    horizontal_walls = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    # Extract vertical walls
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (min_wall_thickness, max_wall_thickness * 3)
    )
    vertical_walls = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    # Combine horizontal and vertical walls
    wall_mask = cv2.bitwise_or(horizontal_walls, vertical_walls)

    # Clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Dilate slightly to connect nearby wall segments
    wall_mask = cv2.dilate(wall_mask, kernel, iterations=1)

    return wall_mask


def detect_doors_from_walls(
    wall_mask: np.ndarray,
    min_gap_width: int = 15,
    max_gap_width: int = 100,
    min_wall_length: int = 80,
) -> List[Tuple[int, int, int, int, str, float]]:
    """Detect door openings as gaps in walls.

    Args:
        wall_mask: Binary mask of walls
        min_gap_width: Minimum gap width in pixels (smaller gaps ignored)
        max_gap_width: Maximum gap width in pixels (larger gaps not doors)
        min_wall_length: Minimum wall length to consider

    Returns:
        List of (x, y, width, height, orientation, gap_width) tuples
        orientation is 'horizontal' or 'vertical'
        gap_width is the actual measured gap in pixels
    """
    doors = []

    # Find contours of walls
    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Check if this is a horizontal wall (long and thin)
        if w > min_wall_length and h < w / 5:
            # Extract the wall region
            wall_region = wall_mask[y:y+h, x:x+w]

            # Find gaps along this wall
            gaps = _find_gaps_in_wall(wall_region, horizontal=True)

            for gap_start, gap_width in gaps:
                if min_gap_width <= gap_width <= max_gap_width:
                    # This is a door opening
                    door_x = x + gap_start
                    door_y = y
                    doors.append((door_x, door_y, gap_width, h, 'vertical', gap_width))

        # Check if this is a vertical wall (tall and thin)
        elif h > min_wall_length and w < h / 5:
            # Extract the wall region
            wall_region = wall_mask[y:y+h, x:x+w]

            # Find gaps along this wall
            gaps = _find_gaps_in_wall(wall_region, horizontal=False)

            for gap_start, gap_width in gaps:
                if min_gap_width <= gap_width <= max_gap_width:
                    # This is a door opening
                    door_x = x
                    door_y = y + gap_start
                    doors.append((door_x, door_y, w, gap_width, 'horizontal', gap_width))

    return doors


def _find_gaps_in_wall(
    wall_region: np.ndarray,
    horizontal: bool = True,
) -> List[Tuple[int, int]]:
    """Find gaps in a wall region.

    Args:
        wall_region: Binary image of wall segment
        horizontal: True if wall runs horizontally, False if vertically

    Returns:
        List of (gap_start_position, gap_width) tuples
    """
    # Project to 1D
    if horizontal:
        # For horizontal walls, check for vertical gaps
        projection = np.max(wall_region, axis=0)
    else:
        # For vertical walls, check for horizontal gaps
        projection = np.max(wall_region, axis=1)

    # Find gaps where projection is zero (no wall pixels)
    is_gap = projection == 0

    # Find continuous gap regions
    gaps = []
    in_gap = False
    gap_start = 0

    for i, gap_val in enumerate(is_gap):
        if gap_val and not in_gap:
            gap_start = i
            in_gap = True
        elif not gap_val and in_gap:
            gap_width = i - gap_start
            gaps.append((gap_start, gap_width))
            in_gap = False

    # Handle gap at the end
    if in_gap:
        gap_width = len(is_gap) - gap_start
        gaps.append((gap_start, gap_width))

    return gaps


def cluster_door_widths(
    door_widths: List[float],
    tolerance: float = 5.0,
) -> List[Tuple[float, int]]:
    """Cluster similar door widths together.

    Args:
        door_widths: List of door widths in pixels
        tolerance: Maximum difference to be in same cluster

    Returns:
        List of (cluster_center, count) tuples, sorted by count
    """
    if not door_widths:
        return []

    # Sort widths
    sorted_widths = sorted(door_widths)

    clusters = []
    current_cluster = [sorted_widths[0]]

    for width in sorted_widths[1:]:
        if width - current_cluster[-1] <= tolerance:
            current_cluster.append(width)
        else:
            # Save current cluster and start new one
            center = np.mean(current_cluster)
            clusters.append((center, len(current_cluster)))
            current_cluster = [width]

    # Don't forget the last cluster
    if current_cluster:
        center = np.mean(current_cluster)
        clusters.append((center, len(current_cluster)))

    # Sort by count (most common first)
    clusters.sort(key=lambda x: x[1], reverse=True)

    return clusters


def visualize_doors(
    image: np.ndarray,
    doors: List[Tuple[int, int, int, int, str, float]],
    wall_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Create a visualization of detected doors.

    Args:
        image: Original image
        doors: List of door detections
        wall_mask: Optional wall mask to overlay

    Returns:
        Visualization image
    """
    vis = image.copy()

    # Ensure we have a color image
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # Draw wall mask in semi-transparent blue if provided
    if wall_mask is not None:
        wall_overlay = np.zeros_like(vis)
        wall_overlay[wall_mask > 0] = [255, 0, 0]  # Blue for walls
        vis = cv2.addWeighted(vis, 0.7, wall_overlay, 0.3, 0)

    # Draw doors
    for i, (x, y, w, h, orientation, gap_width) in enumerate(doors):
        # Draw door rectangle in green
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add label
        label = f"{gap_width:.0f}px"
        cv2.putText(
            vis,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1
        )

    # Add summary
    cv2.putText(
        vis,
        f"Detected {len(doors)} doors",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    return vis
