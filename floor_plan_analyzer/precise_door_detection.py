"""Precise door opening detection by analyzing wall gaps."""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from scipy import ndimage


@dataclass
class WallGap:
    """A gap in a wall line (potential door)."""
    x: int
    y: int
    width_px: int
    orientation: str  # 'horizontal' or 'vertical'
    wall_thickness: int


def detect_precise_door_openings(
    image: np.ndarray,
    min_gap_width: int = 15,
    max_gap_width: int = 60,
) -> List[WallGap]:
    """Detect door openings by finding gaps in wall lines.

    This method focuses on detecting actual architectural openings in walls,
    not door swing arcs or other features.

    Args:
        image: Input floor plan image
        min_gap_width: Minimum door opening width (pixels)
        max_gap_width: Maximum door opening width (pixels)

    Returns:
        List of detected wall gaps (doors)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply strong thresholding to get clean walls
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Invert so walls are white
    binary_inv = cv2.bitwise_not(binary)

    # Detect thick walls (double lines)
    wall_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    walls = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, wall_kernel, iterations=2)

    # Detect horizontal and vertical walls separately
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))

    horizontal_walls = cv2.morphologyEx(walls, cv2.MORPH_OPEN, h_kernel, iterations=2)
    vertical_walls = cv2.morphologyEx(walls, cv2.MORPH_OPEN, v_kernel, iterations=2)

    gaps = []

    # Find gaps in horizontal walls (vertical door openings)
    gaps.extend(find_wall_gaps(horizontal_walls, True, min_gap_width, max_gap_width))

    # Find gaps in vertical walls (horizontal door openings)
    gaps.extend(find_wall_gaps(vertical_walls, False, min_gap_width, max_gap_width))

    return gaps


def find_wall_gaps(
    wall_image: np.ndarray,
    horizontal_wall: bool,
    min_width: int,
    max_width: int,
) -> List[WallGap]:
    """Find gaps in wall lines.

    Args:
        wall_image: Binary image with walls
        horizontal_wall: True if walls are horizontal
        min_width: Minimum gap width
        max_width: Maximum gap width

    Returns:
        List of WallGap objects
    """
    gaps = []

    # Find wall contours
    contours, _ = cv2.findContours(wall_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Check if this is a long wall
        if horizontal_wall:
            if w < 150 or h > 20:  # Not a long horizontal wall
                continue

            wall_thickness = h
            orientation = "vertical"  # Door opens vertically through horizontal wall

            # Extract this wall section
            wall_section = wall_image[y:y+h, x:x+w]

            # Project to 1D (sum along height)
            projection = np.sum(wall_section, axis=0)

            # Find gaps (where projection is low)
            threshold = np.max(projection) * 0.2 if np.max(projection) > 0 else 0

            # Find continuous gap regions
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

                        gaps.append(WallGap(
                            x=gap_x,
                            y=gap_y,
                            width_px=gap_width,
                            orientation=orientation,
                            wall_thickness=wall_thickness,
                        ))

                    in_gap = False

        else:  # Vertical wall
            if h < 150 or w > 20:  # Not a long vertical wall
                continue

            wall_thickness = w
            orientation = "horizontal"  # Door opens horizontally through vertical wall

            # Extract this wall section
            wall_section = wall_image[y:y+h, x:x+w]

            # Project to 1D (sum along width)
            projection = np.sum(wall_section, axis=1)

            # Find gaps
            threshold = np.max(projection) * 0.2 if np.max(projection) > 0 else 0

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

                        gaps.append(WallGap(
                            x=gap_x,
                            y=gap_y,
                            width_px=gap_width,
                            orientation=orientation,
                            wall_thickness=wall_thickness,
                        ))

                    in_gap = False

    return gaps


def filter_door_candidates(gaps: List[WallGap]) -> List[WallGap]:
    """Filter door candidates based on consistency and proximity.

    Args:
        gaps: List of detected gaps

    Returns:
        Filtered list of doors
    """
    if not gaps:
        return []

    # Remove duplicates (gaps too close together)
    filtered = []
    for gap in sorted(gaps, key=lambda g: g.width_px, reverse=True):
        is_duplicate = False
        for existing in filtered:
            dist = np.sqrt((gap.x - existing.x) ** 2 + (gap.y - existing.y) ** 2)
            if dist < 30:
                is_duplicate = True
                break

        if not is_duplicate:
            filtered.append(gap)

    # Filter by width consistency
    if len(filtered) > 3:
        widths = [g.width_px for g in filtered]
        median_width = np.median(widths)
        std_dev = np.std(widths)

        # Keep only gaps within 2 std dev of median
        filtered = [
            g for g in filtered
            if abs(g.width_px - median_width) < 2 * std_dev
        ]

    return filtered
