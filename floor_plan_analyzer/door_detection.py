"""Improved door detection using edge analysis."""

from typing import List, Tuple

import cv2
import numpy as np
from scipy import ndimage


def detect_doors_edge_based(
    image: np.ndarray,
    min_door_width: int = 20,
    max_door_width: int = 80,
) -> List[Tuple[int, int, int, int, str, float]]:
    """Detect doors using edge detection and analysis.

    This approach detects doors by looking for characteristic patterns:
    - Parallel lines (door frame)
    - Arc/curve patterns (door swing)
    - Gaps in walls

    Args:
        image: Input floor plan image
        min_door_width: Minimum door width in pixels
        max_door_width: Maximum door width in pixels

    Returns:
        List of (x, y, width, height, orientation, gap_width) tuples
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Detect edges
    edges = cv2.Canny(gray, 30, 100, apertureSize=3)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_door_width,
        maxLineGap=10
    )

    doors = []

    if lines is None:
        return doors

    # Analyze lines to find door-like patterns
    # Group lines by proximity and orientation
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Determine orientation
        if abs(x2 - x1) > abs(y2 - y1):  # Horizontal
            horizontal_lines.append((min(x1, x2), min(y1, y2), length, (x1, y1, x2, y2)))
        else:  # Vertical
            vertical_lines.append((min(x1, x2), min(y1, y2), length, (x1, y1, x2, y2)))

    # Look for parallel line pairs (door frames)
    # For now, return the lines as potential door locations
    for x, y, length, coords in horizontal_lines:
        if min_door_width <= length <= max_door_width:
            doors.append((x, y, int(length), 5, 'vertical', length))

    for x, y, length, coords in vertical_lines:
        if min_door_width <= length <= max_door_width:
            doors.append((x, y, 5, int(length), 'horizontal', length))

    return doors


def detect_doors_from_binary(
    binary: np.ndarray,
    min_door_width: int = 20,
    max_door_width: int = 80,
) -> List[Tuple[int, int, int, int, str, float]]:
    """Detect doors by analyzing the binary floor plan directly.

    This uses a simpler approach: look for gaps in thick line structures.

    Args:
        binary: Binary image (walls = 255, background = 0)
        min_door_width: Minimum door width in pixels
        max_door_width: Maximum door width in pixels

    Returns:
        List of (x, y, width, height, orientation, gap_width) tuples
    """
    doors = []

    # Create skeleton to find wall centerlines
    skeleton = cv2.ximgproc.thinning(binary)

    # Find endpoints and junctions in skeleton
    # Doors often appear as endpoints or T-junctions

    # For now, use a simpler approach: scan for gaps
    h, w = binary.shape

    # Horizontal scan (find vertical doors/gaps)
    for y in range(10, h - 10, 10):  # Sample every 10 rows
        row = binary[y, :]

        # Find wall segments
        in_wall = False
        wall_start = 0

        for x in range(w):
            if row[x] > 0 and not in_wall:
                wall_start = x
                in_wall = True
            elif row[x] == 0 and in_wall:
                # End of wall, check for gap
                wall_width = x - wall_start

                # Look ahead for next wall
                gap_start = x
                gap_found = False

                for x2 in range(x, min(x + max_door_width + 20, w)):
                    if row[x2] > 0:
                        gap_width = x2 - gap_start

                        if min_door_width <= gap_width <= max_door_width:
                            # Found a potential door
                            # Verify it's consistent vertically
                            consistent_height = _verify_vertical_gap(
                                binary, gap_start, gap_width, y, min_height=30
                            )

                            if consistent_height > 0:
                                doors.append((
                                    gap_start,
                                    y - consistent_height // 2,
                                    gap_width,
                                    consistent_height,
                                    'vertical',
                                    gap_width
                                ))

                        gap_found = True
                        break

                in_wall = False

    # Vertical scan (find horizontal doors/gaps)
    for x in range(10, w - 10, 10):  # Sample every 10 columns
        col = binary[:, x]

        in_wall = False
        wall_start = 0

        for y in range(h):
            if col[y] > 0 and not in_wall:
                wall_start = y
                in_wall = True
            elif col[y] == 0 and in_wall:
                # End of wall, check for gap
                wall_height = y - wall_start

                # Look ahead for next wall
                gap_start = y
                gap_found = False

                for y2 in range(y, min(y + max_door_width + 20, h)):
                    if col[y2] > 0:
                        gap_height = y2 - gap_start

                        if min_door_width <= gap_height <= max_door_width:
                            # Found a potential door
                            # Verify it's consistent horizontally
                            consistent_width = _verify_horizontal_gap(
                                binary, gap_start, gap_height, x, min_width=30
                            )

                            if consistent_width > 0:
                                doors.append((
                                    x - consistent_width // 2,
                                    gap_start,
                                    consistent_width,
                                    gap_height,
                                    'horizontal',
                                    gap_height
                                ))

                        gap_found = True
                        break

                in_wall = False

    # Remove duplicates
    doors = _remove_duplicate_doors(doors)

    return doors


def _verify_vertical_gap(
    binary: np.ndarray,
    gap_x: int,
    gap_width: int,
    center_y: int,
    min_height: int,
) -> int:
    """Verify a vertical gap extends consistently.

    Args:
        binary: Binary image
        gap_x: X position of gap
        gap_width: Width of gap
        center_y: Y position to check from
        min_height: Minimum height required

    Returns:
        Height of consistent gap, or 0 if not valid
    """
    h = binary.shape[0]
    height = 0

    # Check upward
    for y in range(center_y, max(0, center_y - 50), -1):
        gap_region = binary[y, gap_x:gap_x + gap_width]
        if np.mean(gap_region) < 50:  # Mostly empty
            height += 1
        else:
            break

    # Check downward
    for y in range(center_y + 1, min(h, center_y + 50)):
        gap_region = binary[y, gap_x:gap_x + gap_width]
        if np.mean(gap_region) < 50:  # Mostly empty
            height += 1
        else:
            break

    return height if height >= min_height else 0


def _verify_horizontal_gap(
    binary: np.ndarray,
    gap_y: int,
    gap_height: int,
    center_x: int,
    min_width: int,
) -> int:
    """Verify a horizontal gap extends consistently.

    Args:
        binary: Binary image
        gap_y: Y position of gap
        gap_height: Height of gap
        center_x: X position to check from
        min_width: Minimum width required

    Returns:
        Width of consistent gap, or 0 if not valid
    """
    w = binary.shape[1]
    width = 0

    # Check leftward
    for x in range(center_x, max(0, center_x - 50), -1):
        gap_region = binary[gap_y:gap_y + gap_height, x]
        if np.mean(gap_region) < 50:  # Mostly empty
            width += 1
        else:
            break

    # Check rightward
    for x in range(center_x + 1, min(w, center_x + 50)):
        gap_region = binary[gap_y:gap_y + gap_height, x]
        if np.mean(gap_region) < 50:  # Mostly empty
            width += 1
        else:
            break

    return width if width >= min_width else 0


def _remove_duplicate_doors(
    doors: List[Tuple[int, int, int, int, str, float]],
    distance_threshold: int = 30,
) -> List[Tuple[int, int, int, int, str, float]]:
    """Remove duplicate door detections.

    Args:
        doors: List of door detections
        distance_threshold: Maximum distance to consider duplicates

    Returns:
        Filtered list of doors
    """
    if not doors:
        return []

    # Sort by position
    doors = sorted(doors, key=lambda d: (d[0], d[1]))

    filtered = [doors[0]]

    for door in doors[1:]:
        # Check if this door is close to any existing door
        is_duplicate = False

        for existing in filtered:
            dist = np.sqrt((door[0] - existing[0]) ** 2 + (door[1] - existing[1]) ** 2)

            if dist < distance_threshold and door[4] == existing[4]:
                is_duplicate = True
                break

        if not is_duplicate:
            filtered.append(door)

    return filtered
