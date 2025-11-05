"""Improved door detection using simple gap analysis."""

from typing import List, Tuple

import cv2
import numpy as np


def detect_doors_simple(
    image: np.ndarray,
    min_door_width: int = 20,
    max_door_width: int = 200,  # Increased to catch full door openings
    edge_margin: int = 50,
) -> List[Tuple[int, int, int, int, str, float]]:
    """Detect doors using simple gap detection in binary image.

    Args:
        image: Input floor plan image
        min_door_width: Minimum door width in pixels
        max_door_width: Maximum door width in pixels
        edge_margin: Margin from edges to ignore

    Returns:
        List of (x, y, width, height, orientation, gap_width) tuples
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Simple threshold - walls are dark, background is light
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Invert so walls are white, background is black
    binary_inv = cv2.bitwise_not(binary)

    h, w = binary_inv.shape

    # Find all gaps in horizontal scans
    all_gaps = []

    for y in range(edge_margin, h - edge_margin, 10):  # Sample every 10 rows
        row = binary_inv[y, :]

        # Find runs of zeros (gaps/openings)
        in_gap = False
        gap_start = 0
        prev_val = 255

        for x in range(w):
            val = row[x]

            # Detect start of gap
            if prev_val > 128 and val < 128:
                gap_start = x
                in_gap = True
            # Detect end of gap
            elif prev_val < 128 and val > 128 and in_gap:
                gap_width = x - gap_start

                # Check if gap is door-sized
                if min_door_width <= gap_width <= max_door_width:
                    # Check if gap is not at edges
                    if edge_margin < gap_start < w - edge_margin:
                        # Check if there are walls on both sides
                        check_dist = min(10, gap_start, w - x)
                        left_wall = np.mean(row[max(0, gap_start - check_dist):gap_start])
                        right_wall = np.mean(row[x:min(w, x + check_dist)])

                        if left_wall > 100 and right_wall > 100:
                            all_gaps.append((gap_start, y, gap_width))

                in_gap = False

            prev_val = val

    # Cluster gaps that are vertically aligned (same door)
    door_clusters = _cluster_vertical_gaps(all_gaps, x_tolerance=15, min_cluster_size=3)

    # Convert clusters to door detections
    doors = []
    for cluster in door_clusters:
        # Calculate average position and width
        x_positions = [g[0] for g in cluster]
        y_positions = [g[1] for g in cluster]
        widths = [g[2] for g in cluster]

        avg_x = int(np.mean(x_positions))
        min_y = min(y_positions)
        max_y = max(y_positions)
        avg_width = np.mean(widths)

        door_height = max_y - min_y

        # Only include if door spans reasonable height
        # But not too large (which would be a wall boundary)
        if 30 < door_height < 200:  # Doors should be 30-200 pixels tall
            doors.append((
                avg_x,
                min_y,
                int(avg_width),
                door_height,
                'vertical',
                avg_width
            ))

    # Also scan vertically for horizontal doors
    vertical_gaps = []

    for x in range(edge_margin, w - edge_margin, 10):  # Sample every 10 columns
        col = binary_inv[:, x]

        in_gap = False
        gap_start = 0
        prev_val = 255

        for y in range(h):
            val = col[y]

            if prev_val > 128 and val < 128:
                gap_start = y
                in_gap = True
            elif prev_val < 128 and val > 128 and in_gap:
                gap_height = y - gap_start

                if min_door_width <= gap_height <= max_door_width:
                    if edge_margin < gap_start < h - edge_margin:
                        check_dist = min(10, gap_start, h - y)
                        top_wall = np.mean(col[max(0, gap_start - check_dist):gap_start])
                        bottom_wall = np.mean(col[y:min(h, y + check_dist)])

                        if top_wall > 100 and bottom_wall > 100:
                            vertical_gaps.append((x, gap_start, gap_height))

                in_gap = False

            prev_val = val

    # Cluster horizontal doors
    h_door_clusters = _cluster_horizontal_gaps(vertical_gaps, y_tolerance=15, min_cluster_size=3)

    for cluster in h_door_clusters:
        x_positions = [g[0] for g in cluster]
        y_positions = [g[1] for g in cluster]
        heights = [g[2] for g in cluster]

        min_x = min(x_positions)
        max_x = max(x_positions)
        avg_y = int(np.mean(y_positions))
        avg_height = np.mean(heights)

        door_width = max_x - min_x

        # Only include if door spans reasonable width
        # But not too large (which would be a wall boundary)
        if 30 < door_width < 200:  # Doors should be 30-200 pixels wide
            doors.append((
                min_x,
                avg_y,
                door_width,
                int(avg_height),
                'horizontal',
                avg_height
            ))

    return doors


def _cluster_vertical_gaps(
    gaps: List[Tuple[int, int, int]],
    x_tolerance: int = 15,
    min_cluster_size: int = 3,
) -> List[List[Tuple[int, int, int]]]:
    """Cluster gaps that are vertically aligned.

    Args:
        gaps: List of (x, y, width) tuples
        x_tolerance: Maximum x difference to be in same cluster
        min_cluster_size: Minimum gaps to form a valid cluster

    Returns:
        List of gap clusters
    """
    if not gaps:
        return []

    # Sort by x position
    sorted_gaps = sorted(gaps, key=lambda g: g[0])

    clusters = []
    current_cluster = [sorted_gaps[0]]

    for gap in sorted_gaps[1:]:
        # Check if this gap is close to the cluster's average x
        cluster_avg_x = np.mean([g[0] for g in current_cluster])

        if abs(gap[0] - cluster_avg_x) <= x_tolerance:
            current_cluster.append(gap)
        else:
            # Start new cluster
            if len(current_cluster) >= min_cluster_size:
                clusters.append(current_cluster)
            current_cluster = [gap]

    # Don't forget last cluster
    if len(current_cluster) >= min_cluster_size:
        clusters.append(current_cluster)

    return clusters


def _cluster_horizontal_gaps(
    gaps: List[Tuple[int, int, int]],
    y_tolerance: int = 15,
    min_cluster_size: int = 3,
) -> List[List[Tuple[int, int, int]]]:
    """Cluster gaps that are horizontally aligned.

    Args:
        gaps: List of (x, y, height) tuples
        y_tolerance: Maximum y difference to be in same cluster
        min_cluster_size: Minimum gaps to form a valid cluster

    Returns:
        List of gap clusters
    """
    if not gaps:
        return []

    # Sort by y position
    sorted_gaps = sorted(gaps, key=lambda g: g[1])

    clusters = []
    current_cluster = [sorted_gaps[0]]

    for gap in sorted_gaps[1:]:
        cluster_avg_y = np.mean([g[1] for g in current_cluster])

        if abs(gap[1] - cluster_avg_y) <= y_tolerance:
            current_cluster.append(gap)
        else:
            if len(current_cluster) >= min_cluster_size:
                clusters.append(current_cluster)
            current_cluster = [gap]

    if len(current_cluster) >= min_cluster_size:
        clusters.append(current_cluster)

    return clusters
