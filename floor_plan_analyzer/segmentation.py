"""Room segmentation from color-coded floor plans."""

from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage import measure

from .models import RoomInfo


# Actual room colors extracted from the annotated image
# These are RGB values from the scanned floor plan
ROOM_COLORS = {
    "OH": (127, 38, 142),       # Purple - living room (olohuone)
    "TYÖH": (157, 120, 71),     # Tan - work rooms (työhuone)
    "K": (220, 43, 3),          # Red - kitchen (keittiö)
    "MH": (32, 54, 250),        # Dark blue - bedroom (makuuhuone)
    "ET": (169, 210, 103),      # Light green - hallway (eteinen)
    "LASPI": (230, 146, 34),    # Orange - balcony (lasitettu parveke)
    "KPH": (76, 135, 251),      # Light blue - bathroom (kylpyhuone)
    "WC": (143, 253, 255),      # Cyan - toilet
}


def extract_color_mask(
    image: np.ndarray,
    target_color_bgr: Tuple[int, int, int],
    tolerance: int = 30,
) -> np.ndarray:
    """Extract a binary mask for a specific color.

    Args:
        image: Input image in BGR format
        target_color_bgr: Target color (B, G, R)
        tolerance: Color tolerance for matching

    Returns:
        Binary mask where 255 = color match, 0 = no match
    """
    # Convert target color to BGR (OpenCV format)
    b, g, r = target_color_bgr

    # Create mask for pixels within tolerance of target color
    lower = np.array([max(0, b - tolerance), max(0, g - tolerance), max(0, r - tolerance)])
    upper = np.array([min(255, b + tolerance), min(255, g + tolerance), min(255, r + tolerance)])

    mask = cv2.inRange(image, lower, upper)

    return mask


def clean_mask(mask: np.ndarray, min_area: int = 500) -> np.ndarray:
    """Clean up a binary mask by removing small artifacts.

    Args:
        mask: Binary mask
        min_area: Minimum area to keep

    Returns:
        Cleaned binary mask
    """
    # Remove small objects
    cleaned = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    cleaned = ndimage.binary_closing(cleaned, structure=np.ones((5, 5)))

    # Label connected components
    labeled, num_features = ndimage.label(cleaned)

    # Remove small components
    for i in range(1, num_features + 1):
        component_mask = labeled == i
        if np.sum(component_mask) < min_area:
            cleaned[component_mask] = 0

    return cleaned.astype(np.uint8) * 255


def segment_rooms(
    image: np.ndarray,
    room_colors: Dict[str, Tuple[int, int, int]] = None,
    tolerance: int = 30,
    min_area: int = 500,
) -> Dict[str, RoomInfo]:
    """Segment rooms from a color-coded floor plan.

    Args:
        image: Input image in BGR format
        room_colors: Dictionary mapping room names to RGB colors
        tolerance: Color matching tolerance
        min_area: Minimum room area in pixels

    Returns:
        Dictionary mapping room names to RoomInfo objects
    """
    if room_colors is None:
        room_colors = ROOM_COLORS

    rooms = {}

    for room_name, color_rgb in room_colors.items():
        # Convert RGB to BGR for OpenCV
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        # Extract color mask
        mask = extract_color_mask(image, color_bgr, tolerance)

        if np.sum(mask) == 0:
            continue  # No pixels found for this color

        # Clean the mask
        mask = clean_mask(mask, min_area)

        if np.sum(mask) == 0:
            continue  # Nothing left after cleaning

        # Calculate area
        pixel_area = float(np.sum(mask > 0))

        # Find contours for this room
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        # Count total contour points (useful for complexity estimation)
        total_points = sum(len(c) for c in contours)

        rooms[room_name] = RoomInfo(
            name=room_name,
            color_rgb=color_rgb,
            pixel_area=pixel_area,
            contour_points=total_points,
        )

    return rooms


def get_room_contours(
    image: np.ndarray,
    room_colors: Dict[str, Tuple[int, int, int]] = None,
    tolerance: int = 30,
) -> Dict[str, List[np.ndarray]]:
    """Get contours for each room.

    Args:
        image: Input image in BGR format
        room_colors: Dictionary mapping room names to RGB colors
        tolerance: Color matching tolerance

    Returns:
        Dictionary mapping room names to list of contours
    """
    if room_colors is None:
        room_colors = ROOM_COLORS

    room_contours = {}

    for room_name, color_rgb in room_colors.items():
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        mask = extract_color_mask(image, color_bgr, tolerance)

        if np.sum(mask) == 0:
            continue

        mask = clean_mask(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            room_contours[room_name] = contours

    return room_contours


def visualize_segmentation(
    image: np.ndarray,
    rooms: Dict[str, RoomInfo],
    room_contours: Dict[str, List[np.ndarray]] = None,
) -> np.ndarray:
    """Create a visualization of room segmentation.

    Args:
        image: Original image
        rooms: Dictionary of room information
        room_contours: Optional dictionary of room contours

    Returns:
        Visualization image
    """
    vis = image.copy()

    # Draw contours and labels
    for room_name, room_info in rooms.items():
        if room_contours and room_name in room_contours:
            color_bgr = (room_info.color_rgb[2], room_info.color_rgb[1], room_info.color_rgb[0])

            # Draw contours
            cv2.drawContours(vis, room_contours[room_name], -1, color_bgr, 2)

            # Find centroid for label placement
            contour = room_contours[room_name][0]
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw label
                cv2.putText(
                    vis,
                    room_name,
                    (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                )
                cv2.putText(
                    vis,
                    room_name,
                    (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

    return vis
