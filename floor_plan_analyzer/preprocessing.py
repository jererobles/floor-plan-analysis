"""Image preprocessing for floor plan analysis."""

from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

from .models import ProcessingParams


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk.

    Args:
        image_path: Path to the image file

    Returns:
        Image as numpy array in BGR format
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    return img


def detect_yellow_perimeter(
    image: np.ndarray,
    params: Optional[ProcessingParams] = None,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """Detect the yellow perimeter marking the apartment unit.

    Args:
        image: Input image in BGR format
        params: Processing parameters

    Returns:
        Tuple of (binary mask of yellow perimeter, bounding box (x, y, w, h))
    """
    if params is None:
        params = ProcessingParams()

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create mask for yellow color
    yellow_mask = cv2.inRange(
        hsv,
        np.array(params.yellow_lower_hsv),
        np.array(params.yellow_upper_hsv),
    )

    # Clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    # Find the largest contour (should be the perimeter)
    largest_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest_contour) < params.min_contour_area:
        return None, None

    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Create a clean mask for the perimeter
    mask = np.zeros_like(yellow_mask)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)

    return mask, (x, y, w, h)


def crop_to_unit(
    image: np.ndarray,
    bounding_box: Tuple[int, int, int, int],
    padding: int = 10,
) -> np.ndarray:
    """Crop image to the unit's bounding box with optional padding.

    Args:
        image: Input image
        bounding_box: (x, y, width, height) of the unit
        padding: Pixels to add around the bounding box

    Returns:
        Cropped image
    """
    x, y, w, h = bounding_box
    h_img, w_img = image.shape[:2]

    # Apply padding while staying within image bounds
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)

    return image[y1:y2, x1:x2].copy()


def deskew_image(image: np.ndarray, threshold: float = 2.0) -> Tuple[np.ndarray, float]:
    """Detect and correct image skew.

    Args:
        image: Input image
        threshold: Minimum angle (degrees) to correct

    Returns:
        Tuple of (deskewed image, detected angle in degrees)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None:
        return image, 0.0

    # Find dominant angles (should be horizontal/vertical for floor plans)
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta)
        # Normalize to [-45, 45] range
        if angle > 135:
            angle = angle - 180
        elif angle > 45:
            angle = angle - 90
        angles.append(angle)

    # Calculate median angle
    median_angle = float(np.median(angles))

    # Only deskew if angle is significant
    if abs(median_angle) < threshold:
        return image, median_angle

    # Rotate image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)

    # Calculate new image size to avoid cropping
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust rotation matrix for new size
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    deskewed = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))

    return deskewed, median_angle


def preprocess_floor_plan(
    image_path: str,
    crop_to_yellow: bool = True,
    deskew: bool = True,
    params: Optional[ProcessingParams] = None,
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]], float]:
    """Complete preprocessing pipeline for floor plan images.

    Args:
        image_path: Path to the input image
        crop_to_yellow: Whether to crop to yellow perimeter
        deskew: Whether to deskew the image
        params: Processing parameters

    Returns:
        Tuple of (processed image, bounding box if cropped, skew angle)
    """
    # Load image
    image = load_image(image_path)

    bounding_box = None
    skew_angle = 0.0

    # Detect and crop to yellow perimeter
    if crop_to_yellow:
        _, bounding_box = detect_yellow_perimeter(image, params)
        if bounding_box is not None:
            image = crop_to_unit(image, bounding_box)

    # Deskew if requested
    if deskew:
        image, skew_angle = deskew_image(image)

    return image, bounding_box, skew_angle
