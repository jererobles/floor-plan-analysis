"""Dimension extraction from floor plan images."""

import re
from typing import List, Optional, Tuple

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class DimensionMarking:
    """A dimension marking found on the floor plan."""

    text: str
    value_mm: float
    location: Tuple[int, int]
    orientation: str  # 'horizontal' or 'vertical'
    confidence: float


def try_ocr_extraction(image: np.ndarray) -> List[str]:
    """Attempt to extract text using OCR.

    Args:
        image: Input image

    Returns:
        List of extracted text strings
    """
    try:
        import pytesseract

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Enhance contrast for better OCR
        gray = cv2.equalizeHist(gray)

        # Extract text
        text = pytesseract.image_to_string(gray, config="--psm 6 digits")

        # Parse numbers
        numbers = re.findall(r"\d+", text)

        return numbers

    except ImportError:
        print("⚠️  pytesseract not installed. Install with: pip install pytesseract")
        return []
    except Exception as e:
        print(f"⚠️  OCR failed: {e}")
        return []


def find_dimension_lines(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Find dimension lines in the floor plan.

    Dimension lines are typically thin lines with arrows or endpoints.

    Args:
        image: Input image

    Returns:
        List of (x1, y1, x2, y2) line coordinates
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5
    )

    if lines is None:
        return []

    dimension_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Check if line is reasonably straight (horizontal or vertical)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # Horizontal or vertical lines
        if dx > 50 and dy < 10:  # Horizontal
            dimension_lines.append((x1, y1, x2, y2))
        elif dy > 50 and dx < 10:  # Vertical
            dimension_lines.append((x1, y1, x2, y2))

    return dimension_lines


def extract_dimension_from_roi(
    image: np.ndarray, roi: Tuple[int, int, int, int]
) -> Optional[str]:
    """Extract dimension text from a region of interest.

    Args:
        image: Input image
        roi: (x, y, width, height) region of interest

    Returns:
        Extracted dimension text or None
    """
    x, y, w, h = roi

    # Extract ROI
    roi_img = image[y : y + h, x : x + w]

    # Convert to grayscale
    if len(roi_img.shape) == 3:
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_img.copy()

    # Threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Try OCR
    try:
        import pytesseract

        text = pytesseract.image_to_string(
            binary, config="--psm 7 -c tessedit_char_whitelist=0123456789"
        )

        # Extract numbers
        numbers = re.findall(r"\d+", text)

        if numbers:
            return numbers[0]

    except ImportError:
        pass

    return None


def manual_dimension_locations(image_path: str) -> List[DimensionMarking]:
    """Manually identified dimension locations in the floor plan.

    These are dimensions visible in the floor plan that we've identified visually.

    Args:
        image_path: Path to the image (used to determine which markings apply)

    Returns:
        List of known dimension markings
    """
    # For the apartment unit floor plan, we can see:
    # 1. "180" marking which likely means 1800mm based on Finnish standards
    # 2. Various other dimensions on the full building plan

    # Note: These are manually identified from visual inspection
    # and will need to be validated/adjusted based on actual extraction

    markings = []

    # The "180" marking visible at the bottom of the cropped apartment unit
    # This likely refers to a corridor or room width of 1800mm
    if "apartment" in image_path.lower():
        markings.append(
            DimensionMarking(
                text="180",
                value_mm=1800.0,
                location=(0, 0),  # Will be determined by detection
                orientation="horizontal",
                confidence=0.9,  # High confidence this is 1800mm
            )
        )

    return markings


def find_dimension_text_locations(image: np.ndarray) -> List[Tuple[int, int, str]]:
    """Find locations where dimension text appears on the image.

    Args:
        image: Input image

    Returns:
        List of (x, y, text) tuples
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Find text regions using connected components
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_locations = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filter for text-like regions (small, rectangular)
        aspect_ratio = w / float(h) if h > 0 else 0

        if 20 < w < 200 and 10 < h < 50 and 0.5 < aspect_ratio < 5:
            # Try to extract text from this region
            roi = image[max(0, y - 5) : y + h + 5, max(0, x - 5) : x + w + 5]

            try:
                import pytesseract

                text = pytesseract.image_to_string(
                    roi, config="--psm 7 -c tessedit_char_whitelist=0123456789"
                )

                # Extract numbers
                numbers = re.findall(r"\d+", text.strip())

                if numbers and len(numbers[0]) >= 2:  # At least 2 digits
                    text_locations.append((x + w // 2, y + h // 2, numbers[0]))

            except (ImportError, Exception):
                pass

    return text_locations


def parse_dimension_value(text: str) -> float:
    """Parse a dimension text into millimeters.

    Finnish floor plans typically use millimeters without units.
    Common formats:
    - "180" → 1800mm (in centimeters, multiply by 10)
    - "1800" → 1800mm (already in millimeters)
    - "3400" → 3400mm

    Args:
        text: Dimension text

    Returns:
        Value in millimeters
    """
    # Remove any non-digit characters
    digits = re.sub(r"\D", "", text)

    if not digits:
        return 0.0

    value = float(digits)

    # Heuristic: if value is < 1000 and > 10, it's likely in centimeters
    # Finnish architectural drawings often use cm for readability
    if 10 <= value < 1000:
        return value * 10.0  # Convert cm to mm
    else:
        return value  # Already in mm


def visualize_dimension_detection(
    image: np.ndarray, dimensions: List[DimensionMarking]
) -> np.ndarray:
    """Visualize detected dimensions on the image.

    Args:
        image: Input image
        dimensions: List of detected dimensions

    Returns:
        Image with dimension annotations
    """
    vis = image.copy()

    for dim in dimensions:
        x, y = dim.location

        # Draw marker
        cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)

        # Draw text
        label = f"{dim.text} ({dim.value_mm:.0f}mm)"
        cv2.putText(vis, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return vis
