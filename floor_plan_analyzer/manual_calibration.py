"""Manual calibration using known dimensions."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class CalibrationInfo:
    """Calibration information from known measurement."""
    pixel_distance: float
    real_distance_mm: float
    mm_per_pixel: float
    confidence: float = 1.0


def calculate_scale_from_dimension(
    pixel_width: float,
    real_width_mm: float,
) -> float:
    """Calculate mm-to-px ratio from a known dimension.

    Args:
        pixel_width: Width in pixels
        real_width_mm: Real width in millimeters

    Returns:
        mm_per_pixel ratio
    """
    return real_width_mm / pixel_width


def estimate_scale_from_visible_dimension(
    image: np.ndarray,
    expected_dimension_mm: float = 1800.0,
) -> Optional[CalibrationInfo]:
    """Estimate scale by detecting the "180" dimension text in the floor plan.

    The floor plan shows "180" which represents 1800mm (1.8m).
    We need to find this text and measure the line it refers to.

    Args:
        image: Input floor plan image
        expected_dimension_mm: The dimension in mm (default: 1800mm for "180")

    Returns:
        CalibrationInfo or None if dimension cannot be detected
    """
    # This is a simplified implementation
    # In a production system, we'd use OCR to find the text and measure the line

    # For now, return None - this will be implemented when needed
    return None


def apply_empirical_correction(
    calculated_area_m2: float,
    expected_area_m2: float,
    current_scale: float,
) -> float:
    """Calculate corrected scale based on known total area.

    If we know the total area should be a certain value, we can work backwards
    to find the correct scale.

    Args:
        calculated_area_m2: Area calculated with current scale
        expected_area_m2: Expected/known area
        current_scale: Current mm_per_pixel ratio

    Returns:
        Corrected mm_per_pixel ratio
    """
    # Area scales with scale^2
    # expected_area / calculated_area = (corrected_scale / current_scale)^2
    scale_ratio = np.sqrt(expected_area_m2 / calculated_area_m2)
    corrected_scale = current_scale * scale_ratio

    return corrected_scale
