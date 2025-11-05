"""Scale correction based on sanity checks."""

from typing import Optional, Tuple

import numpy as np

from .models import ScaleInfo


def apply_sanity_correction(
    scale_info: ScaleInfo,
    total_area_m2: float,
    expected_area_range: Optional[Tuple[float, float]] = None,
    door_count: Optional[int] = None,
) -> ScaleInfo:
    """Apply sanity corrections to scale estimation.

    This helps correct scale estimates that are clearly wrong based on
    contextual information like expected apartment size or door count.

    Args:
        scale_info: Initial scale estimation
        total_area_m2: Calculated total area in m²
        expected_area_range: Expected area range (min, max) in m²
        door_count: Number of doors detected

    Returns:
        Corrected ScaleInfo
    """
    if expected_area_range is None:
        # Default assumption for typical apartments
        expected_area_range = (50, 150)

    min_area, max_area = expected_area_range

    # Check if total area is within reasonable range
    if min_area <= total_area_m2 <= max_area:
        # Scale is good, no correction needed
        return scale_info

    # Calculate correction factor
    # Use middle of expected range as target
    target_area = (min_area + max_area) / 2

    # Scale correction is based on area ratio
    # Since area scales with scale², we need sqrt
    correction_factor = np.sqrt(target_area / total_area_m2)

    # Apply correction
    corrected_scale = scale_info.mm_per_pixel * correction_factor
    corrected_pixels_per_mm = 1.0 / corrected_scale

    # Reduce confidence since we had to apply a correction
    corrected_confidence = scale_info.confidence * 0.7

    # Add metadata about the correction
    metadata = scale_info.metadata.copy() if scale_info.metadata else {}
    metadata['correction_applied'] = True
    metadata['correction_factor'] = correction_factor
    metadata['original_scale_mm_per_px'] = scale_info.mm_per_pixel
    metadata['original_area_m2'] = total_area_m2
    metadata['target_area_m2'] = target_area

    return ScaleInfo(
        mm_per_pixel=corrected_scale,
        pixels_per_mm=corrected_pixels_per_mm,
        detected_features=scale_info.detected_features + ['corrected'],
        confidence=corrected_confidence,
        metadata=metadata,
    )


def estimate_expected_area_from_doors(door_count: int) -> Tuple[float, float]:
    """Estimate expected apartment area from door count.

    Rough heuristics:
    - Studio (1-3 doors): 20-40 m²
    - 1-bedroom (4-5 doors): 40-60 m²
    - 2-bedroom (6-7 doors): 60-90 m²
    - 3-bedroom (8-10 doors): 90-120 m²
    - 4+ bedroom (11+ doors): 120+ m²

    Args:
        door_count: Number of doors detected

    Returns:
        Tuple of (min_area, max_area) in m²
    """
    if door_count <= 3:
        return (20, 40)
    elif door_count <= 5:
        return (40, 60)
    elif door_count <= 7:
        return (60, 90)
    elif door_count <= 10:
        return (90, 120)
    else:
        return (120, 200)
