"""Data models for floor plan analysis."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class RoomInfo(BaseModel):
    """Information about a single room."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    color_rgb: Tuple[int, int, int]
    pixel_area: float
    area_m2: Optional[float] = None
    contour_points: Optional[int] = None


class ScaleInfo(BaseModel):
    """Scale information for the floor plan."""

    mm_per_pixel: float
    pixels_per_mm: float
    detected_features: list[str] = Field(default_factory=list)
    confidence: float = 0.0

    @property
    def m2_per_pixel2(self) -> float:
        """Square meters per square pixel."""
        return (self.mm_per_pixel / 1000.0) ** 2


class AnalysisResult(BaseModel):
    """Complete floor plan analysis result."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rooms: Dict[str, RoomInfo]
    scale_info: Optional[ScaleInfo] = None
    total_area_m2: Optional[float] = None
    unit_bounds: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height


@dataclass
class ProcessingParams:
    """Parameters for image processing."""

    yellow_lower_hsv: Tuple[int, int, int] = (20, 100, 100)
    yellow_upper_hsv: Tuple[int, int, int] = (30, 255, 255)
    min_contour_area: int = 1000
    door_width_mm_range: Tuple[float, float] = (700.0, 900.0)  # Finnish standard
    wall_thickness_mm: float = 150.0  # Typical wall thickness
