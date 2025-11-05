"""Tests for data models."""

import pytest

from floor_plan_analyzer.models import ProcessingParams, RoomInfo, ScaleInfo


def test_room_info_creation():
    """Test RoomInfo model creation."""
    room = RoomInfo(
        name="OH",
        color_rgb=(128, 0, 128),
        pixel_area=5000.0,
        area_m2=25.0,
        contour_points=150,
    )

    assert room.name == "OH"
    assert room.color_rgb == (128, 0, 128)
    assert room.pixel_area == 5000.0
    assert room.area_m2 == 25.0
    assert room.contour_points == 150


def test_scale_info_creation():
    """Test ScaleInfo model creation."""
    scale = ScaleInfo(
        mm_per_pixel=0.5,
        pixels_per_mm=2.0,
        detected_features=["door_1", "door_2"],
        confidence=0.85,
    )

    assert scale.mm_per_pixel == 0.5
    assert scale.pixels_per_mm == 2.0
    assert len(scale.detected_features) == 2
    assert scale.confidence == 0.85


def test_scale_info_m2_per_pixel2():
    """Test m2_per_pixel2 property calculation."""
    scale = ScaleInfo(
        mm_per_pixel=1.0,
        pixels_per_mm=1.0,
        detected_features=[],
        confidence=1.0,
    )

    # 1 mm/px = 0.001 m/px, so 1 px² = 0.000001 m²
    assert scale.m2_per_pixel2 == pytest.approx(0.000001)


def test_processing_params_defaults():
    """Test ProcessingParams default values."""
    params = ProcessingParams()

    assert params.yellow_lower_hsv == (20, 100, 100)
    assert params.yellow_upper_hsv == (30, 255, 255)
    assert params.min_contour_area == 1000
    assert params.door_width_mm_range == (700.0, 900.0)
    assert params.wall_thickness_mm == 150.0


def test_processing_params_custom():
    """Test ProcessingParams with custom values."""
    params = ProcessingParams(
        min_contour_area=2000,
        door_width_mm_range=(800.0, 900.0),
    )

    assert params.min_contour_area == 2000
    assert params.door_width_mm_range == (800.0, 900.0)
