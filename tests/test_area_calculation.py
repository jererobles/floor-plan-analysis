"""Tests for area calculation module."""

import pytest

from floor_plan_analyzer.area_calculation import (
    calculate_room_areas,
    calculate_total_area,
    get_room_statistics,
)
from floor_plan_analyzer.models import RoomInfo, ScaleInfo


def test_calculate_room_areas():
    """Test area calculation from pixel areas."""
    # Create test rooms
    rooms = {
        "OH": RoomInfo(name="OH", color_rgb=(128, 0, 128), pixel_area=10000.0),
        "K": RoomInfo(name="K", color_rgb=(255, 0, 0), pixel_area=5000.0),
    }

    # Create scale: 0.5 mm/px = 0.0005 m/px
    # So 1 px² = 0.00000025 m²
    scale = ScaleInfo(mm_per_pixel=0.5, pixels_per_mm=2.0, detected_features=[], confidence=1.0)

    # Calculate areas
    updated_rooms = calculate_room_areas(rooms, scale)

    # Check OH room: 10000 px² * 0.00000025 m²/px² = 0.0025 m²
    assert updated_rooms["OH"].area_m2 == pytest.approx(0.0025, rel=0.01)

    # Check K room: 5000 px² * 0.00000025 m²/px² = 0.00125 m²
    assert updated_rooms["K"].area_m2 == pytest.approx(0.00125, rel=0.01)


def test_calculate_total_area():
    """Test total area calculation."""
    rooms = {
        "OH": RoomInfo(name="OH", color_rgb=(128, 0, 128), pixel_area=10000.0, area_m2=25.0),
        "K": RoomInfo(name="K", color_rgb=(255, 0, 0), pixel_area=5000.0, area_m2=15.0),
        "MH": RoomInfo(name="MH", color_rgb=(0, 0, 255), pixel_area=8000.0, area_m2=20.0),
    }

    total = calculate_total_area(rooms)

    assert total == pytest.approx(60.0)


def test_calculate_total_area_with_none():
    """Test total area calculation when some rooms have None area."""
    rooms = {
        "OH": RoomInfo(name="OH", color_rgb=(128, 0, 128), pixel_area=10000.0, area_m2=25.0),
        "K": RoomInfo(name="K", color_rgb=(255, 0, 0), pixel_area=5000.0, area_m2=None),
    }

    total = calculate_total_area(rooms)

    # Should only count OH
    assert total == pytest.approx(25.0)


def test_get_room_statistics():
    """Test room statistics calculation."""
    rooms = {
        "OH": RoomInfo(name="OH", color_rgb=(128, 0, 128), pixel_area=10000.0, area_m2=25.0),
        "K": RoomInfo(name="K", color_rgb=(255, 0, 0), pixel_area=5000.0, area_m2=15.0),
        "MH": RoomInfo(name="MH", color_rgb=(0, 0, 255), pixel_area=8000.0, area_m2=20.0),
    }

    stats = get_room_statistics(rooms)

    assert stats["num_rooms"] == 3
    assert stats["total_area"] == pytest.approx(60.0)
    assert stats["mean_area"] == pytest.approx(20.0)
    assert stats["min_area"] == pytest.approx(15.0)
    assert stats["max_area"] == pytest.approx(25.0)


def test_get_room_statistics_empty():
    """Test room statistics with no rooms."""
    rooms = {}

    stats = get_room_statistics(rooms)

    assert stats == {}


def test_calculate_room_areas_preserves_other_fields():
    """Test that area calculation preserves other room fields."""
    rooms = {
        "OH": RoomInfo(
            name="OH",
            color_rgb=(128, 0, 128),
            pixel_area=10000.0,
            contour_points=150,
        ),
    }

    scale = ScaleInfo(mm_per_pixel=0.5, pixels_per_mm=2.0, detected_features=[], confidence=1.0)

    updated_rooms = calculate_room_areas(rooms, scale)

    # Check that other fields are preserved
    assert updated_rooms["OH"].name == "OH"
    assert updated_rooms["OH"].color_rgb == (128, 0, 128)
    assert updated_rooms["OH"].pixel_area == 10000.0
    assert updated_rooms["OH"].contour_points == 150
    assert updated_rooms["OH"].area_m2 is not None
