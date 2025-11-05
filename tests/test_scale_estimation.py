"""Tests for scale estimation module."""

import numpy as np
import pytest

from floor_plan_analyzer.models import RoomInfo, ScaleInfo
from floor_plan_analyzer.scale_estimation import (
    estimate_scale_from_room_standards,
    estimate_scale_multi_method,
)


def test_estimate_scale_from_room_standards():
    """Test scale estimation using room size standards."""
    # Create mock rooms with known pixel areas
    rooms = {
        "KPH": RoomInfo(name="KPH", color_rgb=(76, 135, 251), pixel_area=130000.0),
        "WC": RoomInfo(name="WC", color_rgb=(143, 253, 255), pixel_area=84000.0),
        "K": RoomInfo(name="K", color_rgb=(220, 43, 3), pixel_area=307000.0),
    }

    # Create a dummy image
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255

    # Estimate scale
    scale_info = estimate_scale_from_room_standards(image, rooms)

    assert scale_info is not None
    assert scale_info.mm_per_pixel > 0
    assert 0.5 < scale_info.confidence <= 1.0
    assert len(scale_info.detected_features) > 0


def test_estimate_scale_from_room_standards_no_standard_rooms():
    """Test scale estimation when no standard rooms are present."""
    # Create rooms without standard rooms
    rooms = {
        "OH": RoomInfo(name="OH", color_rgb=(127, 38, 142), pixel_area=100000.0),
        "MH": RoomInfo(name="MH", color_rgb=(32, 54, 250), pixel_area=80000.0),
    }

    image = np.ones((100, 100, 3), dtype=np.uint8) * 255

    # Should return None as no standard rooms available
    scale_info = estimate_scale_from_room_standards(image, rooms)

    assert scale_info is None


def test_estimate_scale_from_room_standards_single_room():
    """Test scale estimation with only one standard room."""
    rooms = {
        "KPH": RoomInfo(name="KPH", color_rgb=(76, 135, 251), pixel_area=130000.0),
    }

    image = np.ones((100, 100, 3), dtype=np.uint8) * 255

    scale_info = estimate_scale_from_room_standards(image, rooms)

    assert scale_info is not None
    assert scale_info.mm_per_pixel > 0


def test_estimate_scale_from_room_standards_outlier_rejection():
    """Test that outliers are rejected in scale estimation."""
    # Create rooms where one has an outlier size
    rooms = {
        "KPH": RoomInfo(name="KPH", color_rgb=(76, 135, 251), pixel_area=130000.0),
        "WC": RoomInfo(name="WC", color_rgb=(143, 253, 255), pixel_area=1000.0),  # Outlier
        "K": RoomInfo(name="K", color_rgb=(220, 43, 3), pixel_area=307000.0),
    }

    image = np.ones((100, 100, 3), dtype=np.uint8) * 255

    scale_info = estimate_scale_from_room_standards(image, rooms)

    assert scale_info is not None
    # Should use KPH and K, rejecting WC as outlier
    assert "room_WC" not in scale_info.detected_features or len(scale_info.detected_features) == 2


def test_estimate_scale_multi_method_with_rooms():
    """Test multi-method scale estimation with rooms."""
    rooms = {
        "KPH": RoomInfo(name="KPH", color_rgb=(76, 135, 251), pixel_area=130000.0),
        "K": RoomInfo(name="K", color_rgb=(220, 43, 3), pixel_area=307000.0),
    }

    # Create a simple test image
    image = np.ones((500, 500, 3), dtype=np.uint8) * 255

    scale_info = estimate_scale_multi_method(image, params=None, rooms=rooms)

    assert scale_info is not None
    assert scale_info.mm_per_pixel > 0
    # Should use room standards method (highest confidence)
    assert any("room_" in f for f in scale_info.detected_features)


def test_estimate_scale_multi_method_without_rooms():
    """Test multi-method scale estimation without rooms."""
    image = np.ones((500, 500, 3), dtype=np.uint8) * 255

    scale_info = estimate_scale_multi_method(image, params=None, rooms=None)

    assert scale_info is not None
    assert scale_info.mm_per_pixel > 0
    # Should fallback to other methods or default


def test_scale_info_calculation():
    """Test that scale info correctly calculates derived values."""
    # Test case: 5 mm/pixel
    scale = ScaleInfo(
        mm_per_pixel=5.0,
        pixels_per_mm=0.2,
        detected_features=["test"],
        confidence=0.9,
    )

    # 5 mm/px = 0.005 m/px
    # 1 px² = 0.000025 m²
    assert scale.m2_per_pixel2 == pytest.approx(0.000025)


def test_realistic_scale_values():
    """Test scale estimation gives realistic values."""
    # Real pixel areas from our test
    rooms = {
        "KPH": RoomInfo(name="KPH", color_rgb=(76, 135, 251), pixel_area=130560.0),
        "WC": RoomInfo(name="WC", color_rgb=(143, 253, 255), pixel_area=83938.0),
        "K": RoomInfo(name="K", color_rgb=(220, 43, 3), pixel_area=307288.0),
    }

    image = np.ones((1000, 1000, 3), dtype=np.uint8) * 255

    scale_info = estimate_scale_from_room_standards(image, rooms)

    assert scale_info is not None

    # Scale should be reasonable (2-10 mm/pixel for scanned plans)
    assert 2.0 < scale_info.mm_per_pixel < 10.0

    # Check resulting room areas are reasonable
    kph_area_m2 = rooms["KPH"].pixel_area * scale_info.m2_per_pixel2
    assert 3.5 < kph_area_m2 < 5.5  # Bathroom should be 3.5-5.5 m²

    k_area_m2 = rooms["K"].pixel_area * scale_info.m2_per_pixel2
    assert 8.0 < k_area_m2 < 12.0  # Kitchen should be 8-12 m²
