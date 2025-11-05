"""Tests for scale calibration and area calculations."""

import pytest
from pathlib import Path

from floor_plan_analyzer import FloorPlanAnalyzer


def test_apartment_total_area_in_expected_range():
    """Test that the calculated total area is in the expected range for the apartment."""
    analyzer = FloorPlanAnalyzer()
    test_image = "assets/floorplan-scanned-apartment-room-names.png"

    if not Path(test_image).exists():
        pytest.skip("Test image not found")

    result = analyzer.analyze(test_image, save_visualizations=False)

    # Expected area range: 70-90 m² (based on domain knowledge)
    assert result.total_area_m2 is not None
    assert 70.0 <= result.total_area_m2 <= 90.0, (
        f"Total area {result.total_area_m2:.1f} m² is outside expected range 70-90 m²"
    )


def test_door_count_validation():
    """Test that door detection finds the expected number of doors."""
    from floor_plan_analyzer.door_detection import detect_doors
    from floor_plan_analyzer.preprocessing import preprocess_floor_plan
    from floor_plan_analyzer.models import ProcessingParams

    test_image = "assets/floorplan-scanned-apartment-room-names.png"

    if not Path(test_image).exists():
        pytest.skip("Test image not found")

    # Preprocess
    processed_img, _, _ = preprocess_floor_plan(
        test_image,
        crop_to_yellow=True,
        deskew=True,
        params=ProcessingParams(),
    )

    # Detect doors
    doors = detect_doors(processed_img, min_width_px=10, max_width_px=50)

    # Expected: 7-9 doors (allowing for some detection variance, we test 5-12)
    assert 5 <= len(doors) <= 12, (
        f"Detected {len(doors)} doors, expected approximately 7-9"
    )


def test_room_proportions():
    """Test that room proportions are reasonable."""
    analyzer = FloorPlanAnalyzer()
    test_image = "assets/floorplan-scanned-apartment-room-names.png"

    if not Path(test_image).exists():
        pytest.skip("Test image not found")

    result = analyzer.analyze(test_image, save_visualizations=False)

    # OH (living room) should be the largest room
    oh_area = result.rooms["OH"].area_m2
    assert oh_area is not None

    for room_name, room_info in result.rooms.items():
        if room_name != "OH" and room_info.area_m2:
            assert oh_area >= room_info.area_m2, (
                f"OH (living room) should be largest, but {room_name} is larger"
            )

    # WC (toilet) should be one of the smallest rooms
    wc_area = result.rooms["WC"].area_m2
    assert wc_area is not None
    assert wc_area < oh_area / 3, "WC should be much smaller than OH"


def test_scale_confidence():
    """Test that scale estimation has reasonable confidence."""
    analyzer = FloorPlanAnalyzer()
    test_image = "assets/floorplan-scanned-apartment-room-names.png"

    if not Path(test_image).exists():
        pytest.skip("Test image not found")

    result = analyzer.analyze(test_image, save_visualizations=False)

    assert result.scale_info is not None
    assert result.scale_info.confidence >= 0.7, (
        f"Scale confidence {result.scale_info.confidence} is too low"
    )


def test_empirical_scale_value():
    """Test that empirical scale is in the expected range."""
    analyzer = FloorPlanAnalyzer()
    test_image = "assets/floorplan-scanned-apartment-room-names.png"

    if not Path(test_image).exists():
        pytest.skip("Test image not found")

    result = analyzer.analyze(test_image, save_visualizations=False)

    assert result.scale_info is not None
    # Empirical scale for Finnish floor plans: approximately 5-7 mm/pixel
    assert 5.0 <= result.scale_info.mm_per_pixel <= 7.0, (
        f"Scale {result.scale_info.mm_per_pixel} mm/px is outside expected range"
    )
