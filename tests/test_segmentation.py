"""Tests for segmentation module."""

import numpy as np
import pytest

from floor_plan_analyzer.segmentation import (
    clean_mask,
    extract_color_mask,
    segment_rooms,
)


def test_extract_color_mask_exact_match():
    """Test color mask extraction with exact color match."""
    # Create test image with red square
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    image[20:80, 20:80] = [0, 0, 255]  # BGR red

    # Extract red mask
    mask = extract_color_mask(image, (0, 0, 255), tolerance=10)

    # Check that red area is detected
    assert np.sum(mask[20:80, 20:80]) > 0
    # Check that white area is not detected
    assert np.sum(mask[0:10, 0:10]) == 0


def test_extract_color_mask_with_tolerance():
    """Test color mask extraction with tolerance."""
    # Create image with slightly off-red color
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    image[20:80, 20:80] = [10, 10, 245]  # Slightly off red

    # Should not match with low tolerance
    mask_strict = extract_color_mask(image, (0, 0, 255), tolerance=5)
    assert np.sum(mask_strict) == 0

    # Should match with higher tolerance
    mask_loose = extract_color_mask(image, (0, 0, 255), tolerance=20)
    assert np.sum(mask_loose) > 0


def test_clean_mask_removes_small_areas():
    """Test that mask cleaning removes small artifacts."""
    # Create mask with small noise
    mask = np.zeros((100, 100), dtype=np.uint8)

    # Large area (should be kept)
    mask[20:80, 20:80] = 255

    # Small noise (should be removed)
    mask[5:8, 5:8] = 255

    cleaned = clean_mask(mask, min_area=100)

    # Large area should remain
    assert np.sum(cleaned[20:80, 20:80]) > 0
    # Small area should be removed
    assert np.sum(cleaned[5:8, 5:8]) == 0


def test_segment_rooms_empty_image():
    """Test room segmentation on empty image."""
    image = np.ones((500, 500, 3), dtype=np.uint8) * 255

    rooms = segment_rooms(image)

    # Should find no rooms
    assert len(rooms) == 0


def test_segment_rooms_single_color():
    """Test room segmentation with single colored region."""
    image = np.ones((500, 500, 3), dtype=np.uint8) * 255

    # Add a red region (Kitchen)
    image[100:300, 100:300] = [0, 0, 255]  # BGR red

    room_colors = {"K": (255, 0, 0)}  # RGB red
    rooms = segment_rooms(image, room_colors=room_colors, min_area=1000)

    # Should find the kitchen
    assert "K" in rooms
    assert rooms["K"].pixel_area > 0


def test_segment_rooms_multiple_colors():
    """Test room segmentation with multiple colored regions."""
    image = np.ones((500, 500, 3), dtype=np.uint8) * 255

    # Add red region
    image[50:150, 50:150] = [0, 0, 255]  # BGR red
    # Add blue region
    image[200:300, 200:300] = [255, 0, 0]  # BGR blue

    room_colors = {
        "K": (255, 0, 0),  # RGB red
        "MH": (0, 0, 255),  # RGB blue
    }

    rooms = segment_rooms(image, room_colors=room_colors, min_area=1000)

    # Should find both rooms
    assert len(rooms) == 2
    assert "K" in rooms
    assert "MH" in rooms


def test_segment_rooms_respects_min_area():
    """Test that segmentation respects minimum area threshold."""
    image = np.ones((500, 500, 3), dtype=np.uint8) * 255

    # Add small red region
    image[10:20, 10:20] = [0, 0, 255]  # 100 pixels

    room_colors = {"K": (255, 0, 0)}
    rooms = segment_rooms(image, room_colors=room_colors, min_area=200)

    # Should not find room (too small)
    assert len(rooms) == 0


def test_segment_rooms_calculates_area():
    """Test that segmentation calculates pixel area correctly."""
    image = np.ones((500, 500, 3), dtype=np.uint8) * 255

    # Add 100x100 red square = 10000 pixels
    image[50:150, 50:150] = [0, 0, 255]

    room_colors = {"K": (255, 0, 0)}
    rooms = segment_rooms(image, room_colors=room_colors, min_area=1000)

    assert "K" in rooms
    # Should be approximately 10000 pixels
    assert 9500 < rooms["K"].pixel_area < 10500
