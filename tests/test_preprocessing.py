"""Tests for preprocessing module."""

import numpy as np
import pytest

from floor_plan_analyzer.preprocessing import (
    crop_to_unit,
    detect_yellow_perimeter,
    deskew_image,
)


def test_crop_to_unit():
    """Test cropping to unit bounding box."""
    # Create a test image
    image = np.ones((1000, 1000, 3), dtype=np.uint8) * 255

    # Crop to a region
    bounding_box = (100, 200, 300, 400)  # x, y, w, h
    cropped = crop_to_unit(image, bounding_box, padding=0)

    assert cropped.shape == (400, 300, 3)


def test_crop_to_unit_with_padding():
    """Test cropping with padding."""
    image = np.ones((1000, 1000, 3), dtype=np.uint8) * 255

    bounding_box = (100, 200, 300, 400)
    padding = 10
    cropped = crop_to_unit(image, bounding_box, padding=padding)

    # Should be larger due to padding
    assert cropped.shape[0] == 400 + 2 * padding
    assert cropped.shape[1] == 300 + 2 * padding


def test_crop_to_unit_at_edge():
    """Test cropping at image edge doesn't go out of bounds."""
    image = np.ones((1000, 1000, 3), dtype=np.uint8) * 255

    # Bounding box at edge
    bounding_box = (0, 0, 100, 100)
    cropped = crop_to_unit(image, bounding_box, padding=50)

    # Should not exceed image bounds
    assert cropped.shape[0] <= 1000
    assert cropped.shape[1] <= 1000


def test_detect_yellow_perimeter_no_yellow():
    """Test yellow detection with no yellow present."""
    # Create an image with no yellow
    image = np.ones((500, 500, 3), dtype=np.uint8) * 255

    mask, bbox = detect_yellow_perimeter(image)

    assert mask is None
    assert bbox is None


def test_detect_yellow_perimeter_with_yellow():
    """Test yellow detection with yellow rectangle."""
    # Create an image with a yellow rectangle
    image = np.ones((500, 500, 3), dtype=np.uint8) * 255

    # Draw yellow rectangle (in BGR format)
    import cv2

    cv2.rectangle(image, (100, 100), (400, 400), (0, 255, 255), 3)

    mask, bbox = detect_yellow_perimeter(image)

    assert mask is not None
    assert bbox is not None

    x, y, w, h = bbox
    # Should roughly match our rectangle
    assert 90 < x < 110
    assert 90 < y < 110
    assert 290 < w < 310
    assert 290 < h < 310


def test_deskew_image_no_skew():
    """Test deskew on already straight image."""
    # Create a simple test image with horizontal and vertical lines
    image = np.ones((500, 500, 3), dtype=np.uint8) * 255

    import cv2

    # Draw horizontal and vertical lines
    cv2.line(image, (50, 250), (450, 250), (0, 0, 0), 2)
    cv2.line(image, (250, 50), (250, 450), (0, 0, 0), 2)

    deskewed, angle = deskew_image(image, threshold=2.0)

    # Angle should be small
    assert abs(angle) < 2.0


def test_deskew_image_dimensions():
    """Test that deskewed image has valid dimensions."""
    image = np.ones((500, 500, 3), dtype=np.uint8) * 255

    deskewed, angle = deskew_image(image)

    # Should return an image
    assert deskewed.shape[0] > 0
    assert deskewed.shape[1] > 0
    assert deskewed.shape[2] == 3


def test_crop_to_unit_preserves_dtype():
    """Test that cropping preserves image data type."""
    image = np.ones((1000, 1000, 3), dtype=np.uint8) * 128

    bounding_box = (100, 100, 200, 200)
    cropped = crop_to_unit(image, bounding_box)

    assert cropped.dtype == np.uint8
    assert np.all(cropped == 128)
