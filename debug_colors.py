#!/usr/bin/env python3
"""Debug script to extract actual colors from the floor plan."""

import cv2
import numpy as np
from collections import Counter

# Load the processed image
img = cv2.imread("outputs/floorplan-scanned-apartment-room-names_processed.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Get all unique colors
h, w, _ = img.shape
pixels = img_rgb.reshape(-1, 3)

# Count color frequency
color_counts = Counter([tuple(p) for p in pixels])

# Get most common colors (excluding white and black)
common_colors = []
for color, count in color_counts.most_common(100):
    if count > 1000:  # Significant area
        r, g, b = color
        # Skip near-white and near-black
        if not (r > 240 and g > 240 and b > 240) and not (r < 20 and g < 20 and b < 20):
            common_colors.append((color, count))

print("Top 15 colors by frequency (excluding white/black):")
print("RGB Color           | Pixel Count | Approx Area (px²)")
print("-" * 60)

for color, count in common_colors[:15]:
    r, g, b = color
    print(f"({r:3d}, {g:3d}, {b:3d})  |  {count:8d}  |  ~{count:6d}")
