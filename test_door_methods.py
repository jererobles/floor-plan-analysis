"""Test different door detection methods."""

import cv2
import numpy as np
from floor_plan_analyzer.preprocessing import preprocess_floor_plan, binarize_adaptive

# Load image
image_path = "assets/floorplan-scanned-apartment-room-names.png"
processed_img, _, _ = preprocess_floor_plan(image_path, crop_to_yellow=True, deskew=True)

print(f"Image shape: {processed_img.shape}")

# Method 1: Simple binarization and gap finding
print("\n=== Method 1: Binary gap detection ===")
gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Invert so walls are white
binary_inv = cv2.bitwise_not(binary)

cv2.imwrite("outputs/test_binary.png", binary_inv)
print(f"Binary image saved")

# Count transitions in horizontal scans
doors_found = []
h, w = binary_inv.shape

print(f"\nScanning for gaps...")
for y in range(50, h - 50, 20):
    row = binary_inv[y, :]

    # Find runs of zeros (gaps)
    in_gap = False
    gap_start = 0
    prev_val = 255

    for x in range(w):
        val = row[x]

        # Detect transitions
        if prev_val > 128 and val < 128:  # Start of gap
            gap_start = x
            in_gap = True
        elif prev_val < 128 and val > 128 and in_gap:  # End of gap
            gap_width = x - gap_start

            if 20 <= gap_width <= 80:
                # Check if there's wall on both sides
                if gap_start > 10 and x < w - 10:
                    left_wall = np.mean(row[max(0, gap_start - 10):gap_start])
                    right_wall = np.mean(row[x:min(w, x + 10)])

                    if left_wall > 100 and right_wall > 100:
                        doors_found.append((gap_start, y, gap_width))
                        print(f"  Gap at ({gap_start}, {y}), width={gap_width}px")

            in_gap = False

        prev_val = val

print(f"\nFound {len(doors_found)} potential doors")

# Visualize
vis = processed_img.copy()
for gap_x, gap_y, gap_w in doors_found:
    cv2.rectangle(vis, (gap_x, gap_y - 5), (gap_x + gap_w, gap_y + 5), (0, 255, 0), 2)
    cv2.putText(vis, f"{gap_w}px", (gap_x, gap_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

cv2.putText(vis, f"Found {len(doors_found)} gaps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imwrite("outputs/test_simple_gaps.png", vis)
print(f"\nVisualization saved to outputs/test_simple_gaps.png")

# Method 2: Try with adaptive threshold
print("\n=== Method 2: Adaptive threshold ===")
binary_adaptive = binarize_adaptive(processed_img, block_size=15, c=5)
cv2.imwrite("outputs/test_adaptive.png", binary_adaptive)
print("Adaptive binary saved")

# Method 3: Edge detection
print("\n=== Method 3: Edge detection ===")
edges = cv2.Canny(gray, 50, 150)
cv2.imwrite("outputs/test_edges.png", edges)
print("Edges saved")

# Try to find characteristic door patterns
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=20, maxLineGap=10)

if lines is not None:
    print(f"Found {len(lines)} lines")

    line_vis = processed_img.copy()
    for line in lines[:100]:  # Show first 100
        x1, y1, x2, y2 = line[0]
        cv2.line(line_vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imwrite("outputs/test_lines.png", line_vis)
    print("Lines visualization saved")
else:
    print("No lines found")
