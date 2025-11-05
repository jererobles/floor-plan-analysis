"""Main floor plan analyzer class."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .area_calculation import calculate_room_areas, calculate_total_area, generate_area_report
from .models import AnalysisResult, ProcessingParams
from .preprocessing import preprocess_floor_plan
from .scale_estimation import estimate_scale_multi_method
from .segmentation import get_room_contours, segment_rooms, visualize_segmentation


class FloorPlanAnalyzer:
    """Analyze apartment floor plans and calculate room areas."""

    def __init__(
        self,
        params: Optional[ProcessingParams] = None,
        output_dir: Optional[str] = None,
    ):
        """Initialize the analyzer.

        Args:
            params: Processing parameters
            output_dir: Directory for saving output visualizations
        """
        self.params = params or ProcessingParams()
        self.output_dir = Path(output_dir) if output_dir else Path("outputs")
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def analyze(
        self,
        image_path: str,
        crop_to_yellow: bool = True,
        deskew: bool = True,
        save_visualizations: bool = True,
    ) -> AnalysisResult:
        """Analyze a floor plan image.

        Args:
            image_path: Path to the floor plan image
            crop_to_yellow: Whether to crop to yellow perimeter
            deskew: Whether to deskew the image
            save_visualizations: Whether to save visualization outputs

        Returns:
            AnalysisResult with room information and areas
        """
        # Step 1: Preprocess image
        print(f"📸 Loading and preprocessing: {image_path}")
        processed_img, bounding_box, skew_angle = preprocess_floor_plan(
            image_path,
            crop_to_yellow=crop_to_yellow,
            deskew=deskew,
            params=self.params,
        )

        if skew_angle != 0:
            print(f"   Corrected skew: {skew_angle:.2f}°")

        if bounding_box:
            print(f"   Cropped to unit: {bounding_box}")

        # Step 2: Segment rooms
        print("🎨 Segmenting rooms...")
        rooms = segment_rooms(processed_img, tolerance=30, min_area=500)
        print(f"   Found {len(rooms)} rooms: {', '.join(rooms.keys())}")

        # Step 3: Estimate scale
        print("📏 Estimating scale...")
        # Save door visualization if we're saving visualizations
        door_vis_path = None
        if save_visualizations:
            base_name = Path(image_path).stem
            door_vis_path = str(self.output_dir / f"{base_name}_doors.png")

        scale_info = estimate_scale_multi_method(
            processed_img,
            self.params,
            door_visualization_path=door_vis_path
        )
        print(f"   Scale: {scale_info.mm_per_pixel:.4f} mm/pixel")
        print(f"   Method: {', '.join(scale_info.detected_features)}")
        print(f"   Confidence: {scale_info.confidence:.2f}")

        # Step 4: Calculate areas
        print("📐 Calculating areas...")
        rooms = calculate_room_areas(rooms, scale_info)
        total_area = calculate_total_area(rooms)
        print(f"   Total area: {total_area:.2f} m²")

        # Step 5: Create visualizations
        if save_visualizations:
            print("💾 Saving visualizations...")
            self._save_visualizations(
                processed_img, rooms, image_path, bounding_box, scale_info
            )

        # Create result
        result = AnalysisResult(
            rooms=rooms,
            scale_info=scale_info,
            total_area_m2=total_area,
            unit_bounds=bounding_box,
        )

        return result

    def _save_visualizations(
        self,
        image: np.ndarray,
        rooms: dict,
        image_path: str,
        bounding_box: Optional[tuple],
        scale_info,
    ) -> None:
        """Save visualization images.

        Args:
            image: Processed image
            rooms: Room information
            image_path: Original image path
            bounding_box: Bounding box if cropped
            scale_info: Scale information
        """
        base_name = Path(image_path).stem

        # 1. Save processed image
        processed_path = self.output_dir / f"{base_name}_processed.png"
        cv2.imwrite(str(processed_path), image)
        print(f"   ✓ Processed: {processed_path}")

        # 2. Save segmentation visualization
        room_contours = get_room_contours(image, tolerance=30)
        seg_vis = visualize_segmentation(image, rooms, room_contours)
        seg_path = self.output_dir / f"{base_name}_segmentation.png"
        cv2.imwrite(str(seg_path), seg_vis)
        print(f"   ✓ Segmentation: {seg_path}")

        # 3. Save area report as text overlay
        report_img = self._create_report_image(image, rooms, scale_info)
        report_path = self.output_dir / f"{base_name}_report.png"
        cv2.imwrite(str(report_path), report_img)
        print(f"   ✓ Report: {report_path}")

    def _create_report_image(
        self,
        image: np.ndarray,
        rooms: dict,
        scale_info,
    ) -> np.ndarray:
        """Create an image with area report overlay.

        Args:
            image: Input image
            rooms: Room information
            scale_info: Scale information

        Returns:
            Image with report text
        """
        # Create a white background for text
        h, w = image.shape[:2]
        text_width = 400
        combined = np.ones((h, w + text_width, 3), dtype=np.uint8) * 255

        # Place original image
        combined[:h, :w] = image

        # Add text
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 0, 0)
        thickness = 1

        # Title
        cv2.putText(
            combined,
            "ROOM AREAS",
            (w + 10, y_offset),
            font,
            0.6,
            color,
            2,
        )
        y_offset += 30

        # Sort rooms by area
        sorted_rooms = sorted(
            rooms.items(),
            key=lambda x: x[1].area_m2 if x[1].area_m2 else 0,
            reverse=True,
        )

        # Room areas
        for room_name, room_info in sorted_rooms:
            if room_info.area_m2 is not None:
                text = f"{room_name}: {room_info.area_m2:.1f} m2"
                cv2.putText(combined, text, (w + 10, y_offset), font, font_scale, color, thickness)
                y_offset += 25

        # Total
        total = sum(r.area_m2 for r in rooms.values() if r.area_m2)
        y_offset += 10
        cv2.putText(
            combined,
            f"Total: {total:.1f} m2",
            (w + 10, y_offset),
            font,
            0.6,
            color,
            2,
        )
        y_offset += 40

        # Scale info
        cv2.putText(combined, "SCALE INFO", (w + 10, y_offset), font, 0.6, color, 2)
        y_offset += 25
        cv2.putText(
            combined,
            f"{scale_info.mm_per_pixel:.3f} mm/px",
            (w + 10, y_offset),
            font,
            font_scale,
            color,
            thickness,
        )
        y_offset += 25
        cv2.putText(
            combined,
            f"Confidence: {scale_info.confidence:.2f}",
            (w + 10, y_offset),
            font,
            font_scale,
            color,
            thickness,
        )

        return combined

    def print_report(self, result: AnalysisResult) -> None:
        """Print a formatted report of the analysis.

        Args:
            result: Analysis result
        """
        report = generate_area_report(result.rooms, result.total_area_m2 or 0.0)
        print("\n" + report)
