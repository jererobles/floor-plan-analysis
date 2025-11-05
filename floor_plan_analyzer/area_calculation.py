"""Area calculation for floor plan rooms."""

from typing import Dict

from .models import RoomInfo, ScaleInfo


def calculate_room_areas(
    rooms: Dict[str, RoomInfo],
    scale_info: ScaleInfo,
) -> Dict[str, RoomInfo]:
    """Calculate real-world areas for rooms using scale information.

    Args:
        rooms: Dictionary of room information with pixel areas
        scale_info: Scale information for conversion

    Returns:
        Updated dictionary with area_m2 calculated
    """
    updated_rooms = {}

    for room_name, room_info in rooms.items():
        # Calculate area in square meters
        area_m2 = room_info.pixel_area * scale_info.m2_per_pixel2

        # Create updated room info
        updated_rooms[room_name] = RoomInfo(
            name=room_info.name,
            color_rgb=room_info.color_rgb,
            pixel_area=room_info.pixel_area,
            area_m2=area_m2,
            contour_points=room_info.contour_points,
        )

    return updated_rooms


def calculate_total_area(rooms: Dict[str, RoomInfo]) -> float:
    """Calculate total area from all rooms.

    Args:
        rooms: Dictionary of room information with areas

    Returns:
        Total area in square meters
    """
    total = 0.0
    for room_info in rooms.values():
        if room_info.area_m2 is not None:
            total += room_info.area_m2
    return total


def generate_area_report(rooms: Dict[str, RoomInfo], total_area: float) -> str:
    """Generate a formatted report of room areas.

    Args:
        rooms: Dictionary of room information
        total_area: Total apartment area

    Returns:
        Formatted report string
    """
    report_lines = [
        "=" * 50,
        "APARTMENT FLOOR PLAN ANALYSIS",
        "=" * 50,
        "",
    ]

    # Sort rooms by area (largest first)
    sorted_rooms = sorted(
        rooms.items(),
        key=lambda x: x[1].area_m2 if x[1].area_m2 else 0,
        reverse=True,
    )

    # Room details
    report_lines.append("ROOM AREAS:")
    report_lines.append("-" * 50)

    for room_name, room_info in sorted_rooms:
        if room_info.area_m2 is not None:
            percentage = (room_info.area_m2 / total_area * 100) if total_area > 0 else 0
            report_lines.append(
                f"{room_name:12s}: {room_info.area_m2:6.2f} m² ({percentage:5.1f}%)"
            )
        else:
            report_lines.append(f"{room_name:12s}: N/A")

    report_lines.extend(["", "-" * 50, f"{'TOTAL':12s}: {total_area:6.2f} m²", "=" * 50])

    return "\n".join(report_lines)


def get_room_statistics(rooms: Dict[str, RoomInfo]) -> Dict[str, float]:
    """Calculate statistics about room sizes.

    Args:
        rooms: Dictionary of room information

    Returns:
        Dictionary with statistics
    """
    areas = [r.area_m2 for r in rooms.values() if r.area_m2 is not None]

    if not areas:
        return {}

    import numpy as np

    return {
        "mean_area": float(np.mean(areas)),
        "median_area": float(np.median(areas)),
        "min_area": float(np.min(areas)),
        "max_area": float(np.max(areas)),
        "std_area": float(np.std(areas)),
        "total_area": float(np.sum(areas)),
        "num_rooms": len(areas),
    }
