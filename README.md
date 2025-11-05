# Floor Plan Analyzer

Analyze apartment floor plans and calculate room areas from unmarked diagrams.

## Features

- 🏠 Automatic unit extraction from building floor plans
- 🎨 Room segmentation using color-coded labels
- 📏 Scale estimation using standard Finnish building elements (doors)
- 📊 Accurate area calculations in square meters
- 🧪 Comprehensive test coverage

## Installation

```bash
# Install dependencies
pip install -e .

# Install with development tools
pip install -e ".[dev]"

# Install with ML capabilities (optional)
pip install -e ".[ml]"
```

## Usage

```python
from floor_plan_analyzer import FloorPlanAnalyzer

# Analyze a floor plan
analyzer = FloorPlanAnalyzer()
results = analyzer.analyze("assets/floorplan-scanned-apartment-room-names.png")

# Print room areas
for room, area in results.room_areas.items():
    print(f"{room}: {area:.2f} m²")
```

## Project Structure

```
floor-plan-analysis/
├── assets/                    # Input floor plan images
├── floor_plan_analyzer/       # Main package
│   ├── preprocessing.py       # Image preprocessing & cropping
│   ├── segmentation.py        # Room segmentation
│   ├── scale_estimation.py    # Scale detection using doors
│   ├── area_calculation.py    # Area computation
│   └── experiments.py         # Experiment tracking
├── tests/                     # Unit tests
├── experiments/               # Experiment logs
└── outputs/                   # Generated visualizations
```

## Approach

### 1. Scale Estimation
Finnish standard door dimensions:
- Interior doors: 800-900mm width
- Bathroom doors: 700-800mm width
- We detect door openings and use their known dimensions to calculate the mm-to-pixel ratio

### 2. Room Segmentation
- Extract color-coded room labels from annotated image
- Detect room boundaries using contour detection
- Map colors to room names

### 3. Area Calculation
- Calculate pixel area for each room
- Convert to square meters using estimated scale
- Account for any image warping

## Experiments Log

See `experiments/log.md` for detailed records of what works and what doesn't.

## Testing

```bash
pytest
```

## License

MIT
