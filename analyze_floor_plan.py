#!/usr/bin/env python3
"""CLI script for analyzing floor plans."""

import argparse
import sys
from pathlib import Path

from floor_plan_analyzer import FloorPlanAnalyzer
from floor_plan_analyzer.experiments import ExperimentLogger


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze apartment floor plans and calculate room areas"
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to floor plan image",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Don't crop to yellow perimeter",
    )
    parser.add_argument(
        "--no-deskew",
        action="store_true",
        help="Don't deskew the image",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for output visualizations (default: outputs)",
    )
    parser.add_argument(
        "--log-experiment",
        action="store_true",
        help="Log this run as an experiment",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Name for the experiment",
    )

    args = parser.parse_args()

    # Check if image exists
    if not Path(args.image).exists():
        print(f"❌ Error: Image file not found: {args.image}", file=sys.stderr)
        return 1

    try:
        # Create analyzer
        analyzer = FloorPlanAnalyzer(output_dir=args.output_dir)

        # Run analysis
        print(f"\n🏗️  Analyzing floor plan: {args.image}\n")
        result = analyzer.analyze(
            args.image,
            crop_to_yellow=not args.no_crop,
            deskew=not args.no_deskew,
            save_visualizations=True,
        )

        # Print report
        analyzer.print_report(result)

        # Log experiment if requested
        if args.log_experiment:
            logger = ExperimentLogger()
            experiment_name = args.experiment_name or Path(args.image).stem

            metrics = {
                "total_area_m2": result.total_area_m2 or 0.0,
                "num_rooms": len(result.rooms),
                "scale_confidence": result.scale_info.confidence if result.scale_info else 0.0,
            }

            logger.log_experiment(
                name=experiment_name,
                description=f"Analysis of {args.image}",
                result=f"Successfully analyzed {len(result.rooms)} rooms, "
                f"total area: {result.total_area_m2:.2f} m²",
                metrics=metrics,
                success=True,
            )
            print(f"\n📝 Experiment logged: {experiment_name}")

        print(f"\n✅ Analysis complete! Check {args.output_dir}/ for visualizations.\n")
        return 0

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()

        # Log failed experiment if requested
        if args.log_experiment:
            logger = ExperimentLogger()
            experiment_name = args.experiment_name or Path(args.image).stem
            logger.log_experiment(
                name=experiment_name,
                description=f"Analysis of {args.image}",
                result=f"Failed with error: {e}",
                success=False,
            )

        return 1


if __name__ == "__main__":
    sys.exit(main())
