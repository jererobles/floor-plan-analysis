"""Experiment tracking and logging."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentLogger:
    """Logger for tracking what works and what doesn't."""

    def __init__(self, log_dir: str = "experiments"):
        """Initialize the logger.

        Args:
            log_dir: Directory for storing experiment logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.log_file = self.log_dir / "log.md"
        self.json_file = self.log_dir / "experiments.json"

        # Load existing experiments
        self.experiments: List[Dict[str, Any]] = []
        if self.json_file.exists():
            with open(self.json_file, "r") as f:
                self.experiments = json.load(f)

    def log_experiment(
        self,
        name: str,
        description: str,
        result: str,
        metrics: Optional[Dict[str, float]] = None,
        notes: Optional[str] = None,
        success: bool = True,
    ) -> None:
        """Log an experiment.

        Args:
            name: Experiment name
            description: What was tried
            result: What happened
            metrics: Numerical metrics
            notes: Additional notes
            success: Whether the experiment was successful
        """
        experiment = {
            "timestamp": datetime.now().isoformat(),
            "name": name,
            "description": description,
            "result": result,
            "metrics": metrics or {},
            "notes": notes or "",
            "success": success,
        }

        self.experiments.append(experiment)
        self._save_experiments()
        self._update_markdown()

    def _save_experiments(self) -> None:
        """Save experiments to JSON file."""
        with open(self.json_file, "w") as f:
            json.dump(self.experiments, f, indent=2)

    def _update_markdown(self) -> None:
        """Update the markdown log file."""
        with open(self.log_file, "w") as f:
            f.write("# Floor Plan Analysis - Experiment Log\n\n")
            f.write("This log tracks what works and what doesn't in the analysis pipeline.\n\n")

            # Summary statistics
            total = len(self.experiments)
            successful = sum(1 for e in self.experiments if e["success"])
            f.write(f"## Summary\n\n")
            f.write(f"- Total experiments: {total}\n")
            f.write(f"- Successful: {successful}\n")
            f.write(f"- Failed: {total - successful}\n")
            f.write(f"- Success rate: {successful/total*100:.1f}%\n\n" if total > 0 else "\n")

            # Experiments in reverse chronological order
            f.write("## Experiments\n\n")

            for exp in reversed(self.experiments):
                timestamp = datetime.fromisoformat(exp["timestamp"]).strftime("%Y-%m-%d %H:%M")
                status = "✅" if exp["success"] else "❌"

                f.write(f"### {status} {exp['name']} ({timestamp})\n\n")
                f.write(f"**Description:** {exp['description']}\n\n")
                f.write(f"**Result:** {exp['result']}\n\n")

                if exp["metrics"]:
                    f.write("**Metrics:**\n")
                    for key, value in exp["metrics"].items():
                        f.write(f"- {key}: {value}\n")
                    f.write("\n")

                if exp["notes"]:
                    f.write(f"**Notes:** {exp['notes']}\n\n")

                f.write("---\n\n")

    def get_successful_experiments(self) -> List[Dict[str, Any]]:
        """Get all successful experiments.

        Returns:
            List of successful experiments
        """
        return [e for e in self.experiments if e["success"]]

    def get_failed_experiments(self) -> List[Dict[str, Any]]:
        """Get all failed experiments.

        Returns:
            List of failed experiments
        """
        return [e for e in self.experiments if not e["success"]]

    def get_best_metrics(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get experiment with best value for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Experiment with best metric value or None
        """
        experiments_with_metric = [
            e for e in self.experiments if metric_name in e.get("metrics", {})
        ]

        if not experiments_with_metric:
            return None

        return max(experiments_with_metric, key=lambda e: e["metrics"][metric_name])
