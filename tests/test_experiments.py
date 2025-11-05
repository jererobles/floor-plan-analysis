"""Tests for experiments module."""

import json
import tempfile
from pathlib import Path

import pytest

from floor_plan_analyzer.experiments import ExperimentLogger


def test_experiment_logger_creation():
    """Test creating an experiment logger."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(log_dir=tmpdir)

        assert logger.log_dir.exists()
        # Log file is only created when first experiment is logged
        assert logger.log_file.name == "log.md"


def test_log_experiment():
    """Test logging an experiment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(log_dir=tmpdir)

        logger.log_experiment(
            name="test_experiment",
            description="Testing the logger",
            result="Success",
            metrics={"accuracy": 0.95},
            notes="This is a test",
            success=True,
        )

        assert len(logger.experiments) == 1
        assert logger.experiments[0]["name"] == "test_experiment"
        assert logger.experiments[0]["success"] is True


def test_log_multiple_experiments():
    """Test logging multiple experiments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(log_dir=tmpdir)

        for i in range(3):
            logger.log_experiment(
                name=f"exp_{i}",
                description=f"Experiment {i}",
                result="Success",
                success=True,
            )

        assert len(logger.experiments) == 3


def test_get_successful_experiments():
    """Test filtering successful experiments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(log_dir=tmpdir)

        logger.log_experiment("exp1", "Test", "Success", success=True)
        logger.log_experiment("exp2", "Test", "Failed", success=False)
        logger.log_experiment("exp3", "Test", "Success", success=True)

        successful = logger.get_successful_experiments()

        assert len(successful) == 2
        assert all(e["success"] for e in successful)


def test_get_failed_experiments():
    """Test filtering failed experiments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(log_dir=tmpdir)

        logger.log_experiment("exp1", "Test", "Success", success=True)
        logger.log_experiment("exp2", "Test", "Failed", success=False)
        logger.log_experiment("exp3", "Test", "Failed", success=False)

        failed = logger.get_failed_experiments()

        assert len(failed) == 2
        assert all(not e["success"] for e in failed)


def test_get_best_metrics():
    """Test finding experiment with best metric."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(log_dir=tmpdir)

        logger.log_experiment("exp1", "Test", "Result", metrics={"score": 0.8}, success=True)
        logger.log_experiment("exp2", "Test", "Result", metrics={"score": 0.95}, success=True)
        logger.log_experiment("exp3", "Test", "Result", metrics={"score": 0.7}, success=True)

        best = logger.get_best_metrics("score")

        assert best is not None
        assert best["name"] == "exp2"
        assert best["metrics"]["score"] == 0.95


def test_get_best_metrics_missing():
    """Test getting best metric when metric doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(log_dir=tmpdir)

        logger.log_experiment("exp1", "Test", "Result", success=True)

        best = logger.get_best_metrics("nonexistent")

        assert best is None


def test_experiments_persisted():
    """Test that experiments are persisted to disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(log_dir=tmpdir)

        logger.log_experiment("exp1", "Test", "Result", success=True)

        # Create new logger with same directory
        logger2 = ExperimentLogger(log_dir=tmpdir)

        assert len(logger2.experiments) == 1
        assert logger2.experiments[0]["name"] == "exp1"


def test_markdown_log_created():
    """Test that markdown log file is created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(log_dir=tmpdir)

        logger.log_experiment("exp1", "Test", "Result", success=True)

        md_file = Path(tmpdir) / "log.md"
        assert md_file.exists()

        content = md_file.read_text()
        assert "exp1" in content
        assert "Test" in content
