"""
Benchmark Report Manager
========================
Loads and serves the pre-computed benchmark report from data/benchmark_report.json.
The report is generated offline via `python src/run_benchmark.py`.
"""

import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BenchmarkReportManager:
    """Read-only manager for the permanent benchmark report."""

    def __init__(self, data_dir="data", report_file="benchmark_report.json"):
        self.report_path = os.path.join(data_dir, report_file)
        self.report = None
        self._load()

    def _load(self):
        if os.path.exists(self.report_path):
            try:
                with open(self.report_path, "r") as f:
                    self.report = json.load(f)
                logger.info(
                    f"Loaded benchmark report: {self.report['meta']['total_samples']} samples, "
                    f"generated {self.report['meta']['timestamp']}"
                )
            except Exception as e:
                logger.error(f"Failed to load benchmark report: {e}")
                self.report = None
        else:
            logger.warning(
                f"No benchmark report found at {self.report_path}. "
                f"Run `python src/run_benchmark.py` to generate one."
            )

    def get_status(self):
        if self.report is not None:
            return {
                "state": "completed",
                "total_samples": self.report["meta"]["total_samples"],
                "timestamp": self.report["meta"]["timestamp"],
                "device": self.report["meta"]["device"],
            }
        return {"state": "not_available"}

    def get_report(self):
        """Return the full benchmark report (or None)."""
        return self.report

    def get_summary(self):
        """Return a lightweight summary suitable for the frontend cards."""
        if self.report is None:
            return None

        models = {}
        for model_name, model_data in self.report["models"].items():
            agg = model_data["aggregate"]
            models[model_name] = {
                "iou": agg["iou"],
                "f1": agg["f1"],
                "precision": agg["precision"],
                "recall": agg["recall"],
                "latency_ms": agg["latency_ms"],
            }

        return {
            "meta": self.report["meta"],
            "winners": self.report["winners"],
            "models": models,
        }

    def get_histograms(self):
        """Return histogram data for all models."""
        if self.report is None:
            return None

        histograms = {}
        for model_name, model_data in self.report["models"].items():
            histograms[model_name] = {
                "iou": model_data.get("histogram_iou"),
                "f1": model_data.get("histogram_f1"),
            }
        return histograms
