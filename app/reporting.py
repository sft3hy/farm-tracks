import json
import os
import threading
import time
import torch
import numpy as np
import logging
from typing import Dict, Any, List
import cv2

logger = logging.getLogger(__name__)

class PerformanceReportManager:
    def __init__(self, data_dir="data", report_file="performance_report.json"):
        self.data_dir = data_dir
        self.report_path = os.path.join(data_dir, report_file)
        os.makedirs(data_dir, exist_ok=True)
        
        self.status = {
            "state": "idle", # idle, running, completed, error
            "progress": 0,
            "total": 0,
            "error": None,
            "last_run": None
        }
        self.results = None
        self._load_existing_report()

    def _load_existing_report(self):
        if os.path.exists(self.report_path):
            try:
                with open(self.report_path, "r") as f:
                    self.results = json.load(f)
                self.status["state"] = "completed"
                self.status["last_run"] = os.path.getmtime(self.report_path)
                logger.info(f"Loaded existing performance report from {self.report_path}")
            except Exception as e:
                logger.error(f"Failed to load existing report: {e}")

    def get_status(self):
        return self.status

    def get_results(self):
        return self.results

    def start_report_generation(self, index_list, get_model_func, infer_image_func):
        """Starts the report generation in a background thread if not already running."""
        if self.status["state"] == "running":
            return
        
        # If already completed, don't run again unless forced (not implemented)
        if self.status["state"] == "completed":
            return

        self.status["state"] = "running"
        self.status["progress"] = 0
        self.status["total"] = min(250, len(index_list))
        
        thread = threading.Thread(
            target=self._generate_report,
            args=(index_list[:self.status["total"]], get_model_func, infer_image_func),
            daemon=True
        )
        thread.start()
        logger.info(f"Started background performance report generation for {self.status['total']} samples")

    def _generate_report(self, subset, get_model_func, infer_image_func):
        try:
            models = ["unet", "segformer", "sam"]
            report = {
                "summary": {
                    "sample_count": len(subset),
                    "timestamp": time.time(),
                    "winners": {}
                },
                "metrics": {m: {
                    "mIoU": 0, 
                    "mF1": 0, 
                    "mPrecision": 0, 
                    "mRecall": 0, 
                    "avg_inference_ms": 0,
                    "samples": 0
                } for m in models}
            }

            for i, (file_id, _) in enumerate(subset):
                for m_name in models:
                    start_time = time.perf_counter()
                    try:
                        # We call the inference function directly
                        # Use await logic if it's async, but here we are in a thread
                        # Note: infer_image in server.py is async, we need to handle that.
                        import asyncio
                        inf = asyncio.run(infer_image_func(file_id, model=m_name))
                        
                        latency = (time.perf_counter() - start_time) * 1000
                        metrics = inf.get("metrics")
                        
                        if metrics:
                            m_stats = report["metrics"][m_name]
                            m_stats["mIoU"] += metrics.get("mIoU", 0)
                            m_stats["mF1"] += metrics.get("f1Score", 0)
                            m_stats["mPrecision"] += metrics.get("precision", 0) # Need to ensure backend returns these
                            m_stats["mRecall"] += metrics.get("recall", 0)
                            m_stats["avg_inference_ms"] += latency
                            m_stats["samples"] += 1
                    except Exception as e:
                        logger.error(f"Error evaluating {m_name} on {file_id}: {e}")

                self.status["progress"] = i + 1
            
            # Finalize averages
            for m_name in models:
                s = report["metrics"][m_name]["samples"]
                if s > 0:
                    for k in ["mIoU", "mF1", "mPrecision", "mRecall", "avg_inference_ms"]:
                        report["metrics"][m_name][k] = round(report["metrics"][m_name][k] / s, 4)

            # Determine winners
            for metric in ["mIoU", "mF1", "avg_inference_ms"]:
                if metric == "avg_inference_ms": # lower is better
                    best_val = float('inf')
                    winner = None
                    for m_name in models:
                        v = report["metrics"][m_name][metric]
                        if v < best_val and v > 0:
                            best_val = v
                            winner = m_name
                else:
                    best_val = -1
                    winner = None
                    for m_name in models:
                        v = report["metrics"][m_name][metric]
                        if v > best_val:
                            best_val = v
                            winner = m_name
                report["summary"]["winners"][metric] = winner

            self.results = report
            with open(self.report_path, "w") as f:
                json.dump(report, f, indent=2)
            
            self.status["state"] = "completed"
            self.status["last_run"] = time.time()
            logger.info("Performance report generation completed and saved.")

        except Exception as e:
            logger.error(f"Report generation failed: {e}", exc_info=True)
            self.status["state"] = "error"
            self.status["error"] = str(e)
