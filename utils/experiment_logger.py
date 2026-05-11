"""
Experiment Logging
==================
Comprehensive experiment tracking and logging.

Design Rationale:
- Track all experiments with metadata
- Reproducible experiment management
- Support for MLflow (optional) and local logging
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class ExperimentLogger:
    """
    Simple experiment logger for tracking experiments.
    """

    def __init__(self, log_dir: str = "results/logs", experiment_name: Optional[str] = None):
        """
        Initialize experiment logger.

        Args:
            log_dir: Directory for logs
            experiment_name: Optional experiment name
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.experiment_name = experiment_name
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)

        self.metrics = {}
        self.config = {}
        self.artifacts = {}

    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.config.update(config)
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics."""
        if step is not None:
            if step not in self.metrics:
                self.metrics[step] = {}
            self.metrics[step].update(metrics)
        else:
            self.metrics.update(metrics)

        # Save metrics
        metrics_path = self.experiment_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)

    def log_artifact(self, name: str, artifact: Any):
        """Log artifact (model, figure, etc.)."""
        artifact_path = self.experiment_dir / f"{name}.pkl"

        if isinstance(artifact, (pd.DataFrame, np.ndarray)):
            # Save as pickle
            with open(artifact_path, 'wb') as f:
                pickle.dump(artifact, f)
        else:
            # Try to save as pickle
            try:
                with open(artifact_path, 'wb') as f:
                    pickle.dump(artifact, f)
            except Exception:
                # Save as text
                artifact_path = self.experiment_dir / f"{name}.txt"
                with open(artifact_path, 'w') as f:
                    f.write(str(artifact))

        self.artifacts[name] = str(artifact_path)

    def log_summary(self, summary: str):
        """Log text summary."""
        summary_path = self.experiment_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)

    def get_experiment_path(self) -> Path:
        """Get experiment directory path."""
        return self.experiment_dir
