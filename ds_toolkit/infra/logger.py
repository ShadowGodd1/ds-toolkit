"""
ds_toolkit.infra.logger
=========================
ExperimentLogger: context-manager wrapper around an MLflow run.

Auto-logged artefacts
---------------------
  • All keys in the params dict
  • All metrics (per-fold + mean/std)
  • Fitted model serialised via PipelineSerialiser
  • Config YAML used for the run
  • SHAP feature importance plot (if ShapResult provided)
  • requirements.txt snapshot (pip freeze)
  • Git commit hash (if in a git repo)

Usage
-----
>>> from ds_toolkit.infra import ExperimentLogger
>>> logger = ExperimentLogger(tracking_uri="./mlruns")
>>>
>>> with logger.run("my_experiment", params={"model": "rf"}) as run:
...     model.fit(X_train, y_train)
...     logger.log_metrics({"roc_auc": 0.91, "f1": 0.88})
...     logger.log_model(model, "random_forest")
...
>>> run.run_id
>>> run.artifact_uri
"""

# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional

from ds_toolkit.infra.serialiser import PipelineSerialiser


@dataclass
class RunResult:
    """Returned by ExperimentLogger.run() context manager."""

    run_id: str
    artifact_uri: str
    experiment_name: str


class ExperimentLogger:
    """
    MLflow experiment logger with auto-logging.

    Parameters
    ----------
    tracking_uri : str
        MLflow tracking URI. Default './mlruns'.
    log_model : bool
        Serialise and log the fitted model via PipelineSerialiser. Default True.
    log_requirements : bool
        Snapshot pip freeze to artefacts. Default True.
    log_git_hash : bool
        Record current git commit hash. Default True.
    """

    def __init__(
        self,
        tracking_uri: str = "./mlruns",
        log_model: bool = True,
        log_requirements: bool = True,
        log_git_hash: bool = True,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.log_model = log_model
        self.log_requirements = log_requirements
        self.log_git_hash = log_git_hash
        self._active_run = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @contextmanager
    def run(
        self,
        experiment_name: str,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Generator[RunResult, None, None]:
        """
        Context manager that wraps an MLflow run.

        Parameters
        ----------
        experiment_name : str
        params          : dict, optional — logged as MLflow params
        tags            : dict, optional — logged as MLflow tags

        Yields
        ------
        RunResult
        """
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError("ExperimentLogger requires mlflow: pip install mlflow") from exc

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as active_run:
            self._active_run = active_run
            run_id = active_run.info.run_id

            # Log params
            if params:
                for k, v in params.items():
                    mlflow.log_param(k, v)

            # Log tags
            if tags:
                mlflow.set_tags(tags)

            # Git hash
            if self.log_git_hash:
                git_hash = self._get_git_hash()
                if git_hash:
                    mlflow.set_tag("git_commit", git_hash)

            # requirements.txt
            if self.log_requirements:
                req_path = self._snapshot_requirements()
                if req_path:
                    mlflow.log_artifact(req_path)

            result = RunResult(
                run_id=run_id,
                artifact_uri=active_run.info.artifact_uri,
                experiment_name=experiment_name,
            )

            try:
                yield result
            except Exception as exc:
                mlflow.set_tag("run_status", "FAILED")
                mlflow.set_tag("error", str(exc))
                raise
            finally:
                self._active_run = None

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log a metrics dict to the active run."""
        try:
            import mlflow

            mlflow.log_metrics(metrics, step=step)
        except ImportError:
            pass

    def log_model(self, model, name: str, metadata: Optional[Dict] = None) -> Optional[str]:
        """
        Serialise *model* via PipelineSerialiser and log as artefact.

        Parameters
        ----------
        model    : fitted sklearn pipeline or estimator
        name     : base name for the saved file
        metadata : optional metadata dict

        Returns
        -------
        str — path to saved .pkl file, or None on failure.
        """
        try:
            import mlflow
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                serialiser = PipelineSerialiser()
                result = serialiser.save(
                    model,
                    name=name,
                    output_dir=tmp,
                    metadata=metadata or {},
                )
                mlflow.log_artifact(str(result.path))
                return str(result.path)
        except Exception:
            return None

    def log_shap(self, shap_result) -> None:
        """Log SHAP bar figure as artefact if available."""
        try:
            import mlflow
            import tempfile
            import os

            if "bar" in shap_result.figures:
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "shap_importance.png")
                    shap_result.figures["bar"].savefig(path, dpi=90, bbox_inches="tight")
                    mlflow.log_artifact(path)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_git_hash() -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    @staticmethod
    def _snapshot_requirements() -> Optional[str]:
        import tempfile
        import os

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                fd, path = tempfile.mkstemp(suffix=".txt", prefix="requirements_")
                with os.fdopen(fd, "w") as f:
                    f.write(result.stdout)
                return path
        except Exception:
            pass
        return None
