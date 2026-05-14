"""
ds_toolkit.eval.metrics
=========================
MetricsReport: auto-detecting classification and regression metrics.

Classification metrics
----------------------
  accuracy, precision (macro), recall (macro), f1 (macro),
  roc_auc (binary/macro-ovr), pr_auc, log_loss, mcc,
  per-class precision/recall/f1 (multi-class)

Regression metrics
------------------
  rmse, mae, mape, r2, adjusted_r2, max_error, explained_variance

Usage
-----
>>> from ds_toolkit.eval import MetricsReport
>>> reporter = MetricsReport()
>>> result = reporter.report(y_true, y_pred, y_proba=y_proba)
>>> result.display()
>>> result.metrics_df
>>> result.task          # 'clf' or 'reg'
"""
# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    explained_variance_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from ds_toolkit.base import DisplayMixin

Task = Literal["clf", "reg"]


@dataclass
class MetricsResult(DisplayMixin):
    """Output of MetricsReport.report()."""
    metrics_df: pd.DataFrame
    task: str

    def _repr_html_(self) -> str:
        task_label = "Classification" if self.task == "clf" else "Regression"
        return f"""
        <div style="font-family:sans-serif;font-size:13px">
          <div style="font-weight:600;margin-bottom:8px">
            Metrics Report — {task_label}
          </div>
          {self.metrics_df.to_html(border=0)}
        </div>"""


class MetricsReport:
    """
    Auto-detecting metrics reporter.

    Parameters
    ----------
    task : {'clf', 'reg'} or None
        If None (default), task is inferred from y_true:
        ≤ 20 unique float values or integer → clf, else → reg.
    average : str
        Averaging strategy for multi-class classification metrics.
        Default 'macro'.
    n_features : int, optional
        Number of features used in the model. Required for adjusted R².
    """

    def __init__(
        self,
        task: Optional[Task] = None,
        average: str = "macro",
        n_features: Optional[int] = None,
    ) -> None:
        self.task = task
        self.average = average
        self.n_features = n_features

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def report(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        task: Optional[Task] = None,
    ) -> MetricsResult:
        """
        Compute metrics for *y_true* vs *y_pred*.

        Parameters
        ----------
        y_true  : array-like — ground truth
        y_pred  : array-like — predicted labels (clf) or values (reg)
        y_proba : array-like, optional — predicted probabilities (clf only)
        task    : override instance task setting

        Returns
        -------
        MetricsResult
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        resolved_task = task or self.task or self._infer_task(y_true)

        if resolved_task == "clf":
            rows = self._clf_metrics(y_true, y_pred, y_proba)
        else:
            rows = self._reg_metrics(y_true, y_pred)

        metrics_df = pd.DataFrame(rows, columns=["metric", "value"]).set_index("metric")
        return MetricsResult(metrics_df=metrics_df, task=resolved_task)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _clf_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
    ) -> list:
        n_classes = len(np.unique(y_true))
        is_binary = n_classes == 2

        rows = [
            ("accuracy",  round(float(accuracy_score(y_true, y_pred)), 6)),
            ("precision", round(float(precision_score(y_true, y_pred, average=self.average, zero_division=0)), 6)),
            ("recall",    round(float(recall_score(y_true, y_pred, average=self.average, zero_division=0)), 6)),
            ("f1",        round(float(f1_score(y_true, y_pred, average=self.average, zero_division=0)), 6)),
            ("mcc",       round(float(matthews_corrcoef(y_true, y_pred)), 6)),
        ]

        if y_proba is not None:
            try:
                if is_binary:
                    proba_col = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                    rows.append(("roc_auc", round(float(roc_auc_score(y_true, proba_col)), 6)))
                    rows.append(("pr_auc",  round(float(average_precision_score(y_true, proba_col)), 6)))
                else:
                    rows.append(("roc_auc_ovr", round(float(
                        roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                    ), 6)))
                rows.append(("log_loss", round(float(log_loss(y_true, y_proba)), 6)))
            except Exception:
                pass

        return rows

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------

    def _reg_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> list:
        n = len(y_true)
        r2 = float(r2_score(y_true, y_pred))

        rows = [
            ("rmse",              round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6)),
            ("mae",               round(float(mean_absolute_error(y_true, y_pred)), 6)),
            ("r2",                round(r2, 6)),
            ("explained_variance",round(float(explained_variance_score(y_true, y_pred)), 6)),
            ("max_error",         round(float(max_error(y_true, y_pred)), 6)),
        ]

        # MAPE — only when no zeros in y_true
        if not np.any(y_true == 0):
            rows.insert(3, ("mape", round(float(mean_absolute_percentage_error(y_true, y_pred)), 6)))

        # Adjusted R²
        if self.n_features and n > self.n_features + 1:
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - self.n_features - 1)
            rows.insert(3, ("adjusted_r2", round(adj_r2, 6)))

        return rows

    # ------------------------------------------------------------------
    # Task inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_task(y: np.ndarray) -> str:
        if not np.issubdtype(y.dtype, np.number):
            return "clf"
        n_unique = len(np.unique(y))
        if n_unique <= 20 or np.issubdtype(y.dtype, np.integer):
            return "clf"
        return "reg"
