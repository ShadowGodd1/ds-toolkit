"""
ds_toolkit.eval.error_analyser
================================
ErrorAnalyser: segments worst-predicted samples by feature value and
flags cohorts with systematically high error (potential bias).

What it does
------------
1. Scores every prediction with an error metric (misclassification for clf,
   absolute residual for reg).
2. Identifies the top n_worst fraction of observations.
3. For each feature, computes the distribution of values among worst-
   predicted samples vs the rest and flags features with large distribution
   shift (potential sources of bias or systematic failure).

Usage
-----
>>> from ds_toolkit.eval import ErrorAnalyser
>>> analyser = ErrorAnalyser(n_worst=0.1)
>>> result = analyser.analyse(model, X, y)
>>> result.display()
>>> result.segments_df     # per-feature shift statistics
>>> result.worst_df        # the worst-predicted rows
"""

# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ds_toolkit.base import DisplayMixin


@dataclass
class ErrorReport(DisplayMixin):
    """Output of ErrorAnalyser.analyse()."""

    segments_df: pd.DataFrame
    worst_df: pd.DataFrame
    figures: Dict[str, Any] = field(default_factory=dict)
    task: str = "clf"
    n_worst: int = 0

    def _repr_html_(self) -> str:
        return f"""
        <div style="font-family:sans-serif;font-size:13px">
          <div style="font-weight:600;margin-bottom:8px">
            Error Analysis — {self.task.upper()}
            &nbsp;|&nbsp; {self.n_worst} worst-predicted samples
          </div>
          <div style="margin-bottom:8px;font-weight:500">Feature Distribution Shift</div>
          {self.segments_df.to_html(border=0, float_format=lambda x: f"{x:.4f}")}
        </div>"""


class ErrorAnalyser:
    """
    Segment worst-predicted samples and detect bias by feature.

    Parameters
    ----------
    n_worst : float
        Fraction of predictions to treat as 'worst'. Must be in (0, 1).
        Default 0.1 (top 10% by error).
    task : str, optional
        'clf' or 'reg'. Auto-detected if None.
    shift_threshold : float
        Minimum mean shift (numeric) or chi² p-value threshold (categorical)
        to flag a feature. Default 0.1.
    """

    def __init__(
        self,
        n_worst: float = 0.1,
        task: Optional[str] = None,
        shift_threshold: float = 0.1,
    ) -> None:
        if not 0 < n_worst < 1:
            raise ValueError(f"n_worst must be in (0, 1), got {n_worst}")
        self.n_worst = n_worst
        self.task = task
        self.shift_threshold = shift_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> ErrorReport:
        """
        Analyse worst predictions for systematic patterns.

        Parameters
        ----------
        model : fitted estimator
        X     : pd.DataFrame — features
        y     : pd.Series    — true labels/values

        Returns
        -------
        ErrorReport
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")

        y_arr = np.asarray(y)
        resolved_task = self.task or self._infer_task(y_arr)

        y_pred = model.predict(X)
        errors = self._compute_errors(y_arr, y_pred, resolved_task)
        n_worst = max(1, int(len(errors) * self.n_worst))

        worst_idx = np.argsort(errors)[::-1][:n_worst]
        rest_idx = np.argsort(errors)[::-1][n_worst:]

        worst_df = X.iloc[worst_idx].copy()
        worst_df["__error__"] = errors[worst_idx]

        segments_df = self._feature_shift(X, worst_idx, rest_idx)
        figures = self._error_figures(errors, X, worst_idx)

        return ErrorReport(
            segments_df=segments_df,
            worst_df=worst_df,
            figures=figures,
            task=resolved_task,
            n_worst=n_worst,
        )

    # ------------------------------------------------------------------
    # Error computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_errors(y_true, y_pred, task: str) -> np.ndarray:
        if task == "clf":
            return (y_true != y_pred).astype(float)
        return np.abs(y_true - y_pred)

    # ------------------------------------------------------------------
    # Feature shift analysis
    # ------------------------------------------------------------------

    def _feature_shift(
        self,
        X: pd.DataFrame,
        worst_idx: np.ndarray,
        rest_idx: np.ndarray,
    ) -> pd.DataFrame:
        rows = []
        worst = X.iloc[worst_idx]
        rest = X.iloc[rest_idx]

        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                w_mean = float(worst[col].mean())
                r_mean = float(rest[col].mean())
                r_std = float(rest[col].std()) or 1.0
                shift = abs(w_mean - r_mean) / r_std
                flagged = shift > self.shift_threshold
                rows.append(
                    {
                        "feature": col,
                        "type": "numeric",
                        "worst_mean": round(w_mean, 4),
                        "rest_mean": round(r_mean, 4),
                        "std_shift": round(shift, 4),
                        "flagged": flagged,
                    }
                )
            else:
                # Categorical: compare mode and proportion
                w_mode = worst[col].mode().iloc[0] if not worst[col].empty else None
                r_mode = rest[col].mode().iloc[0] if not rest[col].empty else None
                w_top_pct = float((worst[col] == w_mode).mean()) if w_mode else 0.0
                r_top_pct = float((rest[col] == r_mode).mean()) if r_mode else 0.0
                shift = abs(w_top_pct - r_top_pct)
                rows.append(
                    {
                        "feature": col,
                        "type": "categorical",
                        "worst_mean": w_top_pct,
                        "rest_mean": r_top_pct,
                        "std_shift": round(shift, 4),
                        "flagged": shift > self.shift_threshold,
                    }
                )

        df = pd.DataFrame(rows).set_index("feature")
        return df.sort_values("std_shift", ascending=False)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def _error_figures(
        self,
        errors: np.ndarray,
        X: pd.DataFrame,
        worst_idx: np.ndarray,
    ) -> Dict[str, Any]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures = {}

        # Error distribution
        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(errors, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.4)
            threshold = errors[np.sort(worst_idx)[-1]] if len(worst_idx) else errors.max()
            ax.axvline(
                threshold,
                color="red",
                linestyle="--",
                linewidth=1.2,
                label=f"Worst {self.n_worst:.0%} threshold",
            )
            ax.set_xlabel("Error")
            ax.set_ylabel("Count")
            ax.set_title("Error Distribution")
            ax.legend(fontsize=9)
            fig.tight_layout()
            figures["error_distribution"] = fig
            plt.close(fig)
        except Exception:
            pass

        # Worst vs rest feature means (numeric only)
        try:
            numeric_cols = X.select_dtypes(include="number").columns.tolist()[:10]
            if numeric_cols:
                worst_means = X.iloc[worst_idx][numeric_cols].mean()
                rest_idx = np.setdiff1d(np.arange(len(X)), worst_idx)
                rest_means = X.iloc[rest_idx][numeric_cols].mean()

                fig, ax = plt.subplots(figsize=(8, max(4, len(numeric_cols) * 0.5)))
                x = np.arange(len(numeric_cols))
                width = 0.35
                ax.barh(
                    x + width / 2,
                    rest_means.values,
                    width,
                    label="Rest",
                    color="#4C72B0",
                    alpha=0.8,
                )
                ax.barh(
                    x - width / 2,
                    worst_means.values,
                    width,
                    label="Worst",
                    color="#DD4444",
                    alpha=0.8,
                )
                ax.set_yticks(x)
                ax.set_yticklabels(numeric_cols, fontsize=9)
                ax.set_xlabel("Mean value")
                ax.set_title("Worst vs Rest — Feature Means")
                ax.legend()
                fig.tight_layout()
                figures["worst_vs_rest"] = fig
                plt.close(fig)
        except Exception:
            pass

        return figures

    # ------------------------------------------------------------------
    # Task inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_task(y: np.ndarray) -> str:
        if not np.issubdtype(y.dtype, np.number):
            return "clf"
        return "clf" if len(np.unique(y)) <= 20 else "reg"
