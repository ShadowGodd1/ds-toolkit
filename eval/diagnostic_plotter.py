"""
ds_toolkit.eval.diagnostic_plotter
=====================================
DiagnosticPlotter: one-call diagnostic figure suite.

Classification plots
---------------------
  confusion_matrix — heatmap, raw + normalised
  roc_curve        — with AUC annotation
  pr_curve         — precision-recall with AP annotation
  calibration      — reliability diagram (predicted vs actual probability)

Regression plots
-----------------
  residuals_vs_predicted — with zero line
  qq_plot                — quantile-quantile of residuals
  scale_location         — sqrt(|residuals|) vs fitted
  cooks_distance         — influence of each observation

Usage
-----
>>> from ds_toolkit.eval import DiagnosticPlotter
>>> plotter = DiagnosticPlotter()
>>> result = plotter.diagnostics(model, X, y)
>>> result.display()        # renders all figures inline
>>> result.figures          # dict of matplotlib figures
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
class DiagnosticResult(DisplayMixin):
    """Output of DiagnosticPlotter.diagnostics()."""
    figures: Dict[str, Any] = field(default_factory=dict)
    task: str = "clf"

    def _repr_html_(self) -> str:
        import base64, io
        parts = []
        for name, fig in self.figures.items():
            try:
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
                b64 = base64.b64encode(buf.getvalue()).decode()
                parts.append(
                    f'<div style="display:inline-block;margin:6px">'
                    f'<p style="font-size:11px;color:#666;margin:0 0 4px">{name}</p>'
                    f'<img src="data:image/png;base64,{b64}" '
                    f'style="max-width:360px;border:0.5px solid #ddd;border-radius:4px"/>'
                    f'</div>'
                )
            except Exception:
                pass
        return (
            '<div style="font-family:sans-serif">'
            '<div style="font-weight:600;font-size:13px;margin-bottom:10px">'
            f'Diagnostic Plots — {self.task.upper()}'
            f'  ({len(self.figures)} figures)</div>'
            + "".join(parts)
            + "</div>"
        )


class DiagnosticPlotter:
    """
    One-call diagnostic figure generator.

    Parameters
    ----------
    calibration_bins : int
        Number of bins for calibration curve. Default 10.
    """

    def __init__(self, calibration_bins: int = 10) -> None:
        self.calibration_bins = calibration_bins

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diagnostics(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        task: Optional[str] = None,
    ) -> DiagnosticResult:
        """
        Generate a full diagnostic figure suite.

        Parameters
        ----------
        model : fitted estimator
        X     : pd.DataFrame
        y     : pd.Series
        task  : 'clf' or 'reg'. Auto-detected if None.

        Returns
        -------
        DiagnosticResult
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")

        import matplotlib
        matplotlib.use("Agg")

        y_arr = np.asarray(y)
        resolved_task = task or self._infer_task(y_arr)

        if resolved_task == "clf":
            figures = self._clf_plots(model, X, y_arr)
        else:
            figures = self._reg_plots(model, X, y_arr)

        return DiagnosticResult(figures=figures, task=resolved_task)

    # ------------------------------------------------------------------
    # Classification plots
    # ------------------------------------------------------------------

    def _clf_plots(self, model, X, y) -> Dict[str, Any]:
        import matplotlib.pyplot as plt
        from sklearn.metrics import (
            ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
            confusion_matrix,
        )
        try:
            from sklearn.calibration import calibration_curve
        except ImportError:
            from sklearn.metrics import calibration_curve

        figures = {}
        y_pred  = model.predict(X)

        # Confusion matrix
        try:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for ax, normalize in zip(axes, [None, "true"]):
                cm = confusion_matrix(y, y_pred, normalize=normalize)
                disp = ConfusionMatrixDisplay(cm)
                disp.plot(ax=ax, colorbar=False, cmap="Blues")
                ax.set_title("Counts" if normalize is None else "Normalised")
            fig.suptitle("Confusion Matrix", fontsize=12, fontweight="bold")
            fig.tight_layout()
            figures["confusion_matrix"] = fig
            plt.close(fig)
        except Exception:
            pass

        # ROC curve (binary only)
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X)[:, 1]
                fig, ax = plt.subplots(figsize=(6, 5))
                RocCurveDisplay.from_predictions(y, y_proba, ax=ax)
                ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
                ax.set_title("ROC Curve")
                fig.tight_layout()
                figures["roc_curve"] = fig
                plt.close(fig)
        except Exception:
            pass

        # PR curve
        try:
            if hasattr(model, "predict_proba") and len(np.unique(y)) == 2:
                y_proba = model.predict_proba(X)[:, 1]
                fig, ax = plt.subplots(figsize=(6, 5))
                PrecisionRecallDisplay.from_predictions(y, y_proba, ax=ax)
                ax.set_title("Precision-Recall Curve")
                fig.tight_layout()
                figures["pr_curve"] = fig
                plt.close(fig)
        except Exception:
            pass

        # Calibration
        try:
            if hasattr(model, "predict_proba") and len(np.unique(y)) == 2:
                y_proba = model.predict_proba(X)[:, 1]
                prob_true, prob_pred = calibration_curve(
                    y, y_proba, n_bins=self.calibration_bins
                )
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(prob_pred, prob_true, "s-", label="Model", color="#4C72B0")
                ax.plot([0, 1], [0, 1], "k--", label="Perfect", linewidth=0.8)
                ax.set_xlabel("Mean predicted probability")
                ax.set_ylabel("Fraction of positives")
                ax.set_title("Calibration Curve")
                ax.legend()
                fig.tight_layout()
                figures["calibration"] = fig
                plt.close(fig)
        except Exception:
            pass

        return figures

    # ------------------------------------------------------------------
    # Regression plots
    # ------------------------------------------------------------------

    def _reg_plots(self, model, X, y) -> Dict[str, Any]:
        import matplotlib.pyplot as plt
        from scipy import stats

        figures = {}
        y_pred = model.predict(X)
        resid  = y - y_pred

        # Residuals vs Predicted
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y_pred, resid, alpha=0.4, s=15, color="#4C72B0")
            ax.axhline(0, color="red", linewidth=1, linestyle="--")
            ax.set_xlabel("Fitted values")
            ax.set_ylabel("Residuals")
            ax.set_title("Residuals vs Fitted")
            fig.tight_layout()
            figures["residuals_vs_fitted"] = fig
            plt.close(fig)
        except Exception:
            pass

        # Q-Q plot
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
            ax.scatter(osm, osr, alpha=0.5, s=15, color="#4C72B0")
            ax.plot(osm, slope * np.array(osm) + intercept, color="red", linewidth=1)
            ax.set_xlabel("Theoretical quantiles")
            ax.set_ylabel("Sample quantiles")
            ax.set_title("Q-Q Plot of Residuals")
            fig.tight_layout()
            figures["qq_plot"] = fig
            plt.close(fig)
        except Exception:
            pass

        # Scale-Location
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y_pred, np.sqrt(np.abs(resid)), alpha=0.4, s=15, color="#4C72B0")
            ax.set_xlabel("Fitted values")
            ax.set_ylabel("√|Residuals|")
            ax.set_title("Scale-Location")
            fig.tight_layout()
            figures["scale_location"] = fig
            plt.close(fig)
        except Exception:
            pass

        # Cook's distance (approximation via leverage * standardised residual)
        try:
            n, p = len(y), X.shape[1]
            mse    = float(np.mean(resid ** 2))
            std_r  = resid / (np.sqrt(mse) + 1e-10)
            # Approximate leverage as uniform 1/n (full hat matrix too expensive)
            h      = np.full(n, p / n)
            cooks  = (std_r ** 2 / p) * (h / (1 - h + 1e-10) ** 2)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.stem(range(n), cooks, markerfmt=",", linefmt="C0-", basefmt="k-")
            ax.set_xlabel("Observation index")
            ax.set_ylabel("Cook's distance")
            ax.set_title("Cook's Distance")
            fig.tight_layout()
            figures["cooks_distance"] = fig
            plt.close(fig)
        except Exception:
            pass

        return figures

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_task(y: np.ndarray) -> str:
        if not np.issubdtype(y.dtype, np.number):
            return "clf"
        n_unique = len(np.unique(y))
        return "clf" if (n_unique <= 20 or np.issubdtype(y.dtype, np.integer)) else "reg"
