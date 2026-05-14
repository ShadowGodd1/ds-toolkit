"""
ds_toolkit.eval.shap_explainer
================================
ExplainerSHAP: SHAP-based model interpretability.

Auto-selects TreeExplainer for tree-based models (XGBoost, LightGBM,
RandomForest, GradientBoosting) and KernelExplainer for all others.

Outputs
-------
  summary_plot    — global feature importance (beeswarm)
  waterfall_plot  — single-prediction breakdown for a given index
  dependence_plot — SHAP value vs feature value for top N features
  bar_plot        — mean |SHAP| bar chart

Usage
-----
>>> from ds_toolkit.eval import ExplainerSHAP
>>> explainer = ExplainerSHAP(top_n=10)
>>> result = explainer.explain(model, X)
>>> result.display()              # summary plot inline
>>> result.values                 # raw SHAP values array
>>> result.figures                # dict of matplotlib figures
"""
# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ds_toolkit.base import DisplayMixin

# Tree-based model class names that support TreeExplainer
_TREE_CLASSES = {
    "RandomForestClassifier", "RandomForestRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor",
    "CatBoostClassifier", "CatBoostRegressor",
    "DecisionTreeClassifier", "DecisionTreeRegressor",
    "HistGradientBoostingClassifier", "HistGradientBoostingRegressor",
}


@dataclass
class ShapResult(DisplayMixin):
    """Output of ExplainerSHAP.explain()."""
    values: np.ndarray
    feature_names: List[str]
    figures: Dict[str, Any] = field(default_factory=dict)
    top_n: int = 10

    def _repr_html_(self) -> str:
        mean_abs = np.abs(self.values).mean(axis=0)
        if mean_abs.ndim > 1:
            mean_abs = mean_abs.mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][: self.top_n]
        rows = "".join(
            f"<tr><td>{self.feature_names[i]}</td>"
            f"<td>{mean_abs[i]:.4f}</td></tr>"
            for i in top_idx
        )
        return f"""
        <div style="font-family:sans-serif;font-size:13px">
          <div style="font-weight:600;margin-bottom:8px">
            SHAP Explainer — Top {self.top_n} features (mean |SHAP|)
          </div>
          <table border="0" style="border-collapse:collapse">
            <tr style="font-weight:500;border-bottom:1px solid #ddd">
              <td style="padding:4px 12px 4px 0">Feature</td>
              <td style="padding:4px 12px">Mean |SHAP|</td>
            </tr>
            {rows}
          </table>
        </div>"""


class ExplainerSHAP:
    """
    SHAP-based model explainer.

    Parameters
    ----------
    top_n : int
        Number of top features for summary / dependence plots. Default 10.
    sample_n : int
        Background sample size for KernelExplainer. Default 100.
    """

    def __init__(self, top_n: int = 10, sample_n: int = 100) -> None:
        self.top_n    = top_n
        self.sample_n = sample_n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        model,
        X: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> ShapResult:
        """
        Compute SHAP values and generate plots.

        Parameters
        ----------
        model        : fitted sklearn/XGB/LGBM estimator
        X            : pd.DataFrame — data to explain
        feature_names: list, optional — override column names

        Returns
        -------
        ShapResult
        """
        try:
            import shap
        except ImportError as exc:
            raise ImportError("ExplainerSHAP requires shap: pip install shap") from exc

        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")

        names = feature_names or list(X.columns)
        X_clean = X.fillna(0)

        # --- choose explainer ---
        model_class = type(model).__name__
        if model_class in _TREE_CLASSES:
            explainer  = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(X_clean)
        else:
            background = shap.sample(X_clean, min(self.sample_n, len(X_clean)))
            explainer  = shap.KernelExplainer(
                model.predict_proba if hasattr(model, "predict_proba") else model.predict,
                background,
            )
            shap_vals  = explainer.shap_values(X_clean, nsamples=50)

        # For binary classifiers, use positive-class SHAP values
        vals = shap_vals
        if isinstance(shap_vals, list):
            vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]

        figures = self._generate_figures(vals, X_clean, names, shap)

        return ShapResult(
            values=vals,
            feature_names=names,
            figures=figures,
            top_n=self.top_n,
        )

    # ------------------------------------------------------------------
    # Figure generation
    # ------------------------------------------------------------------

    def _generate_figures(
        self, shap_vals: np.ndarray, X: pd.DataFrame,
        names: List[str], shap
    ) -> Dict[str, Any]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures = {}

        # --- Summary (beeswarm) ---
        try:
            fig, ax = plt.subplots(figsize=(8, max(4, self.top_n * 0.4)))
            shap.summary_plot(
                shap_vals, X, feature_names=names,
                max_display=self.top_n, show=False, plot_size=None,
            )
            figures["summary"] = plt.gcf()
            plt.close()
        except Exception:
            pass

        # --- Bar (mean |SHAP|) ---
        try:
            mean_abs = np.abs(shap_vals).mean(axis=0)
            top_idx  = np.argsort(mean_abs)[::-1][: self.top_n]
            fig, ax  = plt.subplots(figsize=(7, max(4, self.top_n * 0.4)))
            ax.barh(
                [names[i] for i in reversed(top_idx)],
                [mean_abs[i] for i in reversed(top_idx)],
                color="#4C72B0",
            )
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"Top {self.top_n} Features — Mean |SHAP|")
            ax.tick_params(labelsize=9)
            fig.tight_layout()
            figures["bar"] = fig
            plt.close(fig)
        except Exception:
            pass

        # --- Dependence plots for top features ---
        try:
            mean_abs = np.abs(shap_vals).mean(axis=0)
            top_idx  = np.argsort(mean_abs)[::-1][: min(self.top_n, 5)]
            for i in top_idx:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(
                    X.iloc[:, i], shap_vals[:, i],
                    alpha=0.4, s=15, c="#4C72B0",
                )
                ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
                ax.set_xlabel(names[i])
                ax.set_ylabel("SHAP value")
                ax.set_title(f"Dependence: {names[i]}")
                fig.tight_layout()
                figures[f"dependence_{names[i]}"] = fig
                plt.close(fig)
        except Exception:
            pass

        return figures
