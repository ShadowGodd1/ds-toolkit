"""
ds_toolkit.models.cv_harness
==============================
CVHarness: runs cross-validation over a list of models and returns
a ranked results DataFrame.

CV strategy auto-selection
---------------------------
  task='clf', balanced   → StratifiedKFold(n_splits=5)
  task='clf', imbalanced → StratifiedKFold + class_weight='balanced'
  task='reg'             → KFold(n_splits=5, shuffle=True)
  task='ts'              → TimeSeriesSplit(n_splits=5)

Output
------
CVResults.results_df columns:
  model, fold, score, fit_time_s
  + aggregated: mean_score, std_score, mean_fit_time

Usage
-----
>>> from ds_toolkit.models import ModelRegistry, CVHarness
>>> models = ModelRegistry.get(task="clf")
>>> harness = CVHarness(task="clf", n_splits=5, scoring="roc_auc")
>>> cv_results = harness.run(models, X_train, y_train)
>>> cv_results.results_df       # full fold-level results
>>> cv_results.summary_df       # one row per model, ranked by mean_score
>>> cv_results.best_model       # (name, fitted estimator)
>>> cv_results.display()        # Jupyter-native ranked table
"""

# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_validate,
)

from ds_toolkit.base import DisplayMixin

Task = Literal["clf", "reg", "ts"]


@dataclass
class CVResults(DisplayMixin):
    """Output of CVHarness.run()."""

    results_df: pd.DataFrame
    summary_df: pd.DataFrame
    best_model: Tuple[str, BaseEstimator]
    task: str
    scoring: str

    def _repr_html_(self) -> str:
        best_name, _ = self.best_model
        return f"""
        <div style="font-family:sans-serif;font-size:13px">
          <div style="font-weight:600;margin-bottom:8px">
            CV Results — {self.scoring} · {self.task.upper()}
            &nbsp;<span style="color:#2e7d32">★ Best: {best_name}</span>
          </div>
          {self.summary_df.to_html(border=0, float_format=lambda x: f"{x:.4f}")}
        </div>"""


class CVHarness:
    """
    Cross-validation harness for multiple models.

    Parameters
    ----------
    task : {'clf', 'reg', 'ts'}
        Task type. Controls CV strategy and default scoring.
    n_splits : int
        Number of CV folds. Default 5.
    scoring : str, optional
        sklearn scoring string. Defaults to 'roc_auc' for clf,
        'r2' for reg/ts.
    n_jobs : int
        Parallel jobs for cross_validate. Default -1.
    imbalance_threshold : float
        If the minority class proportion is below this value, apply
        class_weight='balanced'. Default 0.2.
    verbose : bool
        Print progress per model. Default True.
    """

    def __init__(
        self,
        task: Task = "clf",
        n_splits: int = 5,
        scoring: Optional[str] = None,
        n_jobs: int = -1,
        imbalance_threshold: float = 0.2,
        verbose: bool = True,
    ) -> None:
        self.task = task
        self.n_splits = n_splits
        self.scoring = scoring or ("roc_auc" if task == "clf" else "r2")
        self.n_jobs = n_jobs
        self.imbalance_threshold = imbalance_threshold
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        models: List[Tuple[str, BaseEstimator]],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> CVResults:
        """
        Run cross-validation for each model in *models*.

        Parameters
        ----------
        models : list of (name, estimator) tuples
        X      : pd.DataFrame — features (training data only)
        y      : pd.Series    — target

        Returns
        -------
        CVResults
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")
        if not isinstance(y, pd.Series):
            raise TypeError(f"Expected pd.Series for y, got {type(y).__name__}")

        cv = self._get_cv(y)
        fold_rows: List[dict] = []
        fitted_models: dict = {}

        for name, estimator in models:
            if self.verbose:
                print(f"  CV: {name} ...", end=" ", flush=True)

            est = clone(estimator)
            est = self._apply_class_weight(est, y)

            t0 = time.perf_counter()
            try:
                cv_out = cross_validate(
                    est,
                    X,
                    y,
                    cv=cv,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                    return_train_score=False,
                    error_score="raise",
                )
                scores = cv_out["test_score"]
                fit_times = cv_out["fit_time"]
                elapsed = time.perf_counter() - t0

                for fold_i, (score, ft) in enumerate(zip(scores, fit_times)):
                    fold_rows.append(
                        {
                            "model": name,
                            "fold": fold_i + 1,
                            "score": round(float(score), 6),
                            "fit_time_s": round(float(ft), 3),
                        }
                    )

                # Refit on full training data for the best model later
                est_full = clone(estimator)
                est_full = self._apply_class_weight(est_full, y)
                est_full.fit(X, y)
                fitted_models[name] = est_full

                if self.verbose:
                    print(
                        f"mean={np.mean(scores):.4f}  "
                        f"std={np.std(scores):.4f}  "
                        f"[{elapsed:.1f}s]"
                    )

            except Exception as exc:
                if self.verbose:
                    print(f"FAILED — {exc}")
                fold_rows.append(
                    {
                        "model": name,
                        "fold": -1,
                        "score": np.nan,
                        "fit_time_s": np.nan,
                    }
                )

        results_df = pd.DataFrame(fold_rows)
        summary_df = self._summarise(results_df)
        best_name = summary_df["model"].iloc[0]
        best_est = fitted_models.get(best_name, models[0][1])

        return CVResults(
            results_df=results_df,
            summary_df=summary_df,
            best_model=(best_name, best_est),
            task=self.task,
            scoring=self.scoring,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_cv(self, y: pd.Series):
        if self.task == "ts":
            return TimeSeriesSplit(n_splits=self.n_splits)
        if self.task == "reg":
            return KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

    def _apply_class_weight(self, est: BaseEstimator, y: pd.Series) -> BaseEstimator:
        """Set class_weight='balanced' if task is clf and data is imbalanced."""
        if self.task != "clf":
            return est
        minority_frac = y.value_counts(normalize=True).min()
        if minority_frac < self.imbalance_threshold:
            try:
                est.set_params(class_weight="balanced")
            except Exception:
                pass
        return est

    @staticmethod
    def _summarise(results_df: pd.DataFrame) -> pd.DataFrame:
        summary = (
            results_df[results_df["fold"] > 0]
            .groupby("model")
            .agg(
                mean_score=("score", "mean"),
                std_score=("score", "std"),
                mean_fit_time_s=("fit_time_s", "mean"),
                n_folds=("fold", "count"),
            )
            .reset_index()
            .sort_values("mean_score", ascending=False)
        )
        return summary
