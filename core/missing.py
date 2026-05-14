"""
ds_toolkit.core.missing
========================
MissingHandler: CV-safe per-column imputation strategy selector.

Strategies per column
---------------------
  mean     — numeric mean (default for numeric)
  median   — numeric median
  mode     — most frequent value (default for categorical/object)
  constant — fill with a fixed value
  knn      — KNNImputer (requires scikit-learn >= 1.0)
  mice     — IterativeImputer / MICE (requires scikit-learn >= 1.0)
  none     — leave column untouched

CV safety
---------
Always call .fit() on the training fold only, then .transform() on
validation / test folds. fit_transform() is provided as a shortcut for
the training fold and is equivalent to .fit().transform().

Usage
-----
>>> from ds_toolkit.core import MissingHandler
>>>
>>> # Global strategy for all columns
>>> handler = MissingHandler(strategy="median")
>>> X_train_clean = handler.fit_transform(X_train)
>>> X_val_clean   = handler.transform(X_val)
>>>
>>> # Per-column overrides
>>> handler = MissingHandler(
...     strategy="median",
...     col_strategies={"city": "mode", "note": "constant"},
...     fill_values={"note": "unknown"},
... )
>>> handler.fit(X_train)
>>> X_val_clean = handler.transform(X_val)
"""
# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer

Strategy = Literal["mean", "median", "mode", "constant", "knn", "mice", "none"]


class MissingHandler(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible per-column imputer.

    Parameters
    ----------
    strategy : str
        Global fallback strategy. One of:
        'mean', 'median', 'mode', 'constant', 'knn', 'mice', 'none'.
        Default 'median'.
    col_strategies : dict, optional
        Per-column strategy overrides: {"col_name": "strategy"}.
        Takes precedence over the global strategy.
    fill_values : dict, optional
        Per-column fill values used when strategy='constant'.
        Default fill value is 0 for numeric, 'missing' for others.
    knn_neighbors : int
        Number of neighbours for KNN imputation. Default 5.
    mice_max_iter : int
        Maximum iterations for MICE / IterativeImputer. Default 10.
    """

    def __init__(
        self,
        strategy: Strategy = "median",
        col_strategies: Optional[Dict[str, Strategy]] = None,
        fill_values: Optional[Dict[str, Any]] = None,
        knn_neighbors: int = 5,
        mice_max_iter: int = 10,
    ) -> None:
        self.strategy = strategy
        self.col_strategies = col_strategies or {}
        self.fill_values = fill_values or {}
        self.knn_neighbors = knn_neighbors
        self.mice_max_iter = mice_max_iter

        # Learned state — populated during fit()
        self._fill_map: Dict[str, Any] = {}          # col → scalar fill value
        self._knn_imputer: Optional[KNNImputer] = None
        self._mice_imputer = None
        self._knn_cols: list = []
        self._mice_cols: list = []
        self._fitted = False

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None) -> "MissingHandler":
        """
        Learn fill statistics from *X* (training data only).

        Parameters
        ----------
        X : pd.DataFrame
        y : ignored — present for sklearn pipeline compatibility.
        """
        self._validate_input(X)
        self._fill_map = {}
        self._knn_cols = []
        self._mice_cols = []

        for col in X.columns:
            strat = self._get_strategy(col, X[col])

            if strat == "none":
                continue

            if strat == "knn":
                if pd.api.types.is_numeric_dtype(X[col]):
                    self._knn_cols.append(col)
                else:
                    # Fall back to mode for non-numeric
                    self._fill_map[col] = self._compute_mode(X[col])

            elif strat == "mice":
                if pd.api.types.is_numeric_dtype(X[col]):
                    self._mice_cols.append(col)
                else:
                    self._fill_map[col] = self._compute_mode(X[col])

            elif strat == "mean":
                self._fill_map[col] = float(X[col].mean()) if pd.api.types.is_numeric_dtype(X[col]) else self._compute_mode(X[col])

            elif strat == "median":
                self._fill_map[col] = float(X[col].median()) if pd.api.types.is_numeric_dtype(X[col]) else self._compute_mode(X[col])

            elif strat == "mode":
                self._fill_map[col] = self._compute_mode(X[col])

            elif strat == "constant":
                default = 0 if pd.api.types.is_numeric_dtype(X[col]) else "missing"
                self._fill_map[col] = self.fill_values.get(col, default)

        # Fit KNN imputer on all knn columns jointly
        if self._knn_cols:
            self._knn_imputer = KNNImputer(n_neighbors=self.knn_neighbors)
            self._knn_imputer.fit(X[self._knn_cols])

        # Fit MICE imputer on all mice columns jointly
        if self._mice_cols:
            try:
                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                from sklearn.impute import IterativeImputer
                self._mice_imputer = IterativeImputer(
                    max_iter=self.mice_max_iter, random_state=42
                )
                self._mice_imputer.fit(X[self._mice_cols])
            except ImportError as exc:
                raise ImportError(
                    "MICE imputation requires scikit-learn >= 0.21 with "
                    "IterativeImputer enabled."
                ) from exc

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned imputation to *X*.

        Parameters
        ----------
        X : pd.DataFrame — may be training, validation, or test data.

        Returns
        -------
        pd.DataFrame — imputed copy (original not mutated).
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform().")
        self._validate_input(X)

        out = X.copy()

        # Scalar fills (mean / median / mode / constant)
        if self._fill_map:
            cols_present = {c: v for c, v in self._fill_map.items() if c in out.columns}
            out.fillna(cols_present, inplace=True)

        # KNN
        if self._knn_imputer and self._knn_cols:
            knn_cols_present = [c for c in self._knn_cols if c in out.columns]
            if knn_cols_present:
                imputed = self._knn_imputer.transform(out[knn_cols_present])
                out[knn_cols_present] = imputed

        # MICE
        if self._mice_imputer and self._mice_cols:
            mice_cols_present = [c for c in self._mice_cols if c in out.columns]
            if mice_cols_present:
                imputed = self._mice_imputer.transform(out[mice_cols_present])
                out[mice_cols_present] = imputed

        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_strategy(self, col: str, series: pd.Series) -> str:
        """Resolve the effective strategy for a column."""
        strat = self.col_strategies.get(col, self.strategy)
        # Auto-correct: mean/median on non-numeric → mode
        if strat in ("mean", "median") and not pd.api.types.is_numeric_dtype(series):
            return "mode"
        return strat

    @staticmethod
    def _compute_mode(series: pd.Series) -> Any:
        mode_result = series.mode(dropna=True)
        return mode_result.iloc[0] if not mode_result.empty else np.nan

    @staticmethod
    def _validate_input(X: Any) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")

    def missing_summary(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame summarising missingness in *X* before imputation.
        Useful for reporting / audit.
        """
        n = len(X)
        rows = []
        for col in X.columns:
            n_miss = int(X[col].isna().sum())
            rows.append({
                "column": col,
                "missing_count": n_miss,
                "missing_pct": round(n_miss / n * 100, 2) if n > 0 else 0.0,
                "strategy": self._get_strategy(col, X[col]) if self._fitted else "—",
            })
        return pd.DataFrame(rows).set_index("column")
