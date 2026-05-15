"""
ds_toolkit.features.scaler
============================
Scaler: CV-safe numeric scaling with auto-detection of numeric columns.

Supported strategies
--------------------
  standard  — zero mean, unit variance (StandardScaler)
  minmax    — scale to [0, 1] range (MinMaxScaler)
  robust    — median and IQR based; outlier-resistant (RobustScaler)

CV safety
---------
Always call .fit() on the training fold only. The scaler learns the
statistics (mean/std, min/max, median/IQR) from training data and
applies the same transformation to validation and test data. This
prevents target leakage from distribution statistics.

Usage
-----
>>> from ds_toolkit.features import Scaler
>>> scaler = Scaler(method="standard")
>>> X_train_sc = scaler.fit_transform(X_train)
>>> X_val_sc   = scaler.transform(X_val)
>>> scaler.scaled_cols_       # list of columns that were scaled
>>> scaler.scaling_stats_     # pd.DataFrame — learned stats per column
"""

# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

from typing import List, Literal, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

Method = Literal["standard", "minmax", "robust"]


class Scaler(BaseEstimator, TransformerMixin):
    """
    CV-safe numeric scaler.

    Parameters
    ----------
    method : {'standard', 'minmax', 'robust'}
        Scaling strategy. Default 'standard'.
    cols : list, optional
        Columns to scale. If None, all numeric columns are scaled
        automatically. Boolean columns are always excluded.
    exclude_cols : list, optional
        Columns to explicitly exclude from scaling even if numeric.
        Useful for excluding encoded categoricals, target-encoded cols, etc.
    copy : bool
        If True (default), return a new DataFrame. If False, scale in-place.
    """

    _SCALER_MAP = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }

    def __init__(
        self,
        method: Method = "standard",
        cols: Optional[List[str]] = None,
        exclude_cols: Optional[List[str]] = None,
        copy: bool = True,
    ) -> None:
        if method not in self._SCALER_MAP:
            raise ValueError(f"method must be one of {list(self._SCALER_MAP)}, got '{method}'")
        self.method = method
        self.cols = cols
        self.exclude_cols = exclude_cols or []
        self.copy = copy

        self.scaled_cols_: List[str] = []
        self.scaling_stats_: pd.DataFrame = pd.DataFrame()
        self._scaler = None
        self._fitted = False

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None) -> "Scaler":
        """
        Learn scaling statistics from *X* (training data only).

        Parameters
        ----------
        X : pd.DataFrame
        y : ignored
        """
        self._validate(X)
        self.scaled_cols_ = self._resolve_cols(X)

        if not self.scaled_cols_:
            self._fitted = True
            return self

        self._scaler = self._SCALER_MAP[self.method]()
        self._scaler.fit(X[self.scaled_cols_])
        self.scaling_stats_ = self._build_stats(X[self.scaled_cols_])
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned scaling to *X*.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame — scaled copy (or in-place if copy=False).
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform().")
        self._validate(X)

        if not self.scaled_cols_ or self._scaler is None:
            return X.copy() if self.copy else X

        out = X.copy() if self.copy else X
        present = [c for c in self.scaled_cols_ if c in out.columns]

        if present:
            # Handle columns that were in fit but may be missing in transform
            scaled = self._scaler.transform(out[present])
            out[present] = scaled

        return out

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse the scaling transformation."""
        if not self._fitted:
            raise RuntimeError("Call .fit() before .inverse_transform().")
        self._validate(X)

        out = X.copy() if self.copy else X
        present = [c for c in self.scaled_cols_ if c in out.columns]
        if present and self._scaler is not None:
            out[present] = self._scaler.inverse_transform(out[present])
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_cols(self, X: pd.DataFrame) -> List[str]:
        exclude = set(self.exclude_cols)
        if self.cols is not None:
            return [
                c
                for c in self.cols
                if c in X.columns
                and pd.api.types.is_numeric_dtype(X[c])
                and not pd.api.types.is_bool_dtype(X[c])
                and c not in exclude
            ]
        return [
            c
            for c in X.select_dtypes(include="number").columns
            if not pd.api.types.is_bool_dtype(X[c]) and c not in exclude
        ]

    def _build_stats(self, X: pd.DataFrame) -> pd.DataFrame:
        """Build a human-readable stats table from the fitted scaler."""
        rows = []
        for i, col in enumerate(self.scaled_cols_):
            row = {"column": col, "method": self.method}
            scaler = self._scaler
            if self.method == "standard":
                row["center"] = round(float(scaler.mean_[i]), 4)
                row["scale"] = round(float(scaler.scale_[i]), 4)
            elif self.method == "minmax":
                row["center"] = round(float(scaler.data_min_[i]), 4)
                row["scale"] = round(float(scaler.data_range_[i]), 4)
            elif self.method == "robust":
                row["center"] = round(float(scaler.center_[i]), 4)
                row["scale"] = round(float(scaler.scale_[i]), 4)
            rows.append(row)
        return pd.DataFrame(rows).set_index("column")

    @staticmethod
    def _validate(X) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")
