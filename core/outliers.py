"""
ds_toolkit.core.outliers
=========================
OutlierDetector: detect and handle outliers in numeric columns.

Detection methods
-----------------
  iqr       — Tukey IQR fence  (Q1 - k·IQR, Q3 + k·IQR)
  zscore    — Z-score threshold (default |z| > 3)
  isoforest — Isolation Forest (ensemble, handles multivariate)
  lof       — Local Outlier Factor (density-based)

Action per column
-----------------
  flag — add a new boolean column  <col>_outlier_flag
  cap  — winsorise to the fence / percentile bounds
  drop — drop rows where the column is an outlier

Usage
-----
>>> from ds_toolkit.core import OutlierDetector
>>> detector = OutlierDetector(method="iqr", action="flag", iqr_factor=1.5)
>>> result_df, report = detector.detect(df)
>>> report                        # pd.DataFrame — one row per column
>>>
>>> # Per-column action overrides
>>> detector = OutlierDetector(
...     method="iqr",
...     action="cap",
...     col_actions={"revenue": "drop", "age": "flag"},
... )
>>> result_df, report = detector.detect(df)
"""
# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

Action = Literal["flag", "cap", "drop"]
Method = Literal["iqr", "zscore", "isoforest", "lof"]


class OutlierDetector(BaseEstimator):
    """
    Detect and handle outliers in numeric columns.

    Parameters
    ----------
    method : {'iqr', 'zscore', 'isoforest', 'lof'}
        Detection method. Default 'iqr'.
    action : {'flag', 'cap', 'drop'}
        Default action to take when an outlier is found.
    col_actions : dict, optional
        Per-column action overrides: {"col_name": "action"}.
    iqr_factor : float
        Multiplier for IQR fence. Default 1.5 (Tukey). Use 3.0 for
        "far outliers".
    zscore_threshold : float
        Absolute Z-score threshold for 'zscore' method. Default 3.0.
    isoforest_contamination : float or 'auto'
        Expected proportion of outliers for Isolation Forest.
        Default 'auto'.
    lof_n_neighbors : int
        Number of neighbours for LOF. Default 20.
    cols : list, optional
        Columns to analyse. If None, all numeric columns are used.
    """

    def __init__(
        self,
        method: Method = "iqr",
        action: Action = "flag",
        col_actions: Optional[Dict[str, Action]] = None,
        iqr_factor: float = 1.5,
        zscore_threshold: float = 3.0,
        isoforest_contamination: float | str = "auto",
        lof_n_neighbors: int = 20,
        cols: Optional[list] = None,
    ) -> None:
        self.method = method
        self.action = action
        self.col_actions = col_actions or {}
        self.iqr_factor = iqr_factor
        self.zscore_threshold = zscore_threshold
        self.isoforest_contamination = isoforest_contamination
        self.lof_n_neighbors = lof_n_neighbors
        self.cols = cols

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect and handle outliers in *df*.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        result_df : pd.DataFrame
            DataFrame with outliers handled per the configured action.
        report_df : pd.DataFrame
            Summary table — one row per analysed column with:
            [column, method, n_outliers, pct_outliers, lower_bound, upper_bound, action]
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

        target_cols = self.cols or df.select_dtypes(include="number").columns.tolist()
        target_cols = [c for c in target_cols if c in df.columns]

        out = df.copy()
        report_rows = []

        # Multivariate methods — fit once on all target cols
        multi_mask: Optional[pd.Series] = None
        if self.method == "isoforest":
            multi_mask = self._isoforest_mask(df, target_cols)
        elif self.method == "lof":
            multi_mask = self._lof_mask(df, target_cols)

        # Drop-index accumulator (for 'drop' action — applied once at end)
        rows_to_drop: set = set()

        for col in target_cols:
            series = df[col].dropna()
            if series.empty:
                continue

            # --- compute outlier mask ---
            if self.method == "iqr":
                mask, lower, upper = self._iqr_mask(df[col])
            elif self.method == "zscore":
                mask, lower, upper = self._zscore_mask(df[col])
            else:
                # multivariate: same mask for every column
                mask = multi_mask.reindex(df.index, fill_value=False)
                lower, upper = np.nan, np.nan

            n_outliers = int(mask.sum())
            pct_outliers = round(n_outliers / len(df) * 100, 2)
            col_action = self.col_actions.get(col, self.action)

            # --- apply action ---
            if col_action == "flag":
                out[f"{col}_outlier_flag"] = mask.astype(bool)

            elif col_action == "cap":
                if not np.isnan(lower):
                    out[col] = out[col].clip(lower=lower, upper=upper)
                else:
                    # For multivariate methods, cap at 1st/99th percentile
                    p01 = float(series.quantile(0.01))
                    p99 = float(series.quantile(0.99))
                    out[col] = out[col].clip(lower=p01, upper=p99)
                    lower, upper = p01, p99

            elif col_action == "drop":
                rows_to_drop.update(mask[mask].index.tolist())

            report_rows.append({
                "column": col,
                "method": self.method,
                "n_outliers": n_outliers,
                "pct_outliers": pct_outliers,
                "lower_bound": round(lower, 4) if not np.isnan(lower) else None,
                "upper_bound": round(upper, 4) if not np.isnan(upper) else None,
                "action": col_action,
            })

        # Apply drops once (union of all drop-flagged rows)
        if rows_to_drop:
            out = out.drop(index=list(rows_to_drop)).reset_index(drop=True)

        report_df = pd.DataFrame(report_rows) if report_rows else pd.DataFrame(
            columns=["column", "method", "n_outliers", "pct_outliers",
                     "lower_bound", "upper_bound", "action"]
        )
        return out, report_df

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def _iqr_mask(
        self, series: pd.Series
    ) -> Tuple[pd.Series, float, float]:
        clean = series.dropna()
        q1, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - self.iqr_factor * iqr
        upper = q3 + self.iqr_factor * iqr
        mask = (series < lower) | (series > upper)
        mask = mask.fillna(False)
        return mask, lower, upper

    def _zscore_mask(
        self, series: pd.Series
    ) -> Tuple[pd.Series, float, float]:
        clean = series.dropna()
        mean, std = float(clean.mean()), float(clean.std())
        if std == 0:
            return pd.Series(False, index=series.index), mean, mean
        z = (series - mean) / std
        mask = z.abs() > self.zscore_threshold
        mask = mask.fillna(False)
        lower = mean - self.zscore_threshold * std
        upper = mean + self.zscore_threshold * std
        return mask, lower, upper

    def _isoforest_mask(
        self, df: pd.DataFrame, cols: list
    ) -> pd.Series:
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError as exc:
            raise ImportError("IsolationForest requires scikit-learn.") from exc

        subset = df[cols].select_dtypes(include="number").dropna()
        clf = IsolationForest(
            contamination=self.isoforest_contamination,
            random_state=42,
            n_jobs=-1,
        )
        preds = clf.fit_predict(subset)  # -1 = outlier, 1 = inlier
        mask = pd.Series(preds == -1, index=subset.index)
        return mask.reindex(df.index, fill_value=False)

    def _lof_mask(
        self, df: pd.DataFrame, cols: list
    ) -> pd.Series:
        try:
            from sklearn.neighbors import LocalOutlierFactor
        except ImportError as exc:
            raise ImportError("LOF requires scikit-learn.") from exc

        subset = df[cols].select_dtypes(include="number").dropna()
        clf = LocalOutlierFactor(n_neighbors=self.lof_n_neighbors)
        preds = clf.fit_predict(subset)  # -1 = outlier
        mask = pd.Series(preds == -1, index=subset.index)
        return mask.reindex(df.index, fill_value=False)
