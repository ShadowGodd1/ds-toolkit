"""
ds_toolkit.core.profiler
========================
DataProfiler: one-call dataset summary.

Outputs per column:
  dtype, missing count/%, cardinality, min, max, mean, std,
  skewness, kurtosis, outlier flag (IQR / Z-score / both).

Usage
-----
>>> from ds_toolkit.core import DataProfiler
>>> profiler = DataProfiler(outlier_method="iqr", missing_threshold=0.05)
>>> result = profiler.profile(df)
>>> result.display()
>>> result.summary_df          # pandas DataFrame
>>> result.warnings            # list[str]
"""
# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

from typing import Literal, List

import numpy as np
import pandas as pd
from scipy import stats

from ds_toolkit.base import ProfileResult


class DataProfiler:
    """
    Stateless one-call dataset profiler.

    Parameters
    ----------
    cardinality_threshold : int
        Columns with unique count <= this are treated as categorical.
        Default 50.
    outlier_method : {'iqr', 'zscore', 'both'}
        Method used to flag outlier columns.
    missing_threshold : float
        Fraction above which a missing-% warning is raised (0–1).
    """

    def __init__(
        self,
        cardinality_threshold: int = 50,
        outlier_method: Literal["iqr", "zscore", "both"] = "iqr",
        missing_threshold: float = 0.05,
    ) -> None:
        self.cardinality_threshold = cardinality_threshold
        self.outlier_method = outlier_method
        self.missing_threshold = missing_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(self, df: pd.DataFrame) -> ProfileResult:
        """
        Profile every column in *df*.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        ProfileResult
            .summary_df  — one row per column, all metrics.
            .warnings    — list of human-readable warning strings.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

        rows = []
        for col in df.columns:
            rows.append(self._profile_column(df[col]))

        if not rows:
            summary_df = pd.DataFrame(
                columns=["dtype", "n", "missing_count", "missing_pct",
                         "cardinality", "inferred_role", "min", "max",
                         "mean", "std", "skew", "kurtosis", "outlier_flag"]
            )
            summary_df.index.name = "column"
        else:
            summary_df = pd.DataFrame(rows).set_index("column")
        warnings = self._collect_warnings(summary_df, df)
        return ProfileResult(summary_df=summary_df, warnings=warnings)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _profile_column(self, series: pd.Series) -> dict:
        n = len(series)
        n_missing = int(series.isna().sum())
        pct_missing = round(n_missing / n * 100, 2) if n > 0 else 0.0
        n_unique = int(series.nunique(dropna=True))
        dtype_str = str(series.dtype)

        row: dict = {
            "column": series.name,
            "dtype": dtype_str,
            "n": n,
            "missing_count": n_missing,
            "missing_pct": pct_missing,
            "cardinality": n_unique,
            "inferred_role": self._infer_role(series, n_unique),
        }

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            row.update(self._numeric_stats(clean))
            row["outlier_flag"] = self._detect_outliers(clean)
        else:
            for key in ("min", "max", "mean", "std", "skew", "kurtosis"):
                row[key] = np.nan
            row["outlier_flag"] = False

        return row

    def _infer_role(self, series: pd.Series, n_unique: int) -> str:
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_numeric_dtype(series):
            return "categorical" if n_unique <= self.cardinality_threshold else "numeric"
        return "categorical" if n_unique <= self.cardinality_threshold else "text"

    @staticmethod
    def _numeric_stats(clean: pd.Series) -> dict:
        if clean.empty:
            return {k: np.nan for k in ("min", "max", "mean", "std", "skew", "kurtosis")}
        return {
            "min": round(float(clean.min()), 4),
            "max": round(float(clean.max()), 4),
            "mean": round(float(clean.mean()), 4),
            "std": round(float(clean.std()), 4),
            "skew": round(float(stats.skew(clean)), 4),
            "kurtosis": round(float(stats.kurtosis(clean)), 4),
        }

    def _detect_outliers(self, clean: pd.Series) -> bool:
        """Return True if the column contains outliers by the chosen method."""
        if len(clean) < 4:
            return False

        # Cast to float — boolean series cannot do IQR subtraction arithmetic
        numeric = clean.astype(float)

        iqr_flag = zscore_flag = False

        if self.outlier_method in ("iqr", "both"):
            q1, q3 = float(numeric.quantile(0.25)), float(numeric.quantile(0.75))
            iqr = q3 - q1
            if iqr > 0:
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                iqr_flag = bool(((numeric < lower) | (numeric > upper)).any())

        if self.outlier_method in ("zscore", "both"):
            z = np.abs(stats.zscore(numeric))
            zscore_flag = bool((z > 3).any())

        if self.outlier_method == "both":
            return iqr_flag or zscore_flag
        if self.outlier_method == "iqr":
            return iqr_flag
        return zscore_flag

    def _collect_warnings(self, summary_df: pd.DataFrame, df: pd.DataFrame) -> List[str]:
        warnings: List[str] = []
        n_rows, n_cols = df.shape
        warnings.append(f"Shape: {n_rows:,} rows × {n_cols} columns "
                        f"({df.memory_usage(deep=True).sum() / 1024**2:.1f} MB)")

        threshold_pct = self.missing_threshold * 100
        high_missing = summary_df[summary_df["missing_pct"] > threshold_pct]
        for col, row in high_missing.iterrows():
            warnings.append(
                f"High missingness — '{col}': {row['missing_pct']}% missing "
                f"({int(row['missing_count'])} rows)"
            )

        outlier_cols = summary_df[summary_df["outlier_flag"] == True].index.tolist()
        if outlier_cols:
            warnings.append(
                f"Outlier flag set on {len(outlier_cols)} column(s): "
                + ", ".join(f"'{c}'" for c in outlier_cols)
            )

        constant_cols = summary_df[summary_df["cardinality"] <= 1].index.tolist()
        for col in constant_cols:
            warnings.append(f"Constant or near-constant column: '{col}'")

        return warnings
