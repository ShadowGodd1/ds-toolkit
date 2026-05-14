"""
ds_toolkit.core.typecaster
===========================
TypeCaster: infers and coerces column dtypes automatically.

What it does
------------
  1. Parses date-like string columns to datetime64.
  2. Downcasts int64 / float64 to the smallest safe numeric type.
  3. Converts low-cardinality object columns to pandas Categorical.
  4. Logs every change made so transformations are fully auditable.

Stateless — no fit step required. Safe to call on train, val, and test
independently (no learned state that could cause leakage).

Usage
-----
>>> from ds_toolkit.core import TypeCaster
>>> caster = TypeCaster(cardinality_threshold=50, downcast_numerics=True)
>>> df_typed = caster.cast(df)
>>> caster.change_log          # list of dicts — what changed and why
"""
# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

from typing import List, Optional

import pandas as pd


class TypeCaster:
    """
    Stateless automatic dtype coercion.

    Parameters
    ----------
    cardinality_threshold : int
        Object columns with unique count <= this value are converted to
        pandas Categorical. Default 50.
    downcast_numerics : bool
        If True, downcast int64 → smallest int, float64 → float32 where
        no precision is lost. Default True.
    parse_dates : bool
        If True, attempt to parse object columns that look like dates.
        Default True.
    date_cols : list, optional
        Explicit column names to parse as datetime regardless of content.
        These bypass the heuristic detection.
    infer_bool : bool
        If True, convert binary integer columns (0/1 only) to bool.
        Default False (conservative — explicit opt-in).
    """

    def __init__(
        self,
        cardinality_threshold: int = 50,
        downcast_numerics: bool = True,
        parse_dates: bool = True,
        date_cols: Optional[List[str]] = None,
        infer_bool: bool = False,
    ) -> None:
        self.cardinality_threshold = cardinality_threshold
        self.downcast_numerics = downcast_numerics
        self.parse_dates = parse_dates
        self.date_cols = date_cols or []
        self.infer_bool = infer_bool
        self.change_log: List[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cast(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Infer and coerce dtypes in *df*.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame — retyped copy (original is not mutated).
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

        self.change_log = []
        out = df.copy()

        for col in out.columns:
            original_dtype = str(out[col].dtype)
            out[col] = self._cast_column(col, out[col])
            new_dtype = str(out[col].dtype)
            if new_dtype != original_dtype:
                self.change_log.append({
                    "column": col,
                    "from": original_dtype,
                    "to": new_dtype,
                })

        return out

    def cast_report(self) -> pd.DataFrame:
        """
        Return a DataFrame summarising all dtype changes made in the last
        call to .cast(). Empty if no changes were made.
        """
        if not self.change_log:
            return pd.DataFrame(columns=["column", "from", "to"])
        return pd.DataFrame(self.change_log).set_index("column")

    # ------------------------------------------------------------------
    # Per-column casting logic
    # ------------------------------------------------------------------

    def _cast_column(self, col: str, series: pd.Series) -> pd.Series:
        dtype = series.dtype

        # --- explicit date columns ---
        if col in self.date_cols:
            return self._to_datetime(series)

        # --- already datetime, bool, category: leave alone ---
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return series
        if pd.api.types.is_bool_dtype(dtype):
            return series
        if str(dtype) == "category":
            return series

        # --- numeric: downcast ---
        if pd.api.types.is_integer_dtype(dtype):
            if self.infer_bool and self._is_binary(series):
                return series.astype(bool)
            if self.downcast_numerics:
                return pd.to_numeric(series, downcast="integer")
            return series

        if pd.api.types.is_float_dtype(dtype):
            if self.downcast_numerics:
                downcasted = pd.to_numeric(series, downcast="float")
                return downcasted
            return series

        # --- object / string columns ---
        if dtype == object or pd.api.types.is_string_dtype(dtype):
            # Try datetime parse
            if self.parse_dates and self._looks_like_date(series):
                parsed = self._to_datetime(series)
                if pd.api.types.is_datetime64_any_dtype(parsed):
                    return parsed

            # Try numeric parse
            numeric = self._try_numeric(series)
            if numeric is not None:
                if self.downcast_numerics:
                    return pd.to_numeric(numeric, downcast="float")
                return numeric

            # Low-cardinality → category
            n_unique = series.nunique(dropna=True)
            if n_unique <= self.cardinality_threshold:
                return series.astype("category")

        return series

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_date(series: pd.Series, sample_size: int = 50) -> bool:
        """Heuristic: sample non-null values and check if they parse as dates."""
        sample = series.dropna().head(sample_size)
        if sample.empty:
            return False
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            # At least 80% must parse successfully
            return (parsed.notna().sum() / len(sample)) >= 0.8
        except Exception:
            return False

    @staticmethod
    def _to_datetime(series: pd.Series) -> pd.Series:
        try:
            return pd.to_datetime(series, errors="coerce")
        except Exception:
            return series

    @staticmethod
    def _try_numeric(series: pd.Series) -> Optional[pd.Series]:
        """Attempt to parse an object column as numeric. Returns None on failure."""
        try:
            converted = pd.to_numeric(series, errors="coerce")
            # Require >= 95% successful conversion to avoid false positives
            if converted.notna().sum() / max(len(series), 1) >= 0.95:
                return converted
        except Exception:
            pass
        return None

    @staticmethod
    def _is_binary(series: pd.Series) -> bool:
        """Return True if a numeric column contains only 0 and 1 (ignoring NaN)."""
        unique_vals = set(series.dropna().unique())
        return unique_vals <= {0, 1}
