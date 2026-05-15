"""
ds_toolkit.features.datetime_decomposer
=========================================
DatetimeDecomposer: extracts structured features from datetime columns.

Extracted features per column
------------------------------
  Linear:    year, month, day, day_of_week, hour, minute,
             quarter, week_of_year, is_weekend, is_month_start,
             is_month_end, days_since_epoch

  Cyclical:  month_sin/cos, day_sin/cos, dow_sin/cos,
             hour_sin/cos (if hour varies)

  Optional:  is_holiday (requires the `holidays` package)

Cyclical encodings prevent the model from treating e.g. December (12)
as far from January (1) when they are adjacent in the year cycle.

Stateless — no fit step required.

Usage
-----
>>> from ds_toolkit.features import DatetimeDecomposer
>>> dt = DatetimeDecomposer(cols=["created_at", "updated_at"])
>>> df_expanded = dt.decompose(df)
>>>
>>> # Auto-detect datetime columns
>>> dt = DatetimeDecomposer()
>>> df_expanded = dt.decompose(df)
"""

# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np
import pandas as pd


class DatetimeDecomposer:
    """
    Stateless datetime feature extractor.

    Parameters
    ----------
    cols : list, optional
        Datetime column names to decompose. If None, all datetime64
        columns are auto-detected, and object columns that parse as
        datetime are included too.
    drop_original : bool
        Drop the original datetime column after decomposition. Default True.
    cyclical : bool
        Add sin/cos cyclical encodings for periodic features. Default True.
    add_holidays : bool
        Add is_holiday flag. Requires: pip install holidays. Default False.
    country_code : str
        ISO country code for holiday calendar. Default 'KE' (Kenya).
    epoch : str
        Reference date for days_since_epoch. Default '2000-01-01'.
    """

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        drop_original: bool = True,
        cyclical: bool = True,
        add_holidays: bool = False,
        country_code: str = "KE",
        epoch: str = "2000-01-01",
    ) -> None:
        self.cols = cols
        self.drop_original = drop_original
        self.cyclical = cyclical
        self.add_holidays = add_holidays
        self.country_code = country_code
        self.epoch = pd.Timestamp(epoch)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract datetime features from *df*.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame — original columns + extracted features.
                        Original datetime column dropped if drop_original=True.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

        target_cols = self._resolve_cols(df)
        if not target_cols:
            return df.copy()

        out = df.copy()

        for col in target_cols:
            dt_series = self._to_datetime(out[col], col)
            if dt_series is None:
                continue

            prefix = self._safe_prefix(col)
            new_cols = self._extract(dt_series, prefix)

            for new_col, values in new_cols.items():
                out[new_col] = values

            if self.drop_original:
                out = out.drop(columns=[col])

        return out

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract(self, s: pd.Series, prefix: str) -> dict:
        features: dict = {}
        dt = s.dt

        # --- Linear features ---
        features[f"{prefix}_year"] = dt.year
        features[f"{prefix}_month"] = dt.month
        features[f"{prefix}_day"] = dt.day
        features[f"{prefix}_day_of_week"] = dt.dayofweek  # 0=Mon
        features[f"{prefix}_quarter"] = dt.quarter
        features[f"{prefix}_week_of_year"] = dt.isocalendar().week.astype("int32")
        features[f"{prefix}_is_weekend"] = (dt.dayofweek >= 5).astype("int8")
        features[f"{prefix}_is_month_start"] = dt.is_month_start.astype("int8")
        features[f"{prefix}_is_month_end"] = dt.is_month_end.astype("int8")
        features[f"{prefix}_days_since_epoch"] = (s - self.epoch).dt.days

        # Include hour/minute only if they vary (avoids constant noise cols)
        if dt.hour.nunique() > 1:
            features[f"{prefix}_hour"] = dt.hour
            features[f"{prefix}_minute"] = dt.minute

        # --- Cyclical encodings ---
        if self.cyclical:
            features.update(self._cyclical(dt.month, 12, f"{prefix}_month"))
            features.update(self._cyclical(dt.day, 31, f"{prefix}_day"))
            features.update(self._cyclical(dt.dayofweek, 7, f"{prefix}_dow"))
            if f"{prefix}_hour" in features:
                features.update(self._cyclical(dt.hour, 24, f"{prefix}_hour"))

        # --- Holiday flag ---
        if self.add_holidays:
            features[f"{prefix}_is_holiday"] = self._holiday_flag(s)

        return features

    @staticmethod
    def _cyclical(series: pd.Series, period: int, name: str) -> dict:
        radians = 2 * np.pi * series / period
        return {
            f"{name}_sin": np.sin(radians),
            f"{name}_cos": np.cos(radians),
        }

    def _holiday_flag(self, s: pd.Series) -> pd.Series:
        try:
            import holidays

            country_holidays = holidays.country_holidays(self.country_code)
            return s.dt.date.apply(lambda d: int(d in country_holidays))
        except ImportError:
            import warnings

            warnings.warn(
                "add_holidays=True requires the 'holidays' package: " "pip install holidays",
                ImportWarning,
                stacklevel=3,
            )
            return pd.Series(0, index=s.index)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_cols(self, df: pd.DataFrame) -> List[str]:
        if self.cols is not None:
            return [c for c in self.cols if c in df.columns]
        # Auto-detect: datetime64 + object columns that look like dates
        detected = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                detected.append(col)
            elif df[col].dtype == object:
                sample = df[col].dropna().head(30)
                if sample.empty:
                    continue
                try:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().mean() >= 0.8:
                        detected.append(col)
                except Exception:
                    pass
        return detected

    @staticmethod
    def _to_datetime(series: pd.Series, col: str) -> Optional[pd.Series]:
        if pd.api.types.is_datetime64_any_dtype(series):
            return series
        try:
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().mean() < 0.5:
                import warnings

                warnings.warn(
                    f"DatetimeDecomposer: column '{col}' has < 50% parseable dates. " f"Skipping.",
                    UserWarning,
                    stacklevel=4,
                )
                return None
            return parsed
        except Exception:
            return None

    @staticmethod
    def _safe_prefix(col: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", col).strip("_")
