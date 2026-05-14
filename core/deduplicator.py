"""
ds_toolkit.core.deduplicator
=============================
Deduplicator: exact and optional fuzzy deduplication.

Modes
-----
  exact — drop rows with identical values in the specified key columns.
          Falls back to all columns if keys=None.

  fuzzy — after exact dedup, cluster near-duplicate string rows using
          rapidfuzz token-sort ratio. Rows within the same cluster have
          their duplicates removed (first occurrence kept).
          Requires: pip install rapidfuzz

Usage
-----
>>> from ds_toolkit.core import Deduplicator
>>>
>>> # Exact dedup on all columns
>>> dedup = Deduplicator()
>>> df_clean = dedup.clean(df)
>>>
>>> # Exact dedup on key columns + fuzzy on a name column
>>> dedup = Deduplicator(
...     keys=["patient_id", "visit_date"],
...     fuzzy_cols=["full_name"],
...     fuzzy_threshold=90,
... )
>>> df_clean = dedup.clean(df)
>>> dedup.report()       # pd.DataFrame — summary of removals
"""
# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

from typing import List, Optional

import pandas as pd


class Deduplicator:
    """
    Remove duplicate rows using exact key matching and optional fuzzy
    string similarity.

    Parameters
    ----------
    keys : list, optional
        Column names used for exact deduplication. If None, all columns
        are used.
    keep : {'first', 'last', False}
        Which duplicate to keep. Default 'first'.
    fuzzy_cols : list, optional
        Column names to apply fuzzy deduplication on (after exact dedup).
        Requires rapidfuzz.
    fuzzy_threshold : float
        Similarity score (0–100) above which two rows are considered
        duplicates. Default 90.
    fuzzy_scorer : str
        rapidfuzz scorer name. One of:
        'token_sort_ratio', 'ratio', 'partial_ratio', 'token_set_ratio'.
        Default 'token_sort_ratio'.
    """

    def __init__(
        self,
        keys: Optional[List[str]] = None,
        keep: str = "first",
        fuzzy_cols: Optional[List[str]] = None,
        fuzzy_threshold: float = 90.0,
        fuzzy_scorer: str = "token_sort_ratio",
    ) -> None:
        self.keys = keys
        self.keep = keep
        self.fuzzy_cols = fuzzy_cols or []
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_scorer = fuzzy_scorer

        self._n_before: int = 0
        self._n_after_exact: int = 0
        self._n_after_fuzzy: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate *df*.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame — deduplicated copy (original not mutated).
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

        self._n_before = len(df)
        out = df.copy()

        # 1 — Exact dedup
        subset = self.keys if self.keys else None
        out = out.drop_duplicates(subset=subset, keep=self.keep).reset_index(drop=True)
        self._n_after_exact = len(out)

        # 2 — Fuzzy dedup (column-by-column)
        if self.fuzzy_cols:
            out = self._fuzzy_dedup(out)
        self._n_after_fuzzy = len(out)

        return out

    def report(self) -> pd.DataFrame:
        """
        Return a one-row DataFrame summarising deduplication results.
        Call after .clean().
        """
        return pd.DataFrame([{
            "rows_before": self._n_before,
            "exact_removed": self._n_before - self._n_after_exact,
            "fuzzy_removed": self._n_after_exact - self._n_after_fuzzy,
            "total_removed": self._n_before - self._n_after_fuzzy,
            "rows_after": self._n_after_fuzzy,
        }])

    # ------------------------------------------------------------------
    # Fuzzy deduplication
    # ------------------------------------------------------------------

    def _fuzzy_dedup(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            from rapidfuzz import fuzz, process
        except ImportError as exc:
            raise ImportError(
                "Fuzzy deduplication requires rapidfuzz: pip install rapidfuzz"
            ) from exc

        scorer = getattr(fuzz, self.fuzzy_scorer, fuzz.token_sort_ratio)
        rows_to_drop: set = set()

        for col in self.fuzzy_cols:
            if col not in df.columns:
                continue

            values = df[col].fillna("").astype(str).tolist()
            n = len(values)
            visited: set = set()

            for i in range(n):
                if i in rows_to_drop or i in visited:
                    continue
                visited.add(i)
                for j in range(i + 1, n):
                    if j in rows_to_drop:
                        continue
                    score = scorer(values[i], values[j])
                    if score >= self.fuzzy_threshold:
                        # Keep the 'first' occurrence (i), drop j
                        rows_to_drop.add(j)

        if rows_to_drop:
            df = df.drop(index=list(rows_to_drop)).reset_index(drop=True)

        return df
