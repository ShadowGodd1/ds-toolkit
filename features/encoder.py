"""
ds_toolkit.features.encoder
============================
EncoderFactory: auto-selects encoding strategy per column based on
cardinality, task type, and optional ordered-column metadata.

Decision logic (evaluated in order)
-------------------------------------
1. Column has ordered metadata supplied  → OrdinalEncoder
2. Cardinality <= ohe_threshold           → OneHotEncoder  (sparse=False)
3. Cardinality >  ohe_threshold + target  → TargetEncoder  (CV-safe)
4. Cardinality >  ohe_threshold, no target→ HashingEncoder (fallback)

CV safety
---------
TargetEncoder is the only encoder that touches the target. It is fit
only on training data. Use .fit(X_train, y_train) + .transform(X_val)
— never fit on the full dataset before splitting.

Usage
-----
>>> from ds_toolkit.features import EncoderFactory
>>> enc = EncoderFactory(task="clf", ohe_threshold=15)
>>> X_train_enc = enc.fit_transform(X_train, y_train)
>>> X_val_enc   = enc.transform(X_val)
>>> enc.encoding_map     # dict: column → strategy used
"""
# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted


Task = Literal["clf", "reg"]


class _TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Minimal CV-safe target encoder.
    Encodes each category as the smoothed mean of the target.
    Smoothing prevents high-cardinality rare categories from dominating.
    """

    def __init__(self, smoothing: float = 10.0) -> None:
        self.smoothing = smoothing
        self._global_mean: float = 0.0
        self._col_maps: Dict[str, Dict] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_TargetEncoder":
        self._global_mean = float(y.mean())
        self._col_maps = {}
        for col in X.columns:
            stats = (
                pd.DataFrame({"cat": X[col], "target": y.values})
                .groupby("cat")["target"]
                .agg(["mean", "count"])
            )
            # Smoothed mean: blend category mean toward global mean
            smoother = stats["count"] / (stats["count"] + self.smoothing)
            stats["encoded"] = smoother * stats["mean"] + (1 - smoother) * self._global_mean
            self._col_maps[col] = stats["encoded"].to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, mapping in self._col_maps.items():
            if col in out.columns:
                out[col] = out[col].map(mapping).fillna(self._global_mean)
        return out


class _HashingEncoder(BaseEstimator, TransformerMixin):
    """Stateless hashing encoder — no target required, handles unseen categories."""

    def __init__(self, n_components: int = 16) -> None:
        self.n_components = n_components

    def fit(self, X: pd.DataFrame, y=None) -> "_HashingEncoder":
        self._cols = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for col in self._cols:
            if col not in X.columns:
                continue
            hashed = (
                X[col].astype(str)
                .apply(lambda v: abs(hash(v)) % self.n_components)
            )
            # One-hot over the n_components bins
            dummies = pd.get_dummies(hashed, prefix=col).reindex(
                columns=[f"{col}_{i}" for i in range(self.n_components)], fill_value=0
            )
            frames.append(dummies)
        return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=X.index)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EncoderFactory(BaseEstimator, TransformerMixin):
    """
    Auto-selecting categorical encoder.

    Parameters
    ----------
    task : {'clf', 'reg'}
        Model task. Enables TargetEncoder when a target is available.
    ohe_threshold : int
        Max cardinality for OneHotEncoder. Above → TargetEncoder or hashing.
        Default 15.
    ordered_cols : dict, optional
        Mapping of column name → ordered category list.
        Example: {"size": ["S", "M", "L", "XL"]}
        These columns get OrdinalEncoder regardless of cardinality.
    target_smoothing : float
        Smoothing factor for TargetEncoder. Higher = more shrinkage.
        Default 10.0.
    handle_unknown : str
        OneHotEncoder handle_unknown strategy. Default 'ignore'.
    """

    def __init__(
        self,
        task: Task = "clf",
        ohe_threshold: int = 15,
        ordered_cols: Optional[Dict[str, List]] = None,
        target_smoothing: float = 10.0,
        handle_unknown: str = "ignore",
    ) -> None:
        self.task = task
        self.ohe_threshold = ohe_threshold
        self.ordered_cols = ordered_cols or {}
        self.target_smoothing = target_smoothing
        self.handle_unknown = handle_unknown

        self.encoding_map: Dict[str, str] = {}
        self._fitted_encoders: Dict[str, tuple] = {}  # col → (strategy, encoder)
        self._cat_cols: List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "EncoderFactory":
        self._validate(X)
        self._cat_cols = self._detect_cat_cols(X)
        self._fitted_encoders = {}
        self.encoding_map = {}

        # Separate columns by strategy
        ordinal_cols, ohe_cols, target_cols, hash_cols = [], [], [], []

        for col in self._cat_cols:
            n_unique = X[col].nunique(dropna=True)

            if col in self.ordered_cols:
                ordinal_cols.append(col)
                self.encoding_map[col] = "ordinal"
            elif n_unique <= self.ohe_threshold:
                ohe_cols.append(col)
                self.encoding_map[col] = "ohe"
            elif y is not None:
                target_cols.append(col)
                self.encoding_map[col] = "target"
            else:
                hash_cols.append(col)
                self.encoding_map[col] = "hashing"

        # Fit OrdinalEncoder per column (different category orders)
        for col in ordinal_cols:
            cats = self.ordered_cols[col]
            enc = OrdinalEncoder(
                categories=[cats],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            enc.fit(X[[col]].astype(str))
            self._fitted_encoders[col] = ("ordinal", enc)

        # Fit OneHotEncoder jointly (handles all ohe cols)
        if ohe_cols:
            enc = OneHotEncoder(
                sparse_output=False,
                handle_unknown=self.handle_unknown,
                drop=None,
            )
            enc.fit(X[ohe_cols].astype(str))
            self._fitted_encoders["__ohe__"] = ("ohe", enc, ohe_cols)

        # Fit TargetEncoder jointly
        if target_cols and y is not None:
            enc = _TargetEncoder(smoothing=self.target_smoothing)
            enc.fit(X[target_cols].astype(str), y)
            self._fitted_encoders["__target__"] = ("target", enc, target_cols)

        # Fit HashingEncoder
        if hash_cols:
            enc = _HashingEncoder()
            enc.fit(X[hash_cols].astype(str))
            self._fitted_encoders["__hash__"] = ("hash", enc, hash_cols)

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform().")
        self._validate(X)

        out = X.copy()
        cols_to_drop: List[str] = []

        # Ordinal (per-column)
        for col in self._cat_cols:
            if col in self.ordered_cols and col in self._fitted_encoders:
                _, enc = self._fitted_encoders[col]
                out[col] = enc.transform(out[[col]].astype(str)).ravel()
                continue

        # OHE
        if "__ohe__" in self._fitted_encoders:
            _, enc, ohe_cols = self._fitted_encoders["__ohe__"]
            present = [c for c in ohe_cols if c in out.columns]
            if present:
                encoded = enc.transform(out[present].astype(str))
                feature_names = enc.get_feature_names_out(present)
                ohe_df = pd.DataFrame(encoded, columns=feature_names, index=out.index)
                out = pd.concat([out, ohe_df], axis=1)
                cols_to_drop.extend(present)

        # Target
        if "__target__" in self._fitted_encoders:
            _, enc, target_cols = self._fitted_encoders["__target__"]
            present = [c for c in target_cols if c in out.columns]
            if present:
                out[present] = enc.transform(out[present].astype(str))[present]

        # Hash
        if "__hash__" in self._fitted_encoders:
            _, enc, hash_cols = self._fitted_encoders["__hash__"]
            present = [c for c in hash_cols if c in out.columns]
            if present:
                hashed_df = enc.transform(out[present].astype(str))
                out = pd.concat([out, hashed_df], axis=1)
                cols_to_drop.extend(present)

        if cols_to_drop:
            out = out.drop(columns=[c for c in cols_to_drop if c in out.columns])

        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_cat_cols(X: pd.DataFrame) -> List[str]:
        return [
            col for col in X.columns
            if X[col].dtype == object
            or str(X[col].dtype) == "category"
            or pd.api.types.is_string_dtype(X[col])
        ]

    @staticmethod
    def _validate(X) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")
