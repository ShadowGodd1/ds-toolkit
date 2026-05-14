"""
ds_toolkit.features.interaction_builder
=========================================
InteractionBuilder: generates interaction features between numeric columns.

Feature types
-------------
  polynomial — sklearn PolynomialFeatures (degree 2 or 3, no bias term)
  ratio      — A / (B + epsilon) for all specified pairs (or all combos)
  product    — A * B for all specified pairs (or all combos)

Pruning
-------
After generation, near-zero-variance features are dropped automatically
(variance < variance_threshold). This prevents dimensionality explosion
when degree=2 is applied across many columns.

Optionally, model-based pruning can rank interactions by a quick
RandomForest and keep only the top_k.

Usage
-----
>>> from ds_toolkit.features import InteractionBuilder
>>> builder = InteractionBuilder(
...     cols=["age", "income", "score"],
...     degree=2,
...     include_types=["product", "ratio"],
... )
>>> X_train_int = builder.fit_transform(X_train, y_train)
>>> X_val_int   = builder.transform(X_val)
>>> builder.selected_features_   # list of kept feature names
"""
# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

from itertools import combinations
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


InteractionType = Literal["polynomial", "product", "ratio"]


class InteractionBuilder(BaseEstimator, TransformerMixin):
    """
    Generate and optionally prune interaction features.

    Parameters
    ----------
    cols : list, optional
        Numeric columns to build interactions from. If None, all numeric
        columns in the DataFrame are used.
    degree : int
        Polynomial degree. Only used when 'polynomial' is in include_types.
        Default 2.
    include_types : list
        Which interaction types to generate.
        Options: 'polynomial', 'product', 'ratio'.
        Default ['product', 'ratio'].
    prune_interactions : bool
        Drop near-zero-variance interaction features after generation.
        Default True.
    variance_threshold : float
        Minimum variance for a feature to be kept during pruning.
        Default 1e-4.
    top_k : int, optional
        If set, use a RandomForest to rank interactions and keep only
        the top_k most important. Requires a target y in fit(). Default None.
    ratio_epsilon : float
        Small constant added to denominator to prevent division by zero.
        Default 1e-8.
    """

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        degree: int = 2,
        include_types: Optional[List[InteractionType]] = None,
        prune_interactions: bool = True,
        variance_threshold: float = 1e-4,
        top_k: Optional[int] = None,
        ratio_epsilon: float = 1e-8,
    ) -> None:
        self.cols = cols
        self.degree = degree
        self.include_types = include_types or ["product", "ratio"]
        self.prune_interactions = prune_interactions
        self.variance_threshold = variance_threshold
        self.top_k = top_k
        self.ratio_epsilon = ratio_epsilon

        self.selected_features_: List[str] = []
        self._interaction_cols: List[str] = []   # cols resolved at fit time
        self._fitted = False

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "InteractionBuilder":
        self._validate(X)
        self._interaction_cols = self._resolve_cols(X)

        if not self._interaction_cols:
            self._fitted = True
            return self

        # Generate on training data to determine which features survive pruning
        interactions = self._generate(X[self._interaction_cols])

        if interactions.empty:
            self.selected_features_ = []
            self._fitted = True
            return self

        # Variance pruning
        if self.prune_interactions:
            variances = interactions.var()
            interactions = interactions.loc[:, variances >= self.variance_threshold]

        # Model-based top_k pruning
        if self.top_k and y is not None and not interactions.empty:
            interactions = self._model_prune(interactions, y)

        self.selected_features_ = interactions.columns.tolist()
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform().")
        self._validate(X)

        if not self._interaction_cols or not self.selected_features_:
            return X.copy()

        interactions = self._generate(X[self._interaction_cols])
        # Keep only features selected during fit (handles unseen cols gracefully)
        keep = [c for c in self.selected_features_ if c in interactions.columns]
        missing = [c for c in self.selected_features_ if c not in interactions.columns]

        result = pd.concat([X, interactions[keep]], axis=1)

        # Fill any missing selected features with 0
        for c in missing:
            result[c] = 0.0

        return result

    # ------------------------------------------------------------------
    # Feature generation
    # ------------------------------------------------------------------

    def _generate(self, X: pd.DataFrame) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []

        if "polynomial" in self.include_types:
            frames.append(self._polynomial_features(X))

        if "product" in self.include_types:
            frames.append(self._product_features(X))

        if "ratio" in self.include_types:
            frames.append(self._ratio_features(X))

        if not frames:
            return pd.DataFrame(index=X.index)

        return pd.concat(frames, axis=1)

    def _polynomial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(
            degree=self.degree, include_bias=False, interaction_only=False
        )
        arr = poly.fit_transform(X.fillna(0))
        names = poly.get_feature_names_out(X.columns)
        # Exclude the original columns (they're already in X)
        df = pd.DataFrame(arr, columns=names, index=X.index)
        original = list(X.columns)
        interaction_only = [c for c in df.columns if c not in original]
        return df[interaction_only]

    def _product_features(self, X: pd.DataFrame) -> pd.DataFrame:
        rows: dict = {}
        for a, b in combinations(X.columns, 2):
            name = f"{a}_x_{b}"
            rows[name] = X[a] * X[b]
        return pd.DataFrame(rows, index=X.index)

    def _ratio_features(self, X: pd.DataFrame) -> pd.DataFrame:
        rows: dict = {}
        for a, b in combinations(X.columns, 2):
            name = f"{a}_div_{b}"
            rows[name] = X[a] / (X[b] + self.ratio_epsilon)
        return pd.DataFrame(rows, index=X.index)

    def _model_prune(self, interactions: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Keep top_k features by RandomForest importance."""
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.preprocessing import LabelEncoder

            n_classes = y.nunique()
            if n_classes <= 10:
                le = LabelEncoder()
                y_enc = le.fit_transform(y)
                rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                y_enc = y.values
                rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

            rf.fit(interactions.fillna(0), y_enc)
            importances = pd.Series(rf.feature_importances_, index=interactions.columns)
            top_cols = importances.nlargest(self.top_k).index.tolist()
            return interactions[top_cols]
        except Exception:
            # If pruning fails for any reason, return unpruned
            return interactions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_cols(self, X: pd.DataFrame) -> List[str]:
        if self.cols is not None:
            return [c for c in self.cols if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
        return X.select_dtypes(include="number").columns.tolist()

    @staticmethod
    def _validate(X) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")
