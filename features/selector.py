"""
ds_toolkit.features.selector
==============================
FeatureSelector: multi-method feature selection pipeline.

Pipeline (applied in order)
-----------------------------
1. VarianceThreshold  — drop near-constant features
2. Correlation filter — drop one of each highly-correlated pair
3. Model-based RFECV  — recursive feature elimination with CV
4. SHAP (optional)    — final trim by SHAP value magnitude

Each stage is individually togglable. The pipeline is CV-safe:
all statistics are learned only from the training data passed to fit().

Usage
-----
>>> from ds_toolkit.features import FeatureSelector
>>> selector = FeatureSelector(method="rfecv", task="clf")
>>> X_train_sel = selector.fit_transform(X_train, y_train)
>>> X_val_sel   = selector.transform(X_val)
>>> selector.report()          # pd.DataFrame — what was kept / dropped and why
>>> selector.selected_features_
"""
# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

Method = Literal["variance", "correlation", "rfecv", "shap"]
Task   = Literal["clf", "reg"]


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Multi-stage feature selection pipeline.

    Parameters
    ----------
    method : {'variance', 'correlation', 'rfecv', 'shap'}
        Terminal method. Stages always run in order up to and including
        the chosen method:
          'variance'    → stage 1 only
          'correlation' → stages 1–2
          'rfecv'       → stages 1–3  (default)
          'shap'        → stages 1–4
    task : {'clf', 'reg'}
        Used to select the default estimator for RFECV / SHAP.
    variance_threshold : float
        Minimum variance for a feature to survive stage 1. Default 0.01.
    correlation_threshold : float
        Pearson |r| above which one of a pair is dropped. Default 0.95.
    n_features : int, optional
        Target number of features for RFECV. None lets RFECV choose.
    cv_folds : int
        Number of CV folds used in RFECV. Default 5.
    shap_top_n : int
        Number of features to keep in the SHAP stage. Default 20.
    estimator : sklearn estimator, optional
        Custom estimator for RFECV / SHAP. If None, a default is chosen
        based on `task`.
    n_jobs : int
        Parallel jobs for RFECV. Default -1.
    """

    def __init__(
        self,
        method: Method = "rfecv",
        task: Task = "clf",
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        n_features: Optional[int] = None,
        cv_folds: int = 5,
        shap_top_n: int = 20,
        estimator=None,
        n_jobs: int = -1,
    ) -> None:
        self.method = method
        self.task = task
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.n_features = n_features
        self.cv_folds = cv_folds
        self.shap_top_n = shap_top_n
        self.estimator = estimator
        self.n_jobs = n_jobs

        self.selected_features_: List[str] = []
        self._drop_log: List[dict] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        self._validate(X, y)
        self._drop_log = []
        current_cols = X.select_dtypes(include="number").columns.tolist()

        # Stage 1 — Variance filter
        current_cols = self._stage_variance(X, current_cols)

        if self.method == "variance":
            self.selected_features_ = current_cols
            self._fitted = True
            return self

        # Stage 2 — Correlation filter
        current_cols = self._stage_correlation(X, current_cols)

        if self.method == "correlation":
            self.selected_features_ = current_cols
            self._fitted = True
            return self

        # Stage 3 — RFECV
        current_cols = self._stage_rfecv(X[current_cols], y, current_cols)

        if self.method == "rfecv":
            self.selected_features_ = current_cols
            self._fitted = True
            return self

        # Stage 4 — SHAP
        current_cols = self._stage_shap(X[current_cols], y, current_cols)
        self.selected_features_ = current_cols
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform().")
        self._validate_df(X)
        present = [c for c in self.selected_features_ if c in X.columns]
        return X[present].copy()

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _stage_variance(self, X: pd.DataFrame, cols: List[str]) -> List[str]:
        from sklearn.feature_selection import VarianceThreshold
        vt = VarianceThreshold(threshold=self.variance_threshold)
        subset = X[cols].fillna(0)
        vt.fit(subset)
        mask = vt.get_support()
        kept = [c for c, m in zip(cols, mask) if m]
        dropped = [c for c, m in zip(cols, mask) if not m]
        for c in dropped:
            self._drop_log.append({"feature": c, "stage": "variance",
                                    "reason": f"variance < {self.variance_threshold}"})
        return kept

    def _stage_correlation(self, X: pd.DataFrame, cols: List[str]) -> List[str]:
        if len(cols) < 2:
            return cols
        corr = X[cols].corr(method="pearson").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = set()
        for col in upper.columns:
            high = upper[col][upper[col] > self.correlation_threshold].index.tolist()
            for partner in high:
                if partner not in to_drop:
                    to_drop.add(col)
                    self._drop_log.append({
                        "feature": col,
                        "stage": "correlation",
                        "reason": f"|r| > {self.correlation_threshold} with '{partner}'",
                    })
                    break
        return [c for c in cols if c not in to_drop]

    def _stage_rfecv(
        self, X: pd.DataFrame, y: pd.Series, cols: List[str]
    ) -> List[str]:
        from sklearn.feature_selection import RFECV
        from sklearn.model_selection import StratifiedKFold, KFold

        est = self._get_estimator()
        cv = (
            StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            if self.task == "clf"
            else KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        )
        rfecv = RFECV(
            estimator=est,
            min_features_to_select=self.n_features or 1,
            cv=cv,
            scoring="roc_auc" if self.task == "clf" else "r2",
            n_jobs=self.n_jobs,
        )
        rfecv.fit(X.fillna(0), y)
        mask = rfecv.support_
        kept = [c for c, m in zip(cols, mask) if m]
        dropped = [c for c, m in zip(cols, mask) if not m]
        for c in dropped:
            self._drop_log.append({"feature": c, "stage": "rfecv",
                                    "reason": "eliminated by RFECV"})
        return kept

    def _stage_shap(
        self, X: pd.DataFrame, y: pd.Series, cols: List[str]
    ) -> List[str]:
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "SHAP selection requires shap: pip install shap"
            ) from exc

        est = self._get_estimator()
        est.fit(X.fillna(0), y)

        try:
            explainer = shap.TreeExplainer(est)
            shap_values = explainer.shap_values(X.fillna(0))
        except Exception:
            explainer = shap.KernelExplainer(est.predict, X.fillna(0).head(100))
            shap_values = explainer.shap_values(X.fillna(0))

        if isinstance(shap_values, list):
            # Multi-class: average across classes
            shap_arr = np.abs(np.array(shap_values)).mean(axis=0)
        else:
            shap_arr = np.abs(shap_values)

        mean_shap = pd.Series(shap_arr.mean(axis=0), index=cols)
        top_cols = mean_shap.nlargest(self.shap_top_n).index.tolist()
        dropped = [c for c in cols if c not in top_cols]
        for c in dropped:
            self._drop_log.append({"feature": c, "stage": "shap",
                                    "reason": f"outside top {self.shap_top_n} SHAP features"})
        return top_cols

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> pd.DataFrame:
        """
        Return a DataFrame summarising all dropped features and the stage
        and reason for each removal. Call after .fit().
        """
        if not self._drop_log:
            return pd.DataFrame(columns=["feature", "stage", "reason"])
        return pd.DataFrame(self._drop_log)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_estimator(self):
        if self.estimator is not None:
            return self.estimator
        if self.task == "clf":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=self.n_jobs
            )
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=self.n_jobs
        )

    @staticmethod
    def _validate(X, y) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")
        if not isinstance(y, pd.Series):
            raise TypeError(f"Expected pd.Series for y, got {type(y).__name__}")

    @staticmethod
    def _validate_df(X) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")
