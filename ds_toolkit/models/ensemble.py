"""
ds_toolkit.models.ensemble
============================
EnsembleBuilder: constructs stacking, voting, or weighted blending
ensembles from a list of base models.

Methods
-------
  stack   — StackingClassifier / StackingRegressor with configurable
            meta-learner. Uses out-of-fold predictions to train the
            meta-learner (sklearn default, CV-safe).
  vote    — VotingClassifier (soft vote) / VotingRegressor (mean).
  blend   — Weighted average of predicted probabilities / values.
            Weights can be supplied or auto-derived from CV scores.

Usage
-----
>>> from ds_toolkit.models import ModelRegistry, EnsembleBuilder
>>> models = ModelRegistry.get(task="clf", exclude=["mlp"])
>>> builder = EnsembleBuilder(task="clf", method="stack")
>>> ensemble = builder.build(models, X_train, y_train)
>>> preds = ensemble.predict(X_val)
>>>
>>> # Blending with CV-derived weights
>>> builder = EnsembleBuilder(task="clf", method="blend", cv_folds=5)
>>> ensemble = builder.build(models, X_train, y_train)
>>> proba = ensemble.predict_proba(X_val)
"""
# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

Task   = Literal["clf", "reg"]
Method = Literal["stack", "vote", "blend"]


class EnsembleBuilder:
    """
    Construct ensemble models from a list of base estimators.

    Parameters
    ----------
    task : {'clf', 'reg'}
    method : {'stack', 'vote', 'blend'}
        Ensemble construction method. Default 'stack'.
    meta_learner : str or estimator
        Meta-learner for stacking.
        'lr'    → LogisticRegression (clf) / Ridge (reg)
        'ridge' → Ridge (both tasks)
        Any sklearn estimator is also accepted.
        Default 'lr'.
    cv_folds : int
        Folds for stacking's out-of-fold generation and blend weight
        derivation. Default 5.
    blend_weights : list, optional
        Manual weights for 'blend' method. Must sum to 1 and match
        len(models). If None, weights are derived from CV scores.
    passthrough : bool
        For stacking — whether to pass original features to meta-learner.
        Default False.
    """

    def __init__(
        self,
        task: Task = "clf",
        method: Method = "stack",
        meta_learner = "lr",
        cv_folds: int = 5,
        blend_weights: Optional[List[float]] = None,
        passthrough: bool = False,
    ) -> None:
        self.task = task
        self.method = method
        self.meta_learner = meta_learner
        self.cv_folds = cv_folds
        self.blend_weights = blend_weights
        self.passthrough = passthrough

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        models: List[Tuple[str, BaseEstimator]],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> BaseEstimator:
        """
        Construct and fit an ensemble.

        Parameters
        ----------
        models : list of (name, estimator) tuples — base models
        X      : pd.DataFrame — training features
        y      : pd.Series    — training target

        Returns
        -------
        Fitted ensemble estimator.
        """
        if not models:
            raise ValueError("models list cannot be empty.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")
        if not isinstance(y, pd.Series):
            raise TypeError(f"Expected pd.Series for y, got {type(y).__name__}")

        if self.method == "stack":
            return self._build_stack(models, X, y)
        if self.method == "vote":
            return self._build_vote(models, X, y)
        if self.method == "blend":
            return self._build_blend(models, X, y)
        raise ValueError(f"method must be 'stack', 'vote', or 'blend', got '{self.method}'")

    # ------------------------------------------------------------------
    # Stack
    # ------------------------------------------------------------------

    def _build_stack(
        self, models: List[Tuple[str, BaseEstimator]],
        X: pd.DataFrame, y: pd.Series,
    ) -> BaseEstimator:
        from sklearn.ensemble import StackingClassifier, StackingRegressor

        meta = self._resolve_meta_learner()
        cv   = self._get_cv(y)

        if self.task == "clf":
            ensemble = StackingClassifier(
                estimators=models,
                final_estimator=meta,
                cv=cv,
                passthrough=self.passthrough,
                n_jobs=-1,
            )
        else:
            ensemble = StackingRegressor(
                estimators=models,
                final_estimator=meta,
                cv=cv,
                passthrough=self.passthrough,
                n_jobs=-1,
            )

        ensemble.fit(X, y)
        return ensemble

    # ------------------------------------------------------------------
    # Vote
    # ------------------------------------------------------------------

    def _build_vote(
        self, models: List[Tuple[str, BaseEstimator]],
        X: pd.DataFrame, y: pd.Series,
    ) -> BaseEstimator:
        from sklearn.ensemble import VotingClassifier, VotingRegressor

        if self.task == "clf":
            ensemble = VotingClassifier(estimators=models, voting="soft", n_jobs=-1)
        else:
            ensemble = VotingRegressor(estimators=models, n_jobs=-1)

        ensemble.fit(X, y)
        return ensemble

    # ------------------------------------------------------------------
    # Blend
    # ------------------------------------------------------------------

    def _build_blend(
        self, models: List[Tuple[str, BaseEstimator]],
        X: pd.DataFrame, y: pd.Series,
    ) -> BaseEstimator:
        """
        Fit each base model and return a WeightedBlender wrapper.
        Weights are derived from CV scores unless manually supplied.
        """
        if self.blend_weights is not None:
            if len(self.blend_weights) != len(models):
                raise ValueError(
                    f"blend_weights length ({len(self.blend_weights)}) must "
                    f"match number of models ({len(models)})."
                )
            weights = np.array(self.blend_weights, dtype=float)
        else:
            weights = self._derive_weights(models, X, y)

        # Fit each base model on full training data
        fitted_models = []
        for name, est in models:
            fitted_est = clone(est)
            fitted_est.fit(X, y)
            fitted_models.append((name, fitted_est))

        return _WeightedBlender(
            models=fitted_models,
            weights=weights,
            task=self.task,
        )

    def _derive_weights(
        self, models: List[Tuple[str, BaseEstimator]],
        X: pd.DataFrame, y: pd.Series,
    ) -> np.ndarray:
        """Derive blend weights proportional to CV scores."""
        cv     = self._get_cv(y)
        metric = "roc_auc" if self.task == "clf" else "r2"
        scores = []
        for _, est in models:
            s = cross_val_score(clone(est), X, y, cv=cv, scoring=metric, n_jobs=-1)
            scores.append(max(float(np.mean(s)), 0.0))

        total = sum(scores)
        if total == 0:
            return np.ones(len(models)) / len(models)
        return np.array(scores) / total

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_meta_learner(self) -> BaseEstimator:
        if isinstance(self.meta_learner, str):
            if self.meta_learner == "lr":
                if self.task == "clf":
                    from sklearn.linear_model import LogisticRegression
                    return LogisticRegression(max_iter=1000, random_state=42)
                from sklearn.linear_model import Ridge
                return Ridge(alpha=1.0)
            if self.meta_learner == "ridge":
                from sklearn.linear_model import Ridge
                return Ridge(alpha=1.0)
            raise ValueError(
                f"meta_learner string must be 'lr' or 'ridge', got '{self.meta_learner}'"
            )
        return clone(self.meta_learner)

    def _get_cv(self, y: pd.Series):
        if self.task == "clf":
            return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        return KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)


# ---------------------------------------------------------------------------
# WeightedBlender wrapper — sklearn-compatible
# ---------------------------------------------------------------------------

class _WeightedBlender(BaseEstimator):
    """
    Sklearn-compatible weighted blend of pre-fitted base models.
    Returned by EnsembleBuilder when method='blend'.
    """

    def __init__(
        self,
        models: List[Tuple[str, BaseEstimator]],
        weights: np.ndarray,
        task: str,
    ) -> None:
        self.models  = models
        self.weights = weights
        self.task    = task

    def fit(self, X, y):
        # Already fitted — no-op (supports re-fitting if needed)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.task == "clf":
            probas = self.predict_proba(X)
            return (probas[:, 1] >= 0.5).astype(int)
        preds = np.column_stack([est.predict(X) for _, est in self.models])
        return preds @ self.weights

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "clf":
            raise AttributeError("predict_proba is only available for clf task.")
        probas = np.stack(
            [est.predict_proba(X) for _, est in self.models], axis=0
        )  # (n_models, n_samples, n_classes)
        return np.tensordot(self.weights, probas, axes=[[0], [0]])  # (n_samples, n_classes)

    @property
    def classes_(self):
        return self.models[0][1].classes_
