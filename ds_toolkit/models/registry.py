"""
ds_toolkit.models.registry
============================
ModelRegistry: central catalogue of pre-configured estimators.

Returns a list of (name, estimator) tuples filtered by task type.
Auto-detects which optional boosting libraries are installed and
includes them only when available.

Supported estimators
--------------------
Always available (sklearn):
  LogisticRegression / Ridge
  RandomForestClassifier / RandomForestRegressor
  GradientBoostingClassifier / GradientBoostingRegressor
  ExtraTreesClassifier / ExtraTreesRegressor
  MLPClassifier / MLPRegressor

Optional (installed separately):
  XGBClassifier / XGBRegressor          (xgboost)
  LGBMClassifier / LGBMRegressor        (lightgbm)
  CatBoostClassifier / CatBoostRegressor(catboost)

Usage
-----
>>> from ds_toolkit.models import ModelRegistry
>>> models = ModelRegistry.get(task="clf")
>>> for name, est in models:
...     print(name, est)
>>>
>>> # Just boosting models
>>> boosters = ModelRegistry.get(task="clf", include=["xgboost", "lightgbm"])
>>>
>>> # All except slow ones
>>> fast = ModelRegistry.get(task="clf", exclude=["mlp", "gbm"])
"""
# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from sklearn.base import BaseEstimator

Task = Literal["clf", "reg"]


class ModelRegistry:
    """
    Central catalogue of pre-configured estimators.

    All methods are class-level — instantiation is not required.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def get(
        cls,
        task: Task = "clf",
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> List[Tuple[str, BaseEstimator]]:
        """
        Return a list of (name, estimator) tuples for the given task.

        Parameters
        ----------
        task : {'clf', 'reg'}
        include : list, optional
            Only return estimators whose key is in this list.
            Keys: 'lr', 'rf', 'gbm', 'et', 'mlp', 'xgboost', 'lightgbm', 'catboost'
        exclude : list, optional
            Exclude estimators whose key is in this list.

        Returns
        -------
        List of (name, estimator) tuples.
        """
        if task not in ("clf", "reg"):
            raise ValueError(f"task must be 'clf' or 'reg', got '{task}'")

        catalogue = cls._build_catalogue(task)

        if include:
            catalogue = {k: v for k, v in catalogue.items() if k in include}
        if exclude:
            catalogue = {k: v for k, v in catalogue.items() if k not in exclude}

        return list(catalogue.values())

    @classmethod
    def names(cls, task: Task = "clf") -> List[str]:
        """Return just the estimator names for a given task."""
        return [name for name, _ in cls.get(task=task)]

    @classmethod
    def get_by_name(cls, name: str, task: Task = "clf") -> BaseEstimator:
        """Return a single estimator by name."""
        all_models = dict(cls.get(task=task))
        if name not in all_models:
            raise KeyError(
                f"'{name}' not found. Available: {list(all_models.keys())}"
            )
        return all_models[name]

    # ------------------------------------------------------------------
    # Internal catalogue builders
    # ------------------------------------------------------------------

    @classmethod
    def _build_catalogue(cls, task: Task) -> Dict[str, Tuple[str, BaseEstimator]]:
        catalogue: Dict[str, Tuple[str, BaseEstimator]] = {}

        if task == "clf":
            catalogue.update(cls._sklearn_clf())
            catalogue.update(cls._optional_clf())
        else:
            catalogue.update(cls._sklearn_reg())
            catalogue.update(cls._optional_reg())

        return catalogue

    # --- sklearn classifiers ---

    @staticmethod
    def _sklearn_clf() -> Dict[str, Tuple[str, BaseEstimator]]:
        from sklearn.ensemble import (
            ExtraTreesClassifier,
            GradientBoostingClassifier,
            RandomForestClassifier,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier

        return {
            "lr": ("LogisticRegression", LogisticRegression(
                max_iter=1000, solver="lbfgs", C=1.0, random_state=42
            )),
            "rf": ("RandomForest", RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1
            )),
            "gbm": ("GradientBoosting", GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42
            )),
            "et": ("ExtraTrees", ExtraTreesClassifier(
                n_estimators=200, random_state=42, n_jobs=-1
            )),
            "mlp": ("MLP", MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=300,
                early_stopping=True, random_state=42
            )),
        }

    # --- sklearn regressors ---

    @staticmethod
    def _sklearn_reg() -> Dict[str, Tuple[str, BaseEstimator]]:
        from sklearn.ensemble import (
            ExtraTreesRegressor,
            GradientBoostingRegressor,
            RandomForestRegressor,
        )
        from sklearn.linear_model import Ridge
        from sklearn.neural_network import MLPRegressor

        return {
            "lr": ("Ridge", Ridge(alpha=1.0)),
            "rf": ("RandomForest", RandomForestRegressor(
                n_estimators=200, random_state=42, n_jobs=-1
            )),
            "gbm": ("GradientBoosting", GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42
            )),
            "et": ("ExtraTrees", ExtraTreesRegressor(
                n_estimators=200, random_state=42, n_jobs=-1
            )),
            "mlp": ("MLP", MLPRegressor(
                hidden_layer_sizes=(128, 64), max_iter=300,
                early_stopping=True, random_state=42
            )),
        }

    # --- optional boosting classifiers ---

    @classmethod
    def _optional_clf(cls) -> Dict[str, Tuple[str, BaseEstimator]]:
        out: Dict[str, Tuple[str, BaseEstimator]] = {}

        try:
            from xgboost import XGBClassifier
            out["xgboost"] = ("XGBoost", XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, n_jobs=-1, verbosity=0,
            ))
        except ImportError:
            pass

        try:
            from lightgbm import LGBMClassifier
            out["lightgbm"] = ("LightGBM", LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=63,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbose=-1,
            ))
        except ImportError:
            pass

        try:
            from catboost import CatBoostClassifier
            out["catboost"] = ("CatBoost", CatBoostClassifier(
                iterations=300, learning_rate=0.05, depth=6,
                random_seed=42, verbose=0,
            ))
        except ImportError:
            pass

        return out

    # --- optional boosting regressors ---

    @classmethod
    def _optional_reg(cls) -> Dict[str, Tuple[str, BaseEstimator]]:
        out: Dict[str, Tuple[str, BaseEstimator]] = {}

        try:
            from xgboost import XGBRegressor
            out["xgboost"] = ("XGBoost", XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbosity=0,
            ))
        except ImportError:
            pass

        try:
            from lightgbm import LGBMRegressor
            out["lightgbm"] = ("LightGBM", LGBMRegressor(
                n_estimators=300, learning_rate=0.05, num_leaves=63,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbose=-1,
            ))
        except ImportError:
            pass

        try:
            from catboost import CatBoostRegressor
            out["catboost"] = ("CatBoost", CatBoostRegressor(
                iterations=300, learning_rate=0.05, depth=6,
                random_seed=42, verbose=0,
            ))
        except ImportError:
            pass

        return out
