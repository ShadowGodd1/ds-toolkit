"""
ds_toolkit.models.tuner
=========================
TunerOptuna: Optuna-based hyperparameter search.

Pre-built search spaces
------------------------
  LogisticRegression / Ridge
  RandomForest{Classifier,Regressor}
  GradientBoosting{Classifier,Regressor}
  XGB{Classifier,Regressor}
  LGBM{Classifier,Regressor}

Custom search spaces can be supplied via the `param_space` argument.

Features
--------
  • CV-aware objective (StratifiedKFold / KFold / TimeSeriesSplit)
  • MedianPruner — early stops unpromising trials
  • Parallel trials via n_jobs
  • Optuna study returned for further analysis / visualisation

Usage
-----
>>> from ds_toolkit.models import TunerOptuna
>>> from sklearn.ensemble import RandomForestClassifier
>>>
>>> tuner = TunerOptuna(task="clf", n_trials=50, cv_folds=5)
>>> result = tuner.tune(RandomForestClassifier(), X_train, y_train)
>>> result.best_params
>>> result.best_score
>>> result.study          # optuna.Study object
>>> best_model = RandomForestClassifier(**result.best_params).fit(X_train, y_train)
"""
# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score

Task = Literal["clf", "reg", "ts"]


@dataclass
class TuneResult:
    """Output of TunerOptuna.tune()."""
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    study: Any   # optuna.Study — typed as Any to avoid hard import


class TunerOptuna:
    """
    Optuna-based hyperparameter search.

    Parameters
    ----------
    task : {'clf', 'reg', 'ts'}
    n_trials : int
        Number of Optuna trials. Default 100.
    cv_folds : int
        Cross-validation folds for the objective. Default 5.
    scoring : str, optional
        sklearn scoring string. Defaults to 'roc_auc' / 'r2'.
    timeout : float, optional
        Wall-clock time limit in seconds per study. Default None.
    n_jobs : int
        Parallel Optuna trials. Default 1 (safe default; set -1 for all).
    direction : str
        'maximize' or 'minimize'. Auto-detected from scoring if None.
    param_space : callable, optional
        Custom search space function with signature f(trial, estimator_name) → dict.
        If None, the built-in space for the estimator class is used.
    verbose : bool
        Show Optuna progress. Default False.
    """

    def __init__(
        self,
        task: Task = "clf",
        n_trials: int = 100,
        cv_folds: int = 5,
        scoring: Optional[str] = None,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        direction: Optional[str] = None,
        param_space: Optional[Callable] = None,
        verbose: bool = False,
    ) -> None:
        self.task = task
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.scoring = scoring or ("roc_auc" if task == "clf" else "r2")
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.direction = direction or ("maximize")
        self.param_space = param_space
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tune(
        self,
        estimator: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> TuneResult:
        """
        Run hyperparameter search for *estimator*.

        Parameters
        ----------
        estimator : sklearn-compatible estimator
        X         : pd.DataFrame — training features
        y         : pd.Series    — training target

        Returns
        -------
        TuneResult
        """
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "TunerOptuna requires optuna: pip install optuna"
            ) from exc

        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        cv = self._get_cv(y)
        est_name = type(estimator).__name__
        space_fn = self.param_space or self._get_space(est_name)

        def objective(trial: "optuna.Trial") -> float:
            params = space_fn(trial, est_name)
            est = clone(estimator)
            est.set_params(**params)
            scores = cross_val_score(
                est, X, y,
                cv=cv,
                scoring=self.scoring,
                n_jobs=-1,
                error_score=np.nan,
            )
            return float(np.nanmean(scores))

        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
        study = optuna.create_study(direction=self.direction, pruner=pruner)
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=self.verbose,
        )

        return TuneResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials),
            study=study,
        )

    # ------------------------------------------------------------------
    # CV strategy
    # ------------------------------------------------------------------

    def _get_cv(self, y: pd.Series):
        if self.task == "ts":
            return TimeSeriesSplit(n_splits=self.cv_folds)
        if self.task == "reg":
            return KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

    # ------------------------------------------------------------------
    # Built-in search spaces
    # ------------------------------------------------------------------

    @staticmethod
    def _get_space(est_name: str):
        """Return the appropriate search space function for an estimator name."""
        spaces = {
            "LogisticRegression":            TunerOptuna._space_lr,
            "Ridge":                         TunerOptuna._space_ridge,
            "RandomForestClassifier":        TunerOptuna._space_rf,
            "RandomForestRegressor":         TunerOptuna._space_rf,
            "ExtraTreesClassifier":          TunerOptuna._space_rf,
            "ExtraTreesRegressor":           TunerOptuna._space_rf,
            "GradientBoostingClassifier":    TunerOptuna._space_gbm,
            "GradientBoostingRegressor":     TunerOptuna._space_gbm,
            "XGBClassifier":                 TunerOptuna._space_xgb,
            "XGBRegressor":                  TunerOptuna._space_xgb,
            "LGBMClassifier":                TunerOptuna._space_lgbm,
            "LGBMRegressor":                 TunerOptuna._space_lgbm,
        }
        return spaces.get(est_name, TunerOptuna._space_generic)

    @staticmethod
    def _space_lr(trial, _):
        return {
            "C":       trial.suggest_float("C", 1e-4, 100, log=True),
            "solver":  trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            "max_iter": 1000,
        }

    @staticmethod
    def _space_ridge(trial, _):
        return {"alpha": trial.suggest_float("alpha", 1e-4, 100, log=True)}

    @staticmethod
    def _space_rf(trial, _):
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 500),
            "max_depth":         trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

    @staticmethod
    def _space_gbm(trial, _):
        return {
            "n_estimators":   trial.suggest_int("n_estimators", 50, 500),
            "learning_rate":  trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth":      trial.suggest_int("max_depth", 2, 8),
            "subsample":      trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }

    @staticmethod
    def _space_xgb(trial, _):
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 500),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 2, 10),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    @staticmethod
    def _space_lgbm(trial, _):
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 500),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 20, 300),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "verbose":          -1,
        }

    @staticmethod
    def _space_generic(trial, _):
        """Fallback space — returns empty dict (uses estimator defaults)."""
        return {}
