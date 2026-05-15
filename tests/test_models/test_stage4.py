"""
tests/test_models/test_stage4.py
==================================
Tests for Stage 4: ModelRegistry, CVHarness, TunerOptuna, EnsembleBuilder.
"""
# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from ds_toolkit.models import (
    CVHarness,
    CVResults,
    EnsembleBuilder,
    ModelRegistry,
    TuneResult,
    TunerOptuna,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_data():
    X_arr, y_arr = make_classification(
        n_samples=300, n_features=10, n_informative=5, random_state=42
    )
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(10)])
    y = pd.Series(y_arr, name="target")
    return X, y


@pytest.fixture
def reg_data():
    X_arr, y_arr = make_regression(
        n_samples=300, n_features=10, n_informative=5, random_state=42
    )
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(10)])
    y = pd.Series(y_arr, name="target")
    return X, y


@pytest.fixture
def small_models_clf():
    return [
        ("LR",  LogisticRegression(max_iter=200, random_state=42)),
        ("RF",  RandomForestClassifier(n_estimators=20, random_state=42)),
    ]


@pytest.fixture
def small_models_reg():
    return [
        ("Ridge", Ridge(alpha=1.0)),
        ("RF",    RandomForestRegressor(n_estimators=20, random_state=42)),
    ]


# ============================================================
# ModelRegistry
# ============================================================

class TestModelRegistry:

    def test_get_clf_returns_list(self):
        models = ModelRegistry.get(task="clf")
        assert isinstance(models, list)
        assert len(models) >= 5

    def test_get_reg_returns_list(self):
        models = ModelRegistry.get(task="reg")
        assert isinstance(models, list)
        assert len(models) >= 5

    def test_returns_name_estimator_tuples(self):
        models = ModelRegistry.get(task="clf")
        for item in models:
            assert len(item) == 2
            name, est = item
            assert isinstance(name, str)
            assert hasattr(est, "fit")

    def test_include_filter(self):
        models = ModelRegistry.get(task="clf", include=["lr", "rf"])
        names = [n for n, _ in models]
        assert len(names) == 2
        assert all(n in ["LogisticRegression", "RandomForest"] for n in names)

    def test_exclude_filter(self):
        all_models = ModelRegistry.get(task="clf")
        excl_models = ModelRegistry.get(task="clf", exclude=["mlp"])
        assert len(excl_models) == len(all_models) - 1

    def test_names_method(self):
        names = ModelRegistry.names(task="clf")
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_get_by_name(self):
        est = ModelRegistry.get_by_name("LogisticRegression", task="clf")
        assert hasattr(est, "fit")

    def test_get_by_name_invalid_raises(self):
        with pytest.raises(KeyError):
            ModelRegistry.get_by_name("NonExistentModel", task="clf")

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            ModelRegistry.get(task="clustering")

    def test_estimators_are_unfitted(self):
        models = ModelRegistry.get(task="clf")
        for name, est in models:
            # CatBoost exposes classes_ as a class attribute before fitting
            if "CatBoost" in type(est).__name__:
                continue
            assert not hasattr(est, "classes_"), f"{name} appears pre-fitted"


# ============================================================
# CVHarness
# ============================================================

class TestCVHarness:

    def test_returns_cv_results(self, small_models_clf, clf_data):
        X, y = clf_data
        harness = CVHarness(task="clf", n_splits=3, verbose=False)
        result = harness.run(small_models_clf, X, y)
        assert isinstance(result, CVResults)

    def test_results_df_has_all_folds(self, small_models_clf, clf_data):
        X, y = clf_data
        harness = CVHarness(task="clf", n_splits=3, verbose=False)
        result = harness.run(small_models_clf, X, y)
        # 2 models × 3 folds = 6 rows
        assert len(result.results_df) == 6

    def test_summary_df_ranked_by_score(self, small_models_clf, clf_data):
        X, y = clf_data
        harness = CVHarness(task="clf", n_splits=3, verbose=False)
        result = harness.run(small_models_clf, X, y)
        scores = result.summary_df["mean_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_best_model_is_tuple(self, small_models_clf, clf_data):
        X, y = clf_data
        harness = CVHarness(task="clf", n_splits=3, verbose=False)
        result = harness.run(small_models_clf, X, y)
        name, est = result.best_model
        assert isinstance(name, str)
        assert hasattr(est, "predict")

    def test_best_model_is_fitted(self, small_models_clf, clf_data):
        X, y = clf_data
        harness = CVHarness(task="clf", n_splits=3, verbose=False)
        result = harness.run(small_models_clf, X, y)
        _, est = result.best_model
        preds = est.predict(X)
        assert len(preds) == len(y)

    def test_regression_task(self, small_models_reg, reg_data):
        X, y = reg_data
        harness = CVHarness(task="reg", n_splits=3, scoring="r2", verbose=False)
        result = harness.run(small_models_reg, X, y)
        assert not result.results_df["score"].isna().all()

    def test_non_dataframe_X_raises(self, small_models_clf, clf_data):
        _, y = clf_data
        with pytest.raises(TypeError):
            CVHarness(task="clf", verbose=False).run(small_models_clf, [[1, 2]], y)

    def test_non_series_y_raises(self, small_models_clf, clf_data):
        X, _ = clf_data
        with pytest.raises(TypeError):
            CVHarness(task="clf", verbose=False).run(small_models_clf, X, [0, 1] * 150)

    def test_summary_has_correct_columns(self, small_models_clf, clf_data):
        X, y = clf_data
        harness = CVHarness(task="clf", n_splits=3, verbose=False)
        result = harness.run(small_models_clf, X, y)
        expected = {"model", "mean_score", "std_score", "mean_fit_time_s", "n_folds"}
        assert expected.issubset(set(result.summary_df.columns))

    def test_display_returns_self(self, small_models_clf, clf_data):
        X, y = clf_data
        harness = CVHarness(task="clf", n_splits=3, verbose=False)
        result = harness.run(small_models_clf, X, y)
        assert result.display() is result


# ============================================================
# TunerOptuna
# ============================================================

class TestTunerOptuna:

    def test_returns_tune_result(self, clf_data):
        pytest.importorskip("optuna")
        X, y = clf_data
        tuner = TunerOptuna(task="clf", n_trials=5, cv_folds=3)
        result = tuner.tune(RandomForestClassifier(n_estimators=10), X, y)
        assert isinstance(result, TuneResult)

    def test_best_params_is_dict(self, clf_data):
        pytest.importorskip("optuna")
        X, y = clf_data
        result = TunerOptuna(task="clf", n_trials=5, cv_folds=3).tune(
            RandomForestClassifier(), X, y
        )
        assert isinstance(result.best_params, dict)

    def test_best_score_is_float(self, clf_data):
        pytest.importorskip("optuna")
        X, y = clf_data
        result = TunerOptuna(task="clf", n_trials=5, cv_folds=3).tune(
            RandomForestClassifier(), X, y
        )
        assert isinstance(result.best_score, float)

    def test_n_trials_respected(self, clf_data):
        pytest.importorskip("optuna")
        X, y = clf_data
        result = TunerOptuna(task="clf", n_trials=7, cv_folds=3).tune(
            LogisticRegression(max_iter=200), X, y
        )
        assert result.n_trials == 7

    def test_study_object_returned(self, clf_data):
        optuna = pytest.importorskip("optuna")
        X, y = clf_data
        result = TunerOptuna(task="clf", n_trials=5, cv_folds=3).tune(
            RandomForestClassifier(), X, y
        )
        assert isinstance(result.study, optuna.Study)

    def test_regression_task(self, reg_data):
        pytest.importorskip("optuna")
        X, y = reg_data
        result = TunerOptuna(task="reg", n_trials=5, cv_folds=3).tune(
            RandomForestRegressor(), X, y
        )
        assert isinstance(result.best_params, dict)

    def test_custom_param_space(self, clf_data):
        pytest.importorskip("optuna")
        X, y = clf_data

        def my_space(trial, _):
            return {"C": trial.suggest_float("C", 0.1, 10.0)}

        result = TunerOptuna(
            task="clf", n_trials=5, cv_folds=3, param_space=my_space
        ).tune(LogisticRegression(max_iter=200), X, y)
        assert "C" in result.best_params

    def test_missing_optuna_raises(self, clf_data, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "optuna":
                raise ImportError("No module named 'optuna'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        X, y = clf_data
        with pytest.raises(ImportError, match="optuna"):
            TunerOptuna(task="clf", n_trials=3).tune(LogisticRegression(), X, y)


# ============================================================
# EnsembleBuilder
# ============================================================

class TestEnsembleBuilder:

    def test_stack_clf_predict(self, small_models_clf, clf_data):
        X, y = clf_data
        builder = EnsembleBuilder(task="clf", method="stack", cv_folds=3)
        ensemble = builder.build(small_models_clf, X, y)
        preds = ensemble.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

    def test_stack_clf_predict_proba(self, small_models_clf, clf_data):
        X, y = clf_data
        builder = EnsembleBuilder(task="clf", method="stack", cv_folds=3)
        ensemble = builder.build(small_models_clf, X, y)
        proba = ensemble.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_vote_clf(self, small_models_clf, clf_data):
        X, y = clf_data
        builder = EnsembleBuilder(task="clf", method="vote")
        ensemble = builder.build(small_models_clf, X, y)
        preds = ensemble.predict(X)
        assert len(preds) == len(y)

    def test_blend_clf(self, small_models_clf, clf_data):
        X, y = clf_data
        builder = EnsembleBuilder(task="clf", method="blend", cv_folds=3)
        ensemble = builder.build(small_models_clf, X, y)
        preds = ensemble.predict(X)
        assert len(preds) == len(y)

    def test_blend_manual_weights(self, small_models_clf, clf_data):
        X, y = clf_data
        builder = EnsembleBuilder(
            task="clf", method="blend", blend_weights=[0.3, 0.7]
        )
        ensemble = builder.build(small_models_clf, X, y)
        proba = ensemble.predict_proba(X)
        assert proba.shape[0] == len(y)

    def test_blend_weights_wrong_length_raises(self, small_models_clf, clf_data):
        X, y = clf_data
        builder = EnsembleBuilder(
            task="clf", method="blend", blend_weights=[0.5, 0.3, 0.2]
        )
        with pytest.raises(ValueError, match="blend_weights length"):
            builder.build(small_models_clf, X, y)

    def test_stack_reg(self, small_models_reg, reg_data):
        X, y = reg_data
        builder = EnsembleBuilder(task="reg", method="stack", cv_folds=3)
        ensemble = builder.build(small_models_reg, X, y)
        preds = ensemble.predict(X)
        assert len(preds) == len(y)

    def test_vote_reg(self, small_models_reg, reg_data):
        X, y = reg_data
        builder = EnsembleBuilder(task="reg", method="vote")
        ensemble = builder.build(small_models_reg, X, y)
        preds = ensemble.predict(X)
        assert len(preds) == len(y)

    def test_invalid_method_raises(self, small_models_clf, clf_data):
        X, y = clf_data
        with pytest.raises(ValueError):
            EnsembleBuilder(task="clf", method="bagging").build(small_models_clf, X, y)

    def test_empty_models_raises(self, clf_data):
        X, y = clf_data
        with pytest.raises(ValueError, match="empty"):
            EnsembleBuilder(task="clf").build([], X, y)

    def test_custom_meta_learner(self, small_models_clf, clf_data):
        X, y = clf_data
        meta = LogisticRegression(C=0.5, max_iter=200)
        builder = EnsembleBuilder(task="clf", method="stack", meta_learner=meta, cv_folds=3)
        ensemble = builder.build(small_models_clf, X, y)
        preds = ensemble.predict(X)
        assert len(preds) == len(y)

    def test_non_dataframe_raises(self, small_models_clf, clf_data):
        _, y = clf_data
        with pytest.raises(TypeError):
            EnsembleBuilder(task="clf").build(small_models_clf, [[1, 2]], y)
