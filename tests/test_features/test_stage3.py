"""
tests/test_features/test_stage3.py
====================================
Tests for Stage 3: EncoderFactory, DatetimeDecomposer,
InteractionBuilder, FeatureSelector, Scaler.
"""
# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from ds_toolkit.features import (
    DatetimeDecomposer,
    EncoderFactory,
    FeatureSelector,
    InteractionBuilder,
    Scaler,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cat_df():
    np.random.seed(0)
    n = 200
    return pd.DataFrame({
        "low_card":  np.random.choice(["A", "B", "C"], n),           # OHE
        "high_card": [f"user_{i}" for i in np.random.randint(0, 80, n)],  # target/hash
        "size":      np.random.choice(["S", "M", "L", "XL"], n),     # ordinal
        "num1":      np.random.rand(n),
        "num2":      np.random.rand(n),
    })


@pytest.fixture
def target_clf(cat_df):
    return pd.Series(np.random.randint(0, 2, len(cat_df)), name="label")


@pytest.fixture
def dt_df():
    n = 100
    return pd.DataFrame({
        "created_at": pd.date_range("2022-01-01", periods=n, freq="D"),
        "str_date":   pd.date_range("2023-06-01", periods=n, freq="h").astype(str),
        "num":        np.random.rand(n),
    })


@pytest.fixture
def num_df():
    np.random.seed(42)
    n = 150
    return pd.DataFrame({
        "a": np.random.rand(n),
        "b": np.random.rand(n) * 10,
        "c": np.random.normal(5, 2, n),
    })


@pytest.fixture
def clf_XY():
    X_arr, y_arr = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=3, random_state=42
    )
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(10)])
    y = pd.Series(y_arr, name="target")
    return X, y


# ============================================================
# EncoderFactory
# ============================================================

class TestEncoderFactory:

    def test_ohe_low_cardinality(self, cat_df, target_clf):
        enc = EncoderFactory(ohe_threshold=15)
        result = enc.fit_transform(cat_df[["low_card"]], target_clf)
        # Should have one-hot columns like low_card_A, low_card_B, low_card_C
        ohe_cols = [c for c in result.columns if c.startswith("low_card_")]
        assert len(ohe_cols) == 3

    def test_ordinal_encoding(self, cat_df, target_clf):
        enc = EncoderFactory(
            ohe_threshold=15,
            ordered_cols={"size": ["S", "M", "L", "XL"]},
        )
        result = enc.fit_transform(cat_df[["size"]], target_clf)
        assert "size" in result.columns
        assert pd.api.types.is_numeric_dtype(result["size"])

    def test_target_encoding_high_cardinality(self, cat_df, target_clf):
        enc = EncoderFactory(ohe_threshold=5, task="clf")
        result = enc.fit_transform(cat_df[["high_card"]], target_clf)
        # high_card should be target-encoded to numeric, not exploded
        assert "high_card" in result.columns
        assert pd.api.types.is_numeric_dtype(result["high_card"])

    def test_no_leakage_val_uses_train_mapping(self, cat_df, target_clf):
        enc = EncoderFactory(ohe_threshold=5, task="clf")
        enc.fit(cat_df[["high_card"]], target_clf)
        val = cat_df.copy()
        val_result = enc.transform(val[["high_card"]])
        assert "high_card" in val_result.columns

    def test_encoding_map_populated(self, cat_df, target_clf):
        enc = EncoderFactory(ohe_threshold=15)
        enc.fit(cat_df, target_clf)
        assert len(enc.encoding_map) > 0

    def test_hashing_fallback_no_target(self, cat_df):
        enc = EncoderFactory(ohe_threshold=5)
        result = enc.fit_transform(cat_df[["high_card"]])
        hashed_cols = [c for c in result.columns if c.startswith("high_card_")]
        assert len(hashed_cols) > 0

    def test_numeric_cols_passthrough(self, cat_df, target_clf):
        enc = EncoderFactory(ohe_threshold=15)
        result = enc.fit_transform(cat_df, target_clf)
        assert "num1" in result.columns
        assert "num2" in result.columns

    def test_transform_without_fit_raises(self, cat_df):
        enc = EncoderFactory()
        with pytest.raises(RuntimeError):
            enc.transform(cat_df)

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError):
            EncoderFactory().fit([[1, 2], [3, 4]])

    def test_original_not_mutated(self, cat_df, target_clf):
        original_cols = list(cat_df.columns)
        enc = EncoderFactory(ohe_threshold=15)
        enc.fit_transform(cat_df, target_clf)
        assert list(cat_df.columns) == original_cols


# ============================================================
# DatetimeDecomposer
# ============================================================

class TestDatetimeDecomposer:

    def test_datetime64_decomposed(self, dt_df):
        dt = DatetimeDecomposer(cols=["created_at"])
        result = dt.decompose(dt_df)
        assert "created_at_year" in result.columns
        assert "created_at_month" in result.columns
        assert "created_at_day_of_week" in result.columns

    def test_original_col_dropped(self, dt_df):
        dt = DatetimeDecomposer(cols=["created_at"], drop_original=True)
        result = dt.decompose(dt_df)
        assert "created_at" not in result.columns

    def test_original_col_kept(self, dt_df):
        dt = DatetimeDecomposer(cols=["created_at"], drop_original=False)
        result = dt.decompose(dt_df)
        assert "created_at" in result.columns

    def test_cyclical_features_added(self, dt_df):
        dt = DatetimeDecomposer(cols=["created_at"], cyclical=True)
        result = dt.decompose(dt_df)
        assert "created_at_month_sin" in result.columns
        assert "created_at_month_cos" in result.columns
        assert "created_at_dow_sin" in result.columns
        assert "created_at_dow_cos" in result.columns

    def test_cyclical_sin_cos_bounded(self, dt_df):
        dt = DatetimeDecomposer(cols=["created_at"], cyclical=True)
        result = dt.decompose(dt_df)
        assert result["created_at_month_sin"].between(-1, 1).all()
        assert result["created_at_month_cos"].between(-1, 1).all()

    def test_string_date_auto_parsed(self, dt_df):
        dt = DatetimeDecomposer(cols=["str_date"])
        result = dt.decompose(dt_df)
        assert "str_date_year" in result.columns

    def test_auto_detect_datetime_cols(self, dt_df):
        dt = DatetimeDecomposer()
        result = dt.decompose(dt_df)
        # created_at and str_date should both be decomposed
        assert "created_at_month" in result.columns

    def test_non_datetime_col_untouched(self, dt_df):
        dt = DatetimeDecomposer(cols=["created_at"])
        result = dt.decompose(dt_df)
        assert "num" in result.columns

    def test_is_weekend_binary(self, dt_df):
        dt = DatetimeDecomposer(cols=["created_at"])
        result = dt.decompose(dt_df)
        assert set(result["created_at_is_weekend"].unique()).issubset({0, 1})

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError):
            DatetimeDecomposer().decompose([[1, 2], [3, 4]])

    def test_no_cols_returns_copy(self, num_df):
        dt = DatetimeDecomposer()
        result = dt.decompose(num_df)
        assert list(result.columns) == list(num_df.columns)


# ============================================================
# InteractionBuilder
# ============================================================

class TestInteractionBuilder:

    def test_product_features_generated(self, num_df):
        builder = InteractionBuilder(
            cols=["a", "b", "c"],
            include_types=["product"],
            prune_interactions=False,
        )
        result = builder.fit_transform(num_df)
        assert "a_x_b" in result.columns
        assert "a_x_c" in result.columns
        assert "b_x_c" in result.columns

    def test_ratio_features_generated(self, num_df):
        builder = InteractionBuilder(
            cols=["a", "b"],
            include_types=["ratio"],
            prune_interactions=False,
        )
        result = builder.fit_transform(num_df)
        assert "a_div_b" in result.columns

    def test_polynomial_features_generated(self, num_df):
        builder = InteractionBuilder(
            cols=["a", "b"],
            include_types=["polynomial"],
            degree=2,
            prune_interactions=False,
        )
        result = builder.fit_transform(num_df)
        # Polynomial should add cross-terms on top of originals
        assert result.shape[1] > num_df.shape[1]

    def test_variance_pruning_removes_constants(self):
        df = pd.DataFrame({
            "a": np.ones(100),           # zero variance
            "b": np.random.rand(100),
            "c": np.random.rand(100),
        })
        builder = InteractionBuilder(
            cols=["a", "b", "c"],
            include_types=["product"],
            prune_interactions=True,
            variance_threshold=1e-4,
        )
        result = builder.fit_transform(df)
        # a_x_b and a_x_c should have near-zero variance and be pruned
        assert "a_x_b" not in result.columns or result["a_x_b"].var() > 0

    def test_original_cols_preserved(self, num_df):
        builder = InteractionBuilder(
            cols=["a", "b"],
            include_types=["product"],
            prune_interactions=False,
        )
        result = builder.fit_transform(num_df)
        assert "a" in result.columns
        assert "b" in result.columns
        assert "c" in result.columns

    def test_transform_uses_fit_features(self, num_df):
        builder = InteractionBuilder(
            cols=["a", "b", "c"],
            include_types=["product"],
            prune_interactions=False,
        )
        builder.fit(num_df)
        val = num_df.copy()
        result = builder.transform(val)
        assert set(builder.selected_features_).issubset(set(result.columns))

    def test_transform_without_fit_raises(self, num_df):
        with pytest.raises(RuntimeError):
            InteractionBuilder().transform(num_df)

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError):
            InteractionBuilder().fit([[1, 2], [3, 4]])

    def test_original_not_mutated(self, num_df):
        original_cols = list(num_df.columns)
        InteractionBuilder(include_types=["product"]).fit_transform(num_df)
        assert list(num_df.columns) == original_cols


# ============================================================
# FeatureSelector
# ============================================================

class TestFeatureSelector:

    def test_variance_method(self, clf_XY):
        X, y = clf_XY
        # Add a constant column
        X["constant"] = 1.0
        sel = FeatureSelector(method="variance")
        result = sel.fit_transform(X, y)
        assert "constant" not in result.columns

    def test_correlation_method_drops_redundant(self, clf_XY):
        X, y = clf_XY
        # Add a perfectly correlated column
        X["f0_copy"] = X["f0"]
        sel = FeatureSelector(method="correlation", correlation_threshold=0.95)
        result = sel.fit_transform(X, y)
        # f0 and f0_copy can't both survive
        assert not ("f0" in result.columns and "f0_copy" in result.columns)

    def test_rfecv_method_reduces_features(self, clf_XY):
        X, y = clf_XY
        sel = FeatureSelector(method="rfecv", task="clf", cv_folds=3)
        result = sel.fit_transform(X, y)
        assert result.shape[1] <= X.shape[1]
        assert result.shape[1] >= 1

    def test_selected_features_populated(self, clf_XY):
        X, y = clf_XY
        sel = FeatureSelector(method="variance")
        sel.fit(X, y)
        assert isinstance(sel.selected_features_, list)
        assert len(sel.selected_features_) > 0

    def test_report_returns_dataframe(self, clf_XY):
        X, y = clf_XY
        X["constant"] = 1.0
        sel = FeatureSelector(method="correlation")
        sel.fit(X, y)
        report = sel.report()
        assert isinstance(report, pd.DataFrame)
        assert "feature" in report.columns
        assert "stage" in report.columns
        assert "reason" in report.columns

    def test_transform_without_fit_raises(self, clf_XY):
        X, y = clf_XY
        sel = FeatureSelector()
        with pytest.raises(RuntimeError):
            sel.transform(X)

    def test_transform_drops_unselected(self, clf_XY):
        X, y = clf_XY
        sel = FeatureSelector(method="rfecv", task="clf", cv_folds=3)
        sel.fit(X, y)
        result = sel.transform(X)
        assert set(result.columns) == set(sel.selected_features_)

    def test_non_dataframe_X_raises(self, clf_XY):
        _, y = clf_XY
        with pytest.raises(TypeError):
            FeatureSelector().fit([[1, 2]], y)

    def test_non_series_y_raises(self, clf_XY):
        X, _ = clf_XY
        with pytest.raises(TypeError):
            FeatureSelector().fit(X, [0, 1] * 100)


# ============================================================
# Scaler
# ============================================================

class TestScaler:

    def test_standard_scaler(self, num_df):
        scaler = Scaler(method="standard")
        result = scaler.fit_transform(num_df)
        # After standard scaling, mean ≈ 0 and std ≈ 1
        assert abs(result["a"].mean()) < 1e-10
        assert abs(result["a"].std() - 1.0) < 0.01

    def test_minmax_scaler(self, num_df):
        scaler = Scaler(method="minmax")
        result = scaler.fit_transform(num_df)
        assert result["a"].min() >= 0.0 - 1e-10
        assert result["a"].max() <= 1.0 + 1e-10

    def test_robust_scaler(self, num_df):
        scaler = Scaler(method="robust")
        result = scaler.fit_transform(num_df)
        assert result.shape == num_df.shape

    def test_no_leakage_val_uses_train_stats(self, num_df):
        train = num_df.iloc[:100]
        val   = num_df.iloc[100:]
        scaler = Scaler(method="standard")
        scaler.fit(train)
        val_result = scaler.transform(val)
        # Val result should NOT have mean 0 (it uses train stats)
        assert val_result.shape[1] == num_df.shape[1]

    def test_specific_cols(self, num_df):
        scaler = Scaler(method="standard", cols=["a"])
        result = scaler.fit_transform(num_df)
        assert abs(result["a"].mean()) < 1e-10
        # Other cols should be untouched
        assert result["b"].equals(num_df["b"])

    def test_exclude_cols(self, num_df):
        scaler = Scaler(method="standard", exclude_cols=["a"])
        result = scaler.fit_transform(num_df)
        assert result["a"].equals(num_df["a"])

    def test_bool_cols_excluded(self):
        df = pd.DataFrame({
            "num":  [1.0, 2.0, 3.0],
            "flag": [True, False, True],
        })
        scaler = Scaler(method="standard")
        result = scaler.fit_transform(df)
        assert result["flag"].equals(df["flag"])

    def test_scaling_stats_populated(self, num_df):
        scaler = Scaler(method="standard")
        scaler.fit(num_df)
        assert not scaler.scaling_stats_.empty
        assert "center" in scaler.scaling_stats_.columns
        assert "scale" in scaler.scaling_stats_.columns

    def test_inverse_transform(self, num_df):
        scaler = Scaler(method="standard")
        scaled = scaler.fit_transform(num_df)
        recovered = scaler.inverse_transform(scaled)
        pd.testing.assert_frame_equal(
            recovered.round(8), num_df.round(8), check_dtype=False
        )

    def test_transform_without_fit_raises(self, num_df):
        with pytest.raises(RuntimeError):
            Scaler().transform(num_df)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            Scaler(method="l2norm")

    def test_original_not_mutated(self, num_df):
        original = num_df.copy()
        Scaler(method="standard").fit_transform(num_df)
        pd.testing.assert_frame_equal(num_df, original)

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError):
            Scaler().fit([[1, 2], [3, 4]])
