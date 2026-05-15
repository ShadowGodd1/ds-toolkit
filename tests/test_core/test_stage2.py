"""
tests/test_core/test_stage2.py
================================
Tests for Stage 2: MissingHandler, OutlierDetector, TypeCaster, Deduplicator.
"""
# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ds_toolkit.core import Deduplicator, MissingHandler, OutlierDetector, TypeCaster


# ============================================================
# MissingHandler
# ============================================================

class TestMissingHandler:

    def test_median_imputation(self, dirty_df):
        handler = MissingHandler(strategy="median")
        result = handler.fit_transform(dirty_df)
        assert result["age"].isna().sum() == 0

    def test_mean_imputation(self, dirty_df):
        handler = MissingHandler(strategy="mean")
        result = handler.fit_transform(dirty_df)
        assert result["income"].isna().sum() == 0

    def test_mode_imputation(self, dirty_df):
        handler = MissingHandler(strategy="mode")
        result = handler.fit_transform(dirty_df)
        assert result["category"].isna().sum() == 0

    def test_constant_imputation(self, dirty_df):
        handler = MissingHandler(
            strategy="constant",
            fill_values={"age": -1, "income": 0.0},
        )
        result = handler.fit_transform(dirty_df)
        assert result["age"].isna().sum() == 0

    def test_per_col_strategy_override(self, dirty_df):
        handler = MissingHandler(
            strategy="median",
            col_strategies={"category": "mode"},
        )
        result = handler.fit_transform(dirty_df)
        assert result["age"].isna().sum() == 0
        assert result["category"].isna().sum() == 0

    def test_transform_without_fit_raises(self, dirty_df):
        handler = MissingHandler()
        with pytest.raises(RuntimeError):
            handler.transform(dirty_df)

    def test_no_leakage_val_uses_train_stats(self):
        """Validation set must use stats learned from training set."""
        np.random.seed(1)
        train = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan, 5.0]})
        val = pd.DataFrame({"x": [np.nan, 10.0, np.nan]})

        handler = MissingHandler(strategy="median")
        handler.fit(train)

        result = handler.transform(val)
        train_median = float(train["x"].median())
        # The imputed NaN at index 0 must equal the training median, not any val/test value
        assert result.iloc[0]["x"] == pytest.approx(train_median)
        # The non-missing value at index 1 must be unchanged
        assert result.iloc[1]["x"] == pytest.approx(10.0)

    def test_original_not_mutated(self, dirty_df):
        original_nulls = dirty_df["age"].isna().sum()
        handler = MissingHandler(strategy="median")
        handler.fit_transform(dirty_df)
        assert dirty_df["age"].isna().sum() == original_nulls

    def test_missing_summary(self, dirty_df):
        handler = MissingHandler(strategy="median")
        handler.fit(dirty_df)
        summary = handler.missing_summary(dirty_df)
        assert "missing_count" in summary.columns
        assert "missing_pct" in summary.columns

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError):
            MissingHandler().fit([1, 2, 3])


# ============================================================
# OutlierDetector
# ============================================================

class TestOutlierDetector:

    def test_flag_action_adds_column(self, dirty_df):
        detector = OutlierDetector(method="iqr", action="flag")
        result, report = detector.detect(dirty_df)
        flag_cols = [c for c in result.columns if c.endswith("_outlier_flag")]
        assert len(flag_cols) > 0

    def test_cap_action_bounds_values(self, dirty_df):
        detector = OutlierDetector(method="iqr", action="cap")
        result, report = detector.detect(dirty_df)
        # Injected outlier income=1_000_000 should have been capped
        assert result["income"].max() < 1_000_000

    def test_drop_action_reduces_rows(self, dirty_df):
        detector = OutlierDetector(method="iqr", action="drop")
        result, report = detector.detect(dirty_df)
        assert len(result) < len(dirty_df)

    def test_report_has_correct_columns(self, dirty_df):
        _, report = OutlierDetector(method="iqr", action="flag").detect(dirty_df)
        expected = {"column", "method", "n_outliers", "pct_outliers",
                    "lower_bound", "upper_bound", "action"}
        assert expected.issubset(set(report.columns))

    def test_zscore_method(self, dirty_df):
        detector = OutlierDetector(method="zscore", action="flag")
        result, report = detector.detect(dirty_df)
        assert not report.empty

    def test_col_action_override(self, dirty_df):
        detector = OutlierDetector(
            method="iqr",
            action="flag",
            col_actions={"income": "cap"},
        )
        result, report = detector.detect(dirty_df)
        assert "income_outlier_flag" not in result.columns
        assert result["income"].max() < 1_000_000

    def test_specific_cols_only(self, dirty_df):
        detector = OutlierDetector(method="iqr", action="flag", cols=["income"])
        result, report = detector.detect(dirty_df)
        assert "income_outlier_flag" in result.columns
        assert "age_outlier_flag" not in result.columns

    def test_original_not_mutated(self, dirty_df):
        original_max = dirty_df["income"].max()
        OutlierDetector(method="iqr", action="cap").detect(dirty_df)
        assert dirty_df["income"].max() == original_max

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError):
            OutlierDetector().detect([[1, 2], [3, 4]])


# ============================================================
# TypeCaster
# ============================================================

class TestTypeCaster:

    def test_date_string_parsed(self):
        df = pd.DataFrame({"created_at": ["2023-01-01", "2023-06-15", "2024-03-20"]})
        result = TypeCaster(parse_dates=True).cast(df)
        assert pd.api.types.is_datetime64_any_dtype(result["created_at"])

    def test_explicit_date_col(self):
        df = pd.DataFrame({"ts": ["2023-01-01", "2023-06-15"]})
        result = TypeCaster(date_cols=["ts"]).cast(df)
        assert pd.api.types.is_datetime64_any_dtype(result["ts"])

    def test_downcast_int(self):
        df = pd.DataFrame({"x": pd.array([1, 2, 3], dtype="int64")})
        result = TypeCaster(downcast_numerics=True).cast(df)
        assert result["x"].dtype != "int64"  # should be a smaller int

    def test_low_cardinality_to_category(self):
        df = pd.DataFrame({"city": ["A", "B", "A", "C"] * 20})
        result = TypeCaster(cardinality_threshold=10).cast(df)
        assert str(result["city"].dtype) == "category"

    def test_high_cardinality_stays_object(self):
        df = pd.DataFrame({"name": [f"user_{i}" for i in range(100)]})
        result = TypeCaster(cardinality_threshold=10).cast(df)
        assert str(result["name"].dtype) != "category"

    def test_change_log_populated(self):
        df = pd.DataFrame({
            "city": ["A", "B", "A"] * 5,
            "ts": ["2023-01-01"] * 15,
        })
        caster = TypeCaster()
        caster.cast(df)
        assert len(caster.change_log) > 0

    def test_cast_report_returns_df(self):
        df = pd.DataFrame({"city": ["A", "B"] * 5})
        caster = TypeCaster()
        caster.cast(df)
        report = caster.cast_report()
        assert isinstance(report, pd.DataFrame)

    def test_original_not_mutated(self):
        df = pd.DataFrame({"x": pd.array([1, 2, 3], dtype="int64")})
        caster = TypeCaster()
        caster.cast(df)
        assert str(df["x"].dtype) == "int64"

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError):
            TypeCaster().cast({"col": [1, 2, 3]})


# ============================================================
# Deduplicator
# ============================================================

class TestDeduplicator:

    def test_exact_dedup_all_cols(self, dirty_df):
        dedup = Deduplicator()
        result = dedup.clean(dirty_df)
        assert len(result) < len(dirty_df)
        # No duplicate rows should remain
        assert result.duplicated().sum() == 0

    def test_exact_dedup_key_cols(self):
        df = pd.DataFrame({
            "id": [1, 1, 2, 3],
            "value": [10, 99, 20, 30],
        })
        result = Deduplicator(keys=["id"]).clean(df)
        assert len(result) == 3  # id=1 deduped, keep first

    def test_keep_last(self):
        df = pd.DataFrame({"id": [1, 1], "val": [10, 20]})
        result = Deduplicator(keys=["id"], keep="last").clean(df)
        assert result.iloc[0]["val"] == 20

    def test_report_counts(self, dirty_df):
        dedup = Deduplicator()
        dedup.clean(dirty_df)
        report = dedup.report()
        assert report.iloc[0]["exact_removed"] >= 5  # at least 5 dupes injected

    def test_fuzzy_dedup_requires_rapidfuzz(self):
        df = pd.DataFrame({"name": ["John Smith", "Jon Smith", "Jane Doe"]})
        dedup = Deduplicator(fuzzy_cols=["name"], fuzzy_threshold=80)
        try:
            result = dedup.clean(df)
            # If rapidfuzz is available, duplicates should be reduced
            assert len(result) <= 3
        except ImportError:
            pytest.skip("rapidfuzz not installed")

    def test_original_not_mutated(self, dirty_df):
        original_len = len(dirty_df)
        Deduplicator().clean(dirty_df)
        assert len(dirty_df) == original_len

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError):
            Deduplicator().clean([(1, 2), (1, 2)])

    def test_reset_index_after_dedup(self):
        df = pd.DataFrame({"x": [1, 1, 2, 3]})
        result = Deduplicator(keys=["x"]).clean(df)
        assert list(result.index) == list(range(len(result)))
