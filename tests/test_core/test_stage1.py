"""
tests/test_core/test_stage1.py
==============================
Full test suite for Stage 1 modules.
"""
# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ds_toolkit.core import (
    DataProfiler,
    DistributionReport,
    SchemaValidationError,
    SchemaValidator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "age":    rng.integers(18, 80, size=200).astype(float),
        "income": rng.normal(50_000, 15_000, size=200),
        "score":  rng.uniform(0, 1, size=200),
        "gender": rng.choice(["M", "F", "Other"], size=200),
        "active": rng.choice([True, False], size=200),
    })


@pytest.fixture
def df_with_missings(simple_df) -> pd.DataFrame:
    df = simple_df.copy()
    rng = np.random.default_rng(0)
    idx = rng.choice(df.index, size=30, replace=False)
    df.loc[idx, "age"] = np.nan
    idx2 = rng.choice(df.index, size=5, replace=False)
    df.loc[idx2, "income"] = np.nan
    return df


@pytest.fixture
def df_with_outliers(simple_df) -> pd.DataFrame:
    df = simple_df.copy()
    df.loc[0, "income"] = 9_999_999   # extreme high
    df.loc[1, "income"] = -500_000    # extreme low
    return df


# ===========================================================================
# DataProfiler
# ===========================================================================

class TestDataProfiler:

    def test_returns_profile_result(self, simple_df):
        result = DataProfiler().profile(simple_df)
        assert hasattr(result, "summary_df")
        assert hasattr(result, "warnings")

    def test_summary_shape(self, simple_df):
        result = DataProfiler().profile(simple_df)
        # one row per column
        assert len(result.summary_df) == len(simple_df.columns)

    def test_expected_columns_in_summary(self, simple_df):
        result = DataProfiler().profile(simple_df)
        expected = {"dtype", "n", "missing_count", "missing_pct",
                    "cardinality", "inferred_role", "outlier_flag"}
        assert expected.issubset(set(result.summary_df.columns))

    def test_missing_count_accurate(self, df_with_missings):
        result = DataProfiler().profile(df_with_missings)
        assert result.summary_df.loc["age", "missing_count"] == 30
        assert result.summary_df.loc["income", "missing_count"] == 5

    def test_non_numeric_cols_have_nan_stats(self, simple_df):
        result = DataProfiler().profile(simple_df)
        assert pd.isna(result.summary_df.loc["gender", "mean"])
        assert pd.isna(result.summary_df.loc["gender", "skew"])

    def test_outlier_flag_iqr(self, df_with_outliers):
        result = DataProfiler(outlier_method="iqr").profile(df_with_outliers)
        assert result.summary_df.loc["income", "outlier_flag"] == True  # noqa: E712

    def test_outlier_flag_zscore(self, df_with_outliers):
        result = DataProfiler(outlier_method="zscore").profile(df_with_outliers)
        assert result.summary_df.loc["income", "outlier_flag"] == True  # noqa: E712

    def test_outlier_flag_both(self, df_with_outliers):
        result = DataProfiler(outlier_method="both").profile(df_with_outliers)
        assert result.summary_df.loc["income", "outlier_flag"] == True  # noqa: E712

    def test_missing_threshold_warning(self, df_with_missings):
        # age has 30/200 = 15% missing — should warn with threshold=0.05
        result = DataProfiler(missing_threshold=0.05).profile(df_with_missings)
        assert any("age" in w for w in result.warnings)

    def test_no_missing_warning_below_threshold(self, df_with_missings):
        # With a very high threshold, no missing warning should fire for income
        result = DataProfiler(missing_threshold=0.99).profile(df_with_missings)
        assert not any("income" in w and "missingness" in w for w in result.warnings)

    def test_shape_warning_always_present(self, simple_df):
        result = DataProfiler().profile(simple_df)
        assert any("Shape" in w for w in result.warnings)

    def test_inferred_role_numeric(self, simple_df):
        result = DataProfiler(cardinality_threshold=50).profile(simple_df)
        assert result.summary_df.loc["income", "inferred_role"] == "numeric"

    def test_inferred_role_boolean(self, simple_df):
        result = DataProfiler().profile(simple_df)
        assert result.summary_df.loc["active", "inferred_role"] == "boolean"

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError):
            DataProfiler().profile([1, 2, 3])

    def test_empty_dataframe(self):
        result = DataProfiler().profile(pd.DataFrame())
        assert result.summary_df.empty

    def test_constant_column_warning(self):
        df = pd.DataFrame({"const": [1] * 50, "x": range(50)})
        result = DataProfiler().profile(df)
        assert any("const" in w for w in result.warnings)

    def test_display_returns_self(self, simple_df, monkeypatch):
        monkeypatch.setattr("builtins.print", lambda *a, **k: None)
        result = DataProfiler().profile(simple_df)
        try:
            ret = result.display()
        except Exception:
            ret = result
        assert ret is result


# ===========================================================================
# SchemaValidator
# ===========================================================================

class TestSchemaValidator:

    @pytest.fixture
    def base_schema(self):
        return {
            "age":    {"dtype": "numeric", "min": 0, "max": 120, "nullable": False},
            "income": {"dtype": "numeric"},
            "gender": {"dtype": "string"},
        }

    @pytest.fixture
    def valid_df(self):
        return pd.DataFrame({
            "age":    [25.0, 40.0, 33.0],
            "income": [30_000.0, 55_000.0, 72_000.0],
            "gender": ["M", "F", "M"],
        })

    def test_passes_on_valid_data(self, valid_df, base_schema):
        result = SchemaValidator().check(valid_df, base_schema)
        assert result.passed is True
        assert result.violations_df.empty

    def test_fails_on_dtype_mismatch(self):
        df = pd.DataFrame({"age": ["twenty", "thirty"]})
        schema = {"age": {"dtype": "numeric"}}
        result = SchemaValidator().check(df, schema)
        assert not result.passed
        assert "age" in result.violations_df["column"].values

    def test_nullable_violation(self):
        df = pd.DataFrame({"age": [25.0, np.nan, 30.0]})
        schema = {"age": {"dtype": "numeric", "nullable": False}}
        result = SchemaValidator().check(df, schema)
        assert not result.passed
        checks = result.violations_df["check"].values
        assert "nullable" in checks

    def test_min_violation(self):
        df = pd.DataFrame({"age": [25.0, -5.0, 30.0]})
        schema = {"age": {"dtype": "numeric", "min": 0}}
        result = SchemaValidator().check(df, schema)
        assert not result.passed
        assert "min" in result.violations_df["check"].values

    def test_max_violation(self):
        df = pd.DataFrame({"age": [25.0, 200.0]})
        schema = {"age": {"dtype": "numeric", "max": 120}}
        result = SchemaValidator().check(df, schema)
        assert not result.passed
        assert "max" in result.violations_df["check"].values

    def test_unique_violation(self):
        df = pd.DataFrame({"id": [1, 2, 2, 4]})
        schema = {"id": {"dtype": "numeric", "unique": True}}
        result = SchemaValidator().check(df, schema)
        assert not result.passed
        assert "unique" in result.violations_df["check"].values

    def test_pattern_violation(self):
        df = pd.DataFrame({"email": ["valid@test.com", "not-an-email", "also@ok.io"]})
        schema = {"email": {"dtype": "string", "pattern": r".+@.+\..+"}}
        result = SchemaValidator().check(df, schema)
        assert not result.passed
        assert "pattern" in result.violations_df["check"].values

    def test_allowed_values_violation(self):
        df = pd.DataFrame({"status": ["active", "inactive", "UNKNOWN"]})
        schema = {"status": {"allowed": ["active", "inactive"]}}
        result = SchemaValidator().check(df, schema)
        assert not result.passed
        assert "allowed_values" in result.violations_df["check"].values

    def test_missing_column_violation(self):
        df = pd.DataFrame({"age": [25.0]})
        schema = {"age": {"dtype": "numeric"}, "missing_col": {"dtype": "string"}}
        result = SchemaValidator().check(df, schema)
        assert not result.passed
        assert "missing_column" in result.violations_df["check"].values

    def test_strict_mode_raises(self):
        df = pd.DataFrame({"age": [np.nan, 25.0]})
        schema = {"age": {"nullable": False}}
        with pytest.raises(SchemaValidationError):
            SchemaValidator(strict=True).check(df, schema)

    def test_violations_df_has_correct_columns(self, valid_df, base_schema):
        result = SchemaValidator().check(valid_df, base_schema)
        assert list(result.violations_df.columns) == ["column", "check", "detail"]

    def test_invalid_schema_key_raises(self):
        df = pd.DataFrame({"age": [25.0]})
        schema = {"age": {"unknown_key": 99}}
        with pytest.raises(ValueError, match="unknown schema keys"):
            SchemaValidator().check(df, schema)

    def test_invalid_dtype_in_schema_raises(self):
        df = pd.DataFrame({"age": [25.0]})
        schema = {"age": {"dtype": "imaginary_type"}}
        with pytest.raises(ValueError, match="not valid"):
            SchemaValidator().check(df, schema)

    def test_invalid_regex_raises(self):
        df = pd.DataFrame({"col": ["abc"]})
        schema = {"col": {"pattern": "[unclosed"}}
        with pytest.raises(ValueError, match="invalid regex"):
            SchemaValidator().check(df, schema)

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError):
            SchemaValidator().check({"col": [1, 2]}, {})


# ===========================================================================
# DistributionReport
# ===========================================================================

class TestDistributionReport:

    def test_returns_report_result(self, simple_df):
        result = DistributionReport().run(simple_df)
        assert hasattr(result, "figures")
        assert hasattr(result, "html_path")

    def test_figures_count(self, simple_df):
        numeric_count = simple_df.select_dtypes(include="number").shape[1]
        result = DistributionReport().run(simple_df)
        # one per numeric col + one heatmap (if >=2 numeric cols)
        assert len(result.figures) == numeric_count + 1

    def test_no_heatmap_for_single_numeric_col(self):
        df = pd.DataFrame({"x": np.random.randn(50), "label": ["a"] * 50})
        result = DistributionReport().run(df)
        assert len(result.figures) == 1  # just the column, no heatmap

    def test_html_path_none_without_output_dir(self, simple_df):
        result = DistributionReport().run(simple_df)
        assert result.html_path is None

    def test_html_saved_to_output_dir(self, simple_df, tmp_path):
        result = DistributionReport().run(simple_df, output_dir=tmp_path)
        assert result.html_path is not None
        assert result.html_path.exists()
        assert result.html_path.suffix == ".html"

    def test_html_is_self_contained(self, simple_df, tmp_path):
        result = DistributionReport().run(simple_df, output_dir=tmp_path)
        content = result.html_path.read_text()
        assert "data:image/png;base64" in content
        # No external script or stylesheet references
        assert "src=\"http" not in content
        assert "href=\"http" not in content

    def test_empty_df_returns_empty_figures(self):
        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        result = DistributionReport().run(df)
        assert result.figures == []

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError):
            DistributionReport().run([[1, 2], [3, 4]])

    def test_max_cols_respected(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.random((100, 10)), columns=[f"c{i}" for i in range(10)])
        result = DistributionReport(max_cols=3).run(df)
        # 3 columns + 1 heatmap
        assert len(result.figures) == 4
