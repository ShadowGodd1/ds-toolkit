"""
tests for Stages 5, 6, 7 — Eval, Infra, Reporting.
"""
# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_data():
    X_arr, y_arr = make_classification(
        n_samples=200, n_features=8, n_informative=4, random_state=0
    )
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(8)])
    y = pd.Series(y_arr, name="target")
    return X, y


@pytest.fixture
def reg_data():
    X_arr, y_arr = make_regression(
        n_samples=200, n_features=8, n_informative=4, random_state=0
    )
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(8)])
    y = pd.Series(y_arr, name="target")
    return X, y


@pytest.fixture
def fitted_clf(clf_data):
    X, y = clf_data
    model = RandomForestClassifier(n_estimators=20, random_state=0)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def fitted_reg(reg_data):
    X, y = reg_data
    model = RandomForestRegressor(n_estimators=20, random_state=0)
    model.fit(X, y)
    return model, X, y


# ============================================================
# Stage 5 — MetricsReport
# ============================================================

class TestMetricsReport:
    from ds_toolkit.eval import MetricsReport, MetricsResult

    def test_clf_returns_metrics_result(self, fitted_clf):
        from ds_toolkit.eval import MetricsReport
        model, X, y = fitted_clf
        y_pred  = model.predict(X)
        y_proba = model.predict_proba(X)
        result  = MetricsReport(task="clf").report(y, y_pred, y_proba)
        from ds_toolkit.eval import MetricsResult
        assert isinstance(result, MetricsResult)
        assert result.task == "clf"

    def test_clf_metrics_present(self, fitted_clf):
        from ds_toolkit.eval import MetricsReport
        model, X, y = fitted_clf
        result = MetricsReport(task="clf").report(y, model.predict(X),
                                                   model.predict_proba(X))
        assert "accuracy" in result.metrics_df.index
        assert "f1"       in result.metrics_df.index
        assert "roc_auc"  in result.metrics_df.index

    def test_reg_returns_metrics_result(self, fitted_reg):
        from ds_toolkit.eval import MetricsReport
        model, X, y = fitted_reg
        result = MetricsReport(task="reg").report(y, model.predict(X))
        assert result.task == "reg"
        assert "rmse" in result.metrics_df.index
        assert "r2"   in result.metrics_df.index
        assert "mae"  in result.metrics_df.index

    def test_auto_task_detection_clf(self, fitted_clf):
        from ds_toolkit.eval import MetricsReport
        model, X, y = fitted_clf
        result = MetricsReport().report(y, model.predict(X))
        assert result.task == "clf"

    def test_auto_task_detection_reg(self, fitted_reg):
        from ds_toolkit.eval import MetricsReport
        model, X, y = fitted_reg
        result = MetricsReport().report(y, model.predict(X))
        assert result.task == "reg"

    def test_metrics_values_are_floats(self, fitted_clf):
        from ds_toolkit.eval import MetricsReport
        model, X, y = fitted_clf
        result = MetricsReport(task="clf").report(y, model.predict(X))
        for val in result.metrics_df["value"]:
            assert isinstance(val, float)

    def test_display_returns_self(self, fitted_clf):
        from ds_toolkit.eval import MetricsReport
        model, X, y = fitted_clf
        result = MetricsReport(task="clf").report(y, model.predict(X))
        assert result.display() is result


# ============================================================
# Stage 5 — DiagnosticPlotter
# ============================================================

class TestDiagnosticPlotter:

    def test_clf_returns_diagnostic_result(self, fitted_clf):
        from ds_toolkit.eval import DiagnosticPlotter, DiagnosticResult
        model, X, y = fitted_clf
        result = DiagnosticPlotter().diagnostics(model, X, y, task="clf")
        assert isinstance(result, DiagnosticResult)

    def test_clf_figures_generated(self, fitted_clf):
        from ds_toolkit.eval import DiagnosticPlotter
        model, X, y = fitted_clf
        result = DiagnosticPlotter().diagnostics(model, X, y, task="clf")
        assert len(result.figures) >= 2
        assert "confusion_matrix" in result.figures

    def test_reg_figures_generated(self, fitted_reg):
        from ds_toolkit.eval import DiagnosticPlotter
        model, X, y = fitted_reg
        result = DiagnosticPlotter().diagnostics(model, X, y, task="reg")
        assert len(result.figures) >= 2
        assert "residuals_vs_fitted" in result.figures
        assert "qq_plot" in result.figures

    def test_non_dataframe_raises(self, fitted_clf):
        from ds_toolkit.eval import DiagnosticPlotter
        model, X, y = fitted_clf
        with pytest.raises(TypeError):
            DiagnosticPlotter().diagnostics(model, [[1, 2]], y)

    def test_display_returns_self(self, fitted_clf):
        from ds_toolkit.eval import DiagnosticPlotter
        model, X, y = fitted_clf
        result = DiagnosticPlotter().diagnostics(model, X, y, task="clf")
        assert result.display() is result


# ============================================================
# Stage 5 — ErrorAnalyser
# ============================================================

class TestErrorAnalyser:

    def test_returns_error_report(self, fitted_clf):
        from ds_toolkit.eval import ErrorAnalyser, ErrorReport
        model, X, y = fitted_clf
        result = ErrorAnalyser(n_worst=0.1).analyse(model, X, y)
        assert isinstance(result, ErrorReport)

    def test_worst_df_size(self, fitted_clf):
        from ds_toolkit.eval import ErrorAnalyser
        model, X, y = fitted_clf
        result = ErrorAnalyser(n_worst=0.1).analyse(model, X, y)
        expected = max(1, int(len(y) * 0.1))
        assert result.n_worst == expected
        assert len(result.worst_df) == expected

    def test_segments_df_has_flagged_col(self, fitted_clf):
        from ds_toolkit.eval import ErrorAnalyser
        model, X, y = fitted_clf
        result = ErrorAnalyser(n_worst=0.2).analyse(model, X, y)
        assert "flagged" in result.segments_df.columns
        assert "std_shift" in result.segments_df.columns

    def test_invalid_n_worst_raises(self):
        from ds_toolkit.eval import ErrorAnalyser
        with pytest.raises(ValueError):
            ErrorAnalyser(n_worst=1.5)

    def test_non_dataframe_raises(self, fitted_clf):
        from ds_toolkit.eval import ErrorAnalyser
        model, _, y = fitted_clf
        with pytest.raises(TypeError):
            ErrorAnalyser().analyse(model, [[1, 2]], y)

    def test_reg_task(self, fitted_reg):
        from ds_toolkit.eval import ErrorAnalyser
        model, X, y = fitted_reg
        result = ErrorAnalyser(n_worst=0.1, task="reg").analyse(model, X, y)
        assert result.task == "reg"


# ============================================================
# Stage 6 — PipelineSerialiser
# ============================================================

class TestPipelineSerialiser:

    def test_save_creates_pkl(self, fitted_clf, tmp_path):
        from ds_toolkit.infra import PipelineSerialiser
        model, _, _ = fitted_clf
        s = PipelineSerialiser(output_dir=tmp_path)
        result = s.save(model, name="test_model")
        assert result.path.exists()
        assert result.path.suffix == ".pkl"

    def test_save_creates_metadata(self, fitted_clf, tmp_path):
        from ds_toolkit.infra import PipelineSerialiser
        model, _, _ = fitted_clf
        s = PipelineSerialiser(output_dir=tmp_path)
        result = s.save(model, name="test_model", metadata={"score": 0.91})
        assert result.metadata_path.exists()
        meta = json.loads(result.metadata_path.read_text())
        assert meta["score"] == 0.91
        assert "sha256" in meta

    def test_load_recovers_model(self, fitted_clf, tmp_path):
        from ds_toolkit.infra import PipelineSerialiser
        model, X, y = fitted_clf
        s = PipelineSerialiser(output_dir=tmp_path)
        result  = s.save(model, name="test_model")
        loaded  = s.load(result.path)
        preds_orig   = model.predict(X)
        preds_loaded = loaded.predict(X)
        np.testing.assert_array_equal(preds_orig, preds_loaded)

    def test_checksum_verified_on_load(self, fitted_clf, tmp_path):
        from ds_toolkit.infra import PipelineSerialiser, ChecksumError
        model, _, _ = fitted_clf
        s = PipelineSerialiser(output_dir=tmp_path, checksum_on_load=True)
        result = s.save(model, name="tamper_test")
        # Tamper with the pkl
        result.path.write_bytes(b"tampered_content")
        with pytest.raises(ChecksumError):
            s.load(result.path)

    def test_load_missing_file_raises(self, tmp_path):
        from ds_toolkit.infra import PipelineSerialiser
        s = PipelineSerialiser(tmp_path)
        with pytest.raises(FileNotFoundError):
            s.load(tmp_path / "nonexistent.pkl")

    def test_load_metadata(self, fitted_clf, tmp_path):
        from ds_toolkit.infra import PipelineSerialiser
        model, _, _ = fitted_clf
        s = PipelineSerialiser(output_dir=tmp_path)
        result = s.save(model, name="meta_test", metadata={"version": "v1"})
        meta = s.load_metadata(result.path)
        assert meta["version"] == "v1"
        assert meta["estimator_class"] == "RandomForestClassifier"


# ============================================================
# Stage 6 — ConfigManager
# ============================================================

class TestConfigManager:

    def test_load_yaml(self, tmp_path):
        from ds_toolkit.infra import ConfigManager
        pytest.importorskip("yaml")
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("model:\n  n_estimators: 100\ndata:\n  target: label\n")
        cfg = ConfigManager.load(cfg_file)
        assert cfg.model.n_estimators == 100
        assert cfg.data.target == "label"

    def test_load_json(self, tmp_path):
        from ds_toolkit.infra import ConfigManager
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text('{"model": {"lr": 0.01}, "task": "clf"}')
        cfg = ConfigManager.load(cfg_file)
        assert cfg.model.lr == 0.01
        assert cfg.task == "clf"

    def test_env_var_resolution(self, tmp_path, monkeypatch):
        from ds_toolkit.infra import ConfigManager
        monkeypatch.setenv("MY_SECRET_KEY", "abc123")
        cfg_file = tmp_path / "cfg.json"
        cfg_file.write_text('{"api_key": "${MY_SECRET_KEY}"}')
        cfg = ConfigManager.load(cfg_file)
        assert cfg.api_key == "abc123"

    def test_required_key_missing_raises(self, tmp_path):
        from ds_toolkit.infra import ConfigManager
        cfg_file = tmp_path / "cfg.json"
        cfg_file.write_text('{"model": {"n_estimators": 100}}')
        with pytest.raises(KeyError, match="target_col"):
            ConfigManager.load(cfg_file, required=["data.target_col"])

    def test_from_dict(self):
        from ds_toolkit.infra import ConfigManager
        cfg = ConfigManager.from_dict({"a": 1, "b": {"c": 2}})
        assert cfg.a == 1
        assert cfg.b.c == 2

    def test_meta_stamp_added(self, tmp_path):
        from ds_toolkit.infra import ConfigManager
        cfg_file = tmp_path / "cfg.json"
        cfg_file.write_text('{"x": 1}')
        cfg = ConfigManager.load(cfg_file)
        assert hasattr(cfg, "__meta__")
        assert hasattr(cfg.__meta__, "toolkit_version")

    def test_missing_file_raises(self):
        from ds_toolkit.infra import ConfigManager
        with pytest.raises(FileNotFoundError):
            ConfigManager.load("/nonexistent/config.yaml")

    def test_to_dict_roundtrip(self, tmp_path):
        from ds_toolkit.infra import ConfigManager
        cfg_file = tmp_path / "cfg.json"
        data = {"model": {"n": 50}, "task": "clf"}
        cfg_file.write_text(json.dumps(data))
        cfg  = ConfigManager.load(cfg_file)
        d    = cfg.to_dict()
        assert d["task"] == "clf"
        assert d["model"]["n"] == 50


# ============================================================
# Stage 7 — ModelCard
# ============================================================

class TestModelCard:

    def test_generate_model_card(self, fitted_clf):
        from ds_toolkit.reporting import generate_model_card
        from ds_toolkit.eval import MetricsReport
        model, X, y = fitted_clf
        eval_result = MetricsReport(task="clf").report(y, model.predict(X),
                                                        model.predict_proba(X))
        card = generate_model_card(model, eval_results=eval_result)
        assert card.model_name == "RandomForestClassifier"
        assert card.task == "clf"

    def test_metrics_in_card(self, fitted_clf):
        from ds_toolkit.reporting import generate_model_card
        from ds_toolkit.eval import MetricsReport
        model, X, y = fitted_clf
        eval_result = MetricsReport(task="clf").report(y, model.predict(X))
        card = generate_model_card(model, eval_results=eval_result)
        assert card.metrics is not None
        assert "accuracy" in card.metrics

    def test_to_md(self, fitted_clf):
        from ds_toolkit.reporting import generate_model_card
        from ds_toolkit.eval import MetricsReport
        model, X, y = fitted_clf
        eval_result = MetricsReport(task="clf").report(y, model.predict(X))
        card = generate_model_card(model, eval_results=eval_result)
        md = card.to_md()
        assert "# Model Card" in md
        assert "RandomForestClassifier" in md

    def test_to_html(self, fitted_clf):
        from ds_toolkit.reporting import generate_model_card
        from ds_toolkit.eval import MetricsReport
        model, X, y = fitted_clf
        eval_result = MetricsReport(task="clf").report(y, model.predict(X))
        card = generate_model_card(model, eval_results=eval_result)
        html = card.to_html()
        assert "<div" in html
        assert "RandomForestClassifier" in html

    def test_display_returns_self(self, fitted_clf):
        from ds_toolkit.reporting import generate_model_card
        from ds_toolkit.eval import MetricsReport
        model, X, y = fitted_clf
        eval_result = MetricsReport(task="clf").report(y, model.predict(X))
        card = generate_model_card(model, eval_results=eval_result)
        assert card.display() is card


# ============================================================
# Stage 7 — HTMLExporter
# ============================================================

class TestHTMLExporter:

    def test_creates_html_file(self, fitted_clf, tmp_path):
        from ds_toolkit.reporting import HTMLExporter
        from ds_toolkit.eval import MetricsReport, DiagnosticPlotter
        model, X, y = fitted_clf
        eval_result = MetricsReport(task="clf").report(y, model.predict(X))
        diag_result = DiagnosticPlotter().diagnostics(model, X, y, task="clf")
        exporter = HTMLExporter()
        result = exporter.export(
            tmp_path / "report.html",
            eval_results=eval_result,
            diagnostic_result=diag_result,
        )
        assert result.html_path.exists()

    def test_html_is_self_contained(self, fitted_clf, tmp_path):
        from ds_toolkit.reporting import HTMLExporter
        from ds_toolkit.eval import MetricsReport
        model, X, y = fitted_clf
        eval_result = MetricsReport(task="clf").report(y, model.predict(X))
        result = HTMLExporter().export(
            tmp_path / "report.html", eval_results=eval_result
        )
        content = result.html_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "accuracy" in content

    def test_creates_parent_dirs(self, fitted_clf, tmp_path):
        from ds_toolkit.reporting import HTMLExporter
        from ds_toolkit.eval import MetricsReport
        model, X, y = fitted_clf
        eval_result = MetricsReport(task="clf").report(y, model.predict(X))
        deep_path = tmp_path / "a" / "b" / "c" / "report.html"
        HTMLExporter().export(deep_path, eval_results=eval_result)
        assert deep_path.exists()
