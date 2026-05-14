# Changelog

All notable changes to ds-toolkit are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2025-01-01

### Added

**Stage 1 — Data Understanding & Validation**
- `DataProfiler` — one-call dataset summary: shape, dtypes, memory, missing%, cardinality, skew, kurtosis, outlier flag per column
- `SchemaValidator` — Pydantic-backed schema enforcement with dtype, range, null, uniqueness, regex checks; `strict=True` mode raises on first violation
- `DistributionReport` — auto-generates histograms, KDE plots, QQ plots, box plots, correlation heatmap; exports self-contained HTML

**Stage 2 — Data Cleaning & Preprocessing**
- `MissingHandler` — 7 imputation strategies (mean, median, mode, constant, KNN, MICE, none); per-column overrides; CV-safe fit/transform split
- `OutlierDetector` — IQR, Z-score, Isolation Forest, LOF detection; flag/cap/drop actions per column
- `TypeCaster` — auto datetime parsing, int64/float64 downcast, low-cardinality → category; full change log
- `Deduplicator` — exact dedup on key columns + fuzzy dedup via `rapidfuzz`

**Stage 3 — Feature Engineering**
- `EncoderFactory` — auto-selects OHE / OrdinalEncoder / TargetEncoder (smoothed, CV-safe) / HashingEncoder by cardinality and task
- `DatetimeDecomposer` — year/month/day/dow/quarter/week + sin/cos cyclical encodings + holiday flag; auto-detects datetime columns
- `InteractionBuilder` — product (A×B), ratio (A/B), polynomial interactions; variance pruning; optional RF-based top-k selection
- `FeatureSelector` — 4-stage pipeline: variance threshold → correlation filter → RFECV → SHAP; full drop report per feature
- `Scaler` — standard / minmax / robust; auto-detects numeric columns; excludes booleans; `inverse_transform` support

**Stage 4 — Model Training & Selection**
- `ModelRegistry` — catalogue of pre-configured estimators for clf and reg tasks; optional boosting libraries auto-detected
- `CVHarness` — stratified/KFold/TimeSeriesSplit CV across multiple models; auto class-weight on imbalanced data; ranked summary
- `TunerOptuna` — Optuna hyperparameter search with pre-built search spaces for LR, Ridge, RF, ExtraTrees, GBM, XGBoost, LightGBM
- `EnsembleBuilder` — stacking (configurable meta-learner), voting (soft), weighted blending (CV-derived weights)

**Stage 5 — Evaluation & Diagnostics**
- `MetricsReport` — auto-detects clf/reg; full suite of classification and regression metrics; tidy DataFrame output
- `ExplainerSHAP` — auto-selects TreeExplainer / KernelExplainer; summary, bar, waterfall, dependence plots
- `DiagnosticPlotter` — confusion matrix, ROC, PR curve, calibration (clf); residuals, Q-Q, scale-location, Cook's distance (reg)
- `ErrorAnalyser` — worst-prediction cohort segmentation; feature distribution shift detection; worst vs rest comparison

**Stage 6 — Experiment Tracking & Reproducibility**
- `ExperimentLogger` — MLflow context manager; auto-logs params, metrics, model, SHAP, requirements.txt, git hash
- `ConfigManager` — YAML/JSON loader with `${ENV_VAR}` override syntax; dot-access; required-key validation; version stamp
- `PipelineSerialiser` — versioned .pkl with SHA-256 checksum + metadata sidecar; raises `ChecksumError` on tampered files

**Stage 7 — Reporting & Notebook Output**
- `NotebookReporter` — metric cards, ranked model table, SHAP bar chart, all inline in Jupyter
- `HTMLExporter` — single self-contained HTML export (base64-embedded images); safe to email
- `ModelCard` — Mitchell et al. format; `.display()` / `.to_html()` / `.to_md()` outputs; `generate_model_card()` convenience function

**Infrastructure**
- Full `pyproject.toml` with optional dependency groups: `boosting`, `fuzzy`, `explain`, `tune`, `track`, `all`, `dev`, `docs`
- 209-test suite across all 7 stages using pytest
- GitHub Actions CI (Python 3.9, 3.10, 3.11, 3.12)
- `ruff` + `black` + `mypy` toolchain configuration

---

## [Unreleased]

- `docs/` MkDocs site with per-module API reference
- PyPI publish workflow
- Pre-commit hook configuration
