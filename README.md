# ds-toolkit

**Post-collection data science lifecycle toolkit.**  
From raw DataFrame to evaluated, tracked, and reported model — in composable, Jupyter-native Python.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue" alt="Python versions"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"/>
  <img src="https://img.shields.io/badge/tests-209%20passing-brightgreen" alt="Tests"/>
  <img src="https://img.shields.io/badge/sklearn-compatible-orange" alt="sklearn compatible"/>
</p>

---

## What is ds-toolkit?

`ds-toolkit` is an opinionated, production-ready library that wraps the messy middle of data science work — everything after you have data and before you have a deployed model. It gives you:

- **One-call profiling and validation** before you touch a single row
- **CV-safe preprocessing** that cannot leak across fold boundaries by design
- **Auto-selecting encoders and scalers** that make sensible choices without configuration
- **Multi-model CV harness** that ranks every estimator in one call
- **Optuna-powered tuning** with pre-built search spaces for every major model
- **SHAP explainability** that auto-picks TreeExplainer or KernelExplainer
- **MLflow experiment tracking** as a context manager — zero boilerplate
- **Model cards** generated from your result objects in two lines

Every module is `sklearn`-compatible (`fit` / `transform` / `fit_transform`), returns typed result objects with a `.display()` method that renders inline in Jupyter, and mutates nothing.

---

## Architecture

```
ds_toolkit/
├── core/        # Stage 1–2: profiling, validation, cleaning
├── features/    # Stage 3: encoding, engineering, selection
├── models/      # Stage 4: registry, CV, tuning, ensembles
├── eval/        # Stage 5: metrics, SHAP, plots, error analysis
├── infra/       # Stage 6: experiment logging, config, serialisation
└── reporting/   # Stage 7: notebook output, HTML export, model cards
```

---

## Installation

**Core (no optional deps):**
```bash
pip install ds-toolkit
```

**With boosting libraries:**
```bash
pip install "ds-toolkit[boosting]"      # XGBoost + LightGBM + CatBoost
```

**With tuning + tracking:**
```bash
pip install "ds-toolkit[tune,track]"    # Optuna + MLflow
```

**With SHAP explanations:**
```bash
pip install "ds-toolkit[explain]"       # shap
```

**Everything:**
```bash
pip install "ds-toolkit[all]"
```

**Development install (editable):**
```bash
git clone https://github.com/ShadowGodd1/ds-toolkit.git
cd ds-toolkit
pip install -e ".[dev]"
```

---

## Quick Start — Full Pipeline

```python
import pandas as pd
from ds_toolkit.core import DataProfiler, SchemaValidator, MissingHandler, OutlierDetector, TypeCaster
from ds_toolkit.features import EncoderFactory, DatetimeDecomposer, FeatureSelector, Scaler
from ds_toolkit.models import ModelRegistry, CVHarness, TunerOptuna
from ds_toolkit.eval import MetricsReport, ExplainerSHAP, DiagnosticPlotter, ErrorAnalyser
from ds_toolkit.infra import ExperimentLogger, ConfigManager, PipelineSerialiser
from ds_toolkit.reporting import NotebookReporter, generate_model_card

df = pd.read_csv("data/my_dataset.csv")
target_col = "label"

# ── Stage 1: Understand ──────────────────────────────────────────────────
profile = DataProfiler().profile(df)
profile.display()                          # renders inline in Jupyter

schema = {
    "age":    {"nullable": False, "min": 0, "max": 120},
    "email":  {"regex": r".+@.+\..+"},
}
validation = SchemaValidator().check(df, schema)
validation.display()

# ── Stage 2: Clean ───────────────────────────────────────────────────────
X = df.drop(columns=[target_col])
y = df[target_col]

X = TypeCaster().cast(X)
X, outlier_report = OutlierDetector(method="iqr", action="cap").detect(X)

handler = MissingHandler(strategy="median")
X = handler.fit_transform(X)

# ── Stage 3: Features ────────────────────────────────────────────────────
X = DatetimeDecomposer().decompose(X)

encoder = EncoderFactory(task="clf")
X = encoder.fit_transform(X, y)

scaler = Scaler(method="standard")
X = scaler.fit_transform(X)

selector = FeatureSelector(method="rfecv", task="clf")
X = selector.fit_transform(X, y)

# ── Stage 4: Train ───────────────────────────────────────────────────────
models = ModelRegistry.get(task="clf")

harness = CVHarness(task="clf", n_splits=5, scoring="roc_auc")
cv_results = harness.run(models, X, y)
cv_results.display()

best_name, best_model = cv_results.best_model

# Optional: tune the best model
tuner = TunerOptuna(task="clf", n_trials=100)
tune_result = tuner.tune(best_model, X, y)
best_model.set_params(**tune_result.best_params)
best_model.fit(X, y)

# ── Stage 5: Evaluate ────────────────────────────────────────────────────
y_pred  = best_model.predict(X)
y_proba = best_model.predict_proba(X)

metrics = MetricsReport(task="clf").report(y, y_pred, y_proba)
metrics.display()

shap_result = ExplainerSHAP(top_n=10).explain(best_model, X)
shap_result.display()

diag = DiagnosticPlotter().diagnostics(best_model, X, y)
diag.display()

errors = ErrorAnalyser(n_worst=0.1).analyse(best_model, X, y)
errors.display()

# ── Stage 6: Track ───────────────────────────────────────────────────────
logger = ExperimentLogger(tracking_uri="./mlruns")

with logger.run("my_experiment", params={"model": best_name}) as run:
    logger.log_metrics(metrics.metrics_df["value"].to_dict())
    logger.log_model(best_model, name=best_name)
    logger.log_shap(shap_result)

serialiser = PipelineSerialiser(output_dir="./models")
save_result = serialiser.save(best_model, name=best_name)

# ── Stage 7: Report ──────────────────────────────────────────────────────
NotebookReporter().display(cv_results, metrics, shap_result)

card = generate_model_card(
    best_model,
    cv_results=cv_results,
    eval_results=metrics,
    shap_result=shap_result,
    error_report=errors,
    experiment_info={"run_id": run.run_id},
)
card.display()
print(card.to_md())    # export as Markdown
```

---

## Stage Reference

### Stage 1 — Data Understanding & Validation

#### `DataProfiler`

One-call dataset summary: shape, dtypes, memory, missing%, cardinality, skew, kurtosis, outlier flag.

```python
from ds_toolkit.core import DataProfiler

profiler = DataProfiler(
    cardinality_threshold=50,   # columns with ≤N unique values → categorical
    outlier_method="iqr",       # 'iqr' | 'zscore' | 'both'
    missing_threshold=0.05,     # warn if missing% exceeds this
)
result = profiler.profile(df)
result.display()                # Jupyter inline
result.summary_df               # pd.DataFrame — one row per column
result.warnings                 # list[str]
```

#### `SchemaValidator`

Pydantic-backed schema enforcement. Raises or returns a violations report.

```python
from ds_toolkit.core import SchemaValidator

schema = {
    "age":    {"dtype": "numeric", "nullable": False, "min": 0, "max": 120},
    "email":  {"regex": r".+@.+\..+"},
    "status": {"allowed": ["active", "inactive"]},
    "id":     {"unique": True, "nullable": False},
}
result = SchemaValidator(strict=False).check(df, schema)
result.passed           # bool
result.violations_df    # pd.DataFrame — [column, check, detail]
```

#### `DistributionReport`

Auto-generates histograms, KDE plots, QQ plots, box plots, and correlation heatmap. Exports self-contained HTML.

```python
from ds_toolkit.core import DistributionReport

result = DistributionReport().run(df, output_dir="reports/")
result.html_path        # Path to saved HTML
result.display()        # inline in Jupyter
```

---

### Stage 2 — Data Cleaning & Preprocessing

#### `MissingHandler`

Per-column imputation — CV-safe (fit on train only).

```python
from ds_toolkit.core import MissingHandler

handler = MissingHandler(
    strategy="median",                      # global fallback
    col_strategies={"city": "mode",         # per-column overrides
                    "note": "constant"},
    fill_values={"note": "unknown"},
    knn_neighbors=5,
)
X_train_clean = handler.fit_transform(X_train)
X_val_clean   = handler.transform(X_val)    # uses train statistics
```

Supported strategies: `mean`, `median`, `mode`, `constant`, `knn`, `mice`, `none`.

#### `OutlierDetector`

```python
from ds_toolkit.core import OutlierDetector

detector = OutlierDetector(
    method="iqr",                           # 'iqr' | 'zscore' | 'isoforest' | 'lof'
    action="cap",                           # 'flag' | 'cap' | 'drop'
    col_actions={"revenue": "drop"},        # per-column action override
    iqr_factor=1.5,
)
result_df, report = detector.detect(df)
```

#### `TypeCaster`

```python
from ds_toolkit.core import TypeCaster

caster = TypeCaster(
    cardinality_threshold=50,   # object cols with ≤N unique → category
    downcast_numerics=True,     # int64 → smallest safe int
    parse_dates=True,           # detect and parse date strings
)
df_typed = caster.cast(df)
caster.change_log               # list of {column, from, to}
```

#### `Deduplicator`

```python
from ds_toolkit.core import Deduplicator

dedup = Deduplicator(
    keys=["patient_id", "visit_date"],  # exact dedup keys
    fuzzy_cols=["full_name"],           # fuzzy dedup columns (requires rapidfuzz)
    fuzzy_threshold=90,
)
df_clean = dedup.clean(df)
dedup.report()                          # pd.DataFrame — rows removed
```

---

### Stage 3 — Feature Engineering

#### `EncoderFactory`

Auto-selects encoding by cardinality and task type.

| Condition | Strategy |
|---|---|
| Column has ordered metadata | `OrdinalEncoder` |
| Cardinality ≤ `ohe_threshold` (default 15) | `OneHotEncoder` |
| Cardinality > threshold + target available | `TargetEncoder` (smoothed, CV-safe) |
| Cardinality > threshold + no target | `HashingEncoder` |

```python
from ds_toolkit.features import EncoderFactory

enc = EncoderFactory(
    task="clf",
    ohe_threshold=15,
    ordered_cols={"size": ["S", "M", "L", "XL"]},
)
X_train_enc = enc.fit_transform(X_train, y_train)
X_val_enc   = enc.transform(X_val)
enc.encoding_map                # dict: column → strategy used
```

#### `DatetimeDecomposer`

```python
from ds_toolkit.features import DatetimeDecomposer

dt = DatetimeDecomposer(
    cols=["created_at"],        # None = auto-detect all datetime cols
    cyclical=True,              # add sin/cos encodings for month, dow, hour
    add_holidays=True,          # requires: pip install holidays
    country_code="KE",          # ISO country code for holiday calendar
)
df_expanded = dt.decompose(df)
# Adds: created_at_year, _month, _day, _day_of_week, _is_weekend,
#       _month_sin, _month_cos, _dow_sin, _dow_cos, ...
```

#### `InteractionBuilder`

```python
from ds_toolkit.features import InteractionBuilder

builder = InteractionBuilder(
    cols=["age", "income", "score"],
    include_types=["product", "ratio"],  # 'polynomial' | 'product' | 'ratio'
    prune_interactions=True,             # drop near-zero-variance interactions
    top_k=20,                            # optional: RF-based top-k selection
)
X_train_int = builder.fit_transform(X_train, y_train)
X_val_int   = builder.transform(X_val)
builder.selected_features_              # list of surviving feature names
```

#### `FeatureSelector`

Multi-stage pipeline: variance → correlation → RFECV → SHAP (each stage toggleable).

```python
from ds_toolkit.features import FeatureSelector

selector = FeatureSelector(
    method="rfecv",             # 'variance' | 'correlation' | 'rfecv' | 'shap'
    task="clf",
    correlation_threshold=0.95,
    cv_folds=5,
)
X_train_sel = selector.fit_transform(X_train, y_train)
X_val_sel   = selector.transform(X_val)
selector.selected_features_     # list of kept features
selector.report()               # pd.DataFrame — [feature, stage, reason]
```

#### `Scaler`

```python
from ds_toolkit.features import Scaler

scaler = Scaler(
    method="standard",           # 'standard' | 'minmax' | 'robust'
    exclude_cols=["id", "flag"], # never scale these
)
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
scaler.scaling_stats_            # pd.DataFrame — center/scale per column
```

---

### Stage 4 — Model Training & Selection

#### `ModelRegistry`

```python
from ds_toolkit.models import ModelRegistry

models = ModelRegistry.get(task="clf")           # all available
models = ModelRegistry.get(task="clf",
    include=["lr", "rf", "xgboost"])             # only these
models = ModelRegistry.get(task="clf",
    exclude=["mlp"])                             # all except these
```

Built-in keys: `lr`, `rf`, `gbm`, `et`, `mlp`, `xgboost`, `lightgbm`, `catboost`

#### `CVHarness`

```python
from ds_toolkit.models import CVHarness

harness = CVHarness(
    task="clf",
    n_splits=5,
    scoring="roc_auc",
    verbose=True,
)
cv_results = harness.run(models, X_train, y_train)
cv_results.summary_df           # ranked by mean_score
cv_results.best_model           # (name, fitted estimator)
cv_results.display()            # inline table in Jupyter
```

CV strategy is auto-selected:

| Condition | Strategy |
|---|---|
| `task='clf'`, balanced | `StratifiedKFold(n_splits=5)` |
| `task='clf'`, imbalanced | `StratifiedKFold` + `class_weight='balanced'` |
| `task='reg'` | `KFold(n_splits=5, shuffle=True)` |
| `task='ts'` | `TimeSeriesSplit(n_splits=5)` |

#### `TunerOptuna`

```python
from ds_toolkit.models import TunerOptuna
from sklearn.ensemble import RandomForestClassifier

tuner = TunerOptuna(
    task="clf",
    n_trials=100,
    cv_folds=5,
    scoring="roc_auc",
)
result = tuner.tune(RandomForestClassifier(), X_train, y_train)
result.best_params               # dict — apply with model.set_params(**result.best_params)
result.best_score
result.study                     # optuna.Study for further analysis
```

Pre-built search spaces: `LogisticRegression`, `Ridge`, `RandomForest`, `ExtraTrees`, `GradientBoosting`, `XGBoost`, `LightGBM`.

#### `EnsembleBuilder`

```python
from ds_toolkit.models import EnsembleBuilder

builder = EnsembleBuilder(
    task="clf",
    method="stack",              # 'stack' | 'vote' | 'blend'
    meta_learner="lr",           # 'lr' | 'ridge' | any sklearn estimator
    cv_folds=5,
)
ensemble = builder.build(models, X_train, y_train)
preds = ensemble.predict(X_val)
proba = ensemble.predict_proba(X_val)
```

---

### Stage 5 — Evaluation & Diagnostics

#### `MetricsReport`

```python
from ds_toolkit.eval import MetricsReport

result = MetricsReport(task="clf").report(y_true, y_pred, y_proba=y_proba)
result.metrics_df                # pd.DataFrame — metric → value
result.display()
```

| Task | Primary Metrics | Secondary Metrics |
|---|---|---|
| Binary clf | ROC-AUC, F1, Precision, Recall | Log-loss, MCC, PR-AUC |
| Multi-class clf | Macro F1, Accuracy | Per-class P/R/F1 |
| Regression | RMSE, MAE, R² | MAPE, Adj. R², Max error |

#### `ExplainerSHAP`

```python
from ds_toolkit.eval import ExplainerSHAP

result = ExplainerSHAP(top_n=10).explain(model, X)
result.display()                 # summary plot inline
result.values                    # raw SHAP values (n_samples × n_features)
result.figures                   # dict: 'summary', 'bar', 'dependence_<col>'
```

Auto-selects `TreeExplainer` for tree-based models, `KernelExplainer` for all others.

#### `DiagnosticPlotter`

```python
from ds_toolkit.eval import DiagnosticPlotter

result = DiagnosticPlotter().diagnostics(model, X, y)
result.display()
result.figures                   # dict of matplotlib figures
```

**Classification:** confusion matrix (raw + normalised), ROC curve, PR curve, calibration plot  
**Regression:** residuals vs fitted, Q-Q plot, scale-location, Cook's distance

#### `ErrorAnalyser`

```python
from ds_toolkit.eval import ErrorAnalyser

result = ErrorAnalyser(n_worst=0.1).analyse(model, X, y)
result.segments_df               # feature distribution shift: worst vs rest
result.worst_df                  # the n_worst mis-predicted rows
result.display()
```

---

### Stage 6 — Experiment Tracking & Reproducibility

#### `ExperimentLogger`

```python
from ds_toolkit.infra import ExperimentLogger

logger = ExperimentLogger(tracking_uri="./mlruns")

with logger.run("my_experiment", params={"model": "rf", "n_estimators": 200}) as run:
    model.fit(X_train, y_train)
    logger.log_metrics({"roc_auc": 0.91, "f1": 0.87})
    logger.log_model(model, name="random_forest")
    logger.log_shap(shap_result)

print(run.run_id)
print(run.artifact_uri)
```

Auto-logged per run: params, metrics, model artifact, SHAP plot, requirements.txt snapshot, git commit hash.

#### `ConfigManager`

```python
from ds_toolkit.infra import ConfigManager

# config/experiment.yaml:
# model:
#   n_estimators: 200
#   task: clf
# data:
#   target_col: ${TARGET_COL}   # resolved from env var

cfg = ConfigManager.load(
    "config/experiment.yaml",
    required=["data.target_col", "model.task"],
)
cfg.model.n_estimators           # 200
cfg.data.target_col              # value from $TARGET_COL
```

#### `PipelineSerialiser`

```python
from ds_toolkit.infra import PipelineSerialiser

serial = PipelineSerialiser(output_dir="./models")

# Save with SHA-256 checksum + metadata sidecar
result = serial.save(
    pipeline,
    name="rf_v1",
    metadata={"roc_auc": 0.91, "trained_on": "2024-01-15"},
)
print(result.path)               # ./models/rf_v1_20240115_143022.pkl
print(result.checksum)           # SHA-256 hex

# Load — raises ChecksumError if file was tampered
model = serial.load(result.path)
```

---

### Stage 7 — Reporting & Notebook Output

#### `NotebookReporter`

```python
from ds_toolkit.reporting import NotebookReporter

NotebookReporter().display(
    cv_results=cv_results,
    eval_results=metrics,
    shap_result=shap_result,
    title="Patient Readmission Model — v1",
)
```

#### `HTMLExporter`

```python
from ds_toolkit.reporting import HTMLExporter

result = HTMLExporter().export(
    output_path="reports/experiment_v1.html",
    cv_results=cv_results,
    eval_results=metrics,
    shap_result=shap_result,
    diagnostic_result=diag,
    title="Experiment Report",
)
# Self-contained HTML — no external deps, safe to email
```

#### `ModelCard`

```python
from ds_toolkit.reporting import generate_model_card

card = generate_model_card(
    model,
    cv_results=cv_results,
    eval_results=metrics,
    shap_result=shap_result,
    error_report=errors,
    experiment_info={"run_id": run.run_id, "git_hash": "a1b2c3d"},
)
card.display()                   # inline in Jupyter
card.to_md()                     # Markdown string
card.to_html()                   # HTML string
```

---

## Design Principles

1. **No side effects.** Every module accepts a DataFrame or model and returns a new object. Nothing is mutated in place.
2. **CV-safety by default.** Anything that touches the target (`TargetEncoder`, `MissingHandler`, `Scaler`, `FeatureSelector`) has a `fit` / `transform` split. Fit on train. Transform on val/test.
3. **Jupyter-native.** Every result object has a `.display()` method that renders rich HTML inline. Nothing requires a separate report step.
4. **Stack-agnostic.** XGBoost, LightGBM, CatBoost, and all sklearn estimators are first-class citizens across every stage.
5. **Optional dependencies stay optional.** `shap`, `optuna`, `mlflow`, `rapidfuzz`, and the boosting libraries are never imported at the top level. They are imported at call time and fail with a clear install message.

---

## Running Tests

```bash
# All 209 tests
pytest

# Specific stage
pytest tests/test_core/
pytest tests/test_features/
pytest tests/test_models/
pytest tests/test_eval/

# With coverage
pytest --cov=ds_toolkit --cov-report=html
```

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Quick contribution flow:**
```bash
git clone https://github.com/ShadowGodd1/ds-toolkit.git
cd ds-toolkit
pip install -e ".[dev]"
git checkout -b feature/my-feature
# make changes
pytest
git push origin feature/my-feature
# open a Pull Request
```

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**Adnan Mohamud**  
CEO & Founder, [PataDoc](https://patadoc.com) — The Partner in Health in Your Hand  
[github.com/ShadowGodd1](https://github.com/ShadowGodd1)
