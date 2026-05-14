# Quick Start

A complete end-to-end pipeline from raw DataFrame to model card.

```python
import pandas as pd
from ds_toolkit.core import (
    DataProfiler, SchemaValidator,
    MissingHandler, OutlierDetector, TypeCaster,
)
from ds_toolkit.features import (
    EncoderFactory, DatetimeDecomposer,
    FeatureSelector, Scaler,
)
from ds_toolkit.models import ModelRegistry, CVHarness, TunerOptuna
from ds_toolkit.eval import (
    MetricsReport, ExplainerSHAP,
    DiagnosticPlotter, ErrorAnalyser,
)
from ds_toolkit.infra import ExperimentLogger, PipelineSerialiser
from ds_toolkit.reporting import NotebookReporter, generate_model_card

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/patients.csv")
X  = df.drop(columns=["readmitted"])
y  = df["readmitted"]

# ── Stage 1: Understand ──────────────────────────────────────────────────────
DataProfiler().profile(df).display()

SchemaValidator().check(df, {
    "age":    {"nullable": False, "min": 0, "max": 120},
    "gender": {"allowed": ["M", "F", "Other"]},
}).display()

# ── Stage 2: Clean ───────────────────────────────────────────────────────────
X = TypeCaster().cast(X)
X, _ = OutlierDetector(method="iqr", action="cap").detect(X)
X    = MissingHandler(strategy="median").fit_transform(X)

# ── Stage 3: Features ────────────────────────────────────────────────────────
X = DatetimeDecomposer().decompose(X)
X = EncoderFactory(task="clf").fit_transform(X, y)
X = Scaler(method="standard").fit_transform(X)
X = FeatureSelector(method="rfecv", task="clf").fit_transform(X, y)

# ── Stage 4: Train ───────────────────────────────────────────────────────────
cv_results = CVHarness(task="clf", scoring="roc_auc").run(
    ModelRegistry.get(task="clf"), X, y
)
cv_results.display()

best_name, best_model = cv_results.best_model

# ── Stage 5: Evaluate ────────────────────────────────────────────────────────
metrics = MetricsReport(task="clf").report(
    y, best_model.predict(X), best_model.predict_proba(X)
)
metrics.display()

shap = ExplainerSHAP(top_n=10).explain(best_model, X)
shap.display()

DiagnosticPlotter().diagnostics(best_model, X, y).display()
ErrorAnalyser(n_worst=0.1).analyse(best_model, X, y).display()

# ── Stage 6: Track ───────────────────────────────────────────────────────────
logger = ExperimentLogger("./mlruns")
with logger.run("readmission_v1", params={"model": best_name}) as run:
    logger.log_metrics(metrics.metrics_df["value"].to_dict())
    logger.log_model(best_model, name=best_name)

PipelineSerialiser("./models").save(best_model, name=best_name)

# ── Stage 7: Report ──────────────────────────────────────────────────────────
NotebookReporter().display(cv_results, metrics, shap)

generate_model_card(
    best_model,
    cv_results=cv_results,
    eval_results=metrics,
    shap_result=shap,
).display()
```
