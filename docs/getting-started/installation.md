# Installation

## Requirements

- Python 3.9, 3.10, 3.11, or 3.12
- pip ≥ 21

---

## Core install

Installs the base toolkit with sklearn, pandas, numpy, scipy, matplotlib, seaborn, pydantic, and IPython.

```bash
pip install dstoolkit-adnan
```

---

## Optional dependency groups

ds-toolkit keeps heavy optional libraries out of the required dependencies. Install only what you need.

| Group | Libraries installed | When you need it |
|---|---|---|
| `boosting` | XGBoost, LightGBM, CatBoost | `ModelRegistry` boosting estimators |
| `fuzzy` | rapidfuzz | `Deduplicator` fuzzy dedup |
| `explain` | shap | `ExplainerSHAP`, `FeatureSelector(method='shap')` |
| `tune` | optuna | `TunerOptuna` |
| `track` | mlflow, pyyaml | `ExperimentLogger`, `ConfigManager` (YAML) |
| `all` | all of the above | full pipeline |
| `dev` | pytest, black, ruff, mypy | contributing / development |
| `docs` | mkdocs, mkdocs-material, mkdocstrings | building these docs |

```bash
pip install "dstoolkit-adnan[boosting]"
pip install "dstoolkit-adnan[tune,track]"
pip install "dstoolkit-adnan[explain]"
pip install "dstoolkit-adnan[all]"
```

---

## Development install

```bash
git clone https://github.com/ShadowGodd1/ds-toolkit.git
cd ds-toolkit
pip install -e ".[dev]"
```

Verify:
```bash
pytest
# Expected: 209 passed
```
