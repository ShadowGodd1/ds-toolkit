# ds-toolkit

**Post-collection data science lifecycle toolkit.**  
From raw DataFrame to evaluated, tracked, and reported model — in composable, Jupyter-native Python.

---

## What is ds-toolkit?

`ds-toolkit` is an opinionated, production-ready library that covers every stage of data science work after you have data and before you have a deployed model.

It is built on four convictions:

1. **Preprocessing must be CV-safe by default.** Any transformer that touches the target must have a hard `fit()` / `transform()` split with no way to accidentally leak.
2. **Every result should render inline in Jupyter.** No separate report step. Every result object has a `.display()` method that produces rich HTML output.
3. **Optional dependencies must stay optional.** `shap`, `optuna`, `mlflow`, `catboost` are imported at call time with clear install instructions — not silently required at import.
4. **Nothing is mutated.** Every module accepts a DataFrame or model and returns a new object.

---

## Architecture

```
ds_toolkit/
├── core/        Stage 1–2  profiling, validation, cleaning
├── features/    Stage 3    encoding, engineering, selection
├── models/      Stage 4    registry, CV, tuning, ensembles
├── eval/        Stage 5    metrics, SHAP, plots, error analysis
├── infra/       Stage 6    experiment logging, config, serialisation
└── reporting/   Stage 7    notebook output, HTML export, model cards
```

---

## Quick Install

```bash
pip install ds-toolkit           # core
pip install "ds-toolkit[all]"    # everything
```

See [Installation](getting-started/installation.md) for full options.

---

## Author

**Adnan Mohamud** — CEO & Founder, [PataDoc](https://patadoc.com)  
[github.com/ShadowGodd1](https://github.com/ShadowGodd1)

---

## License

[MIT](https://github.com/ShadowGodd1/ds-toolkit/blob/main/LICENSE)
