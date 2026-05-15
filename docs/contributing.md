# Contributing to ds-toolkit

Thank you for your interest in contributing. ds-toolkit is an open-source project maintained by [Adnan Mohamud](https://github.com/ShadowGodd1).

---

## Before You Start

- Check the [issue tracker](https://github.com/ShadowGodd1/ds-toolkit/issues) to see if your bug or feature request already exists
- For significant changes, open an issue first to discuss the approach
- Small fixes (typos, doc improvements, test additions) can go straight to a PR

---

## Development Setup

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/<your-username>/ds-toolkit.git
cd ds-toolkit

# 2. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 3. Install optional dependencies for full test coverage
pip install -e ".[all]"

# 4. Verify the full test suite passes
pytest
# Expected: 209 passed
```

---

## Project Structure

```
ds_toolkit/
├── core/           # Stage 1–2: profiling, validation, cleaning
├── features/       # Stage 3: encoding, engineering, selection
├── models/         # Stage 4: registry, CV, tuning, ensembles
├── eval/           # Stage 5: metrics, SHAP, plots, error analysis
├── infra/          # Stage 6: experiment logging, config, serialisation
└── reporting/      # Stage 7: notebook output, HTML export, model cards

tests/
├── conftest.py               # shared fixtures
├── test_core/
│   ├── test_stage1.py
│   └── test_stage2.py
├── test_features/
│   └── test_stage3.py
├── test_models/
│   └── test_stage4.py
└── test_eval/
    └── test_stages567.py
```

---

## Contribution Guidelines

### Code style

- **Formatter:** `black` (line length 100)
- **Linter:** `ruff`
- **Type hints:** required on all public methods

```bash
black ds_toolkit/
ruff check ds_toolkit/ --fix
mypy ds_toolkit/
```

### Module design rules

Every new module must follow the existing patterns:

| Rule | Detail |
|---|---|
| No side effects | Accept DataFrame/model, return new object — never mutate inputs |
| CV-safe | If a transformer touches the target, it must have `fit()` / `transform()` split |
| sklearn-compatible | Implement `fit`, `transform`, `fit_transform` (or `BaseEstimator` subclass) |
| Jupyter-native | Result objects must have `_repr_html_()` and a `display()` method |
| Optional deps | Import heavy optional libs (shap, optuna, mlflow) inside methods, not at module top level |
| Author header | Every `.py` file must include the author/license header block |

### File header template

Every Python file must begin with:
```python
"""
ds_toolkit.<module>.<filename>
===============================
<one-line description>
...
"""

# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT
```

### Tests

- Every new public method must have at least one test
- Tests go in the matching `tests/test_<stage>/` directory
- Use the shared fixtures in `tests/conftest.py` where possible
- Tests must pass on Python 3.9, 3.10, 3.11, 3.12
- Do not use `unittest` — pytest only

```bash
# Run tests for a specific stage
pytest tests/test_core/ -v

# Run with coverage
pytest --cov=ds_toolkit --cov-report=html
```

---

## Pull Request Process

1. Branch from `main`:
   ```bash
   git checkout -b feature/my-feature
   # or
   git checkout -b fix/issue-123
   ```

2. Make your changes following the guidelines above

3. Run the full test suite:
   ```bash
   pytest
   black --check ds_toolkit/
   ruff check ds_toolkit/
   ```

4. Update `CHANGELOG.md` under `[Unreleased]`

5. Push your branch and open a PR against `main`

6. Fill in the PR template — describe what changed and why

---

## Adding a New Module

1. Create `ds_toolkit/<stage>/<module_name>.py`
2. Add the author/license header
3. Implement the class following the design rules above
4. Export from `ds_toolkit/<stage>/__init__.py`
5. Write tests in `tests/test_<stage>/`
6. Add a section to `README.md` under the correct stage
7. Add an entry to `CHANGELOG.md`

---

## Reporting Bugs

Open an issue with:
- Python version and OS
- ds-toolkit version (`python -c "import ds_toolkit; print(ds_toolkit.__version__)"`)
- Minimal reproducible example
- Full traceback

---

## Code of Conduct

Be respectful. Constructive criticism of code is welcome; personal attacks are not.

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
