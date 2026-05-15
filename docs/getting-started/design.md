# Design Principles

ds-toolkit is built around five hard rules. Every module follows all five.

---

## 1. No side effects

Every module accepts a DataFrame or model object and returns a **new** transformed object.  
The input is never mutated.

```python
# This is guaranteed safe — df is unchanged
df_typed = TypeCaster().cast(df)
assert df.equals(original_df)   # always True
```

---

## 2. CV-safety is a first-class concern

Any transformer that learns from data — including from the target — **must** have a `fit()` / `transform()` split.

```python
# WRONG — leaks validation statistics into training
handler = MissingHandler()
X_all_clean = handler.fit_transform(X)         # learns from val/test too
X_train, X_val = train_test_split(X_all_clean) # too late

# RIGHT — statistics learned on train only
X_train, X_val = train_test_split(X)
handler = MissingHandler()
X_train_clean = handler.fit_transform(X_train) # learns only here
X_val_clean   = handler.transform(X_val)       # applies train stats
```

Affected modules: `MissingHandler`, `Scaler`, `EncoderFactory` (TargetEncoder), `FeatureSelector`.

---

## 3. Jupyter-native output

Every result object has:
- `_repr_html_()` — renders automatically in Jupyter cells
- `.display()` — explicit render, returns `self` for chaining

```python
profile = DataProfiler().profile(df)
profile.display()        # explicit
profile                  # also renders in Jupyter automatically
```

---

## 4. Stack-agnostic

XGBoost, LightGBM, CatBoost, and all sklearn estimators are treated identically across every stage.  
`CVHarness`, `TunerOptuna`, `ExplainerSHAP`, and `EnsembleBuilder` all handle any of them transparently.

---

## 5. Optional dependencies are truly optional

Heavy libraries — `shap`, `optuna`, `mlflow`, `rapidfuzz`, `catboost` — are never imported at module level.  
They are imported at call time and fail with a clear, actionable message:

```python
# What you see if shap is not installed
>>> ExplainerSHAP().explain(model, X)
ImportError: ExplainerSHAP requires shap: pip install shap
```

This means `import ds_toolkit` is always fast and never fails due to an optional dep.
