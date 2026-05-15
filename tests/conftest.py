"""
tests/conftest.py
=================
Shared pytest fixtures for the ds_toolkit test suite.
"""
# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Small clean DataFrame
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "age":      np.random.randint(18, 80, n).astype("int64"),
        "income":   np.random.normal(55_000, 15_000, n),
        "score":    np.random.uniform(0, 1, n),
        "category": np.random.choice(["A", "B", "C"], n),
        "city":     np.random.choice(["Nairobi", "Mombasa", "Kisumu"], n),
        "label":    np.random.randint(0, 2, n),
    })


# ---------------------------------------------------------------------------
# DataFrame with intentional issues
# ---------------------------------------------------------------------------

@pytest.fixture
def dirty_df() -> pd.DataFrame:
    np.random.seed(7)
    n = 300
    df = pd.DataFrame({
        "age":        np.random.randint(18, 80, n).astype("int64"),
        "income":     np.random.normal(50_000, 20_000, n),
        "score":      np.random.uniform(0, 1, n),
        "category":   np.random.choice(["A", "B", "C", None], n),
        "signup_date": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
        "email":      [f"user{i}@example.com" for i in range(n)],
        "label":      np.random.randint(0, 2, n),
    })

    # Inject missing values
    missing_idx = np.random.choice(n, 30, replace=False)
    df.loc[missing_idx, "age"] = np.nan
    df.loc[missing_idx[:10], "income"] = np.nan

    # Inject outliers
    df.loc[0, "income"] = 1_000_000
    df.loc[1, "income"] = -500_000

    # Inject duplicates
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)

    return df


# ---------------------------------------------------------------------------
# Minimal numeric-only DataFrame
# ---------------------------------------------------------------------------

@pytest.fixture
def numeric_df() -> pd.DataFrame:
    np.random.seed(0)
    return pd.DataFrame({
        "x": np.random.normal(0, 1, 100),
        "y": np.random.normal(5, 2, 100),
        "z": np.random.exponential(2, 100),
    })
