"""
ds_toolkit.core.validator
=========================
SchemaValidator: Pydantic-backed schema enforcement.

Checks per column:
  dtype compatibility, value ranges, null constraints,
  uniqueness, regex pattern matching.

Usage
-----
>>> from ds_toolkit.core import SchemaValidator
>>> schema = {
...     "age":    {"dtype": "numeric", "min": 0, "max": 120, "nullable": False},
...     "email":  {"dtype": "string",  "pattern": r".+@.+\..+"},
...     "status": {"dtype": "string",  "allowed": ["active", "inactive"]},
...     "id":     {"dtype": "numeric", "unique": True, "nullable": False},
... }
>>> validator = SchemaValidator(strict=False)
>>> result = validator.check(df, schema)
>>> result.display()
>>> result.passed          # bool
>>> result.violations_df   # pd.DataFrame
"""
# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd

from ds_toolkit.base import ValidationResult


# ---------------------------------------------------------------------------
# Column schema model (pure Python — no Pydantic runtime dependency for core)
# ---------------------------------------------------------------------------

_VALID_DTYPES = {"numeric", "string", "boolean", "datetime", "category", "any"}


class ColumnSchema:
    """
    Validated column-level schema definition.

    Accepted keys
    -------------
    dtype       : str   — one of 'numeric', 'string', 'boolean', 'datetime', 'category', 'any'
    nullable    : bool  — whether NaN/None is allowed (default True)
    unique      : bool  — whether all values must be unique (default False)
    min         : float — minimum value (numeric only)
    max         : float — maximum value (numeric only)
    min_length  : int   — minimum string length (string only)
    max_length  : int   — maximum string length (string only)
    pattern     : str   — regex pattern every non-null value must match (string only)
    allowed     : list  — exhaustive allowed-value list
    """

    def __init__(self, col_name: str, definition: Dict[str, Any]) -> None:
        self.col_name = col_name
        unknown = set(definition) - {
            "dtype", "nullable", "unique", "min", "max",
            "min_length", "max_length", "pattern", "allowed",
        }
        if unknown:
            raise ValueError(
                f"Column '{col_name}' has unknown schema keys: {sorted(unknown)}"
            )

        dtype = definition.get("dtype", "any")
        if dtype not in _VALID_DTYPES:
            raise ValueError(
                f"Column '{col_name}': dtype='{dtype}' is not valid. "
                f"Use one of {sorted(_VALID_DTYPES)}"
            )

        self.dtype: str = dtype
        self.nullable: bool = bool(definition.get("nullable", True))
        self.unique: bool = bool(definition.get("unique", False))
        self.min: Optional[float] = definition.get("min")
        self.max: Optional[float] = definition.get("max")
        self.min_length: Optional[int] = definition.get("min_length")
        self.max_length: Optional[int] = definition.get("max_length")
        self.pattern: Optional[str] = definition.get("pattern")
        self.allowed: Optional[list] = definition.get("allowed")

        if self.pattern:
            try:
                re.compile(self.pattern)
            except re.error as exc:
                raise ValueError(
                    f"Column '{col_name}': invalid regex pattern '{self.pattern}': {exc}"
                ) from exc


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class SchemaValidator:
    """
    Validate a DataFrame against a column-level schema dict.

    Parameters
    ----------
    strict : bool
        If True, raise SchemaValidationError on the first violation.
        If False (default), collect all violations and return them.
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Dict[str, Any]],
    ) -> ValidationResult:
        """
        Validate *df* against *schema*.

        Parameters
        ----------
        df     : pd.DataFrame
        schema : dict mapping column name → constraint dict

        Returns
        -------
        ValidationResult
            .passed        — True if zero violations found.
            .violations_df — DataFrame with one row per violation.

        Raises
        ------
        SchemaValidationError
            Only when strict=True and at least one violation is found.
        TypeError
            If df is not a pd.DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

        parsed = {
            col: ColumnSchema(col, defn)
            for col, defn in schema.items()
        }

        violations: List[Dict[str, str]] = []

        for col, col_schema in parsed.items():
            if col not in df.columns:
                v = self._violation(col, "missing_column",
                                    f"Column '{col}' defined in schema but absent from DataFrame")
                violations.append(v)
                if self.strict:
                    raise SchemaValidationError(violations)
                continue

            series = df[col]
            col_violations = self._check_column(series, col_schema)
            violations.extend(col_violations)

            if self.strict and col_violations:
                raise SchemaValidationError(violations)

        violations_df = pd.DataFrame(
            violations,
            columns=["column", "check", "detail"],
        ) if violations else pd.DataFrame(columns=["column", "check", "detail"])

        return ValidationResult(passed=len(violations) == 0, violations_df=violations_df)

    # ------------------------------------------------------------------
    # Per-column checks
    # ------------------------------------------------------------------

    def _check_column(
        self, series: pd.Series, schema: ColumnSchema
    ) -> List[Dict[str, str]]:
        violations: List[Dict[str, str]] = []
        col = schema.col_name

        # --- dtype ---
        dtype_ok = self._check_dtype(series, schema.dtype)
        if not dtype_ok:
            violations.append(self._violation(
                col, "dtype",
                f"Expected dtype '{schema.dtype}', got '{series.dtype}'"
            ))

        # --- nullable ---
        n_null = int(series.isna().sum())
        if not schema.nullable and n_null > 0:
            violations.append(self._violation(
                col, "nullable",
                f"{n_null} null value(s) found; column is marked non-nullable"
            ))

        # --- uniqueness ---
        if schema.unique:
            n_dup = int(series.dropna().duplicated().sum())
            if n_dup > 0:
                violations.append(self._violation(
                    col, "unique",
                    f"{n_dup} duplicate value(s) found; column must be unique"
                ))

        # Work only on non-null values for remaining checks
        clean = series.dropna()

        # --- numeric range ---
        if schema.min is not None or schema.max is not None:
            if pd.api.types.is_numeric_dtype(series):
                if schema.min is not None:
                    n_below = int((clean < schema.min).sum())
                    if n_below:
                        violations.append(self._violation(
                            col, "min",
                            f"{n_below} value(s) below minimum {schema.min}"
                        ))
                if schema.max is not None:
                    n_above = int((clean > schema.max).sum())
                    if n_above:
                        violations.append(self._violation(
                            col, "max",
                            f"{n_above} value(s) above maximum {schema.max}"
                        ))
            else:
                violations.append(self._violation(
                    col, "range_inapplicable",
                    "min/max constraints require a numeric column"
                ))

        # --- string length ---
        if schema.min_length is not None or schema.max_length is not None:
            if pd.api.types.is_string_dtype(series) or series.dtype == object:
                lengths = clean.astype(str).str.len()
                if schema.min_length is not None:
                    n_short = int((lengths < schema.min_length).sum())
                    if n_short:
                        violations.append(self._violation(
                            col, "min_length",
                            f"{n_short} value(s) shorter than min_length={schema.min_length}"
                        ))
                if schema.max_length is not None:
                    n_long = int((lengths > schema.max_length).sum())
                    if n_long:
                        violations.append(self._violation(
                            col, "max_length",
                            f"{n_long} value(s) longer than max_length={schema.max_length}"
                        ))

        # --- regex pattern ---
        if schema.pattern:
            if pd.api.types.is_string_dtype(series) or series.dtype == object:
                rx = re.compile(schema.pattern)
                matches = clean.astype(str).str.match(rx)
                n_fail = int((~matches).sum())
                if n_fail:
                    violations.append(self._violation(
                        col, "pattern",
                        f"{n_fail} value(s) do not match pattern '{schema.pattern}'"
                    ))

        # --- allowed values ---
        if schema.allowed is not None:
            bad = clean[~clean.isin(schema.allowed)]
            if len(bad):
                sample = sorted(bad.astype(str).unique()[:5].tolist())
                violations.append(self._violation(
                    col, "allowed_values",
                    f"{len(bad)} value(s) not in allowed set. Sample: {sample}"
                ))

        return violations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_dtype(series: pd.Series, expected: str) -> bool:
        if expected == "any":
            return True
        if expected == "numeric":
            return pd.api.types.is_numeric_dtype(series)
        if expected == "string":
            return pd.api.types.is_string_dtype(series) or series.dtype == object
        if expected == "boolean":
            return pd.api.types.is_bool_dtype(series)
        if expected == "datetime":
            return pd.api.types.is_datetime64_any_dtype(series)
        if expected == "category":
            return str(series.dtype) == "category"
        return True

    @staticmethod
    def _violation(col: str, check: str, detail: str) -> Dict[str, str]:
        return {"column": col, "check": check, "detail": detail}


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class SchemaValidationError(Exception):
    """Raised by SchemaValidator when strict=True and violations are found."""

    def __init__(self, violations: List[Dict[str, str]]) -> None:
        self.violations = violations
        n = len(violations)
        super().__init__(f"Schema validation failed with {n} violation(s).")
