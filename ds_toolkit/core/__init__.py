"""
ds_toolkit.core
===============
Stage 1 — Data Understanding & Validation
Stage 2 — Data Cleaning & Preprocessing

Stage 1 exports
---------------
DataProfiler        — one-call dataset summary
SchemaValidator     — Pydantic-backed schema enforcement
DistributionReport  — histogram / KDE / QQ / heatmap export

Stage 2 exports (coming next)
------------------------------
MissingHandler      — CV-safe per-column imputation
OutlierDetector     — IQR / Z-score / IsoForest / LOF detection
TypeCaster          — auto dtype inference and coercion
Deduplicator        — exact and fuzzy deduplication
"""

# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from ds_toolkit.core.profiler import DataProfiler
from ds_toolkit.core.validator import SchemaValidator, SchemaValidationError
from ds_toolkit.core.distribution import DistributionReport
from ds_toolkit.core.missing import MissingHandler
from ds_toolkit.core.outliers import OutlierDetector
from ds_toolkit.core.typecaster import TypeCaster
from ds_toolkit.core.deduplicator import Deduplicator

__all__ = [
    # Stage 1
    "DataProfiler",
    "SchemaValidator",
    "SchemaValidationError",
    "DistributionReport",
    # Stage 2
    "MissingHandler",
    "OutlierDetector",
    "TypeCaster",
    "Deduplicator",
]
