"""
ds_toolkit.eval
================
Stage 5 — Evaluation & Diagnostics.
"""

# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from ds_toolkit.eval.metrics import MetricsReport, MetricsResult
from ds_toolkit.eval.shap_explainer import ExplainerSHAP, ShapResult
from ds_toolkit.eval.diagnostic_plotter import DiagnosticPlotter, DiagnosticResult
from ds_toolkit.eval.error_analyser import ErrorAnalyser, ErrorReport

__all__ = [
    "MetricsReport",
    "MetricsResult",
    "ExplainerSHAP",
    "ShapResult",
    "DiagnosticPlotter",
    "DiagnosticResult",
    "ErrorAnalyser",
    "ErrorReport",
]
