"""
ds_toolkit
==========
Post-collection data science lifecycle toolkit.

Covers every stage from raw DataFrame to evaluated, tracked, and reported
model — as reusable, composable Python modules with Jupyter-native outputs.

Stages
------
1. core        — profiling, validation, cleaning
2. features    — encoding, engineering, selection
3. models      — registry, CV, tuning, ensembles
4. eval        — metrics, SHAP, plots, error analysis
5. infra       — experiment logging, config, serialisation
6. reporting   — notebook output, HTML export, model cards

Quick start
-----------
>>> from ds_toolkit.core import DataProfiler, MissingHandler
>>> from ds_toolkit.features import EncoderFactory, FeatureSelector
>>> from ds_toolkit.models import ModelRegistry, CVHarness
>>> from ds_toolkit.eval import MetricsReport, DiagnosticPlotter
>>> from ds_toolkit.reporting import generate_model_card
>>>
>>> profile = DataProfiler().profile(df)
>>> profile.display()

Links
-----
Repository : https://github.com/ShadowGodd1/ds-toolkit
Docs       : https://ShadowGodd1.github.io/ds-toolkit
Author     : Adnan Mohamud — CEO & Founder, PataDoc (https://patadoc.com)
License    : MIT
"""

# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

__version__ = "1.0.4"
__author__ = "Adnan Mohamud"
__email__ = "adnan@patadoc.com"
__license__ = "MIT"
__url__ = "https://github.com/ShadowGodd1/ds-toolkit"
