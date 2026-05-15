"""
ds_toolkit.features
====================
Stage 3 — Feature Engineering.

Modules
-------
EncoderFactory       — cardinality-aware categorical encoder
DatetimeDecomposer   — datetime → linear + cyclical features
InteractionBuilder   — product, ratio, and polynomial interactions
FeatureSelector      — variance → correlation → RFECV → SHAP pipeline
Scaler               — CV-safe numeric scaling
"""

# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from ds_toolkit.features.encoder import EncoderFactory
from ds_toolkit.features.datetime_decomposer import DatetimeDecomposer
from ds_toolkit.features.interaction_builder import InteractionBuilder
from ds_toolkit.features.selector import FeatureSelector
from ds_toolkit.features.scaler import Scaler

__all__ = [
    "EncoderFactory",
    "DatetimeDecomposer",
    "InteractionBuilder",
    "FeatureSelector",
    "Scaler",
]
