"""
Core — inferencia TRIBE v2 y convenciones de la malla cortical.
"""

from core.ordering import (
    FSAAVERAGE5_VERTICES,
    FSAAVERAGE5_VERTICES_PER_HEMI,
    HEMODYNAMIC_OFFSET_SECONDS,
    TR_SECONDS,
    join_hemispheres,
    split_hemispheres,
)
from core.tribe_model import TribePredictor

__all__ = [
    "FSAAVERAGE5_VERTICES",
    "FSAAVERAGE5_VERTICES_PER_HEMI",
    "HEMODYNAMIC_OFFSET_SECONDS",
    "TR_SECONDS",
    "TribePredictor",
    "join_hemispheres",
    "split_hemispheres",
]
