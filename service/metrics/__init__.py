"""
Métricas de activación neural, ROIs anatómicas y estadística.
"""

from service.metrics.engagement import (
    ActivationSummary,
    Provenance,
    fraction_above_reference,
    summarize_activation,
    temporal_profile,
)
from service.metrics.roi import RoiIndex
from service.metrics.stats import (
    PairedTestResult,
    compare_parcels,
    paired_block_permutation,
    parcel_means,
)

__all__ = [
    "ActivationSummary",
    "Provenance",
    "RoiIndex",
    "PairedTestResult",
    "compare_parcels",
    "fraction_above_reference",
    "paired_block_permutation",
    "parcel_means",
    "summarize_activation",
    "temporal_profile",
]
