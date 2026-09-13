"""
Capa de servicio agnóstica de protocolo.

Toda la lógica de dominio vive aquí. ``mcp/`` y ``api/`` son fronteras delgadas que
solo traducen entre su protocolo y esta capa.
"""

from service.errors import (
    AdsServiceError,
    GpuOom,
    HfAccessDenied,
    InvalidInput,
    JobNotFound,
    LicenseRequired,
    MediaTooLarge,
    MediaUnsupported,
    ModelUnavailable,
    QuotaExceeded,
    TextToSpeechUnavailable,
)
from service.serializers import downsample, to_jsonable

__all__ = [
    "AdsServiceError",
    "GpuOom",
    "HfAccessDenied",
    "InvalidInput",
    "JobNotFound",
    "LicenseRequired",
    "MediaTooLarge",
    "MediaUnsupported",
    "ModelUnavailable",
    "QuotaExceeded",
    "TextToSpeechUnavailable",
    "downsample",
    "to_jsonable",
]
