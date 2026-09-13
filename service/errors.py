"""
Taxonomía de errores accionable por máquina.

Sin códigos estables, un agente no puede distinguir "reintentar" de "corregir la
entrada" de "rendirse", y degrada a reintentos ciegos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AdsServiceError(Exception):
    """Error con código estable, reintentabilidad y sugerencia de corrección."""

    code: str
    message: str
    retryable: bool = False
    hint: Optional[str] = None

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "hint": self.hint,
        }


class InvalidInput(AdsServiceError):
    def __init__(self, message: str, hint: Optional[str] = None):
        super().__init__("INVALID_INPUT", message, retryable=False, hint=hint)


class MediaUnsupported(AdsServiceError):
    def __init__(self, message: str, hint: Optional[str] = None):
        super().__init__("MEDIA_UNSUPPORTED", message, retryable=False, hint=hint)


class MediaTooLarge(AdsServiceError):
    def __init__(self, message: str, hint: Optional[str] = None):
        super().__init__("MEDIA_TOO_LARGE", message, retryable=False, hint=hint)


class TextToSpeechUnavailable(AdsServiceError):
    def __init__(self, message: str = "Texto a voz no disponible"):
        super().__init__(
            "TEXT_TO_SPEECH_UNAVAILABLE",
            message,
            retryable=True,
            hint=(
                "La entrada de texto requiere gTTS (Google TTS) y red. "
                "Usa una entrada de audio o video, o reintenta con red disponible."
            ),
        )


class HfAccessDenied(AdsServiceError):
    def __init__(self, message: str = "Acceso a HuggingFace denegado"):
        super().__init__(
            "HF_ACCESS_DENIED",
            message,
            retryable=False,
            hint=(
                "Las features de texto usan Llama-3.2 (gated). "
                "Solicita acceso y configura HF_TOKEN."
            ),
        )


class JobNotFound(AdsServiceError):
    def __init__(self, job_id: str):
        super().__init__("JOB_NOT_FOUND", f"Job {job_id} no existe", retryable=False)


class QuotaExceeded(AdsServiceError):
    def __init__(self, message: str = "Cuota excedida"):
        super().__init__("QUOTA_EXCEEDED", message, retryable=False)


class ModelUnavailable(AdsServiceError):
    def __init__(self, message: str = "Modelo no disponible"):
        super().__init__("MODEL_UNAVAILABLE", message, retryable=True)


class GpuOom(AdsServiceError):
    def __init__(self, message: str = "VRAM agotada"):
        super().__init__(
            "GPU_OOM", message, retryable=True, hint="Reintentar con menos trabajo en vuelo."
        )


class LicenseRequired(AdsServiceError):
    def __init__(self, message: str = "Uso comercial no autorizado"):
        super().__init__(
            "LICENSE_REQUIRED",
            message,
            retryable=False,
            hint=(
                "TRIBE v2 es CC-BY-NC-4.0 (no comercial). "
                "Configura LICENSE_MODE=research o aporta META_LICENSE_REF."
            ),
        )
