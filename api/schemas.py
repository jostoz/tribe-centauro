"""
Esquemas de la API REST.

Nota sobre unidades: **no existe un "score de engagement" 0-100**. Todos los valores
numéricos están en unidades crudas del modelo y van acompañados de ``provenance``.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PlatformEnum(str, Enum):
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    FACEBOOK = "facebook"
    YOUTUBE = "youtube"


class SourceKind(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"


class Provenance(BaseModel):
    """Procedencia y límites de validez de los números devueltos."""

    units: str = "raw_model_activation"
    calibrated: bool = False
    validity: str = "population-average, research-use-only"
    model_version: str = "unknown"
    comparability: str = "within-batch-only"
    tr_seconds: float = 1.0
    hemodynamic_offset_seconds: float = 5.0


class ActivationMetrics(BaseModel):
    """Métricas de activación en unidades crudas (no calibradas)."""

    mean_abs_activation: float
    peak_abs_activation: float
    activation_std: float
    mean_signed_activation: float
    n_timesteps: int
    n_vertices: int


class NetworkActivation(BaseModel):
    """Activación media de una red funcional (Schaefer 7Networks)."""

    mean_abs_activation: float
    peak_abs_activation: float
    n_vertices: int


class ValidateRequest(BaseModel):
    source_kind: SourceKind
    platform: PlatformEnum = PlatformEnum.INSTAGRAM
    duration_seconds: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None


class ValidationIssue(BaseModel):
    severity: str  # blocker | warning | info
    field: str
    message: str


class ValidateResponse(BaseModel):
    platform: str
    is_valid: bool
    issues: List[ValidationIssue] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    text: Optional[str] = None
    platform: PlatformEnum = PlatformEnum.INSTAGRAM


class AnalyzeResponse(BaseModel):
    source_kind: str
    platform: str
    metrics: ActivationMetrics
    networks: Dict[str, NetworkActivation]
    temporal_profile: List[float] = Field(
        ..., description="Activación media absoluta por timestep, diezmada"
    )
    peaks: List[Dict[str, float]] = Field(default_factory=list)
    dips: List[int] = Field(default_factory=list)
    provenance: Provenance
    warnings: List[str] = Field(default_factory=list)


class CompareRequest(BaseModel):
    a_video_path: Optional[str] = None
    a_audio_path: Optional[str] = None
    a_text: Optional[str] = None
    b_video_path: Optional[str] = None
    b_audio_path: Optional[str] = None
    b_text: Optional[str] = None
    block_size: int = 5
    n_permutations: int = 5000


class PermutationResult(BaseModel):
    observed_difference: float
    p_value: float
    ci_low: float
    ci_high: float
    significant: bool
    n_blocks: int
    block_size: int
    n_permutations: int
    note: str


class CompareResponse(BaseModel):
    source_kind: str
    difference_units: str
    overall: PermutationResult
    per_network: Dict[str, PermutationResult]
    interpretation: str
    provenance: Provenance


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    license_mode: str
    atlas_loaded: bool
    version: str
