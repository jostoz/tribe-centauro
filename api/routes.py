"""
Endpoints REST — adaptador delgado sobre ``service/`` y ``core/``.

Este módulo NO contiene lógica de dominio: solo traduce HTTP ↔ servicio.

Diferencias con la versión anterior:
- No inicializa el modelo en tiempo de import (antes lo hacía y tumbaba el proceso).
- No invoca un "score de engagement" inexistente.
- Las rutas de subida se sanean (antes: ``open(temp_dir / file.filename)`` → path traversal).
- No expone endpoints que devolvían 501.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from ads.validator import AdValidator
from api.schemas import (
    ActivationMetrics,
    AnalyzeRequest,
    AnalyzeResponse,
    CompareRequest,
    CompareResponse,
    HealthResponse,
    NetworkActivation,
    PermutationResult,
    Provenance,
    ValidateRequest,
    ValidateResponse,
    ValidationIssue,
)
from core.ordering import ALIGNMENT_CONVENTION, FSAAVERAGE5_VERTICES
from core.tribe_model import TribePredictor
from service.errors import AdsServiceError, InvalidInput, LicenseRequired
from service.metrics.engagement import (
    Provenance as ProvenanceInternal,
    find_dips,
    find_peaks,
    summarize_activation,
    temporal_profile,
)
from service.metrics.roi import RoiIndex
from service.metrics.stats import compare_parcels
from service.serializers import downsample

logger = logging.getLogger(__name__)

router = APIRouter()

VERSION = "2.0.0"

# Estado perezoso: nada de cargar el modelo al importar el módulo.
_predictor: Optional[TribePredictor] = None
_roi_index: Optional[RoiIndex] = None

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


def license_mode() -> str:
    return os.getenv("LICENSE_MODE", "research").lower()


def assert_license_allows_use() -> None:
    """TRIBE v2 es CC-BY-NC-4.0: el uso comercial requiere licencia explícita."""
    mode = license_mode()
    if mode == "commercial" and not os.getenv("META_LICENSE_REF"):
        raise LicenseRequired(
            "LICENSE_MODE=commercial sin META_LICENSE_REF configurado"
        )


def get_predictor() -> TribePredictor:
    global _predictor
    if _predictor is None:
        _predictor = TribePredictor(
            checkpoint_dir=os.getenv("MODEL_NAME", "facebook/tribev2"),
            cache_folder=os.getenv("CACHE_DIR", "./cache"),
            device=os.getenv("DEVICE", "auto"),
        )
    return _predictor


def get_roi_index() -> RoiIndex:
    global _roi_index
    if _roi_index is None:
        _roi_index = RoiIndex.from_schaefer(
            os.getenv("ATLAS_DIR", "data/atlas/schaefer200")
        )
    return _roi_index


def _internal_provenance() -> ProvenanceInternal:
    return ProvenanceInternal(model_version=os.getenv("MODEL_NAME", "facebook/tribev2"))


def _provenance_response(p: ProvenanceInternal) -> Provenance:
    return Provenance(**p.as_dict())


@router.get("/health", response_model=HealthResponse)
async def health_check():
    predictor = get_predictor()
    try:
        get_roi_index()
        atlas_ok = True
    except Exception:  # noqa: BLE001
        atlas_ok = False

    return HealthResponse(
        status="ok" if predictor.is_loaded else "model_not_loaded",
        model_loaded=predictor.is_loaded,
        device=predictor.device,
        license_mode=license_mode(),
        atlas_loaded=atlas_ok,
        version=VERSION,
    )


@router.post("/api/ads/validate", response_model=ValidateResponse)
async def validate_ad(request: ValidateRequest):
    """Pre-flight barato: sin GPU, sin modelo."""
    ad_info = {}
    if request.duration_seconds is not None or request.width is not None:
        ad_info["video"] = {
            "duration": request.duration_seconds or 0.0,
            "width": request.width or 0,
            "height": request.height or 0,
            "fps": request.fps or 0,
        }

    is_valid, issues = AdValidator.validate_for_platform(ad_info, request.platform.value)
    recs = AdValidator.get_recommendations(request.platform.value).get("tips", [])

    return ValidateResponse(
        platform=request.platform,
        is_valid=is_valid,
        issues=[ValidationIssue(severity="blocker", field="media", message=m) for m in issues],
        recommendations=recs,
    )


@router.post("/api/ads/analyze", response_model=AnalyzeResponse)
async def analyze_ad(request: AnalyzeRequest):
    """Analiza un creativo: una sola fuente (video, audio o texto)."""
    assert_license_allows_use()

    predictor = get_predictor()
    roi = get_roi_index()
    provenance = _internal_provenance()

    try:
        preds, metadata = predictor.predict(
            video_path=request.video_path,
            audio_path=request.audio_path,
            text=request.text,
        )
    except AdsServiceError as exc:
        raise HTTPException(status_code=422, detail=exc.as_dict()) from exc

    summary = summarize_activation(preds, provenance)
    profile = temporal_profile(preds)

    networks = {}
    for net in sorted(roi.networks):
        ts = roi.timeseries(preds, net)
        networks[net] = NetworkActivation(
            mean_abs_activation=float(abs(ts).mean()),
            peak_abs_activation=float(abs(ts).max()),
            n_vertices=int(roi.networks[net].sum()),
        )

    warnings = []
    if metadata.get("alignment") != ALIGNMENT_CONVENTION:
        warnings.append(
            "La alineación temporal no está verificada en esta corrida: no interpretar "
            "los timesteps como segundos del estímulo."
        )
    if metadata.get("n_segments") and metadata["n_segments"] > 1:
        warnings.append(
            f"El estímulo se troceó en {metadata['n_segments']} segmentos "
            "(ChunkEvents 30-60 s); revisar continuidad."
        )

    return AnalyzeResponse(
        source_kind=metadata["source"],
        platform=request.platform,
        metrics=ActivationMetrics(
            mean_abs_activation=summary.mean_abs_activation,
            peak_abs_activation=summary.peak_abs_activation,
            activation_std=summary.activation_std,
            mean_signed_activation=summary.mean_signed_activation,
            n_timesteps=summary.n_timesteps,
            n_vertices=summary.n_vertices,
        ),
        networks=networks,
        temporal_profile=downsample(profile, max_points=120),
        peaks=find_peaks(profile),
        dips=find_dips(profile),
        provenance=_provenance_response(provenance),
        warnings=warnings,
    )


@router.post("/api/ads/compare", response_model=CompareResponse)
async def compare_ads(request: CompareRequest):
    """Comparación honesta: permutación por bloques sobre parcelas, no t-test aplanado."""
    assert_license_allows_use()

    predictor = get_predictor()
    roi = get_roi_index()
    provenance = _internal_provenance()

    try:
        preds_a, _ = predictor.predict(
            video_path=request.a_video_path,
            audio_path=request.a_audio_path,
            text=request.a_text,
        )
        preds_b, meta_b = predictor.predict(
            video_path=request.b_video_path,
            audio_path=request.b_audio_path,
            text=request.b_text,
        )
    except AdsServiceError as exc:
        raise HTTPException(status_code=422, detail=exc.as_dict()) from exc

    if preds_a.shape[0] != preds_b.shape[0]:
        raise HTTPException(
            status_code=422,
            detail=InvalidInput(
                f"Los creativos tienen distinta duración en timesteps: "
                f"{preds_a.shape[0]} vs {preds_b.shape[0]}",
                hint="Recorta ambos a la misma duración para una comparación pareada.",
            ).as_dict(),
        )

    result = compare_parcels(
        preds_a,
        preds_b,
        roi,
        block_size=request.block_size,
        n_permutations=request.n_permutations,
    )

    return CompareResponse(
        source_kind=meta_b["source"],
        difference_units=result["difference_units"],
        overall=PermutationResult(**result["overall"]),
        per_network={k: PermutationResult(**v) for k, v in result["per_network"].items()},
        interpretation=result["interpretation"],
        provenance=_provenance_response(provenance),
    )


@router.post("/api/upload")
async def upload_media(file: UploadFile = File(...)):
    """Sube un medio. El nombre se sanea para evitar path traversal."""
    original = Path(file.filename or "upload")
    safe_name = _SAFE_NAME.sub("_", original.name)[:120]
    if not safe_name or safe_name in {".", ".."}:
        safe_name = "upload"

    target_dir = Path(os.getenv("UPLOAD_DIR", Path(tempfile.gettempdir()) / "tribe_ads"))
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"

    # Verificar contención: el destino debe quedar dentro del directorio permitido.
    if target_dir.resolve() not in target.resolve().parents:
        raise HTTPException(status_code=400, detail="Nombre de archivo inválido")

    size = 0
    with open(target, "wb") as fh:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                fh.close()
                target.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Excede el máximo de {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                )
            fh.write(chunk)

    return {"filename": safe_name, "path": str(target), "bytes": size}
