"""
Métricas de activación neural — SIN score calibrado.

Contexto
--------
El código anterior exponía un "Neural Engagement Score" 0-100 calculado como::

    nes = min(100, np.mean(np.abs(predictions)) * 100)

Eso no mide "engagement". Mide la magnitud media absoluta de la salida del modelo,
multiplicada por 100 y recortada en 100. Su valor depende de la escala arbitraria de
las predicciones de TRIBE v2, no de ninguna propiedad del creativo publicitario, y
nunca se validó contra ningún resultado de campaña. Los umbrales por plataforma
(60/65/50/55) no tenían origen.

Un agente que optimice contra ese número optimizará contra ruido escalado.

Qué exponemos en su lugar
-------------------------
Magnitudes crudas, en unidades del modelo, con procedencia explícita:

    mean_abs_activation    - activación media absoluta (unidades crudas)
    peak_abs_activation    - máximo absoluto
    activation_std         - desviación estándar espacial
    fraction_above_ref     - fracción de vértices por encima de una referencia

Y **comparaciones relativas** entre creativos procesados por el mismo pipeline
(ver ``service/metrics/stats.py``), que son defendibles incluso sin calibración.

La calibración contra resultados reales se acumula vía ``submit_feedback`` y se aplica
en ``service/metrics/calibration.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

VALIDITY = "population-average, research-use-only"


@dataclass(frozen=True)
class Provenance:
    """Procedencia y límites de validez de cualquier número que devolvamos.

    Se adjunta SIEMPRE a las respuestas. Sin esto, un agente no puede distinguir una
    medición de una estimación no validada.

    ``hemodynamic_offset_seconds`` es el retardo que el checkpoint **ya aplicó** al
    construir su objetivo de entrenamiento: ``preds[k]`` es la respuesta al segundo ``k``
    del estímulo (``core.ordering.ALIGNMENT_CONVENTION``). No hay que restarlo.
    """

    units: str = "raw_model_activation"
    calibrated: bool = False
    validity: str = VALIDITY
    model_version: str = "unknown"
    comparability: str = "within-batch-only"
    tr_seconds: float = 1.0
    hemodynamic_offset_seconds: float = 5.0

    def as_dict(self) -> Dict:
        return {
            "units": self.units,
            "calibrated": self.calibrated,
            "validity": self.validity,
            "model_version": self.model_version,
            "comparability": self.comparability,
            "tr_seconds": self.tr_seconds,
            "hemodynamic_offset_seconds": self.hemodynamic_offset_seconds,
        }


@dataclass
class ActivationSummary:
    """Resumen de activación en unidades crudas."""

    mean_abs_activation: float
    peak_abs_activation: float
    activation_std: float
    mean_signed_activation: float
    n_timesteps: int
    n_vertices: int
    provenance: Provenance = field(default_factory=Provenance)

    def as_dict(self) -> Dict:
        return {
            "mean_abs_activation": self.mean_abs_activation,
            "peak_abs_activation": self.peak_abs_activation,
            "activation_std": self.activation_std,
            "mean_signed_activation": self.mean_signed_activation,
            "n_timesteps": self.n_timesteps,
            "n_vertices": self.n_vertices,
            "provenance": self.provenance.as_dict(),
        }


def summarize_activation(
    predictions: np.ndarray,
    provenance: Optional[Provenance] = None,
) -> ActivationSummary:
    """Resumen de activación en unidades crudas. No produce ningún score 0-100."""
    preds = np.asarray(predictions, dtype=np.float64)
    _require_2d(preds)

    return ActivationSummary(
        mean_abs_activation=float(np.mean(np.abs(preds))),
        peak_abs_activation=float(np.max(np.abs(preds))),
        activation_std=float(np.std(preds)),
        mean_signed_activation=float(np.mean(preds)),
        n_timesteps=int(preds.shape[0]),
        n_vertices=int(preds.shape[1]),
        provenance=provenance or Provenance(),
    )


def temporal_profile(predictions: np.ndarray) -> np.ndarray:
    """Perfil temporal: activación media absoluta por timestep. Forma ``(n_timesteps,)``.

    Es una magnitud cruda, no normalizada. No se reescala a 0-100.
    """
    preds = np.asarray(predictions, dtype=np.float64)
    _require_2d(preds)
    return np.mean(np.abs(preds), axis=1)


def fraction_above_reference(
    predictions: np.ndarray, reference: float
) -> float:
    """Fracción de valores con |activación| > ``reference``.

    ``reference`` debe venir de una ejecución previa del mismo pipeline (p. ej. el
    percentil 95 de un creativo de control), nunca de un valor inventado.
    """
    preds = np.asarray(predictions, dtype=np.float64)
    return float(np.mean(np.abs(preds) > reference))


def find_peaks(
    temporal: np.ndarray, relative_threshold: float = 1.3
) -> list:
    """Timesteps con activación por encima de ``relative_threshold`` x la media.

    El umbral es **relativo a la propia serie**, no un valor absoluto de engagement.
    Un pico significa "este instante destaca dentro de este creativo", no "este
    instante es bueno". Devuelve ``[{"timestep": int, "value": float}, ...]``.
    """
    series = np.asarray(temporal, dtype=np.float64).ravel()
    if series.size == 0:
        return []
    cutoff = float(np.mean(series)) * relative_threshold
    idx = np.flatnonzero(series > cutoff)
    # Se ordena por magnitud: los agentes suelen querer los más salientes primero.
    idx = idx[np.argsort(-series[idx])]
    return [{"timestep": int(i), "value": float(series[i])} for i in idx]


def find_dips(
    temporal: np.ndarray, relative_threshold: float = 0.7
) -> list:
    """Timesteps con activación por debajo de ``relative_threshold`` x la media.

    Igual que :func:`find_peaks`, el umbral es relativo a la serie. Un valle es un
    candidato a recorte, no una prueba de que el instante sea malo.
    """
    series = np.asarray(temporal, dtype=np.float64).ravel()
    if series.size == 0:
        return []
    cutoff = float(np.mean(series)) * relative_threshold
    return [int(i) for i in np.flatnonzero(series < cutoff)]



def _require_2d(preds: np.ndarray) -> None:
    if preds.ndim != 2:
        raise ValueError(
            f"Se esperaba (n_timesteps, n_vertices), recibido {preds.shape}"
        )
