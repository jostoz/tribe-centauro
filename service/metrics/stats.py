"""
Comparación estadística entre creativos, corregida por estructura espacio-temporal.

El código anterior hacía::

    t_stat, p_value = stats.ttest_ind(np.abs(pred_a).flatten(),
                                      np.abs(pred_b).flatten())

Sobre ~20484 vértices x T timesteps. Dos errores:

1. Trata cada vértice como una observación independiente. Los vértices corticales
   están fuertemente autocorrelacionados espacialmente (vértices vecinos comparten
   señal), así que el N efectivo es órdenes de magnitud menor que 20484*T.
   Con N inflado, el p-valor es ~0 para *cualquier* par de creativos: todo sale
   "significativo".
2. Un t-test asume observaciones independientes; aquí no lo son.

Qué hacemos en su lugar
-----------------------
Agregamos a parcelas (elimina la pseudo-réplica espacial) y aplicamos una prueba de
permutación por bloques temporales. El bloqueo preserva la autocorrelación temporal:
permutar timesteps sueltos rompería la estructura de la señal y volvería a inflar la
significancia.

Resultado: un p-valor que puede ser grande. Eso es correcto y esperado.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PairedTestResult:
    """Resultado de una prueba pareada por bloques."""

    observed_difference: float
    p_value: float
    ci_low: float
    ci_high: float
    n_blocks: int
    block_size: int
    n_permutations: int
    metric: str

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def as_dict(self) -> Dict:
        return {
            "observed_difference": self.observed_difference,
            "p_value": self.p_value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "significant": self.significant,
            "n_blocks": self.n_blocks,
            "block_size": self.block_size,
            "n_permutations": self.n_permutations,
            "metric": self.metric,
            "note": (
                "Prueba de permutación por bloques temporales sobre series agregadas "
                "a parcelas. No es un t-test sobre vértices independientes."
            ),
        }


def parcel_means(predictions: np.ndarray, roi_index) -> np.ndarray:
    """Agrega ``(n_timesteps, 20484)`` a ``(n_timesteps, n_parcelas)``.

    La agregación a parcelas elimina la pseudo-réplica espacial: en lugar de 20484
    vértices correlacionados usamos ~200 unidades anatómicamente disjuntas.
    """
    preds = np.asarray(predictions, dtype=np.float64)
    masks = list(roi_index.parcels.values())
    return np.stack([preds[:, m].mean(axis=1) for m in masks], axis=1)


def paired_block_permutation(
    a: np.ndarray,
    b: np.ndarray,
    block_size: int = 5,
    n_permutations: int = 5000,
    metric: str = "mean",
    seed: int = 0,
) -> PairedTestResult:
    """Prueba de permutación pareada con bloques temporales.

    Parameters
    ----------
    a, b:
        Series de forma ``(n_timesteps,)`` o ``(n_timesteps, n_unidades)`` del MISMO
        creativo emparejado temporalmente. Deben tener igual longitud.
    block_size:
        Tamaño de bloque en timesteps. Con TR=1 s, 5 bloques ≈ 5 s, suficiente para
        preservar la autocorrelación hemodinámica residual.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(f"Formas incompatibles: {a.shape} vs {b.shape}")
    if a.ndim == 1:
        a = a[:, None]
        b = b[:, None]
    if a.ndim != 2:
        raise ValueError(f"Se esperaba 1D o 2D, recibido {a.shape}")

    n_timesteps = a.shape[0]
    if n_timesteps < 2 * block_size:
        raise ValueError(
            f"Se requieren al menos {2 * block_size} timesteps para bloques de "
            f"{block_size}; hay {n_timesteps}"
        )

    diff = _aggregate(b, metric) - _aggregate(a, metric)

    block_means = _block_reduce(diff, block_size)  # (n_blocks, n_units)
    n_blocks, n_units = block_means.shape

    # El estadístico es la media sobre todos los bloques y unidades. Se calcula desde
    # las medias de bloque (no desde el array crudo) para que observado y nulo sean
    # exactamente la misma función de los datos.
    observed = float(block_means.mean())

    rng = np.random.default_rng(seed)
    # Sign-flip por bloque: todas las unidades de un bloque se invierten juntas, lo que
    # preserva la estructura de dependencia espacial dentro del bloque.
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n_blocks))
    perm_stats = (signs @ block_means).sum(axis=1) / (n_blocks * n_units)

    extreme = np.abs(perm_stats) >= abs(observed) - 1e-12
    # Corrección +1: nunca reportar p=0 exacto con un nº finito de permutaciones.
    p_value = float((extreme.sum() + 1) / (n_permutations + 1))

    ci_low, ci_high = _bootstrap_ci(block_means, rng, n_permutations)

    return PairedTestResult(
        observed_difference=observed,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        n_blocks=n_blocks,
        block_size=block_size,
        n_permutations=n_permutations,
        metric=metric,
    )


def compare_parcels(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    roi_index,
    block_size: int = 5,
    n_permutations: int = 5000,
) -> Dict:
    """Comparación completa entre dos creativos, a nivel de parcela y de red.

    Devuelve diferencias en unidades crudas, no un "porcentaje de mejora" sobre un
    score no calibrado.
    """
    pm_a = parcel_means(predictions_a, roi_index)
    pm_b = parcel_means(predictions_b, roi_index)

    overall = paired_block_permutation(
        pm_a, pm_b, block_size=block_size, n_permutations=n_permutations
    )

    per_network = {}
    for net in sorted(roi_index.networks):
        ts_a = roi_index.timeseries(predictions_a, net)
        ts_b = roi_index.timeseries(predictions_b, net)
        per_network[net] = paired_block_permutation(
            ts_a, ts_b, block_size=block_size, n_permutations=n_permutations
        ).as_dict()

    return {
        "overall": overall.as_dict(),
        "per_network": per_network,
        "difference_units": "raw_model_activation",
        "interpretation": (
            "difference = B - A en unidades crudas del modelo. Sin calibración "
            "contra resultados de campaña, un valor positivo NO implica mejor CTR."
        ),
    }


def _aggregate(x: np.ndarray, metric: str) -> np.ndarray:
    if metric == "mean":
        return x
    if metric == "abs":
        return np.abs(x)
    raise ValueError(f"Métrica desconocida: {metric}")


def _block_reduce(x: np.ndarray, block_size: int) -> np.ndarray:
    n_timesteps, n_units = x.shape
    n_blocks = n_timesteps // block_size
    trimmed = x[: n_blocks * block_size]
    return trimmed.reshape(n_blocks, block_size, n_units).mean(axis=1)


def _bootstrap_ci(
    block_means: np.ndarray, rng: np.random.Generator, n_resamples: int
) -> Tuple[float, float]:
    n_blocks = block_means.shape[0]
    idx = rng.integers(0, n_blocks, size=(n_resamples, n_blocks))
    resampled = block_means[idx].mean(axis=(1, 2))
    lo, hi = np.percentile(resampled, [2.5, 97.5])
    return float(lo), float(hi)
