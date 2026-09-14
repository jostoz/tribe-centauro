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


# ---------------------------------------------------------------------------
# Dos grupos independientes de creativos (p. ej. rápido vs lento), con covariables
# ---------------------------------------------------------------------------


@dataclass
class GroupTestResult:
    """Prueba de permutación entre dos grupos independientes, por unidad (red/parcela)."""

    statistic: np.ndarray      # coeficiente del grupo por unidad
    p_values: np.ndarray       # p bilateral por unidad
    n_a: int
    n_b: int
    n_permutations: int
    adjusted_for: Optional[str] = None

    def holm(self) -> np.ndarray:
        """p-valores ajustados por Holm–Bonferroni (controla el error de familia)."""
        return holm_bonferroni(self.p_values)

    def as_dict(self, units: Optional[list] = None) -> Dict:
        units = units or list(range(len(self.p_values)))
        adj = self.holm()
        return {
            unit: {
                "statistic": float(self.statistic[i]),
                "p_value": float(self.p_values[i]),
                "p_holm": float(adj[i]),
                "significant_holm": bool(adj[i] < 0.05),
            }
            for i, unit in enumerate(units)
        }


def group_permutation_test(
    values: np.ndarray,
    is_group_a: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    n_permutations: int = 10000,
    seed: int = 0,
    covariate_name: Optional[str] = None,
) -> GroupTestResult:
    """Permutación bilateral para **dos grupos independientes** sobre ``n_units`` salidas.

    ``values`` es ``(n_ads, n_units)`` (p. ej. shares de redes por anuncio) y ``is_group_a``
    ``(n_ads,)`` booleano. El estadístico es el coeficiente de la dummy de grupo en una
    regresión con las covariables controladas.

    Con covariables se usa el esquema de **Freedman–Lane**: se permutan los residuos del
    modelo reducido (sin el grupo) y se recalcula el estadístico del modelo completo. Eso
    preserva el ajuste y evita el error de permutar la etiqueta con una covariable
    correlacionada (que produciría falsos positivos, como demuestra el test del módulo).
    """
    y = np.asarray(values, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    labels = np.asarray(is_group_a, dtype=bool)
    n = y.shape[0]
    if labels.shape[0] != n:
        raise ValueError(f"labels {labels.shape} no coincide con values {y.shape}")
    if covariates is not None:
        cov = np.asarray(covariates, dtype=float)
        if cov.ndim == 1:
            cov = cov[:, None]
        if cov.shape[0] != n:
            raise ValueError("covariates y values deben tener el mismo nº de filas")

    ones = np.ones((n, 1))
    full = np.hstack([ones, labels[:, None].astype(float)] + ([cov] if covariates is not None else []))
    reduced = np.hstack([ones] + ([cov] if covariates is not None else []))

    beta_full, *_ = np.linalg.lstsq(full, y, rcond=None)
    stat_obs = beta_full[1].copy()

    pinv_full = np.linalg.pinv(full)
    rng = np.random.default_rng(seed)
    count = np.zeros(y.shape[1], dtype=int)

    if covariates is None:
        # sin covariables: permutar la etiqueta y reajustar
        for _ in range(n_permutations):
            perm = rng.permutation(n)
            y_perm = y[perm]
            beta, *_ = np.linalg.lstsq(full, y_perm, rcond=None)
            count += np.abs(beta[1]) >= np.abs(stat_obs)
    else:
        # Freedman–Lane: permutar residuos del modelo reducido y reajustar el completo
        beta_red, *_ = np.linalg.lstsq(reduced, y, rcond=None)
        fitted = reduced @ beta_red
        resid = y - fitted
        for _ in range(n_permutations):
            resid_perm = resid[rng.permutation(n)]
            beta = pinv_full @ (fitted + resid_perm)
            count += np.abs(beta[1]) >= np.abs(stat_obs)

    p = (1.0 + count) / (1.0 + n_permutations)
    return GroupTestResult(
        statistic=stat_obs,
        p_values=p,
        n_a=int(labels.sum()),
        n_b=int(n - labels.sum()),
        n_permutations=n_permutations,
        adjusted_for=covariate_name,
    )


def holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """p-valores ajustados por Holm–Bonferroni, en el orden original.

    Paso descendente con corrección de monotonía: garantiza que ningún p ajustado quede por
    debajo de uno ajustado previamente más pequeño.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted
