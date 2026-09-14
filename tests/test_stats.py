"""Tests de la comparación estadística y de `parcel_means`.

El objeto del bug es el **falso positivo**: el código anterior trataba cada vértice
como una observación independiente. Como los vértices corticales vecinos están
fuertemente correlacionados, el N efectivo es mucho menor que 20484 x T, y el t-test
declara diferencias significativas entre creativos que son dos muestras del mismo
proceso.
"""

import numpy as np
import pytest
from scipy import stats

from core.ordering import FSAAVERAGE5_VERTICES
from service.metrics.roi import RoiIndex
from service.metrics.stats import (
    compare_parcels,
    group_permutation_test,
    holm_bonferroni,
    paired_block_permutation,
)

ATLAS_DIR = "data/atlas/schaefer200"


@pytest.fixture(scope="module")
def roi_index():
    return RoiIndex.from_schaefer(ATLAS_DIR)


def _correlated_pair(seed=0, n_timesteps=60, n_blocks=100, repeat=200):
    """Dos muestras INDEPENDIENTES del mismo proceso con autocorrelación espacial.

    Cada campo se genera con ``n_blocks`` valores independientes repetidos en bloques
    contiguos. La diferencia real entre ambos es cero por construcción; el N
    independiente real es ``n_blocks``, no ``n_blocks * repeat``.
    """
    rng = np.random.default_rng(seed)
    coarse_a = rng.random((n_timesteps, n_blocks))
    coarse_b = rng.random((n_timesteps, n_blocks))
    return coarse_a, coarse_b, np.repeat(coarse_a, repeat, axis=1), np.repeat(coarse_b, repeat, axis=1)


def test_legacy_ttest_produces_a_false_positive():
    """Dos muestras del MISMO proceso, declaradas significativamente distintas.

    Es la evidencia central: el p-valor no mide un efecto, mide cuántos vértices hay.
    """
    _, _, a_exp, b_exp = _correlated_pair(seed=0)
    assert a_exp.shape[1] == 20000

    _, p_value = stats.ttest_ind(a_exp.flatten(), b_exp.flatten())
    assert p_value < 1e-6, (
        f"el t-test aplanado no puede detectar que no hay diferencia (p={p_value})"
    )


def test_permutation_on_true_independent_units_does_not_flag_it():
    """Los mismos datos, agregados a sus unidades independientes reales: no significativo."""
    coarse_a, coarse_b, _, _ = _correlated_pair(seed=0)

    result = paired_block_permutation(coarse_a, coarse_b, block_size=5, n_permutations=2000)
    assert result.p_value > 0.05, (
        f"dos muestras del mismo proceso no deben ser significativas (p={result.p_value})"
    )


def test_detects_a_large_consistent_effect():
    rng = np.random.default_rng(1)
    a = rng.random((60, 40))
    b = a + 2.0 + rng.normal(0, 0.1, size=a.shape)

    result = paired_block_permutation(a, b, block_size=5, n_permutations=2000)
    assert result.p_value < 0.05
    assert result.observed_difference == pytest.approx(2.0, abs=0.2)


def test_confidence_interval_brackets_the_effect():
    rng = np.random.default_rng(2)
    a = rng.random((80, 20))
    b = a + 0.3 + rng.normal(0, 0.05, size=a.shape)

    result = paired_block_permutation(a, b, block_size=5, n_permutations=2000)
    assert result.ci_low < result.observed_difference < result.ci_high
    assert result.ci_high - result.ci_low > 0


def test_p_value_never_exactly_zero():
    """Con permutaciones finitas, p=0 exacto es una afirmación falsa."""
    a = np.zeros((60, 10))
    b = np.ones((60, 10))

    result = paired_block_permutation(a, b, block_size=5, n_permutations=1000)
    assert result.p_value > 0


def test_requires_matching_shapes():
    with pytest.raises(ValueError, match="Formas incompatibles"):
        paired_block_permutation(np.zeros(60), np.zeros(50), block_size=5)


def test_rejects_series_shorter_than_two_blocks():
    with pytest.raises(ValueError, match="timesteps"):
        paired_block_permutation(np.zeros(6), np.zeros(6), block_size=5)


def test_parcel_means_removes_spatial_pseudo_replication(roi_index):
    """Agregar a parcelas reduce 20484 vértices correlacionados a ~200 unidades."""
    rng = np.random.default_rng(5)
    preds = rng.random((30, FSAAVERAGE5_VERTICES))

    pm = __import__(
        "service.metrics.stats", fromlist=["parcel_means"]
    ).parcel_means(preds, roi_index)

    assert pm.shape == (30, len(roi_index.parcels))
    assert pm.shape[1] < FSAAVERAGE5_VERTICES / 50


def test_compare_parcels_reports_raw_units_and_all_networks(roi_index):
    rng = np.random.default_rng(3)
    a = rng.random((40, FSAAVERAGE5_VERTICES))
    b = rng.random((40, FSAAVERAGE5_VERTICES))

    result = compare_parcels(a, b, roi_index, block_size=5, n_permutations=500)

    assert result["difference_units"] == "raw_model_activation"
    assert "p_value" in result["overall"]
    assert set(result["per_network"]) == set(roi_index.networks)
    # Sin calibración no debe declararse un ganador de campaña.
    assert "winner" not in result
    assert "improvement_percentage" not in str(result)


# --- dos grupos independientes con covariable (Grupo A2) ---------------------


def test_holm_bonferroni_matches_worked_example():
    # Ejemplo canónico: p = [0.01, 0.04, 0.03] -> ajustados [0.03, 0.06, 0.06]
    adj = holm_bonferroni(np.array([0.01, 0.04, 0.03]))
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    # Nunca menor que el p crudo, nunca mayor que 1, y monótono respecto al orden.
    assert np.all(adj >= np.array([0.01, 0.04, 0.03]) - 1e-12)
    assert np.all(adj <= 1.0)


def test_group_permutation_detects_real_separation():
    rng = np.random.default_rng(0)
    n = 40
    labels = np.array([True] * 20 + [False] * 20)
    values = rng.normal(0, 1, size=(n, 1)) + labels[:, None] * 3.0

    res = group_permutation_test(values, labels, n_permutations=2000, seed=0)

    assert res.p_values[0] < 0.01
    assert res.n_a == 20 and res.n_b == 20


def test_group_permutation_does_not_invent_an_effect():
    rng = np.random.default_rng(1)
    labels = np.array([True] * 15 + [False] * 15)
    values = rng.normal(0, 1, size=(30, 1))  # ruido puro, sin efecto de grupo

    res = group_permutation_test(values, labels, n_permutations=2000, seed=0)

    assert res.p_values[0] > 0.05


def test_group_permutation_adjusts_for_correlated_covariate():
    """Una covariable correlacionada con el grupo produce un falso positivo si no se ajusta.

    Es la razón de usar Freedman–Lane: aquí el efecto real es SOLO la duración, y la
    etiqueta está correlacionada con ella. Sin ajustar sale "significativo"; ajustando, no.
    """
    rng = np.random.default_rng(7)
    n = 60
    duration = rng.uniform(10, 60, n)
    labels = rng.random(n) < (duration / 80.0)          # etiqueta correlacionada
    values = (0.10 * duration + rng.normal(0, 0.2, n))[:, None]  # efecto = solo duración

    unadjusted = group_permutation_test(values, labels, n_permutations=2000, seed=0)
    adjusted = group_permutation_test(
        values, labels, covariates=duration[:, None], n_permutations=2000, seed=0,
        covariate_name="duration",
    )

    assert unadjusted.p_values[0] < 0.05, "el confusor debería dar un falso positivo"
    assert adjusted.p_values[0] > 0.05, "al ajustar, el efecto de grupo debe desaparecer"
    assert adjusted.adjusted_for == "duration"
