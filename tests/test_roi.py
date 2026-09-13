"""Tests del índice de ROI contra el atlas Schaefer 2018 real (fsaverage5)."""

import numpy as np
import pytest

from core.ordering import FSAAVERAGE5_VERTICES, FSAAVERAGE5_VERTICES_PER_HEMI
from service.metrics.roi import RoiIndex

ATLAS_DIR = "data/atlas/schaefer200"

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(ATLAS_DIR).exists(),
    reason="Atlas Schaefer 200 no descargado",
)


@pytest.fixture(scope="module")
def roi_index():
    return RoiIndex.from_schaefer(ATLAS_DIR)


def test_covers_full_fsaverage5_mesh(roi_index):
    """Las máscaras deben estar en la malla que realmente predice el modelo."""
    assert roi_index.n_vertices == FSAAVERAGE5_VERTICES == 20484
    for mask in roi_index.parcels.values():
        assert mask.shape == (FSAAVERAGE5_VERTICES,)


def test_all_seven_networks_present(roi_index):
    """Schaefer 7Networks debe producir exactamente 7 redes.

    Regresión: un regex codicioso sobre las etiquetas capturaba
    'DorsAttn_Post' en lugar de 'DorsAttn' y dejaba solo 2 redes detectadas.
    """
    assert set(roi_index.networks) == {
        "Vis",
        "SomMot",
        "DorsAttn",
        "SalVentAttn",
        "Limbic",
        "Cont",
        "Default",
    }


def test_two_hundred_parcels(roi_index):
    assert len(roi_index.parcels) == 200


def test_masks_are_boolean_and_nonempty(roi_index):
    for name, mask in roi_index.parcels.items():
        assert mask.dtype == np.bool_, name
        assert mask.any(), name


def test_hemispheres_do_not_overlap(roi_index):
    left = np.zeros(FSAAVERAGE5_VERTICES, dtype=bool)
    left[:FSAAVERAGE5_VERTICES_PER_HEMI] = True
    for name, mask in roi_index.parcels.items():
        if name.endswith("_1") or True:
            pass
    # Ninguna parcela puede tener vértices en ambos hemisferios.
    for name, mask in roi_index.parcels.items():
        n_left = int((mask & left).sum())
        n_right = int((mask & ~left).sum())
        assert (n_left == 0) or (n_right == 0), f"{name} cruza hemisferios"


def test_no_contiguous_slice_is_an_anatomical_region(roi_index):
    """Evidencia de por qué el ROI anterior era falso.

    ``slice(0, 2000)`` estaba etiquetado como 'córtex visual'. En realidad atraviesa
    las 7 redes funcionales y ~100 de las 200 parcelas.
    """
    legacy = np.zeros(FSAAVERAGE5_VERTICES, dtype=bool)
    legacy[:2000] = True

    networks_touched = [
        net for net, mask in roi_index.networks.items() if (legacy & mask).any()
    ]
    assert len(networks_touched) == 7, "un slice contiguo cruza todas las redes"

    parcels_touched = sum(
        1 for mask in roi_index.parcels.values() if (legacy & mask).any()
    )
    assert parcels_touched > 50

    # El slice NO es visual: la red Default (no visual) ocupa más que la visual.
    default_overlap = int((legacy & roi_index.networks["Default"]).sum())
    visual_overlap = int((legacy & roi_index.networks["Vis"]).sum())
    assert default_overlap > visual_overlap


def test_timeseries_shape_and_value(roi_index):
    rng = np.random.default_rng(0)
    preds = rng.random((20, FSAAVERAGE5_VERTICES))

    ts = roi_index.timeseries(preds, "Vis")
    assert ts.shape == (20,)

    # Debe ser la media sobre los vértices de la máscara.
    expected = preds[:, roi_index.networks["Vis"]].mean(axis=1)
    np.testing.assert_allclose(ts, expected)


def test_unknown_roi_raises_keyerror(roi_index):
    with pytest.raises(KeyError):
        roi_index.mask("visual_cortex")  # nombre del código anterior, ya inexistente


def test_timeseries_rejects_wrong_vertex_count(roi_index):
    with pytest.raises(ValueError, match="20484"):
        roi_index.timeseries(np.zeros((10, 5000)), "Vis")
