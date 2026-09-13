"""
Índice de regiones de interés (ROI) sobre la malla fsaverage5.

Reemplaza el mapeo fabricado del código anterior::

    roi_mapping = {"visual_cortex": slice(0, 2000), ...}   # ANATÓMICAMENTE FALSO

por máscaras booleanas derivadas de un atlas anotado sobre la MISMA malla que
predice el modelo (Schaefer 2018, 200 parcelas, 7 redes, fsaverage5).

Por qué importa para publicidad
-------------------------------
Un slice contiguo no es una región cerebral. Las parcelas de Schaefer sí lo son, y
vienen agrupadas en 7 redes funcionales con significado directo para neuromarketing:

    Vis            - corteza visual: atención a estímulo visual, cambio de escena
    SomMot         - sensoriomotor: acción, movimiento, gestos
    DorsAttn       - atención dorsal: orientación atencional, saliencia espacial
    SalVentAttn    - atención ventral/saliencia: "esto me importa", captura de atención
    Limbic         - límbico: valencia afectiva, respuesta emocional
    Cont           - control: esfuerzo cognitivo, carga de procesamiento
    Default        - red por defecto: narrativa, autoproyección, memoria episódica

Los nombres se extraen de las etiquetas del propio atlas (``7Networks_<HEMI>_<Red>_<N>``);
no se codifican a mano.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Nombres de red tal como aparecen en las etiquetas del atlas Schaefer 2018.
SCHAEFFER_2018_NETWORKS = (
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default",
)

# Las etiquetas de Schaefer 2018 tienen dos formas:
#   "7Networks_LH_Vis_1"                -> red + índice
#   "7Networks_LH_DorsAttn_Post_1"      -> red + subregión + índice
# La red es SIEMPRE el tercer campo. Un `.+` codicioso capturaría
# "DorsAttn_Post" y dejaría 5 de las 7 redes sin detectar (verificado).
_LABEL_RE = re.compile(r"^7Networks_(?P<hemi>LH|RH)_(?P<network>[A-Za-z]+)(?:_[A-Za-z]+)*_\d+$")


class RoiIndex:
    """Máscaras booleanas de ROI sobre fsaverage5, derivadas de un atlas anotado.

    Attributes
    ----------
    parcels : Dict[str, np.ndarray]
        Máscara booleana ``(20484,)`` por parcela (p. ej. ``"7Networks_LH_Vis_1"``).
    networks : Dict[str, np.ndarray]
        Máscara booleana ``(20484,)`` por red funcional (p. ej. ``"Vis"``).
    """

    def __init__(self, parcels: Dict[str, np.ndarray], networks: Dict[str, np.ndarray]):
        self.parcels = parcels
        self.networks = networks
        self.n_vertices = int(next(iter(parcels.values())).shape[0])
        self._validate()

    def _validate(self) -> None:
        for name, mask in self.parcels.items():
            if mask.shape != (self.n_vertices,):
                raise ValueError(f"Parcela {name}: forma {mask.shape} inesperada")
            if mask.dtype != np.bool_:
                raise ValueError(f"Parcela {name}: se esperaba máscara booleana")
            if not mask.any():
                raise ValueError(f"Parcela {name}: máscara vacía")

    @classmethod
    def from_schaefer(
        cls, atlas_dir: str | Path, n_parcels: int = 200, yeo_networks: int = 7
    ) -> "RoiIndex":
        """Carga el atlas Schaefer 2018 desde archivos ``.annot`` de fsaverage5.

        Parameters
        ----------
        atlas_dir:
            Directorio con ``lh.Schaefer2018_{n}Parcels_{k}Networks_order.annot``
            y su contraparte ``rh.``.
        """
        atlas_dir = Path(atlas_dir)
        stem = f"Schaefer2018_{n_parcels}Parcels_{yeo_networks}Networks_order.annot"

        hemi_labels: Dict[str, np.ndarray] = {}
        hemi_names: Dict[str, List[str]] = {}

        for hemi in ("lh", "rh"):
            path = atlas_dir / f"{hemi}.{stem}"
            if not path.exists():
                raise FileNotFoundError(
                    f"Falta el atlas {path}. Descargar los .annot de fsaverage5 desde "
                    "ThomasYeoLab/CBIG (Parcellations/FreeSurfer5.3/fsaverage5/label/)."
                )
            labels, _ctab, names = _read_annot(path)
            hemi_labels[hemi] = labels
            hemi_names[hemi] = [
                n.decode() if isinstance(n, bytes) else str(n) for n in names
            ]

        full_labels = np.concatenate([hemi_labels["lh"], hemi_labels["rh"]])

        parcels: Dict[str, np.ndarray] = {}
        for hemi in ("lh", "rh"):
            names = hemi_names[hemi]
            # El índice del nombre coincide con el id de etiqueta. El id 0 es el
            # muro medial ("Background+FreeSurfer_Defined_Medial_Wall"), no una parcela.
            for label_id, name in enumerate(names):
                if label_id == 0 or "Medial_Wall" in name or name in ("unknown", "???"):
                    continue
                mask_hemi = hemi_labels[hemi] == label_id
                if not mask_hemi.any():
                    continue
                parcels[name] = _place_in_full(mask_hemi, hemi)

        networks: Dict[str, np.ndarray] = {}
        for net in SCHAEFFER_2018_NETWORKS:
            masks = [
                m for name, m in parcels.items() if _network_of(name) == net
            ]
            if masks:
                networks[net] = np.logical_or.reduce(masks)

        logger.info(
            "Atlas cargado: %d parcelas, %d redes, %d vértices",
            len(parcels),
            len(networks),
            full_labels.shape[0],
        )
        return cls(parcels=parcels, networks=networks)

    def mask(self, name: str) -> np.ndarray:
        """Devuelve la máscara de una parcela o red; error explícito si no existe."""
        if name in self.parcels:
            return self.parcels[name]
        if name in self.networks:
            return self.networks[name]
        raise KeyError(
            f"ROI {name!r} desconocido. Redes: {sorted(self.networks)}; "
            f"parcelas: {len(self.parcels)}"
        )

    def timeseries(self, predictions: np.ndarray, name: str) -> np.ndarray:
        """Serie temporal de activación media de un ROI, forma ``(n_timesteps,)``.

        ``predictions`` debe tener forma ``(n_timesteps, 20484)``.
        """
        preds = np.asarray(predictions)
        if preds.ndim != 2 or preds.shape[1] != self.n_vertices:
            raise ValueError(
                f"Se esperaba (n_timesteps, {self.n_vertices}), recibido {preds.shape}"
            )
        return preds[:, self.mask(name)].mean(axis=1)

    def all_networks_timeseries(self, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """Series temporales de todas las redes funcionales."""
        return {net: self.timeseries(predictions, net) for net in self.networks}

    def as_dict(self) -> Dict[str, List[int]]:
        """Representación serializable: índices de vértice por ROI."""
        out = {}
        for name, mask in {**self.parcels, **self.networks}.items():
            out[name] = np.flatnonzero(mask).tolist()
        return out


def _read_annot(path: Path):
    """Lee un ``.annot`` de FreeSurfer sin depender de nilearn."""
    import nibabel

    return nibabel.freesurfer.read_annot(str(path), orig_ids=False)


def _network_of(name: str) -> Optional[str]:
    """Extrae la red funcional del nombre de una parcela Schaefer."""
    m = _LABEL_RE.match(name)
    if m:
        return m.group("network")
    return None


def _place_in_full(mask_hemi: np.ndarray, hemi: str) -> np.ndarray:
    """Coloca una máscara de hemisferio en el array completo [izq | der]."""
    from core.ordering import FSAAVERAGE5_VERTICES, FSAAVERAGE5_VERTICES_PER_HEMI

    if mask_hemi.shape != (FSAAVERAGE5_VERTICES_PER_HEMI,):
        raise ValueError(
            f"Hemisferio {hemi}: {mask_hemi.shape[0]} vértices, "
            f"se esperaban {FSAAVERAGE5_VERTICES_PER_HEMI}"
        )

    full = np.zeros(FSAAVERAGE5_VERTICES, dtype=bool)
    if hemi == "lh":
        full[:FSAAVERAGE5_VERTICES_PER_HEMI] = mask_hemi
    else:
        full[FSAAVERAGE5_VERTICES_PER_HEMI:] = mask_hemi
    return full
