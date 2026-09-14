"""Análisis neural TRIBE v2 con modelo RESIDENTE.

Carga TRIBE (+ V-JEPA) una sola vez y procesa N anuncios. Cargar el modelo por
anuncio es lo que hacía inviable escalar (descargas + inicialización repetidas).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from discovery.gpu import free_gpu

DEFAULT_CHECKPOINT = "facebook/tribev2"
DEFAULT_CACHE = "./cache"
DEFAULT_ATLAS = "data/atlas/schaefer200"


class NeuralAnalyzer:
    """Modelo TRIBE residente: ``load()`` una vez, ``analyze()`` por anuncio."""

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        cache_folder: str = DEFAULT_CACHE,
        atlas_dir: str = DEFAULT_ATLAS,
    ) -> None:
        self.checkpoint = checkpoint
        self.cache_folder = cache_folder
        self.atlas_dir = atlas_dir
        self._model = None
        self._roi = None

    def load(self) -> None:
        if self._model is not None:
            return
        from core.tribe_model import resolve_checkpoint_dir
        from service.metrics.roi import RoiIndex
        from tribev2.demo_utils import TribeModel

        local_dir = resolve_checkpoint_dir(self.checkpoint, self.cache_folder)
        self._model = TribeModel.from_pretrained(
            local_dir, cache_folder=self.cache_folder, device="auto"
        )
        # Windows: sin esto el DataLoader spawnea N_CPUS workers que agotan commit
        self._model.data.num_workers = 0
        if Path(self.atlas_dir).is_dir():
            self._roi = RoiIndex.from_schaefer(self.atlas_dir)

    def extract_features(self, video_path: str | Path) -> int:
        """Fuerza y cachea las features (V-JEPA/audio) del anuncio SIN forward.

        Iterar el dataloader ejecuta los extractores y escribe sus features en la
        caché de neuralset. Separar esto del forward permite: (a) codificar todos los
        anuncios seguidos (GPU ocupada en la tarea pesada), y (b) que los forwards
        posteriores sean baratos y repetibles. Devuelve el nº de batches iterados.
        """
        if self._model is None:
            raise RuntimeError("llama a load() antes de extract_features()")
        import pandas as pd
        from tribev2.demo_utils import get_audio_and_text_events

        event = {
            "type": "Video",
            "filepath": str(video_path),
            "start": 0,
            "timeline": "default",
            "subject": "default",
        }
        events = get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
        loader = self._model.data.get_loaders(events=events, split_to_build="all")["all"]
        n = 0
        for _batch in loader:  # dispara la extracción; el resultado queda cacheado
            n += 1
        return n

    def analyze(self, video_path: str | Path, with_vertices: bool = False) -> Dict:
        """Perfil por red funcional (unidades crudas) + red dominante.

        Con ``with_vertices=True`` incluye ``vertex_mean_abs`` (20484,) — el **patrón completo**
        por vértice en lugar de los 7 promedios de red. Es la lectura multivariada del modelo:
        reducir a 7 shares tira el 99.99 % de la señal.
        """
        if self._model is None:
            raise RuntimeError("llama a load() antes de analyze()")
        import pandas as pd
        from tribev2.demo_utils import get_audio_and_text_events

        event = {
            "type": "Video",
            "filepath": str(video_path),
            "start": 0,
            "timeline": "default",
            "subject": "default",
        }
        events = get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
        preds, _ = self._model.predict(events=events, verbose=False)
        out: Dict = {"shape": list(preds.shape)}
        if with_vertices:
            out["vertex_mean_abs"] = np.mean(np.abs(preds), axis=0)  # (20484,)
        if self._roi is not None:
            nets = {
                net: round(float(np.mean(np.abs(ts))), 5)
                for net, ts in self._roi.all_networks_timeseries(preds).items()
            }
            out["networks"] = nets
            out["red_dominante"] = max(nets, key=nets.get) if nets else ""
        return out

    def unload(self) -> None:
        self._model = None
        self._roi = None
        free_gpu()
