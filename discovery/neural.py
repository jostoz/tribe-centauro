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

    def analyze(self, video_path: str | Path) -> Dict:
        """Perfil por red funcional (unidades crudas) + red dominante."""
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
