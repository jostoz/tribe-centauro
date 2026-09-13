"""
Analisis de un anuncio real (ruta VIDEO) — validacion de la ruta visual.

Ejecuta el pipeline audiovisual de TRIBE v2 sobre un .mp4 real (video + audio,
sin transcripcion de texto para evitar Llama-3.2 gated) y produce un informe:
forma de las predicciones, perfil temporal de atencion y activacion por red
funcional (Schaefer 2018, 7 redes).

Uso:
    .venv/Scripts/python.exe scripts/analyze_video.py ads/comercial.mp4
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ordering import FSAAVERAGE5_VERTICES  # noqa: E402
from service.metrics.engagement import find_peaks, temporal_profile  # noqa: E402
from service.metrics.roi import RoiIndex  # noqa: E402


def main() -> int:
    import torch

    from tribev2.demo_utils import TribeModel, get_audio_and_text_events

    from core.tribe_model import resolve_checkpoint_dir

    video_path = Path(sys.argv[1] if len(sys.argv) > 1 else "ads/comercial.mp4")
    if not video_path.is_file():
        print(f"No existe el video: {video_path}")
        return 1

    print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()}")
    print(f"video: {video_path} ({video_path.stat().st_size/1e6:.1f} MB)\n")

    local_dir = resolve_checkpoint_dir("facebook/tribev2", "./cache")
    t0 = time.time()
    model = TribeModel.from_pretrained(local_dir, cache_folder="./cache", device="auto")
    model.data.num_workers = 0
    print(f"modelo cargado en {time.time() - t0:.1f}s\n")

    event = {
        "type": "Video",
        "filepath": str(video_path),
        "start": 0,
        "timeline": "default",
        "subject": "default",
    }
    print("construyendo eventos audiovisuales (sin texto/Llama)...")
    events = get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
    print(f"  tipos de evento: {events.type.unique().tolist()}  | filas: {len(events)}")

    print("\ninfiriendo (ruta video)...")
    t0 = time.time()
    preds, segments = model.predict(events=events, verbose=True)
    elapsed = time.time() - t0

    print("\n=== RESULTADO ===")
    print(f"preds.shape : {preds.shape}  (esperado (n_seg, {FSAAVERAGE5_VERTICES}))")
    print(f"n_segments  : {len(segments)}")
    print(f"rango       : [{preds.min():.4f}, {preds.max():.4f}]")
    print(f"tiempo inf  : {elapsed:.1f}s")

    if preds.shape[1] != FSAAVERAGE5_VERTICES:
        print("FALLO: numero de vertices inesperado")
        return 1

    # Perfil temporal de atencion + picos
    prof = temporal_profile(preds)
    peaks = find_peaks(prof, relative_threshold=1.3)
    print("\n--- perfil temporal (activacion media |.| por timestep ~ 1s) ---")
    print("    (preds[k] = respuesta al segundo k; stimulus-aligned, precision ~±1.5 s)")
    print("  t(s): " + " ".join(f"{v:.3f}" for v in prof[:30]))
    print(f"  picos de atencion (>1.3x media): {[p['timestep'] for p in peaks]}")

    # Activacion por red funcional (Schaefer)
    print("\n--- activacion media por red funcional (unidades crudas) ---")
    try:
        roi = RoiIndex.from_schaefer("data/atlas/schaefer200")
        rows = []
        for net, ts in roi.all_networks_timeseries(preds).items():
            rows.append(
                {
                    "red": net,
                    "media_abs": round(float(np.mean(np.abs(ts))), 5),
                    "pico_abs": round(float(np.max(np.abs(ts))), 5),
                    "t_pico": int(np.argmax(np.abs(ts))),
                }
            )
        df = pd.DataFrame(rows).sort_values("media_abs", ascending=False)
        print(df.to_string(index=False))
        print(
            "\nInterpretacion (constructo): Vis=atencion visual, SalVentAttn=captura de\n"
            "atencion, Limbic=valencia emocional, Default=narrativa. Valores en unidades\n"
            "crudas, no calibrados, comparables solo dentro de este lote."
        )
    except FileNotFoundError as exc:
        print(f"(atlas no disponible, se omite ROI: {exc})")

    print("\n=== OK ruta video ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
