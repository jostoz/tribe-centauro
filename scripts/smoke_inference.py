"""
Smoke test de inferencia real sobre GPU.

Verifica el entorno completo: descarga del checkpoint, extracción de features,
inferencia en GPU y la forma del resultado.

Usa ``audio_only=True`` para saltar transcripción (whisperx) y features de texto
(Llama-3.2), que requieren acceso gated. Esto permite validar el pipeline de
inferencia sin depender de Llama.

Uso:
    .venv/Scripts/python.exe scripts/smoke_inference.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ordering import FSAAVERAGE5_VERTICES  # noqa: E402

DURATION_S = 12
SAMPLE_RATE = 16000


def make_audio(path: Path) -> Path:
    """Genera un audio sintético no estacionario (12 s)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, DURATION_S, DURATION_S * SAMPLE_RATE, endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
    sf.write(str(path), (tone * envelope).astype(np.float32), SAMPLE_RATE)
    return path


def main() -> int:
    import torch
    from tribev2.demo_utils import TribeModel, get_audio_and_text_events

    from core.tribe_model import resolve_checkpoint_dir

    print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("AVISO: sin GPU, la inferencia será muy lenta")

    audio = make_audio(Path("data/samples/tone_12s.wav"))
    print(f"audio: {audio} ({audio.stat().st_size} bytes)")

    print("\n[1/3] resolviendo checkpoint (workaround del bug de Windows)...")
    t0 = time.time()
    local_dir = resolve_checkpoint_dir("facebook/tribev2", "./cache")
    print(f"      checkpoint local: {local_dir} ({time.time() - t0:.1f}s)")

    print("\n[2/3] cargando modelo en GPU...")
    t0 = time.time()
    model = TribeModel.from_pretrained(
        local_dir, cache_folder="./cache", device="auto"
    )
    print(f"      cargado en {time.time() - t0:.1f}s")

    # En Windows, el DataLoader spawnea N_CPUS procesos worker que cargan las
    # DLLs de CUDA cada uno, agotando la commit memory -> OSError WinError 1455
    # ("paging file too small"). Con un solo evento no hay nada que paralelizar,
    # así que forzamos carga en el proceso principal.
    model.data.num_workers = 0
    print(f"      num_workers forzado a {model.data.num_workers} (estabilidad Windows)")

    print("\n[2/3] construyendo eventos (audio_only, sin Llama/whisperx)...")
    event = {
        "type": "Audio",
        "filepath": str(audio),
        "start": 0,
        "timeline": "default",
        "subject": "default",
    }
    events = get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
    print(f"      tipos de evento: {events.type.unique().tolist()}")
    print(f"      filas: {len(events)}")

    print("\n[3/3] infiriendo...")
    t0 = time.time()
    preds, segments = model.predict(events=events, verbose=False)
    elapsed = time.time() - t0

    print(f"\n=== RESULTADO ===")
    print(f"preds.shape    : {preds.shape}  (esperado (n_segments, {FSAAVERAGE5_VERTICES}))")
    print(f"dtype          : {preds.dtype}")
    print(f"n_segments     : {len(segments)}")
    print(f"rango valores  : [{preds.min():.4f}, {preds.max():.4f}]")
    print(f"tiempo         : {elapsed:.1f}s ({elapsed / DURATION_S:.2f}s por segundo de audio)")

    if preds.shape[1] != FSAAVERAGE5_VERTICES:
        print(f"\nFALLO: se esperaban {FSAAVERAGE5_VERTICES} vértices")
        return 1

    seg = segments[0]
    attrs = [a for a in dir(seg) if not a.startswith("_")]
    print(f"\n--- estructura de un segmento ---")
    print(f"attrs: {attrs}")
    for name in ("offset", "duration", "ns_events", "start", "type", "frequency"):
        if hasattr(seg, name):
            print(f"  {name:12} = {getattr(seg, name)!r}")

    offsets = [getattr(s, "offset", None) for s in segments]
    print(f"\n--- eje temporal (clave para el mapeo timestep->segundo) ---")
    print(f"offsets[:10]   : {offsets[:10]}")
    print(f"offsets únicos : {len(set(offsets))} de {len(offsets)}")
    print(f"n_segments={len(segments)} para {DURATION_S}s de audio")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
