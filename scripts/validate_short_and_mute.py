"""
Fase 0.5 - Validacion go/no-go del producto publicitario.

Verifica empiricamente dos supuestos criticos del plan (MCP_ADS_SERVICE_PLAN.md):

  1. Clips cortos (6-15 s) frente a ChunkEvents(min_duration=30): que NO se
     descarten ni degeneren, y que produzcan preds validas (n_segments, 20484).
  2. Anuncio mudo (silencio total) frente a RemoveMissing(): que la ruta
     audio_only siga produciendo inferencia (no vacia, no crash) y que el
     silencio se distinga de una senal tonal.

Ruta validada: audio_only=True (audio/video sin transcripcion), que es la
ruta comercial limpia (sin dependencia de Llama-3.2 gated ni gTTS).

Uso:
    .venv/Scripts/python.exe scripts/validate_short_and_mute.py
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

SAMPLE_RATE = 16000
OUT_DIR = Path("data/samples")


def make_tone(path: Path, duration_s: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration_s, int(duration_s * SAMPLE_RATE), endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
    sf.write(str(path), (tone * envelope).astype(np.float32), SAMPLE_RATE)
    return path


def make_silence(path: Path, duration_s: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros(int(duration_s * SAMPLE_RATE), dtype=np.float32)
    sf.write(str(path), samples, SAMPLE_RATE)
    return path


def main() -> int:
    import torch

    from tribev2.demo_utils import TribeModel, get_audio_and_text_events

    from core.tribe_model import resolve_checkpoint_dir

    print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("AVISO: sin GPU")

    local_dir = resolve_checkpoint_dir("facebook/tribev2", "./cache")
    t0 = time.time()
    model = TribeModel.from_pretrained(local_dir, cache_folder="./cache", device="auto")
    model.data.num_workers = 0  # estabilidad Windows (evita WinError 1455)
    print(f"modelo cargado en {time.time() - t0:.1f}s\n")

    # (etiqueta, generador, duracion)
    cases = [
        ("tono_6s", make_tone, 6.0),
        ("tono_10s", make_tone, 10.0),
        ("tono_15s", make_tone, 15.0),
        ("mudo_6s", make_silence, 6.0),
        ("mudo_10s", make_silence, 10.0),
        ("mudo_15s", make_silence, 15.0),
    ]

    results = []
    means = {}
    for label, gen, dur in cases:
        wav = gen(OUT_DIR / f"val_{label}.wav", dur)
        event = {
            "type": "Audio",
            "filepath": str(wav),
            "start": 0,
            "timeline": "default",
            "subject": "default",
        }
        try:
            events = get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
            n_events = len(events)
            t0 = time.time()
            preds, segments = model.predict(events=events, verbose=False)
            elapsed = time.time() - t0
            shape_ok = preds.shape[1] == FSAAVERAGE5_VERTICES
            nonempty = preds.shape[0] > 0
            expected_segs = int(round(dur))
            segs_ok = len(segments) == expected_segs
            mean_abs = float(np.mean(np.abs(preds)))
            means[label] = mean_abs
            results.append(
                {
                    "caso": label,
                    "dur": dur,
                    "n_eventos": n_events,
                    "preds_shape": str(preds.shape),
                    "n_segments": len(segments),
                    "segs_esperados": expected_segs,
                    "shape_ok": shape_ok,
                    "no_vacio": nonempty,
                    "segs_ok": segs_ok,
                    "mean_abs": round(mean_abs, 5),
                    "t_s": round(elapsed, 2),
                    "estado": "OK" if (shape_ok and nonempty) else "FALLO",
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "caso": label,
                    "dur": dur,
                    "estado": f"EXCEPCION: {type(exc).__name__}: {exc}",
                }
            )

    df = pd.DataFrame(results)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("=== RESULTADOS FASE 0.5 ===")
    print(df.to_string(index=False))

    # Veredicto
    short_ok = all(
        r.get("shape_ok") and r.get("no_vacio")
        for r in results
        if str(r["caso"]).startswith("tono")
    )
    mute_ok = all(
        r.get("shape_ok") and r.get("no_vacio")
        for r in results
        if str(r["caso"]).startswith("mudo")
    )

    # El silencio deberia distinguirse del tono (senal != ruido plano)
    distinct = None
    if means:
        pairs = [(f"tono_{d}", f"mudo_{d}") for d in ("6s", "10s", "15s")]
        diffs = [
            abs(means[a] - means[b])
            for a, b in pairs
            if a in means and b in means
        ]
        if diffs:
            distinct = max(diffs) > 1e-4

    print("\n=== VEREDICTO ===")
    print(f"clips cortos 6-15s producen preds validas : {short_ok}")
    print(f"anuncio mudo produce preds validas        : {mute_ok}")
    print(f"tono se distingue de mudo (mean_abs)       : {distinct}")
    print(f"GO/NO-GO ruta audio: {'GO' if (short_ok and mute_ok) else 'NO-GO'}")

    return 0 if (short_ok and mute_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
