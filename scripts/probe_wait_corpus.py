"""
Corpus: ¿la ESPERA antes de la primera palabra predice la activación inicial?

Hipótesis (intuición de negocio, patrón MrBeast): tardar más en arrancar el
contenido/la voz "engancha" más. El barrido sintético (P6 de
``probe_edit_feedback.py``) la apoya: con 0/1/3/5/8 s de espera, la activación
media sube monótonamente 0.104 -> 0.174. Aquí se replica sobre los anuncios
reales de ``data/ads/`` con DOS medidas independientes de la espera:

  1. ``silencio_inicial_s``  — onset por energía (CPU, sin modelo).
  2. ``primera_palabra_s``   — onset de VOZ (Whisper, timestamps por palabra).

Y se cruzan con la activación del modelo en los primeros TR (ruta audio, barata).

Uso:
    .venv/Scripts/python.exe scripts/probe_wait_corpus.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ordering import FSAAVERAGE5_VERTICES  # noqa: E402
from service.metrics.engagement import temporal_profile  # noqa: E402

ADS_DIR = Path("data/ads")
OUT_DIR = Path("data/discovery/probe")
AUDIO_DIR = OUT_DIR / "corpus_audio"
SR = 16000


def leading_silence(x: np.ndarray, frame_s: float = 0.02, thr: float = 3e-3,
                    hold_s: float = 0.15) -> float:
    """Segundos hasta el primer sonido sostenido por encima del umbral."""
    n = int(frame_s * SR)
    hold = int(hold_s / frame_s)
    if len(x) < n * hold:
        return float("nan")
    frames = x[: len(x) // n * n].reshape(-1, n)
    rms = np.sqrt((frames**2).mean(axis=1))
    above = rms > thr
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= hold:
            return round((i - hold + 1) * frame_s, 3)
    return float("nan")


def pearson(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3:
        return float("nan")
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def asr_first_word(pipe, wav: Path, batch_size: int):
    """Onset de la primera palabra. OJO: con ``batch_size > 1`` los timestamps por
    palabra de Whisper se corrompen (dan 0.0 en todos los audios); usar batch=1."""
    x, _ = sf.read(str(wav), dtype="float32")
    out = pipe(
        x,
        return_timestamps="word",
        chunk_length_s=30,
        batch_size=batch_size,
        generate_kwargs={"task": "transcribe", "language": "spanish"},
    )
    first = None
    for c in out.get("chunks", []):
        ts = c.get("timestamp") or (None, None)
        if ts[0] is not None and c.get("text", "").strip():
            first = float(ts[0])
            break
    return first, out["text"].strip()[:70]


def main() -> int:
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ads = sorted(ADS_DIR.glob("*.mp4"))
    if not ads:
        print(f"no hay .mp4 en {ADS_DIR}")
        return 1
    print(f"anuncios: {len(ads)}  ·  torch {torch.__version__} cuda={torch.cuda.is_available()}")

    asr_only = "--asr-only" in sys.argv
    if asr_only:
        prev = json.loads((OUT_DIR / "wait_corpus.json").read_text(encoding="utf-8"))
        rows = prev["filas"]
        print("modo --asr-only: se reusan las métricas TRIBE y el silencio por energía")
    else:
        rows = {}
        for ad in ads:
            from discovery.transcribe import extract_audio

            x = extract_audio(ad)
            wav = AUDIO_DIR / f"{ad.stem}.wav"
            sf.write(str(wav), x, SR)
            rows[ad.name] = {
                "dur_s": round(len(x) / SR, 2),
                "silencio_inicial_s": leading_silence(x),
                "rms_global": round(float(np.sqrt((x**2).mean())), 5),
            }
        print("\nfase 1 (energía) lista")

    # ---------- Fase 2 (GPU): primera palabra (Whisper) -------------------
    from discovery.transcribe import _pipe, unload as unload_whisper

    pipe = _pipe()
    t0 = time.time()
    for ad in ads:
        try:
            first, text = asr_first_word(pipe, AUDIO_DIR / f"{ad.stem}.wav", batch_size=1)
            rows[ad.name]["primera_palabra_s"] = first
            rows[ad.name]["texto"] = text
        except Exception as exc:  # noqa: BLE001
            rows[ad.name]["primera_palabra_s"] = None
            rows[ad.name]["texto"] = f"ERROR {type(exc).__name__}: {exc}"[:70]
    print(f"fase 2 (voz) lista en {time.time() - t0:.0f}s")
    unload_whisper()

    if asr_only:
        df = pd.DataFrame(rows).T
        df.index.name = "anuncio"
        espera = pd.to_numeric(df["primera_palabra_s"], errors="coerce").to_numpy(float)
        finito = np.isfinite(espera)
        resumen = {
            "n": int(len(df)),
            "primera_palabra_s": {
                "min": float(np.nanmin(espera)) if finito.any() else None,
                "median": float(np.nanmedian(espera)) if finito.any() else None,
                "max": float(np.nanmax(espera)) if finito.any() else None,
            },
            "r_espera_vs_act_t0_2": pearson(espera, df["act_t0_2"].to_numpy(float)),
            "r_espera_vs_act_media": pearson(espera, df["act_media"].to_numpy(float)),
            "r_silencio_energia_vs_act_t0_2": pearson(
                df["silencio_inicial_s"].to_numpy(float), df["act_t0_2"].to_numpy(float)
            ),
            "filas": rows,
        }
        (OUT_DIR / "wait_corpus.json").write_text(
            json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pd.set_option("display.width", 220)
        print(df[["dur_s", "silencio_inicial_s", "primera_palabra_s", "act_t0_2", "act_media"]].to_string())
        print(f"\n  r(1ª palabra, act t0-2)     = {resumen['r_espera_vs_act_t0_2']:.3f}")
        print(f"  r(1ª palabra, act media)    = {resumen['r_espera_vs_act_media']:.3f}")
        print(f"  r(silencio energía, act t0-2) = {resumen['r_silencio_energia_vs_act_t0_2']:.3f}")
        return 0

    # ---------- Fase 3 (GPU): TRIBE ruta audio ----------------------------
    import tribev2.demo_utils as du

    from core.tribe_model import resolve_checkpoint_dir

    local_dir = resolve_checkpoint_dir("facebook/tribev2", "./cache")
    t0 = time.time()
    model = du.TribeModel.from_pretrained(local_dir, cache_folder="./cache", device="auto")
    model.data.num_workers = 0
    print(f"TRIBE cargado en {time.time() - t0:.1f}s")
    for ad in ads:
        wav = AUDIO_DIR / f"{ad.stem}.wav"
        event = {
            "type": "Audio",
            "filepath": str(wav),
            "start": 0,
            "timeline": "default",
            "subject": "default",
        }
        events = du.get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
        preds, _ = model.predict(events=events, verbose=False)
        preds = np.asarray(preds)
        assert preds.shape[1] == FSAAVERAGE5_VERTICES
        prof = temporal_profile(preds)
        np.save(OUT_DIR / f"corpus_preds_{ad.stem}.npy", preds)
        rows[ad.name].update(
            {
                "n_tr": len(prof),
                "act_t0_2": round(float(prof[:3].mean()), 5),
                "act_media": round(float(prof.mean()), 5),
                "pico_t": int(np.argmax(prof)),
                "pico": round(float(prof.max()), 5),
            }
        )
    print("fase 3 (TRIBE) lista")

    df = pd.DataFrame(rows).T
    df.index.name = "anuncio"
    df = df.sort_values("primera_palabra_s", na_position="last")
    pd.set_option("display.width", 220)
    print("\n=== CORPUS: espera vs activación inicial ===")
    cols = ["dur_s", "silencio_inicial_s", "primera_palabra_s", "n_tr",
            "act_t0_2", "act_media", "pico_t", "pico"]
    print(df[cols].to_string())

    espera = pd.to_numeric(df["primera_palabra_s"], errors="coerce").to_numpy(float)
    resumen = {
        "n": int(len(df)),
        "primera_palabra_s": {
            "min": float(np.nanmin(espera)) if np.isfinite(espera).any() else None,
            "median": float(np.nanmedian(espera)) if np.isfinite(espera).any() else None,
            "max": float(np.nanmax(espera)) if np.isfinite(espera).any() else None,
        },
        "r_espera_vs_act_t0_2": pearson(espera, df["act_t0_2"].to_numpy(float)),
        "r_espera_vs_act_media": pearson(espera, df["act_media"].to_numpy(float)),
        "r_silencio_energia_vs_act_t0_2": pearson(
            df["silencio_inicial_s"].to_numpy(float), df["act_t0_2"].to_numpy(float)
        ),
        "filas": rows,
    }
    (OUT_DIR / "wait_corpus.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== CORRELACIONES ===")
    print(f"  espera antes de la 1ª palabra: min={resumen['primera_palabra_s']['min']}s "
          f"mediana={resumen['primera_palabra_s']['median']}s max={resumen['primera_palabra_s']['max']}s")
    print(f"  r(espera, activación t0-2) = {resumen['r_espera_vs_act_t0_2']:.3f}")
    print(f"  r(espera, activación media) = {resumen['r_espera_vs_act_media']:.3f}")
    print(f"  r(silencio por energía, activación t0-2) = {resumen['r_silencio_energia_vs_act_t0_2']:.3f}")
    print(f"\nartefactos: {OUT_DIR}/wait_corpus.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
