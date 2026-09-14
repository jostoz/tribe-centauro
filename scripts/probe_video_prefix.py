"""
¿El efecto "prefijo sin información" existe en el canal de VIDEO?

El canal de audio premia el SILENCIO DIGITAL al inicio (P7): 3 s de ceros suben
la activación t0-2 de 0.096 a 0.186, y basta ruido de sala a -50 dBFS para que
desaparezca. La práctica real de video (set-piece visual antes de la primera
palabra) NO es silencio digital: es imagen + música. Así que hay que probar si
el canal de video tiene el mismo comportamiento.

Variantes (primeros 12 s de ``ads/comercial.mp4``):
  v_base         — el clip tal cual.
  v_negro        — 3 s de negro + silencio digital, luego el clip.
  v_negro_ruido  — 3 s de negro + ruido rosa -50 dBFS, luego el clip.

Uso:
    .venv/Scripts/python.exe scripts/probe_video_prefix.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imageio_ffmpeg  # noqa: E402

from core.ordering import FSAAVERAGE5_VERTICES  # noqa: E402
from service.metrics.engagement import temporal_profile  # noqa: E402

SRC = Path("ads/comercial.mp4")
OUT_DIR = Path("data/discovery/probe/video")
CLIP_S = 12.0
PREFIX_S = 3.0
FF = imageio_ffmpeg.get_ffmpeg_exe()


def build_variants() -> dict[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "v_base": OUT_DIR / "v_base.mp4",
        "v_negro": OUT_DIR / "v_negro.mp4",
        "v_negro_ruido": OUT_DIR / "v_negro_ruido.mp4",
    }
    common = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
              "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"]

    subprocess.run([FF, "-y", "-i", str(SRC), "-t", str(CLIP_S), *common,
                    str(out["v_base"])], check=True, capture_output=True)

    black = ["-f", "lavfi", "-i",
             f"color=c=black:s=720x1280:r=2997/100:d={PREFIX_S}"]
    subprocess.run(
        [FF, "-y", *black, "-i", str(SRC), "-filter_complex",
         f"[0:v]setsar=1[v0];[1:v]trim=0:{CLIP_S},setpts=PTS-STARTPTS[v1];"
         "[v0][v1]concat=n=2:v=1:a=0[v];"
         f"[1:a]atrim=0:{CLIP_S},asetpts=PTS-STARTPTS,"
         f"adelay={int(PREFIX_S * 1000)}:all=1[a]",
         "-map", "[v]", "-map", "[a]", *common, str(out["v_negro"])],
        check=True, capture_output=True,
    )

    noise = ["-f", "lavfi", "-i", f"anoisesrc=d={PREFIX_S}:c=pink:a=0.003:r=44100"]
    subprocess.run(
        [FF, "-y", *black, *noise, "-i", str(SRC), "-filter_complex",
         f"[0:v]setsar=1[v0];[2:v]trim=0:{CLIP_S},setpts=PTS-STARTPTS[v1];"
         "[v0][v1]concat=n=2:v=1:a=0[v];"
         "[1:a]aformat=channel_layouts=stereo,atrim=0:3,asetpts=PTS-STARTPTS[a0];"
         f"[2:a]atrim=0:{CLIP_S},asetpts=PTS-STARTPTS[a1];"
         "[a0][a1]concat=n=2:v=0:a=1[a]",
         "-map", "[v]", "-map", "[a]", *common, str(out["v_negro_ruido"])],
        check=True, capture_output=True,
    )
    return out


def main() -> int:
    import torch

    from tribev2.demo_utils import TribeModel, get_audio_and_text_events

    from core.tribe_model import resolve_checkpoint_dir

    print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()}")
    variants = build_variants()
    for name, p in variants.items():
        print(f"  {name:14s} {p.stat().st_size/1e6:6.2f} MB  {p}")

    local_dir = resolve_checkpoint_dir("facebook/tribev2", "./cache")
    t0 = time.time()
    model = TribeModel.from_pretrained(local_dir, cache_folder="./cache", device="auto")
    model.data.num_workers = 0
    print(f"\nmodelo cargado en {time.time() - t0:.1f}s\n")

    rows = {}
    for name, path in variants.items():
        event = {
            "type": "Video",
            "filepath": str(path),
            "start": 0,
            "timeline": "default",
            "subject": "default",
        }
        events = get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
        t0 = time.time()
        preds, _ = model.predict(events=events, verbose=False)
        preds = np.asarray(preds)
        assert preds.shape[1] == FSAAVERAGE5_VERTICES, preds.shape
        prof = temporal_profile(preds)
        np.save(OUT_DIR / f"preds_{name}.npy", preds)
        rows[name] = {
            "n_tr": int(len(prof)),
            "transitorio_t0_2": round(float(prof[:3].mean()), 5),
            "media_global": round(float(prof.mean()), 5),
            "pico_t": int(np.argmax(prof)),
            "pico": round(float(prof.max()), 5),
            "perfil_t0_9": [round(float(v), 4) for v in prof[:10]],
            "t_s": round(time.time() - t0, 1),
        }
        print(f"  {name:14s} t0-2={rows[name]['transitorio_t0_2']:.4f} "
              f"(t={rows[name]['t_s']}s)")

    (OUT_DIR / "video_prefix.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== VIDEO: prefijo sin información (3 s) antes del mismo clip ===")
    print(f"  {'variante':>14} {'t0-2':>8} {'media':>8} {'pico@t':>7} {'pico':>8}   perfil t0..t9")
    for name, row in rows.items():
        print(f"  {name:>14} {row['transitorio_t0_2']:>8.4f} {row['media_global']:>8.4f}"
              f" {row['pico_t']:>7d} {row['pico']:>8.4f}   "
              + " ".join(f"{v:.3f}" for v in row["perfil_t0_9"]))
    print(f"\nartefactos: {OUT_DIR}/video_prefix.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
