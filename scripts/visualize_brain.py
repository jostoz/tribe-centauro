"""Visualiza la respuesta cerebral predicha por TRIBE v2 para un anuncio.

Genera:
  - brain_peak.png       : mapa cortical multi-vista del timestep de mayor activación
  - brain_peak_*.html    : superficie 3D interactiva (WebGL, nilearn view_surf)
  - brain_movie.gif      : animación segundo-a-segundo (hemisferio izq, vista lateral)

Convención temporal: ``preds[k]`` es la respuesta al **segundo k** del anuncio
(``core.ordering.ALIGNMENT_CONVENTION == "stimulus-aligned"``; el offset hemodinámico de
5 s ya lo aplica el checkpoint, NO hay que restarlo). Precisión de la localización
absoluta: **≈ ±1.5 s** — las etiquetas ``t=Ns`` no son de precisión sub-segundo.
Ver docs/FASE_0.5_ALINEACION.md.

Uso:
    .venv/Scripts/python.exe scripts/visualize_brain.py ads/comercial.mp4
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ordering import FSAAVERAGE5_VERTICES  # noqa: E402

OUT = Path("data/discovery")
HEMI = FSAAVERAGE5_VERTICES // 2  # 10242


def get_preds(video_path: Path) -> np.ndarray:
    cache = OUT / f"{video_path.stem}_preds.npy"
    if cache.exists():
        print(f"preds en caché: {cache}")
        return np.load(cache)
    import pandas as pd

    from core.tribe_model import resolve_checkpoint_dir
    from tribev2.demo_utils import TribeModel, get_audio_and_text_events

    print("corriendo TRIBE v2 (ruta video)...")
    local_dir = resolve_checkpoint_dir("facebook/tribev2", "./cache")
    model = TribeModel.from_pretrained(local_dir, cache_folder="./cache", device="auto")
    model.data.num_workers = 0
    event = {"type": "Video", "filepath": str(video_path), "start": 0,
             "timeline": "default", "subject": "default"}
    events = get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
    t0 = time.time()
    preds, _ = model.predict(events=events, verbose=False)
    print(f"inferencia en {time.time()-t0:.1f}s -> {preds.shape}")
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(cache, preds)
    return preds


def main() -> int:
    from nilearn import datasets, plotting

    video = Path(sys.argv[1] if len(sys.argv) > 1 else "ads/comercial.mp4")
    preds = get_preds(video)
    OUT.mkdir(parents=True, exist_ok=True)

    prof = np.mean(np.abs(preds), axis=1)
    t = int(np.argmax(prof))
    print(f"timestep pico: t={t}s (de {preds.shape[0]}) — stimulus-aligned, ±1.5 s")
    L, R = preds[t, :HEMI], preds[t, HEMI:]

    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    vmax = float(np.percentile(np.abs(preds), 99))

    # 1. PNG multi-vista (lateral + medial, ambos hemisferios)
    fig, axes = plt.subplots(2, 2, subplot_kw={"projection": "3d"}, figsize=(11, 9))
    specs = [
        (fs.infl_left, L, fs.sulc_left, "left", "lateral", axes[0, 0], "Izq lateral"),
        (fs.infl_right, R, fs.sulc_right, "right", "lateral", axes[0, 1], "Der lateral"),
        (fs.infl_left, L, fs.sulc_left, "left", "medial", axes[1, 0], "Izq medial"),
        (fs.infl_right, R, fs.sulc_right, "right", "medial", axes[1, 1], "Der medial"),
    ]
    for mesh, vals, bg, hemi, view, ax, title in specs:
        plotting.plot_surf_stat_map(
            mesh, vals, hemi=hemi, bg_map=bg, view=view, cmap="hot",
            vmax=vmax, threshold=vmax * 0.15, colorbar=False, axes=ax,
        )
        ax.set_title(title, fontsize=11)
    fig.suptitle(
        f"{video.stem} — activación cortical en t={t}s ±1.5s (TRIBE v2, stimulus-aligned)",
        fontsize=13,
    )
    png = OUT / "brain_peak.png"
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[png] {png}")

    # 2. HTML 3D interactivo (WebGL) por hemisferio
    for mesh, vals, bg, name in [
        (fs.infl_left, L, fs.sulc_left, "left"),
        (fs.infl_right, R, fs.sulc_right, "right"),
    ]:
        view = plotting.view_surf(
            mesh, vals, bg_map=bg, cmap="hot", symmetric_cmap=False,
            vmax=vmax, threshold=vmax * 0.15, title=f"{video.stem} t={t}s ({name})",
        )
        html = OUT / f"brain_peak_{name}.html"
        view.save_as_html(str(html))
        print(f"[html3d] {html}")

    # 3. Animación segundo-a-segundo (izq lateral)
    try:
        from PIL import Image

        frames = []
        for ti in range(preds.shape[0]):
            f, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(4, 4))
            plotting.plot_surf_stat_map(
                fs.infl_left, preds[ti, :HEMI], hemi="left", bg_map=fs.sulc_left,
                view="lateral", cmap="hot", vmax=vmax, threshold=vmax * 0.15,
                colorbar=False, axes=ax,
            )
            ax.set_title(f"t={ti}s", fontsize=10)
            f.canvas.draw()
            buf = np.asarray(f.canvas.buffer_rgba())[:, :, :3]
            frames.append(Image.fromarray(buf.copy()))
            plt.close(f)
        gif = OUT / "brain_movie.gif"
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=350, loop=0)
        print(f"[gif] {gif} ({len(frames)} frames)")
    except Exception as exc:  # noqa: BLE001
        print(f"[gif] omitido: {type(exc).__name__}: {exc}")

    print("\n=== visualizaciones listas en data/discovery/ ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
