"""3D interactivo y ANIMADO de la respuesta cerebral a lo largo del anuncio.

Superficie cortical (ambos hemisferios) con slider de tiempo + botón play:
cada frame es un segundo (TR) del anuncio. WebGL puro (plotly), sin servidor.

Uso:
    .venv/Scripts/python.exe scripts/brain_3d_movie.py [preds.npy] [salida.html]
    (por defecto usa data/discovery/comercial_preds.npy)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from nilearn import datasets, surface

OUT = Path("data/discovery")

# colormap bipolar: negro/rojo (neg) -> gris (0) -> naranja/amarillo/blanco (pos)
COLORSCALE = [
    [0.0, "#000000"], [0.12, "#5c0000"], [0.30, "#e00000"],
    [0.50, "#9a9a9a"], [0.70, "#ffa500"], [0.86, "#ffff00"], [1.0, "#ffffff"],
]


def main() -> int:
    npy = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "comercial_preds.npy"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT / "brain_movie_3d.html"
    preds = np.load(npy)  # (T, 20484)
    T, V = preds.shape

    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    cl, fl = surface.load_surf_mesh(fs.infl_left)
    cr, fr = surface.load_surf_mesh(fs.infl_right)
    cr = cr.copy()
    cr[:, 0] += (cl[:, 0].max() - cr[:, 0].min()) + 10  # separar hemisferios
    coords = np.vstack([cl, cr])
    faces = np.vstack([fl, fr + len(cl)])
    vmax = float(np.percentile(np.abs(preds), 99))

    def mesh(t: int, with_geom: bool) -> go.Mesh3d:
        kw = dict(
            intensity=preds[t],
            cmin=-vmax, cmax=vmax, colorscale=COLORSCALE,
            showscale=with_geom, colorbar=dict(title="activación") if with_geom else None,
            lighting=dict(ambient=0.55, diffuse=0.6, specular=0.1),
            flatshading=False, name=f"t={t}s",
        )
        if with_geom:
            kw.update(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            )
        return go.Mesh3d(**kw)

    frames = [go.Frame(data=[mesh(t, False)], name=str(t), traces=[0]) for t in range(T)]
    fig = go.Figure(data=[mesh(0, True)], frames=frames)

    steps = [
        dict(method="animate", label=f"{t}s",
             args=[[str(t)], dict(mode="immediate",
                                  frame=dict(duration=0, redraw=True),
                                  transition=dict(duration=0))])
        for t in range(T)
    ]
    fig.update_layout(
        title="comercial Telcel — respuesta cortical a lo largo del anuncio (TRIBE v2, 1 frame = 1 s)",
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data",
        ),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.05, y=0.05, xanchor="left",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=350, redraw=True),
                                      fromcurrent=True, transition=dict(duration=0))]),
                dict(label="⏸ Pausa", method="animate",
                     args=[[None], dict(mode="immediate",
                                        frame=dict(duration=0, redraw=False))]),
            ],
        )],
        sliders=[dict(active=0, currentvalue=dict(prefix="segundo: "), steps=steps,
                      x=0.15, len=0.8)],
        margin=dict(l=0, r=0, t=40, b=0),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn", auto_play=False)
    print(f"[3d movie] {out}  ({out.stat().st_size//1024} KB, {T} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
