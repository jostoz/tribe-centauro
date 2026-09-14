"""¿Qué propiedades ANOTABLES de un anuncio predicen su memorabilidad? (sin modelo, sin GPU)

Sirve de **contraste de expectativas**: si las anotaciones humanas (ritmo, emociones, tono,
complejidad, personas, marca, duración) apenas explican `recall_score`, entonces pretender que un
modelo cerebral extraiga de ahí una señal fuerte es una expectativa mal puesta.

Con n≈2 183 **todo va a salir "significativo"**: el reporte enfatiza **tamaños de efecto**
(ρ de Spearman, η²), no p-valores. Los p van con corrección de Holm y se leen como control.

Uso:
    .venv/Scripts/python.exe scripts/analyze_lambda_annotations.py
"""

from __future__ import annotations

import ast
import io
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.metrics.stats import holm_bonferroni  # noqa: E402

_API = "https://huggingface.co/api/datasets/behavior-in-the-wild/LAMBDA/tree/main/data"
_BASE = "https://huggingface.co/datasets/behavior-in-the-wild/LAMBDA/resolve/main"
_OUT = Path("data/lambda")


def load() -> pd.DataFrame:
    files = [f["path"] for f in json.load(urllib.request.urlopen(_API))]
    frames = [
        pd.read_parquet(io.BytesIO(urllib.request.urlopen(f"{_BASE}/{f}").read()))
        for f in files
    ]
    df = pd.concat(frames, ignore_index=True)
    ad = pd.json_normalize(df["ad_details"])
    return pd.concat([df.drop(columns=["ad_details"]), ad], axis=1)


def _scenes(v):
    """`Scenes` llega como ndarray (json_normalize ya lo parseó) o como str."""
    try:
        s = ast.literal_eval(v) if isinstance(v, str) else v
        return list(s) if hasattr(s, "__len__") else []
    except Exception:  # noqa: BLE001
        return []


def _secs(v) -> float:
    m = re.search(r"(\d+)", str(v))
    return float(m.group(1)) if m else np.nan


_HUMAN = re.compile(r"human|people|person|man\b|woman|child|face|friend|family", re.I)


def features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["recall"] = pd.to_numeric(df["recall_score"], errors="coerce")
    out["duration_s"] = df["Duration"].apply(_secs)
    out["orientation"] = df["Orientation"].astype(str).str.strip().str.lower().replace({"": np.nan})
    out["pace"] = df["Pace"].astype(str).str.strip().str.lower().replace({"": np.nan})
    # OJO: `Audio` es la TRANSCRIPCION (texto hablado), no un tipo de audio.
    out["has_transcript"] = df["Audio"].astype(str).str.strip().ne("")
    out["brand"] = df["Brand"].astype(str).str.strip()

    parsed = df["Scenes"].apply(_scenes)
    out["n_scenes"] = parsed.apply(len)
    out["scenes_per_s"] = out["n_scenes"] / out["duration_s"]
    out["human_present"] = parsed.apply(
        lambda ss: any(_HUMAN.search(str(s.get("Description", "")) + " " + str(s.get("Tags", ""))) for s in ss)
    )
    for field, name in (("Tone", "tone"), ("Emotions", "emotion"),
                        ("Visual Complexity", "complexity"), ("Photography Style", "photo_style")):
        out[name] = parsed.apply(
            lambda ss, f=field: (Counter(
                str(s.get(f, "")).strip().lower() for s in ss if str(s.get(f, "")).strip()
            ).most_common(1) or [("", 0)])[0][0] or np.nan
        )
    return out


def main() -> int:
    f = features(load())
    f = f.dropna(subset=["recall"])
    print(f"anuncios: {len(f)} | recall medio {f['recall'].mean():.3f} (SD {f['recall'].std():.3f})")

    filas = []

    # --- continuas: Spearman ---
    for col in ("n_scenes", "duration_s", "scenes_per_s"):
        d = f[[col, "recall"]].dropna()
        rho, p = sps.spearmanr(d[col], d["recall"])
        filas.append({"variable": col, "tipo": "continua", "n": len(d),
                      "efecto": rho, "metric": "rho", "p": p})

    # --- categóricas: Kruskal + eta^2 ---
    for col in ("pace", "orientation", "has_transcript", "tone", "emotion", "complexity",
                "photo_style", "human_present", "brand"):
        d = f[[col, "recall"]].dropna()
        grupos = [g["recall"].values for _, g in d.groupby(col) if len(g) >= 15]
        if len(grupos) < 2:
            continue
        h, p = sps.kruskal(*grupos)
        allv = d["recall"]
        ss_between = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in grupos)
        eta2 = ss_between / ((allv - allv.mean()) ** 2).sum()
        filas.append({"variable": col, "tipo": "categorica", "n": len(d), "k": len(grupos),
                      "efecto": eta2, "metric": "eta2", "p": p})

    r = pd.DataFrame(filas)
    r["p_holm"] = holm_bonferroni(r["p"].values)
    r["signif_holm"] = r["p_holm"] < 0.05
    r = r.sort_values("efecto", key=lambda s: s.abs(), ascending=False)

    print(f"\n{'variable':14} {'tipo':11} {'n':>5} {'efecto':>8} {'métrica':>8} {'p Holm':>9}  sig")
    for _, x in r.iterrows():
        print(f"{x['variable']:14} {x['tipo']:11} {int(x['n']):>5} {x['efecto']:+8.3f} "
              f"{x['metric']:>8} {x['p_holm']:9.4f}  {'sí' if x['signif_holm'] else 'no'}")

    top = r.iloc[0]
    print(f"\nMayor efecto: {top['variable']} con {top['metric']} = {top['efecto']:+.3f} "
          f"→ explica ~{top['efecto']**2*100 if top['metric']=='rho' else top['efecto']*100:.1f} % de la varianza")

    print("\n--- Pace: recall por grupo ---")
    print(f.groupby("pace")["recall"].agg(["count", "mean", "std"]).round(3).to_string())
    print("\n--- brand: las 8 con más anuncios ---")
    top_b = f["brand"].value_counts().head(8).index
    print(f[f["brand"].isin(top_b)].groupby("brand")["recall"].agg(["count", "mean"]).round(3).to_string())

    _OUT.mkdir(parents=True, exist_ok=True)
    r.to_json(_OUT / "annotation_recall_assoc.json", orient="records", force_ascii=False, indent=2)
    print(f"\nguardado: {_OUT / 'annotation_recall_assoc.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
