"""Construye la submuestra estratificada de LAMBDA lista para ingerir.

LAMBDA (Long-Term Ad Memorability, WACV 2025) trae 2 183 anuncios con `recall_score` de 1 749
participantes y `Pace` anotado por humanos. Aquí:

1. Se leen las anotaciones desde HuggingFace (parquet directo: **no** requiere el paquete
   `datasets`, solo pyarrow, para no tocar el entorno).
2. Se elige una **submuestra estratificada por `Pace`** (todos los `high` + N de `medium`/`low`),
   porque el grupo rápido es el cuello de potencia.
3. Se **verifica liveness** de cada enlace (link rot medido: ~27 %) y se descartan los caídos.
4. Se escriben el manifiesto y la lista de URLs para el pipeline.

Uso:
    .venv/Scripts/python.exe scripts/lambda_subsample.py --per-group 160
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from discovery import fetch  # noqa: E402

OUT = Path("data/lambda")
_API = "https://huggingface.co/api/datasets/behavior-in-the-wild/LAMBDA/tree/main/data"
_BASE = "https://huggingface.co/datasets/behavior-in-the-wild/LAMBDA/resolve/main"


def load_annotations() -> pd.DataFrame:
    files = [f["path"] for f in json.load(urllib.request.urlopen(_API))]
    frames = [
        pd.read_parquet(io.BytesIO(urllib.request.urlopen(f"{_BASE}/{f}").read()))
        for f in files
    ]
    df = pd.concat(frames, ignore_index=True)
    return pd.concat([df, pd.json_normalize(df["ad_details"])], axis=1)


def _n_scenes(v) -> int:
    try:
        return len(ast.literal_eval(v)) if isinstance(v, str) else len(v)
    except Exception:  # noqa: BLE001
        return 0


def build_subsample(df: pd.DataFrame, per_group: int, seed: int) -> pd.DataFrame:
    """Todos los `high` + `per_group` de cada uno de los otros grupos."""
    df = df[df["youtube_id"].notna()].copy()
    parts = []
    for pace, g in df.groupby(df["Pace"].str.strip().str.lower()):
        if pace not in ("high", "medium", "low"):
            continue
        n = len(g) if pace == "high" else min(per_group, len(g))
        parts.append(g.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Submuestra estratificada de LAMBDA.")
    ap.add_argument("--per-group", type=int, default=160, help="anuncios por grupo medium/low")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="verificación de enlaces en paralelo")
    args = ap.parse_args()

    df = load_annotations()
    print(f"LAMBDA: {len(df)} anuncios | Pace: {df['Pace'].value_counts().to_dict()}")

    sub = build_subsample(df, args.per_group, args.seed)
    print(f"submuestra candidata: {len(sub)} | Pace: {sub['Pace'].value_counts().to_dict()}")

    urls = [f"https://www.youtube.com/watch?v={y}" for y in sub["youtube_id"]]
    print(f"verificando liveness de {len(urls)} enlaces (link rot medido ~27 %)...")
    live = fetch.fetch_stats(urls, workers=args.workers)
    live_ids = {r["id"] for r in live}
    print(f"vivos: {len(live_ids)}/{len(urls)} ({100*len(live_ids)/len(urls):.0f} %)")

    keep = sub[sub["youtube_id"].isin(live_ids)].copy()
    keep["n_scenes"] = keep["Scenes"].apply(_n_scenes)
    cols = ["youtube_id", "recall_score", "Pace", "Audio", "Brand", "Duration",
            "Orientation", "Title", "n_scenes"]
    keep = keep[[c for c in cols if c in keep.columns]]

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "subsample.json"
    manifest.write_text(keep.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    (OUT / "urls.txt").write_text(
        "\n".join(f"https://www.youtube.com/watch?v={y}" for y in keep["youtube_id"]),
        encoding="utf-8",
    )

    print(f"\nsubmuestra final (viva): {len(keep)}")
    print(keep.groupby("Pace")["recall_score"].agg(["count", "mean"]).round(3).to_string())
    secs = 0
    if len(keep):
        secs = int(keep["Duration"].astype(str).str.extract(r"(\d+)").astype(float).sum().iloc[0])
    print(f"\nsegundos de video (estimado por Duration): {secs} (~{secs * 5.2 / 3600:.1f} h de GPU para neural)")
    print(f"manifiesto: {manifest}")
    print(f"URLs      : {OUT / 'urls.txt'}")
    print("\nPara ingerir el contenido (sin GPU salvo el VLM):")
    print(f"  .venv/Scripts/python.exe -m discovery.pipeline --corpus lambda "
          f"--vlm-batch 3 --urls $(cat {OUT / 'urls.txt'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
