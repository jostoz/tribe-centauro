"""Construye el manifiesto multi-marca para el inventario de Creative Intelligence.

Para cada marca: busca en YouTube, filtra por duración de anuncio (6–60 s), deduplica entre
marcas (un vídeo se asigna a la primera que lo encuentre) y **verifica que el enlace esté vivo**
(el link rot medido en este tipo de corpus es ~30 %).

Escribe `data/brands/<slug>.txt` (una URL por línea) y `data/brands/manifest.json`.

Uso:
    .venv/Scripts/python.exe scripts/build_brand_manifest.py --per-brand 18
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import yt_dlp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from discovery import fetch  # noqa: E402

OUT = Path("data/brands")

# Marcas por categoría (el cliente elige su categoría y sus competidores)
MARCAS = {
    "telecom": ["AT&T México", "Movistar México"],
    "bebidas": ["Pepsi México", "Jarritos", "Corona", "Tecate"],
    "alimentos": ["Sabritas", "Bimbo", "Marinela", "McDonald's México"],
    "retail": ["Walmart México", "Liverpool", "Elektra", "OXXO"],
    "banca": ["BBVA México", "Banorte", "Santander México"],
    "autos": ["Nissan México", "Kia México"],
    "tecnologia": ["Samsung México", "Xiaomi México"],
}


def slug(name: str) -> str:
    s = unicodedata.normalize("NFKD", name.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def candidatos(marca: str, n: int) -> list:
    opts = {"quiet": True, "no_warnings": True, "skip_download": True,
            "extract_flat": True, "playlistend": n + 12, "ignoreerrors": True}
    out, vistos = [], set()
    for q in (f"comercial {marca}", f"{marca} anuncio mexico"):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(f"ytsearch{n + 12}:{q}", download=False)
        except Exception:  # noqa: BLE001
            continue
        for e in (info or {}).get("entries", []) or []:
            if not e or not e.get("id") or e["id"] in vistos:
                continue
            d = e.get("duration") or 0
            if 6 <= d <= 60:
                vistos.add(e["id"])
                out.append((e["id"], d))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Manifiesto multi-marca para el inventario.")
    ap.add_argument("--per-brand", type=int, default=18)
    ap.add_argument("--workers", type=int, default=3, help="bajo a propósito: ver THROTTLE en fetch.py")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    asignados: dict = {}
    usados: set = set()
    manifiesto = {}

    for cat, marcas in MARCAS.items():
        for marca in marcas:
            cands = [(i, d) for i, d in candidatos(marca, args.per_brand) if i not in usados]
            cands = cands[: args.per_brand]
            if not cands:
                print(f"  {marca:22} sin candidatos")
                continue
            urls = [f"https://www.youtube.com/watch?v={i}" for i, _ in cands]
            vivos = fetch.fetch_stats(urls, workers=args.workers)
            vivos_ids = [r["id"] for r in vivos]
            usados.update(vivos_ids)
            asignados[marca] = vivos_ids
            p = OUT / f"{slug(marca)}.txt"
            p.write_text("\n".join(f"https://www.youtube.com/watch?v={i}" for i in vivos_ids),
                         encoding="utf-8")
            manifiesto[marca] = {"categoria": cat, "n": len(vivos_ids),
                                 "candidatos": len(cands), "archivo": str(p)}
            print(f"  {marca:22} {cat:11} candidatos {len(cands):3} → vivos {len(vivos_ids):3}")

    (OUT / "manifest.json").write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    total = sum(v["n"] for v in manifiesto.values())
    print(f"\nmarcas: {len(manifiesto)} | anuncios vivos: {total}")
    print(f"manifiesto: {OUT / 'manifest.json'}")
    for m, v in manifiesto.items():
        print(f"  --corpus {slug(m)} --urls-file {v['archivo']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
