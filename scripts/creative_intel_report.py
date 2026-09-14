"""Genera los datos del informe de Creative Intelligence (Nivel 1: sin TRIBE, sin licencia).

Produce, con datos reales del store y de las anotaciones de LAMBDA:

  A. Inventario creativo      (n, duraciones, cadencia anual por corpus)
  B. Perfil de construcción   (ritmo, caras, texto en pantalla, temas, marca, tono) — del VLM
  C. Rendimiento público      (vistas/día, tasa de likes) con sus salvedades
  D. Benchmark de categoría   (LAMBDA: 2 183 anuncios, anotaciones humanas)
  E. Comparativa de duración  (único eje objetivo comparable entre ambos)

Escribe `data/reports/creative_intel.json` y volca las tablas por stdout para el informe.

Uso:
    .venv/Scripts/python.exe scripts/creative_intel_report.py
"""

from __future__ import annotations

import ast
import io
import json
import re
import sys
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from discovery import store  # noqa: E402

OUT = Path("data/reports")
_API = "https://huggingface.co/api/datasets/behavior-in-the-wild/LAMBDA/tree/main/data"
_BASE = "https://huggingface.co/datasets/behavior-in-the-wild/LAMBDA/resolve/main"


def _plain(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s).lower()) if not unicodedata.combining(c))


def seccion_a(con, corpus: list) -> dict:
    out = {}
    for c in corpus:
        ads = store.all_ads(con, c)
        durs = sorted(int(a["duration"]) for a in ads if a.get("duration"))
        years = Counter((a.get("upload_date") or "")[:4] for a in ads if a.get("upload_date"))
        out[c] = {
            "n": len(ads),
            "dur_min": durs[0] if durs else None,
            "dur_mediana": durs[len(durs) // 2] if durs else None,
            "dur_max": durs[-1] if durs else None,
            "fecha_min": min((a.get("upload_date") for a in ads if a.get("upload_date")), default=None),
            "fecha_max": max((a.get("upload_date") for a in ads if a.get("upload_date")), default=None),
            "por_anio": dict(sorted(years.items())),
        }
    return out


def seccion_b(con, corpus: list) -> dict:
    out = {}
    for c in corpus:
        ads = [a for a in store.all_ads(con, c) if a.get("understanding")]
        ritmo = Counter((a["understanding"] or {}).get("ritmo") for a in ads)
        caras = sum(1 for a in ads if (a["understanding"] or {}).get("hay_caras"))
        texto = sum(1 for a in ads if (a["understanding"] or {}).get("hay_texto_en_pantalla"))
        temas = Counter(t for a in ads for t in ((a["understanding"] or {}).get("temas") or []))
        # El VLM devuelve a veces caracteres sueltos como "elemento de marca": se filtran.
        marca = Counter(
            m for a in ads for m in ((a["understanding"] or {}).get("marca_elementos") or [])
            if len(str(m).strip()) > 2
        )
        tono = Counter((a["understanding"] or {}).get("tono_emocional") for a in ads)
        out[c] = {
            "n": len(ads),
            "ritmo": dict(ritmo.most_common()),
            "pct_caras": round(100 * caras / max(len(ads), 1)),
            "pct_texto_pantalla": round(100 * texto / max(len(ads), 1)),
            "temas_top": temas.most_common(6),
            "marca_top": marca.most_common(6),
            "tono_top": tono.most_common(5),
        }
    return out


def seccion_c(con, corpus: list) -> dict:
    rows = store.stats_table(con)
    out = {}
    for c in corpus:
        ids = {a["id"] for a in store.all_ads(con, c)}
        r = [x for x in rows if x["id"] in ids and x["views"]]
        if not r:
            out[c] = {}
            continue
        vpd = sorted(x["views_per_day"] for x in r if x["views_per_day"])
        lr = sorted(x["like_rate"] for x in r if x["like_rate"])
        out[c] = {
            "n": len(r),
            "vpd_mediana": round(vpd[len(vpd) // 2]) if vpd else None,
            "vpd_max": round(vpd[-1]) if vpd else None,
            "like_rate_mediana_pct": round(100 * lr[len(lr) // 2], 3) if lr else None,
            "n_con_likes": len(lr),
        }
    return out


def seccion_d() -> dict:
    files = [f["path"] for f in json.load(urllib.request.urlopen(_API))]
    df = pd.concat([
        pd.read_parquet(io.BytesIO(urllib.request.urlopen(f"{_BASE}/{f}").read())) for f in files
    ], ignore_index=True)
    ad = pd.json_normalize(df["ad_details"])
    d = pd.DataFrame({
        "pace": ad["Pace"].astype(str).str.strip().str.lower(),
        "dur": ad["Duration"].apply(lambda v: int(re.search(r"(\d+)", str(v)).group(1)) if re.search(r"(\d+)", str(v)) else 0),
        "brand": ad["Brand"].astype(str).str.strip(),
        "recall": df["recall_score"],
    })
    def nsc(v):
        try:
            s = ast.literal_eval(v) if isinstance(v, str) else v
            return len(s)
        except Exception:  # noqa: BLE001
            return 0
    d["n_scenes"] = ad["Scenes"].apply(nsc)
    return {
        "n": len(d),
        "n_marcas": d["brand"].nunique(),
        "pace": {k: int(v) for k, v in d["pace"].value_counts().items()},
        "dur_mediana": int(d["dur"].median()),
        "recall_medio": round(float(d["recall"].mean()), 3),
        "recall_por_pace": {k: round(float(g["recall"].mean()), 3) for k, g in d.groupby("pace")},
        "escenas_medias": round(float(d["n_scenes"].mean()), 2),
        "escenas_max": int(d["n_scenes"].max()),
        "marcas_top": d["brand"].value_counts().head(8).to_dict(),
    }


def seccion_e(a: dict, d: dict) -> dict:
    out = {}
    for c, v in a.items():
        if v["dur_mediana"]:
            out[c] = {"dur_mediana": v["dur_mediana"], "vs_benchmark": v["dur_mediana"] - d["dur_mediana"]}
    out["benchmark"] = {"dur_mediana": d["dur_mediana"]}
    return out


def main() -> int:
    con = store.connect()
    store.init(con)
    corpus = ["telcel", "cocacola"]

    a = seccion_a(con, corpus)
    b = seccion_b(con, corpus)
    c = seccion_c(con, corpus)
    d = seccion_d()
    e = seccion_e(a, d)

    print("=== A. INVENTARIO CREATIVO ===")
    for k, v in a.items():
        print(f"{k:9} n={v['n']:3} | duración {v['dur_min']}-{v['dur_mediana']}-{v['dur_max']}s "
              f"| {v['fecha_min']}..{v['fecha_max']} | por año {v['por_anio']}")

    print("\n=== B. PERFIL DE CONSTRUCCIÓN (VLM) ===")
    for k, v in b.items():
        print(f"{k}: ritmo={v['ritmo']} | caras {v['pct_caras']}% | texto en pantalla {v['pct_texto_pantalla']}%")
        print(f"        temas: {v['temas_top']}")
        print(f"        marca: {v['marca_top']}")
        print(f"        tono : {v['tono_top']}")

    print("\n=== C. RENDIMIENTO PÚBLICO (proxy, ver límites) ===")
    for k, v in c.items():
        print(f"{k:9} n={v.get('n')} | vistas/día mediana {v.get('vpd_mediana')} (máx {v.get('vpd_max')}) "
              f"| like rate mediana {v.get('like_rate_mediana_pct')}% ({v.get('n_con_likes')} con likes)")

    print("\n=== D. BENCHMARK DE CATEGORÍA (LAMBDA, anotación humana) ===")
    print(f"n={d['n']} anuncios | {d['n_marcas']} marcas | duración mediana {d['dur_mediana']}s "
          f"| recall medio {d['recall_medio']}")
    print(f"pace: {d['pace']}")
    print(f"recall por pace: {d['recall_por_pace']}")
    print(f"escenas: media {d['escenas_medias']} (máx {d['escenas_max']} → CENSURADO)")
    print(f"marcas top: {d['marcas_top']}")

    print("\n=== E. DURACIÓN vs BENCHMARK ===")
    print(json.dumps(e, ensure_ascii=False))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "creative_intel.json").write_text(
        json.dumps({"A_inventario": a, "B_perfil": b, "C_publico": c,
                    "D_benchmark": d, "E_duracion": e}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nguardado: {OUT / 'creative_intel.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
