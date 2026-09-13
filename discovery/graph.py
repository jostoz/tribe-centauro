"""Grafo de contenidos de los anuncios (networkx) + export interactivo.

Nodos:
  - ``ad``      un anuncio (con título, duración, transcript, perfil neural si existe)
  - ``obj``     objeto detectado        - ``pers``  personaje
  - ``tema``    tema                    - ``marca`` elemento de marca
Aristas:
  - anuncio -> atributo (aparece en)
  - anuncio <-> anuncio (peso = nº de atributos compartidos)
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List

import networkx as nx

from discovery import entities as _ent

_COLORS = {
    "ad": "#e63946",
    "obj": "#457b9d",
    "pers": "#f4a261",
    "tema": "#2a9d8f",
    "marca": "#8338ec",
}


def build_graph(records: List[dict]) -> nx.Graph:
    G = nx.Graph()
    ad_attrs: Dict[str, set] = {}
    for r in records:
        aid = str(r["id"])
        neural = r.get("neural", {}) or {}
        G.add_node(
            aid,
            kind="ad",
            label=r.get("title") or aid,
            duration=r.get("duration"),
            url=r.get("url"),
            resumen=(r.get("understanding", {}) or {}).get("resumen", ""),
            ritmo=(r.get("understanding", {}) or {}).get("ritmo", ""),
            tono=(r.get("understanding", {}) or {}).get("tono_emocional", ""),
            red_dominante=neural.get("red_dominante", ""),
        )
        attrs = set()
        u = r.get("understanding", {}) or {}
        for kind, canon, raw in _ent.extract_entities(u):
            node = f"{kind}:{canon}"
            if node not in G:
                G.add_node(node, kind=kind, label=raw)
            G.add_edge(aid, node, kind="tiene")
            attrs.add(node)
        ad_attrs[aid] = attrs

    # aristas anuncio-anuncio por atributos compartidos
    for a, b in itertools.combinations(ad_attrs, 2):
        shared = ad_attrs[a] & ad_attrs[b]
        if shared:
            G.add_edge(a, b, kind="similar", weight=len(shared), shared=sorted(shared))
    return G


def to_json(G: nx.Graph) -> dict:
    return {
        "nodes": [{"id": n, **d} for n, d in G.nodes(data=True)],
        "edges": [{"source": u, "target": v, **d} for u, v, d in G.edges(data=True)],
    }


def export(G: nx.Graph, out_stem: str | Path) -> dict:
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    data = to_json(G)
    json_path = out_stem.with_suffix(".json")
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    graphml_path = out_stem.with_suffix(".graphml")
    # graphml no admite listas/None: serializar valores complejos a str
    H = nx.Graph()
    for n, d in G.nodes(data=True):
        H.add_node(n, **{k: ("" if v is None else v if isinstance(v, (int, float, str)) else json.dumps(v, ensure_ascii=False)) for k, v in d.items()})
    for u, v, d in G.edges(data=True):
        H.add_edge(u, v, **{k: (val if isinstance(val, (int, float, str)) else json.dumps(val, ensure_ascii=False)) for k, val in d.items()})
    nx.write_graphml(H, graphml_path)
    html_path = out_stem.with_suffix(".html")
    html_path.write_text(_html(data), encoding="utf-8")
    return {"json": str(json_path), "graphml": str(graphml_path), "html": str(html_path)}


def _html(data: dict) -> str:
    vis_nodes = [
        {
            "id": n["id"],
            "label": (n.get("label") or n["id"])[:40],
            "group": n.get("kind", "ad"),
            "color": _COLORS.get(n.get("kind", "ad"), "#999"),
            "title": (n.get("resumen") or n.get("label") or n["id"]),
            "value": 6 if n.get("kind") == "ad" else 3,
        }
        for n in data["nodes"]
    ]
    vis_edges = [
        {
            "from": e["source"],
            "to": e["target"],
            "value": e.get("weight", 1),
            "dashes": e.get("kind") == "similar",
        }
        for e in data["edges"]
    ]
    payload = json.dumps({"nodes": vis_nodes, "edges": vis_edges}, ensure_ascii=False)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Grafo de anuncios - Centauro</title>"
        "<script src='https://unpkg.com/vis-network/standalone/umd/vis-network.min.js'></script>"
        "<style>body{margin:0;font-family:sans-serif}#net{width:100vw;height:100vh}"
        "#leg{position:fixed;top:8px;left:8px;background:#fff;padding:8px;border:1px solid #ccc;font-size:12px}</style>"
        "</head><body><div id='leg'>"
        "<b>Leyenda</b><br>"
        "<span style='color:#e63946'>&#9679;</span> anuncio "
        "<span style='color:#457b9d'>&#9679;</span> objeto "
        "<span style='color:#f4a261'>&#9679;</span> personaje "
        "<span style='color:#2a9d8f'>&#9679;</span> tema "
        "<span style='color:#8338ec'>&#9679;</span> marca<br>"
        "linea punteada = anuncios similares</div>"
        "<div id='net'></div><script>"
        f"const data={payload};"
        "const c=document.getElementById('net');"
        "new vis.Network(c,{nodes:new vis.DataSet(data.nodes),edges:new vis.DataSet(data.edges)},"
        "{physics:{stabilization:true},nodes:{shape:'dot',scaling:{min:3,max:20}},"
        "edges:{smooth:false,color:{opacity:0.4}}});"
        "</script></body></html>"
    )
