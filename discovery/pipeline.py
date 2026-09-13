"""Orquestación por FASES DE MODELO: fetch -> transcribe -> understand -> neural -> graph.

Regla de oro en 1 GPU (ver docs/PLAN_ESCALABILIDAD.md): procesar **por fase de modelo,
no por anuncio**. Cada modelo (Whisper, Qwen, TRIBE) se carga UNA vez por corrida y se
procesan todos los anuncios de esa fase; luego se descarga y se libera VRAM. Así el
modelo se carga 3 veces en total, no 3×N.

Idempotencia: cada etapa se cachea por contenido (sha256 del video) y los resultados
persisten en SQLite; re-correr no repite cómputo ni descargas (`--no-cache` lo fuerza).

Ejemplos:
    .venv/Scripts/python.exe -m discovery.pipeline --channel @Telcel --n 12
    .venv/Scripts/python.exe -m discovery.pipeline --search "Telcel anuncio" --n 8 --neural
    .venv/Scripts/python.exe -m discovery.pipeline --graph-only
    .venv/Scripts/python.exe -m discovery.pipeline --cache-stats
    .venv/Scripts/python.exe -m discovery.pipeline --prune-cache 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from discovery import cache, entities, fetch, graph, store  # noqa: E402

OUT = Path("data/discovery")


class Metrics:
    """Tiempos por etapa y aciertos de caché, para medir escalabilidad con datos."""

    def __init__(self) -> None:
        self.rows: List[Dict] = []

    def add(self, stage: str, ad_id: str, seconds: float, cached: bool) -> None:
        self.rows.append({"stage": stage, "ad": ad_id, "s": seconds, "cached": cached})

    def report(self) -> None:
        if not self.rows:
            return
        stages = sorted({r["stage"] for r in self.rows})
        print("\n=== métricas por etapa ===")
        print(f"{'etapa':12} {'n':>4} {'caché':>6} {'total_s':>9} {'s/ann':>7}")
        for st in stages:
            rs = [r for r in self.rows if r["stage"] == st]
            tot = sum(r["s"] for r in rs)
            hits = sum(1 for r in rs if r["cached"])
            print(f"{st:12} {len(rs):>4} {hits:>6} {tot:>9.1f} {tot/max(len(rs),1):>7.2f}")


def _set_hf_home() -> None:
    """Permite mover la caché de modelos a otro disco (evita llenar C:)."""
    alt = os.environ.get("CENTAURO_HF_HOME")
    if alt:
        os.environ.setdefault("HF_HOME", alt)


def _upsert(con, rec: dict) -> None:
    u = rec.get("understanding")
    ents = entities.extract_entities(u) if u else None
    store.upsert_ad(con, rec, ents)


def _video_ok(a: dict) -> bool:
    vp = a.get("video_path")
    return bool(vp) and Path(vp).is_file()


# ---------------------------------------------------------------------------
# Fase 0 — fetch (CPU/red, en paralelo)
# ---------------------------------------------------------------------------
def phase_fetch(con, urls: List[str], workers: int) -> List[dict]:
    existing = {a["url"]: a for a in store.all_ads(con) if a.get("url")}
    to_download = [u for u in urls if not (u in existing and _video_ok(existing[u]))]
    skipped = len(urls) - len(to_download)
    if skipped:
        print(f"[fetch] {skipped} ya descargados (resume)")
    if to_download:
        print(f"[fetch] descargando {len(to_download)} en paralelo (workers={workers})...")
        for rec in fetch.download_many(to_download, workers=workers):
            _upsert(con, rec)
    ads = [a for a in store.all_ads(con) if a.get("url") in set(urls)]
    return [a for a in ads if _video_ok(a)]


# ---------------------------------------------------------------------------
# Fase 1 — transcripción (Whisper residente)
# ---------------------------------------------------------------------------
def phase_transcribe(con, ads: List[dict], model: str, language: str, m: Metrics, use_cache: bool) -> None:
    pend = [a for a in ads if not use_cache or not a.get("transcript")]
    if not pend:
        print("[transcribe] ya completo (resume)")
        return
    print(f"[transcribe] {len(pend)} anuncios (Whisper residente)...")
    from discovery import transcribe

    for a in pend:
        t0 = time.time()
        key = cache.content_key(a["video_path"], "transcribe", f"{model}|lang={language}")
        cached = None if not use_cache else cache.get("transcribe", key)
        if cached is not None:
            a["transcript"] = cached
            m.add("transcribe", a["id"], time.time() - t0, True)
            _upsert(con, a)
            continue
        a["transcript"] = transcribe.transcribe(a["video_path"], model=model, language=language or None)
        cache.put("transcribe", key, a["transcript"])
        _upsert(con, a)
        m.add("transcribe", a["id"], time.time() - t0, False)
    transcribe.unload()
    print("      [transcribe] modelo descargado")


# ---------------------------------------------------------------------------
# Fase 2 — comprensión (Qwen residente)
# ---------------------------------------------------------------------------
def phase_understand(
    con, ads: List[dict], model_id: str, frames: int, m: Metrics, use_cache: bool, batch_size: int
) -> None:
    # con --no-cache (use_cache=False) se ignora también el resume del store
    pend = [a for a in ads if not use_cache or not a.get("understanding")]
    if not pend:
        print("[understand] ya completo (resume)")
        return

    # 1) resolver desde caché (barato, sin GPU)
    todo = []
    for a in pend:
        key = cache.content_key(a["video_path"], "understand", f"{model_id}|frames={frames}")
        cached = None if not use_cache else cache.get("understand", key)
        if cached is not None:
            a["understanding"] = cached
            _upsert(con, a)
            m.add("understand", a["id"], 0.0, True)
        else:
            todo.append((a, key))

    if not todo:
        print(f"[understand] {len(pend)} desde caché (resume)")
        return

    print(f"[understand] {len(todo)} anuncios ({model_id}, {frames} frames, batch={batch_size})...")
    from discovery import understand

    t0 = time.time()
    results = understand.understand_batch(
        [a["video_path"] for a, _ in todo],
        n_frames=frames,
        model_id=model_id,
        batch_size=batch_size,
    )
    total = time.time() - t0
    per = total / max(len(todo), 1)
    ok = 0
    for (a, key), res in zip(todo, results):
        if not isinstance(res, dict) or "_error" in res:
            err = res.get("_error") if isinstance(res, dict) else res
            print(f"      [understand] {a['id']} FALLO: {err}")
            continue
        a["understanding"] = res
        cache.put("understand", key, res)
        _upsert(con, a)
        m.add("understand", a["id"], per, False)
        ok += 1
    print(f"      [understand] {ok}/{len(todo)} en {total:.1f}s ({per:.1f}s/anuncio amortizado)")
    understand.unload()
    print("      [understand] modelo descargado")


# ---------------------------------------------------------------------------
# Fase 3 — perfil neural (TRIBE residente)
# ---------------------------------------------------------------------------
def phase_neural(con, ads: List[dict], m: Metrics, use_cache: bool) -> None:
    pend = [a for a in ads if not use_cache or not a.get("neural")]
    if not pend:
        print("[neural] ya completo (resume)")
        return
    print(f"[neural] {len(pend)} anuncios (TRIBE residente)...")
    from discovery.neural import NeuralAnalyzer

    an = NeuralAnalyzer()
    an.load()
    for a in pend:
        t0 = time.time()
        key = cache.content_key(a["video_path"], "neural", "tribev2|default")
        cached = None if not use_cache else cache.get("neural", key)
        if cached is not None:
            a["neural"] = cached
            m.add("neural", a["id"], time.time() - t0, True)
            _upsert(con, a)
            continue
        try:
            a["neural"] = an.analyze(a["video_path"])
            cache.put("neural", key, a["neural"])
            _upsert(con, a)
            m.add("neural", a["id"], time.time() - t0, False)
        except Exception as exc:  # noqa: BLE001 - un anuncio roto no debe tumbar la fase
            print(f"      [neural] {a['id']} FALLO: {type(exc).__name__}: {exc}")
    an.unload()
    print("      [neural] modelo descargado")


def build_and_export_graph(con) -> dict:
    records = store.all_ads(con)
    paths = graph.export(graph.build_graph(records), OUT / "graph")
    print(f"\n[graph] {len(records)} anuncios -> {paths['html']}")
    tops = store.top_entities(con, limit=8)
    if tops:
        print("[graph] entidades más frecuentes: " + ", ".join(f"{c}({n})" for c, n in tops))
    return paths


def run(args: argparse.Namespace) -> int:
    _set_hf_home()

    if args.cache_stats:
        st = cache.stats()
        print("caché por etapa:", st or "(vacía)")
        return 0
    if args.prune_cache is not None:
        n = cache.prune(args.prune_cache)
        print(f"caché: {n} entradas borradas (> {args.prune_cache} días)")
        return 0

    con = store.connect()
    store.init(con)

    if args.graph_only:
        build_and_export_graph(con)
        return 0

    urls: List[str] = list(args.urls or [])
    if args.search:
        urls += fetch.search_urls(args.search, args.n)
    if args.channel:
        urls += fetch.channel_urls(args.channel, args.n)
    if not urls:
        print("Nada que hacer. Usa --urls, --search o --channel.")
        return 1

    m = Metrics()
    t_start = time.time()
    ads = phase_fetch(con, urls, args.workers)
    print(f"[fetch] {len(ads)} anuncios con video disponible")
    if not ads:
        return 1

    if not args.no_transcribe:
        phase_transcribe(con, ads, args.whisper_model, args.language, m, not args.no_cache)
    if not args.no_understand:
        phase_understand(con, ads, args.qwen_model, args.frames, m, not args.no_cache, args.vlm_batch)
    if args.neural:
        phase_neural(con, ads, m, not args.no_cache)

    build_and_export_graph(con)
    m.report()
    print(f"\ntotal: {time.time() - t_start:.1f}s")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Descubrimiento y análisis de anuncios (Centauro).")
    src = p.add_argument_group("fuente")
    src.add_argument("--urls", nargs="*", help="URLs de YouTube")
    src.add_argument("--search", help="consulta de búsqueda en YouTube")
    src.add_argument("--channel", help="handle de canal, p.ej. @Telcel")
    src.add_argument("--n", type=int, default=8, help="nº de resultados de search/channel")
    p.add_argument("--workers", type=int, default=4, help="descargas en paralelo")
    p.add_argument("--frames", type=int, default=12, help="frames por anuncio para el VLM")
    p.add_argument("--vlm-batch", type=int, default=2,
                   help="anuncios por forward del VLM (adaptativo: baja solo si hay OOM)")
    p.add_argument("--qwen-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--whisper-model", default="openai/whisper-small")
    p.add_argument("--language", default="spanish", help="idioma de transcripción ('' = auto)")
    p.add_argument("--no-transcribe", action="store_true")
    p.add_argument("--no-understand", action="store_true")
    p.add_argument("--neural", action="store_true", help="perfil neural TRIBE (lento, opt-in)")
    p.add_argument("--no-cache", action="store_true", help="ignora la caché y recalcula")
    p.add_argument("--graph-only", action="store_true", help="reconstruye el grafo desde el store")
    p.add_argument("--cache-stats", action="store_true", help="muestra el tamaño de la caché")
    p.add_argument("--prune-cache", type=float, default=None, metavar="DÍAS",
                   help="borra entradas de caché más antiguas que N días")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
