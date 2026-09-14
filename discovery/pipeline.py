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
from typing import Dict, List, Optional

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
    """Única vía de escritura de un anuncio al store.

    Persiste metadata + entidades y, si el registro trae métricas públicas (viene así de
    la descarga), también las stats. Al centralizarlo aquí, ningún camino de escritura
    (fetch, transcribe, understand, neural) puede olvidarse del enriquecimiento.
    """
    u = rec.get("understanding")
    ents = entities.extract_entities(u) if u else None
    store.upsert_ad(con, rec, ents)
    if rec.get("view_count") is not None or rec.get("like_count") is not None:
        store.upsert_stats(con, rec)


def _video_ok(a: dict) -> bool:
    vp = a.get("video_path")
    return bool(vp) and Path(vp).is_file()


# ---------------------------------------------------------------------------
# Fase 0 — fetch (CPU/red, en paralelo)
# ---------------------------------------------------------------------------
def phase_fetch(con, urls: List[str], workers: int, corpus: Optional[str] = None) -> List[dict]:
    existing = {a["url"]: a for a in store.all_ads(con) if a.get("url")}
    to_download = [u for u in urls if not (u in existing and _video_ok(existing[u]))]
    skipped = len(urls) - len(to_download)
    if skipped:
        print(f"[fetch] {skipped} ya descargados (resume)")
    if to_download:
        print(f"[fetch] descargando {len(to_download)} en paralelo (workers={workers})...")
        for rec in fetch.download_many(to_download, workers=workers):
            if corpus:
                rec["corpus"] = corpus
            _upsert(con, rec)  # metadata + stats públicas (vienen en el propio download)
    ads = [a for a in store.all_ads(con) if a.get("url") in set(urls)]
    return [a for a in ads if _video_ok(a)]


# ---------------------------------------------------------------------------
# Fase 0b — métricas públicas (red, sin GPU)
# ---------------------------------------------------------------------------
STATS_TTL_HOURS = 24.0


def phase_stats(con, ads: List[dict], workers: int, force: bool, use_cache: bool) -> None:
    """Refresca vistas/likes/comentarios públicos de los anuncios.

    Es red-only y barato (no toca la GPU). Se refresca si ``stats_updated_at`` supera el
    TTL, si no hay dato, o si se fuerza con ``--refresh-stats``. Las derivadas
    (vistas/día, tasas) se calculan al leer con ``store.stats_table()``.
    """
    from datetime import datetime, timezone

    todo = []
    for a in ads:
        if not a.get("url"):
            continue
        if force or not use_cache:
            todo.append(a)
            continue
        age = None
        if a.get("stats_updated_at"):
            try:
                ts = datetime.fromisoformat(a["stats_updated_at"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
            except ValueError:
                age = None
        if age is None or age >= STATS_TTL_HOURS:
            todo.append(a)
    if not todo:
        print(f"[stats] al día ({len(ads)} anuncios, TTL {STATS_TTL_HOURS:.0f} h)")
        return
    print(f"[stats] refrescando métricas públicas de {len(todo)}...")
    recs = fetch.fetch_stats([a["url"] for a in todo], workers=workers)
    for r in recs:
        store.upsert_stats(con, r)
    print(f"[stats] {len(recs)} actualizados")


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
        key = cache.content_key(a["video_path"], "transcribe", f"{model}|lang={language}|v2-nospeech")
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
    """Fase neural en DOS pasadas, con un solo load del modelo.

    A) **features**: codifica V-JEPA/audio de TODOS los anuncios (la parte cara,
       minutos por anuncio) y las deja cacheadas.
    B) **forwards**: ejecuta el forward de TRIBE para todos (barato con features
       en caché; ~1-2 s por anuncio).

    Así la GPU se dedica a la tarea pesada de forma contigua en vez de alternar
    encode→forward→encode, y los forwards quedan repetibles sin re-codificar.
    """
    pend = [a for a in ads if not use_cache or not a.get("neural")]
    if not pend:
        print("[neural] ya completo (resume)")
        return
    print(f"[neural] {len(pend)} anuncios (TRIBE residente, 2 pasadas)...")
    from discovery.neural import NeuralAnalyzer

    an = NeuralAnalyzer()
    an.load()

    # --- Pasada A: features (V-JEPA/audio) de todos ---
    todo = []
    for a in pend:
        fkey = cache.content_key(a["video_path"], "features", "tribev2|vjepa")
        t0 = time.time()
        if use_cache and cache.get("features", fkey) is not None:
            m.add("features", a["id"], 0.0, True)
        else:
            try:
                n = an.extract_features(a["video_path"])
                cache.put("features", fkey, {"batches": n})
                m.add("features", a["id"], time.time() - t0, False)
            except Exception as exc:  # noqa: BLE001
                print(f"      [neural] {a['id']} features FALLO: {type(exc).__name__}: {exc}")
                continue
        todo.append(a)

    # --- Pasada B: forwards (features ya en caché) ---
    for a in todo:
        t0 = time.time()
        nkey = cache.content_key(a["video_path"], "neural", "tribev2|default")
        cached = None if not use_cache else cache.get("neural", nkey)
        if cached is not None:
            a["neural"] = cached
            m.add("neural", a["id"], time.time() - t0, True)
            _upsert(con, a)
            continue
        try:
            a["neural"] = an.analyze(a["video_path"])
            cache.put("neural", nkey, a["neural"])
            _upsert(con, a)
            m.add("neural", a["id"], time.time() - t0, False)
        except Exception as exc:  # noqa: BLE001 - un anuncio roto no debe tumbar la fase
            print(f"      [neural] {a['id']} FALLO: {type(exc).__name__}: {exc}")
    an.unload()
    print("      [neural] modelo descargado")


def build_and_export_graph(con, corpus: Optional[str] = None) -> dict:
    records = store.all_ads(con, corpus)
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
        build_and_export_graph(con, args.corpus)
        return 0

    if args.stats_only:
        ads_all = [a for a in store.all_ads(con, args.corpus) if a.get("url")]
        phase_stats(con, ads_all, args.workers, args.refresh_stats, not args.no_cache)
        return 0

    if args.all:
        ads = [a for a in store.all_ads(con, args.corpus) if _video_ok(a)]
        print(f"[store] {len(ads)} anuncios con video disponible" + (f" (corpus={args.corpus})" if args.corpus else ""))
        if not ads:
            return 1
    else:
        urls: List[str] = list(args.urls or [])
        if args.search:
            urls += fetch.search_urls(args.search, args.n)
        if args.channel:
            urls += fetch.channel_urls(args.channel, args.n)
        if not urls:
            print("Nada que hacer. Usa --urls, --search, --channel o --all.")
            return 1
        ads = phase_fetch(con, urls, args.workers, args.corpus)
        print(f"[fetch] {len(ads)} anuncios con video disponible")
        if not ads:
            return 1

    m = Metrics()
    t_start = time.time()

    if not args.no_stats:
        phase_stats(con, ads, args.workers, args.refresh_stats, not args.no_cache)

    if not args.no_transcribe:
        phase_transcribe(con, ads, args.whisper_model, args.language, m, not args.no_cache)
    if not args.no_understand:
        phase_understand(con, ads, args.qwen_model, args.frames, m, not args.no_cache, args.vlm_batch)
    if args.neural:
        phase_neural(con, ads, m, not args.no_cache)

    build_and_export_graph(con, args.corpus)
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
    p.add_argument("--all", action="store_true",
                   help="procesa TODOS los anuncios ya presentes en el store (sin URLs)")
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
    p.add_argument("--no-stats", action="store_true", help="no refrescar métricas públicas")
    p.add_argument("--refresh-stats", action="store_true",
                   help="forzar refresco de métricas públicas (ignora el TTL de 24 h)")
    p.add_argument("--stats-only", action="store_true",
                   help="solo refresca métricas públicas y sale (red, sin GPU)")
    p.add_argument("--corpus", help="etiqueta del conjunto (p. ej. telcel, cocacola) para separar corpora")
    p.add_argument("--graph-only", action="store_true", help="reconstruye el grafo desde el store")
    p.add_argument("--cache-stats", action="store_true", help="muestra el tamaño de la caché")
    p.add_argument("--prune-cache", type=float, default=None, metavar="DÍAS",
                   help="borra entradas de caché más antiguas que N días")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
