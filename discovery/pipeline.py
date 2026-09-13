"""Orquestación: fetch -> transcribe -> understand -> graph [-> neural TRIBE].

Ejemplos:
    # Descubrir y analizar por búsqueda
    .venv/Scripts/python.exe -m discovery.pipeline --search "Telcel anuncio" --n 8

    # Un canal
    .venv/Scripts/python.exe -m discovery.pipeline --channel @Telcel --n 12

    # URLs concretas, sin transcripción, con perfil neural TRIBE
    .venv/Scripts/python.exe -m discovery.pipeline --urls https://youtu.be/XXXX --neural

    # Solo reconstruir el grafo desde registros ya guardados
    .venv/Scripts/python.exe -m discovery.pipeline --graph-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from discovery import fetch, graph  # noqa: E402

OUT = Path("data/discovery")
REC_DIR = OUT / "records"


def _save_record(rec: dict) -> None:
    REC_DIR.mkdir(parents=True, exist_ok=True)
    (REC_DIR / f"{rec['id']}.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_records() -> List[dict]:
    if not REC_DIR.exists():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in REC_DIR.glob("*.json")]


def neural_profile(video_path: str) -> dict:
    """Perfil neural por red funcional (TRIBE v2) para un anuncio."""
    import numpy as np
    import pandas as pd

    from core.tribe_model import resolve_checkpoint_dir
    from service.metrics.roi import RoiIndex
    from tribev2.demo_utils import TribeModel, get_audio_and_text_events

    local_dir = resolve_checkpoint_dir("facebook/tribev2", "./cache")
    model = TribeModel.from_pretrained(local_dir, cache_folder="./cache", device="auto")
    model.data.num_workers = 0  # estabilidad Windows (skill centauro-gpu-inference)
    event = {
        "type": "Video",
        "filepath": str(video_path),
        "start": 0,
        "timeline": "default",
        "subject": "default",
    }
    events = get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
    preds, _ = model.predict(events=events, verbose=False)
    roi = RoiIndex.from_schaefer("data/atlas/schaefer200")
    nets = {
        net: round(float(np.mean(np.abs(ts))), 5)
        for net, ts in roi.all_networks_timeseries(preds).items()
    }
    dominant = max(nets, key=nets.get) if nets else ""
    return {"networks": nets, "red_dominante": dominant, "shape": list(preds.shape)}


def run(args: argparse.Namespace) -> int:
    if args.graph_only:
        records = _load_records()
        if not records:
            print("No hay registros en", REC_DIR)
            return 1
        paths = graph.export(graph.build_graph(records), OUT / "graph")
        print("Grafo:", json.dumps(paths, indent=2))
        return 0

    # 1. Descubrir URLs
    urls: List[str] = list(args.urls or [])
    if args.search:
        urls += fetch.search_urls(args.search, args.n)
    if args.channel:
        urls += fetch.channel_urls(args.channel, args.n)
    if not urls:
        print("Nada que descargar. Usa --urls, --search o --channel.")
        return 1
    print(f"[fetch] {len(urls)} URLs -> descargando...")
    records = fetch.download(urls)
    print(f"[fetch] {len(records)} anuncios descargados.")

    # 2-4. Por anuncio: transcript + understanding (+ neural)
    for i, rec in enumerate(records, 1):
        vid = rec["video_path"]
        if not Path(vid).is_file():
            print(f"  ({i}) sin archivo, se omite: {rec['id']}")
            continue
        tag = f"({i}/{len(records)}) {rec.get('title', rec['id'])[:50]}"
        if not args.no_transcribe:
            print(f"  {tag} :: transcribiendo...")
            from discovery import transcribe

            t0 = time.time()
            rec["transcript"] = transcribe.transcribe(vid)
            print(f"      script en {time.time()-t0:.1f}s "
                  f"({len(rec['transcript'].get('text',''))} chars)")
        if not args.no_understand:
            print(f"  {tag} :: entendiendo (Qwen2.5-VL)...")
            from discovery import understand

            t0 = time.time()
            rec["understanding"] = understand.understand(
                vid, n_frames=args.frames, model_id=args.qwen_model
            )
            print(f"      contenido en {time.time()-t0:.1f}s")
        if args.neural:
            print(f"  {tag} :: perfil neural TRIBE...")
            t0 = time.time()
            try:
                rec["neural"] = neural_profile(vid)
                print(f"      neural en {time.time()-t0:.1f}s "
                      f"(dominante: {rec['neural']['red_dominante']})")
            except Exception as exc:  # noqa: BLE001
                print(f"      neural FALLO: {type(exc).__name__}: {exc}")
        _save_record(rec)

    # 5. Grafo
    all_records = _load_records()
    paths = graph.export(graph.build_graph(all_records), OUT / "graph")
    print(f"\n[graph] {len(all_records)} anuncios -> {json.dumps(paths, indent=2)}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Descubrimiento y análisis de anuncios (Centauro).")
    src = p.add_argument_group("fuente")
    src.add_argument("--urls", nargs="*", help="URLs de YouTube")
    src.add_argument("--search", help="consulta de búsqueda en YouTube")
    src.add_argument("--channel", help="handle de canal, p.ej. @Telcel")
    src.add_argument("--n", type=int, default=8, help="nº de resultados de search/channel")
    p.add_argument("--frames", type=int, default=12, help="frames por anuncio para el VLM")
    p.add_argument("--qwen-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--no-transcribe", action="store_true")
    p.add_argument("--no-understand", action="store_true")
    p.add_argument("--neural", action="store_true", help="añade perfil neural TRIBE (lento)")
    p.add_argument("--graph-only", action="store_true", help="reconstruye el grafo desde registros")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
