"""Caché content-addressed e idempotencia por etapa.

Clave = sha256(bytes del video ‖ etapa ‖ parámetros del modelo). Si el resultado de
una etapa ya existe, se reutiliza: re-correr el pipeline no repite cómputo ni
descargas (contrato de idempotencia del futuro `analyze_creative` del MCP).

Ubicación: ``<raíz>/<etapa>/<clave>.json`` (raíz por defecto ``data/discovery/cache``,
configurable con la env ``CENTAURO_CACHE``).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_FILE_HASH: Dict[Tuple[str, int, int], str] = {}


def root() -> Path:
    return Path(os.environ.get("CENTAURO_CACHE", "data/discovery/cache"))


def file_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    """sha256 de los bytes del archivo, memoizado por (ruta, mtime, tamaño)."""
    p = Path(path)
    st = p.stat()
    memo = (str(p.resolve()), int(st.st_mtime), int(st.st_size))
    if memo in _FILE_HASH:
        return _FILE_HASH[memo]
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    digest = h.hexdigest()
    _FILE_HASH[memo] = digest
    return digest


def content_key(video_path: str | Path, stage: str, params: str = "") -> str:
    """Clave estable de una etapa para un video concreto."""
    h = hashlib.sha256()
    h.update(file_hash(video_path).encode())
    h.update(b"\x00")
    h.update(stage.encode())
    h.update(b"\x00")
    h.update(params.encode())
    return h.hexdigest()[:32]


def _entry(stage: str, key: str) -> Path:
    return root() / stage / f"{key}.json"


def get(stage: str, key: str) -> Optional[Any]:
    p = _entry(stage, key)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None  # entrada corrupta -> se recalcula


def put(stage: str, key: str, value: Any) -> None:
    p = _entry(stage, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)  # atómico: nunca queda un JSON a medias


def stats() -> Dict[str, int]:
    """Nº de entradas y bytes por etapa."""
    r = root()
    out: Dict[str, int] = {}
    if not r.exists():
        return out
    for stage_dir in r.iterdir():
        if stage_dir.is_dir():
            files = list(stage_dir.glob("*.json"))
            out[stage_dir.name] = sum(f.stat().st_size for f in files)
    return out


def prune(older_than_days: float) -> int:
    """Borra entradas de caché más antiguas que N días. Devuelve nº de archivos."""
    r = root()
    if not r.exists():
        return 0
    cutoff = time.time() - older_than_days * 86400
    removed = 0
    for stage_dir in r.iterdir():
        if not stage_dir.is_dir():
            continue
        for f in stage_dir.glob("*.json"):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
                removed += 1
    return removed


def clear() -> None:
    r = root()
    if r.exists():
        shutil.rmtree(r)
