"""
Serialización segura a JSON.

Motivo: los payloads anteriores contenían ``np.int64``, ``np.bool_`` y ``np.float64``,
que no son serializables por ``json`` y rompen el transporte JSON-RPC de MCP.
Además, NaN/Infinity no son JSON válido.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

MAX_SEQUENCE_LENGTH = 2048


def to_jsonable(obj: Any) -> Any:
    """Convierte recursivamente a tipos nativos JSON.

    Los vectores se truncan a ``MAX_SEQUENCE_LENGTH`` para que un agente no reciba
    20484 valores por timestep en su contexto. Para datos completos se expone un
    recurso aparte, no el payload de la herramienta.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, (int, np.integer)):
        return int(obj)

    if isinstance(obj, (float, np.floating)):
        return _finite(float(obj))

    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        items = [to_jsonable(v) for v in obj]
        if len(items) > MAX_SEQUENCE_LENGTH:
            return items[:MAX_SEQUENCE_LENGTH]
        return items

    if hasattr(obj, "as_dict") and callable(obj.as_dict):
        return to_jsonable(obj.as_dict())

    return str(obj)


def downsample(series: Any, max_points: int = 120) -> list:
    """Reduce una serie a ``max_points`` por media de bloques.

    Un agente no necesita 1 punto por segundo durante 120 s en su contexto; necesita
    la forma de la curva. Los datos completos viven en un recurso.
    """
    arr = np.asarray(series, dtype=np.float64).ravel()
    if arr.size <= max_points:
        return to_jsonable(arr)

    n_blocks = max_points
    block_size = arr.size // n_blocks
    trimmed = arr[: n_blocks * block_size]
    reduced = trimmed.reshape(n_blocks, block_size).mean(axis=1)
    return to_jsonable(reduced)


def _finite(value: float) -> Any:
    if math.isfinite(value):
        return value
    # NaN/Inf no son JSON válido: se degradan a null explícito.
    return None
