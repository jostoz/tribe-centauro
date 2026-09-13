"""Utilidades de runtime de GPU compartidas por las fases del pipeline."""

from __future__ import annotations

import gc


def free_gpu() -> None:
    """Libera VRAM y recolecta basura tras descargar un modelo de una fase.

    Obligatorio en Windows single-GPU: entre fases (Whisper -> Qwen -> TRIBE) hay
    que soltar el modelo anterior o la suma de pesos agota VRAM/commit
    (ver skill centauro-gpu-inference).
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001 - nunca debe romper por limpieza
        pass


def vram_report() -> str:
    """One-liner con la VRAM usada/reservada, para logs de fase."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "cuda=no"
        alloc = torch.cuda.memory_allocated() / 1e9
        resv = torch.cuda.memory_reserved() / 1e9
        return f"vram {alloc:.2f}G alloc / {resv:.2f}G reservada"
    except Exception:  # noqa: BLE001
        return "vram n/a"
