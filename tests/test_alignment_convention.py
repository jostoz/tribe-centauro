"""Contrato de alineación temporal: ``preds[k]`` es la respuesta al segundo ``k``.

Estos tests fijan la semántica de la que depende la convención —el objetivo de
entrenamiento del checkpoint— para que un cambio aguas arriba en ``neuralset`` o en el
``config.yaml`` publicado rompa el build en lugar de corromper en silencio la
interpretación temporal de cada respuesta.

La verificación **empírica** (que el modelo efectivamente aprendió esa convención sobre
anuncios reales) vive en ``scripts/validate_alignment.py`` y sus números en
``docs/FASE_0.5_ALINEACION.md``. Requiere GPU y no corre en CI.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from core.ordering import ALIGNMENT_CONVENTION, HEMODYNAMIC_OFFSET_SECONDS
from core.tribe_model import portable_config_text

CHECKPOINT_CONFIG = Path("cache/checkpoint/facebook__tribev2/config.yaml")


def test_offset_declared_in_the_released_checkpoint():
    """El checkpoint publicado declara ``offset: 5`` en el extractor de fMRI."""
    if not CHECKPOINT_CONFIG.is_file():
        pytest.skip("checkpoint no descargado")

    text = portable_config_text(CHECKPOINT_CONFIG.read_text(encoding="utf-8"), "cache")
    config = yaml.load(text, Loader=yaml.UnsafeLoader)
    neuro = config["data"]["neuro"]

    assert neuro["name"] == "FmriExtractor"
    assert neuro["offset"] == HEMODYNAMIC_OFFSET_SECONDS == 5.0
    assert neuro["frequency"] == 1.0  # 1 TR = 1 s


def test_training_target_reads_bold_shifted_by_offset(monkeypatch):
    """La ventana de estímulo ``[s, s+D]`` se aparea con BOLD crudo ``[s+5, s+5+D)``.

    Es la razón de que el índice de ``preds`` ya esté alineado al estímulo: el modelo
    aprende ``estímulo(t) -> BOLD(t+5)``, y el BOLD en ``t+5`` es la respuesta al
    estímulo del segundo ``t`` (la HRF pica ~5 s después del evento neural).
    """
    from neuralset.extractors.neuro import FmriExtractor, TimedArray

    offset = int(HEMODYNAMIC_OFFSET_SECONDS)
    raw = np.arange(60, dtype=np.float32).reshape(1, 60)  # BOLD crudo: muestra i = i
    bold = TimedArray(data=raw, frequency=1, start=0, duration=60)

    def stub(self, events):
        for _ in events:
            yield bold

    monkeypatch.setattr(FmriExtractor, "_get_data", stub)
    extractor = FmriExtractor(offset=HEMODYNAMIC_OFFSET_SECONDS, frequency=1)

    for start in (0, 7):
        target = next(
            iter(extractor._get_timed_arrays([object()], start=start, duration=10))
        )
        window = target.overlap(start=start, duration=4)
        assert window.data.ravel().tolist() == list(
            range(start + offset, start + offset + 4)
        )


def test_convention_is_not_unverified():
    """El valor publicado dejó de ser ``unverified``: la fase está cerrada."""
    assert ALIGNMENT_CONVENTION == "stimulus-aligned"
