"""Tests del contrato de fuentes del wrapper TRIBE v2.

No requieren el modelo real: inyectan un doble para verificar el despacho de fuentes,
que es donde el código anterior fallaba en el 100% de las llamadas.
"""

from pathlib import Path

import numpy as np
import pytest

from core.ordering import FSAAVERAGE5_VERTICES
from core.tribe_model import TribePredictor
from service.errors import InvalidInput, MediaUnsupported, ModelUnavailable


class FakeTribeModel:
    """Doble que registra los kwargs recibidos por get_events_dataframe."""

    def __init__(self, n_timesteps=60, n_vertices=FSAAVERAGE5_VERTICES):
        self.n_timesteps = n_timesteps
        self.n_vertices = n_vertices
        self.events_kwargs = None
        self.text_path_existed = None

    def get_events_dataframe(self, **kwargs):
        self.events_kwargs = kwargs
        # Verifica que el .txt temporal existe en el momento de la llamada.
        if "text_path" in kwargs:
            self.text_path_existed = Path(kwargs["text_path"]).exists()
        return {"events": kwargs}

    def predict(self, events=None):
        rng = np.random.default_rng(0)
        preds = rng.random((self.n_timesteps, self.n_vertices))
        segments = [{"type": "Video", "start": 0, "duration": self.n_timesteps}]
        return preds, segments


@pytest.fixture
def predictor():
    p = TribePredictor(checkpoint_dir="fake", cache_folder=".cache_test")
    p._model = FakeTribeModel()
    return p


@pytest.fixture
def video_file(tmp_path):
    f = tmp_path / "ad.mp4"
    f.write_bytes(b"\x00")
    return str(f)


@pytest.fixture
def audio_file(tmp_path):
    f = tmp_path / "ad.wav"
    f.write_bytes(b"\x00")
    return str(f)


def test_video_source_passes_only_video_path(predictor, video_file):
    preds, meta = predictor.predict(video_path=video_file)

    assert "video_path" in predictor._model.events_kwargs
    assert "audio_path" not in predictor._model.events_kwargs
    assert "text_path" not in predictor._model.events_kwargs
    assert preds.shape == (60, FSAAVERAGE5_VERTICES)
    assert meta["source"] == "video_path"


def test_audio_source_passes_only_audio_path(predictor, audio_file):
    _, meta = predictor.predict(audio_path=audio_file)

    assert "audio_path" in predictor._model.events_kwargs
    assert "video_path" not in predictor._model.events_kwargs
    assert meta["source"] == "audio_path"


def test_text_source_writes_a_txt_file_and_cleans_up(predictor):
    _, meta = predictor.predict(text="Compra ahora. Oferta limitada.")

    kwargs = predictor._model.events_kwargs
    assert "text_path" in kwargs, "el modelo requiere text_path, no text"
    assert Path(kwargs["text_path"]).suffix == ".txt"
    assert predictor._model.text_path_existed is True

    # El temporal se elimina tras la llamada.
    assert not Path(kwargs["text_path"]).exists()
    assert meta["source"] == "text"


def test_metadata_declares_temporal_conventions(predictor, video_file):
    _, meta = predictor.predict(video_path=video_file)

    assert meta["tr_seconds"] == 1.0
    assert meta["hemodynamic_offset_seconds"] == 5.0
    assert meta["mesh"] == "fsaverage5"
    assert meta["alignment"] == "unverified"


def test_zero_sources_is_rejected(predictor):
    with pytest.raises(InvalidInput, match="exactamente una fuente"):
        predictor.predict()


def test_multiple_sources_are_rejected(predictor, video_file, audio_file):
    """El código anterior enviaba hasta tres fuentes; la API acepta exactamente una."""
    with pytest.raises(InvalidInput, match="exactamente una fuente"):
        predictor.predict(video_path=video_file, audio_path=audio_file, text="hola")


def test_wrong_extension_is_rejected(predictor, tmp_path):
    bad = tmp_path / "ad.avi".replace("ad.avi", "ad.xyz")
    bad.write_bytes(b"\x00")
    with pytest.raises(MediaUnsupported, match="no soportada"):
        predictor.predict(video_path=str(bad))


def test_missing_file_is_rejected(predictor):
    with pytest.raises(InvalidInput, match="no encontrado"):
        predictor.predict(video_path="no/existe.mp4")


def test_empty_text_is_rejected(predictor):
    with pytest.raises(InvalidInput, match="vacío"):
        predictor.predict(text="   ")


def test_wrong_vertex_count_is_surfaced(tmp_path):
    p = TribePredictor(checkpoint_dir="fake")
    p._model = FakeTribeModel(n_vertices=5000)
    f = tmp_path / "ad.mp4"
    f.write_bytes(b"\x00")

    with pytest.raises(ModelUnavailable, match="fsaverage5"):
        p.predict(video_path=str(f))
