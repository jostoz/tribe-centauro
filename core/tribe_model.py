"""
Wrapper de inferencia de TRIBE v2 — reparado contra la API real de Meta.

Qué estaba roto en la versión anterior
--------------------------------------
1. ``get_events_dataframe(video_path=..., audio_path=..., text=...)``: la API acepta
   **exactamente una** fuente y lanza ``ValueError`` con cero o más de una.
   Además el kwarg correcto es ``text_path`` (archivo ``.txt``), no ``text``.
2. ``self.model.to(device)``: ``TribeModel`` ya devuelve el modelo en device y en
   ``eval()`` desde ``from_pretrained``. Además ``TribeModel`` no es ``nn.Module``.
3. ``predict_multimodal`` era conceptualmente imposible: el video ya incluye su audio
   (Meta lo extrae del propio track), y no hay forma de aportar tres modalidades.

Contrato real (verificado en ``tribev2/demo_utils.py``)
-------------------------------------------------------
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=..., device="auto")
    df = model.get_events_dataframe(video_path=Path("clip.mp4"))   # exactamente una fuente
    preds, segments = model.predict(events=df)
    # preds.shape == (n_timesteps, 20484) ; 1 TR = 1 s ; fsaverage5
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from core.ordering import (
    ALIGNMENT_CONVENTION,
    FSAAVERAGE5_VERTICES,
    HEMODYNAMIC_OFFSET_SECONDS,
    TR_SECONDS,
)
from service.errors import (
    HfAccessDenied,
    InvalidInput,
    MediaUnsupported,
    ModelUnavailable,
    TextToSpeechUnavailable,
)

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg"}
TEXT_SUFFIXES = {".txt"}


class TribePredictor:
    """Inferencia TRIBE v2 sobre video, audio o texto (una fuente por llamada)."""

    def __init__(
        self,
        checkpoint_dir: str = "facebook/tribev2",
        cache_folder: str = "./cache",
        device: str = "auto",
        model_version: Optional[str] = None,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.cache_folder = cache_folder
        self.device = device
        self.model_version = model_version or checkpoint_dir
        self._model: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Carga el checkpoint. El device se resuelve AQUÍ, no con ``.to()`` después."""
        if self._model is not None:
            return
        try:
            from tribev2.demo_utils import TribeModel
        except ImportError as exc:
            raise ModelUnavailable(
                "tribev2 no está instalado. Instalar desde git: "
                'uv pip install "tribev2 @ git+https://github.com/facebookresearch/tribev2.git"'
            ) from exc

        local_dir = resolve_checkpoint_dir(self.checkpoint_dir, self.cache_folder)
        logger.info("Cargando TRIBE v2 desde %s en %s", local_dir, self.device)
        try:
            self._model = TribeModel.from_pretrained(
                local_dir,
                cache_folder=self.cache_folder,
                device=self.device,
            )
        except Exception as exc:  # noqa: BLE001 - se mapea a taxonomía
            raise _map_load_error(exc) from exc
        logger.info("TRIBE v2 cargado")

    def predict(
        self,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        text: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Ejecuta la inferencia sobre EXACTAMENTE una fuente.

        Returns
        -------
        (predictions, metadata)
            ``predictions`` tiene forma ``(n_timesteps, 20484)`` y está en unidades
            crudas del modelo. ``metadata`` declara la procedencia y las convenciones
            temporales.
        """
        self.load()

        provided = {
            "video_path": video_path,
            "audio_path": audio_path,
            "text": text,
        }
        active = [k for k, v in provided.items() if v is not None]
        if len(active) != 1:
            raise InvalidInput(
                f"Se requiere exactamente una fuente (video_path, audio_path o text); "
                f"recibidas: {active or 'ninguna'}",
                hint=(
                    "El video ya incluye su pista de audio. Para combinar narrativa y "
                    "locución, usa un único video."
                ),
            )

        source = active[0]
        resolver = {
            "video_path": self._resolve_video,
            "audio_path": self._resolve_audio,
            "text": self._resolve_text,
        }[source]
        kwargs, cleanup = resolver(provided[source])

        try:
            df = self._model.get_events_dataframe(**kwargs)
            preds, segments = self._model.predict(events=df)
        except Exception as exc:  # noqa: BLE001
            raise _map_predict_error(exc) from exc
        finally:
            cleanup()

        preds = np.asarray(preds)
        self._validate_shape(preds)

        metadata = {
            "source": source,
            "n_timesteps": int(preds.shape[0]),
            "n_vertices": int(preds.shape[1]),
            "n_segments": len(segments) if segments is not None else None,
            "segments": _segments_summary(segments),
            "tr_seconds": TR_SECONDS,
            "hemodynamic_offset_seconds": HEMODYNAMIC_OFFSET_SECONDS,
            "alignment": ALIGNMENT_CONVENTION,
            "mesh": "fsaverage5",
            "model_version": self.model_version,
            "device": self.device,
        }
        logger.info(
            "Predicción lista: %s en %s", preds.shape, self.device
        )
        return preds, metadata

    # ------------------------------------------------------------------ fuentes

    def _resolve_video(self, path: str):
        p = _require_file(path, VIDEO_SUFFIXES, "video")
        return {"video_path": p}, _noop

    def _resolve_audio(self, path: str):
        p = _require_file(path, AUDIO_SUFFIXES, "audio")
        return {"audio_path": p}, _noop

    def _resolve_text(self, text: str):
        """El modelo requiere un archivo ``.txt``; convierte texto a voz vía gTTS."""
        if not text or not text.strip():
            raise InvalidInput("El texto está vacío")

        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        )
        tmp.write(text)
        tmp.close()
        return {"text_path": Path(tmp.name)}, _make_cleanup(tmp.name)

    @staticmethod
    def _validate_shape(preds: np.ndarray) -> None:
        if preds.ndim != 2:
            raise ModelUnavailable(
                f"Se esperaba (n_timesteps, n_vertices), el modelo devolvió {preds.shape}"
            )
        if preds.shape[1] != FSAAVERAGE5_VERTICES:
            raise ModelUnavailable(
                f"Se esperaban {FSAAVERAGE5_VERTICES} vértices (fsaverage5), "
                f"el modelo devolvió {preds.shape[1]}. "
                "Las máscaras de ROI construidas sobre fsaverage5 no aplicarían."
            )


_POSIXPATH_TAG = "!!python/object/apply:pathlib.PosixPath"


def portable_config_text(text: str, local_cache_root: str) -> str:
    """Reescribe un ``config.yaml`` de TRIBE v2 para que sea cargable en Windows.

    Dos problemas del checkpoint oficial:

    1. Se serializó en Linux con ``yaml.UnsafeLoader`` e incluye objetos
       ``pathlib.PosixPath``, que Windows no puede instanciar::

           NotImplementedError: cannot instantiate 'PosixPath' on your system

    2. Contiene rutas absolutas de los servidores de Meta
       (``/checkpoint/sdascoli/...``) que nunca existen en otra máquina.

    Cada ``PosixPath`` se sustituye por una cadena, y las rutas absolutas se
    reubican bajo ``local_cache_root`` para que la caché se escriba dentro del
    proyecto en lugar de crear directorios en la raíz de la unidad.
    """
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _POSIXPATH_TAG in line:
            prefix = line.split(_POSIXPATH_TAG)[0]  # p. ej. "                    folder: "
            parts: list[str] = []
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("- "):
                parts.append(lines[j].strip()[2:].strip())
                j += 1

            # Semántica de PosixPath('/', 'checkpoint', ...) -> '/checkpoint/...'
            components = [p for p in parts if p not in ("/", "")]
            posix_path = "/" + "/".join(components)

            if posix_path.startswith("/"):
                local = f"{local_cache_root.rstrip('/')}/{posix_path.lstrip('/')}"
            else:
                local = posix_path

            out.append(f'{prefix}"{local}"')
            i = j
            continue

        if line.strip().startswith("folder: /"):
            indent, _, value = line.partition("folder:")
            value = value.strip()
            local = f"{local_cache_root.rstrip('/')}/{value.lstrip('/')}"
            out.append(f'{indent}folder: "{local}"')
            i += 1
            continue

        out.append(line)
        i += 1

    return "\n".join(out) + "\n"


def _make_config_portable(config_path: Path, cache_folder: str) -> None:
    """Aplica :func:`portable_config_text` en sitio al config descargado."""
    local_root = (Path(cache_folder) / "meta_checkpoint_cache").as_posix()
    original = config_path.read_text(encoding="utf-8")
    rewritten = portable_config_text(original, local_root)
    if rewritten != original:
        config_path.write_text(rewritten, encoding="utf-8")
        logger.info("config.yaml reescrito para portabilidad Windows")


def resolve_checkpoint_dir(checkpoint_dir: str, cache_folder: str) -> Path:
    """Devuelve un directorio local con ``config.yaml`` y ``best.ckpt``.

    Workaround de un bug de tribev2 en Windows: ``from_pretrained`` hace
    ``Path(checkpoint_dir)`` y luego ``str()`` para obtener el repo id de
    HuggingFace. En Windows eso convierte ``"facebook/tribev2"`` en
    ``"facebook\\tribev2"``, que HF rechaza con ``HFValidationError``.

    Por eso descargamos el checkpoint explícitamente y pasamos una ruta local.
    """
    path = Path(checkpoint_dir)
    if path.exists() and (path / "config.yaml").exists():
        return path

    from huggingface_hub import snapshot_download

    # Se usa el repo id tal cual (con '/'), nunca a través de Path.
    repo_id = checkpoint_dir.replace("\\", "/")
    target = Path(cache_folder) / "checkpoint" / repo_id.replace("/", "__")
    logger.info("Descargando checkpoint %s -> %s", repo_id, target)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            allow_patterns=["config.yaml", "best.ckpt"],
        )
    except Exception as exc:  # noqa: BLE001
        raise _map_load_error(exc) from exc

    if not (target / "config.yaml").exists():
        raise ModelUnavailable(f"El checkpoint descargado no tiene config.yaml: {target}")

    _make_config_portable(target / "config.yaml", cache_folder)
    return target


def _require_file(path: str, allowed: set, kind: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise InvalidInput(f"Archivo de {kind} no encontrado: {p}")
    if p.suffix.lower() not in allowed:
        raise MediaUnsupported(
            f"Extensión {p.suffix!r} no soportada para {kind}. Soportadas: "
            f"{sorted(allowed)}"
        )
    return p


def _segments_summary(segments) -> Optional[list]:
    """Resumen de segmentos.

    Relevante porque TRIBE v2 trocea estímulos con ``ChunkEvents(max_duration=60,
    min_duration=30)``: un anuncio corto (6-15 s) puede caer en una sola pieza o
    comportarse de forma no documentada. ``n_segments`` lo hace observable en vez
    de silencioso.
    """
    if segments is None:
        return None
    summary = []
    for seg in list(segments)[:50]:
        if isinstance(seg, dict):
            summary.append(
                {
                    "start": seg.get("start"),
                    "duration": seg.get("duration"),
                    "types": seg.get("type"),
                }
            )
        else:
            summary.append({"repr": str(seg)[:120]})
    return summary


def _map_load_error(exc: Exception) -> Exception:
    text = str(exc).lower()
    if "gated" in text or "401" in text or "403" in text or "access" in text:
        return HfAccessDenied(str(exc))
    if "gtts" in text or "connection" in text or "langdetect" in text:
        return TextToSpeechUnavailable(str(exc))
    return ModelUnavailable(f"No se pudo cargar TRIBE v2: {exc}")


def _map_predict_error(exc: Exception) -> Exception:
    text = str(exc).lower()
    if isinstance(exc, ValueError) and "exactly one" in text:
        return InvalidInput(str(exc))
    if "cuda" in text and "memory" in text:
        from service.errors import GpuOom

        return GpuOom(str(exc))
    if "gtts" in text or "langdetect" in text or "connection" in text:
        return TextToSpeechUnavailable(str(exc))
    if "gated" in text or "401" in text:
        return HfAccessDenied(str(exc))
    return ModelUnavailable(f"Fallo en inferencia: {exc}")


def _noop() -> None:
    return None


def _make_cleanup(path: str):
    def _cleanup() -> None:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:  # pragma: no cover - mejor esfuerzo
            pass

    return _cleanup
