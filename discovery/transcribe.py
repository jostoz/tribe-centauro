"""Transcripción (el 'script' del anuncio) con Whisper vía transformers, en GPU.

Evita faster-whisper/ctranslate2 (frágil en Windows). Extrae el audio con el
ffmpeg empaquetado por imageio-ffmpeg (no requiere ffmpeg en PATH).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import imageio_ffmpeg
import numpy as np
import soundfile as sf

_PIPE = None


def extract_audio(video_path: str | Path, sr: int = 16000) -> np.ndarray:
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    try:
        subprocess.run(
            [ff, "-y", "-i", str(video_path), "-ac", "1", "-ar", str(sr), "-vn", str(tmp)],
            check=True,
            capture_output=True,
        )
        audio, _ = sf.read(str(tmp), dtype="float32")
    finally:
        tmp.unlink(missing_ok=True)
    return audio


def _pipe(model: str = "openai/whisper-small"):
    global _PIPE
    if _PIPE is None:
        import torch
        from transformers import pipeline

        use_cuda = torch.cuda.is_available()
        _PIPE = pipeline(
            "automatic-speech-recognition",
            model=model,
            device=0 if use_cuda else -1,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
        )
    return _PIPE


def transcribe(
    video_path: str | Path,
    model: str = "openai/whisper-small",
    language: Optional[str] = "spanish",
) -> dict:
    """Devuelve ``{"text": str, "segments": [{start, end, text}], "has_speech": bool}``."""
    audio = extract_audio(video_path)
    if audio.size == 0 or float(np.max(np.abs(audio))) < 1e-3:
        return {"text": "", "segments": [], "has_speech": False}
    pipe = _pipe(model)
    gen_kwargs = {"task": "transcribe"}
    if language:
        gen_kwargs["language"] = language
    out = pipe(
        audio,
        return_timestamps=True,
        chunk_length_s=30,
        batch_size=8,
        generate_kwargs=gen_kwargs,
    )
    segments = [
        {
            "start": (c["timestamp"] or [None, None])[0],
            "end": (c["timestamp"] or [None, None])[1],
            "text": c["text"].strip(),
        }
        for c in out.get("chunks", [])
    ]
    text = out["text"].strip()
    return {"text": text, "segments": segments, "has_speech": bool(text)}
