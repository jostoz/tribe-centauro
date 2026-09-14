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


def unload() -> None:
    """Suelta el modelo Whisper y libera VRAM (fin de la fase de transcripción)."""
    global _PIPE
    _PIPE = None
    from discovery.gpu import free_gpu

    free_gpu()


def _no_speech_prob(pipe, audio: np.ndarray) -> float:
    """Probabilidad de "sin voz" según Whisper (token <|nospeech|> del 1er paso).

    Es el mecanismo estándar de Whisper para esto y reutiliza el modelo ya cargado:
    evita depender de un VAD externo (silero-vad falla en este Windows por DLL).
    """
    import torch

    feats = pipe.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features
    dtype = next(pipe.model.parameters()).dtype
    feats = feats.to(pipe.device, dtype=dtype)
    with torch.inference_mode():
        out = pipe.model.generate(
            feats, return_dict_in_generate=True, output_scores=True, max_new_tokens=4
        )
    probs = torch.softmax(out.scores[0][0].float(), dim=-1)
    nid = pipe.tokenizer.convert_tokens_to_ids("<|nospeech|>")
    return float(probs[nid])


def transcribe(
    video_path: str | Path,
    model: str = "openai/whisper-small",
    language: Optional[str] = "spanish",
) -> dict:
    """Devuelve ``{"text", "segments", "has_speech", "no_speech_prob"}``.

    ``has_speech`` se decide por ``no_speech_prob`` de Whisper, **no** por si hay texto:
    Whisper transcribe/alucina texto sobre música, así que "hay texto" no implica "hay voz".
    """
    audio = extract_audio(video_path)
    if audio.size == 0 or float(np.max(np.abs(audio))) < 1e-3:
        return {"text": "", "segments": [], "has_speech": False, "no_speech_prob": 1.0}
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
    p = _no_speech_prob(pipe, audio)
    return {
        "text": text,
        "segments": segments,
        "has_speech": bool(p < 0.5),
        "no_speech_prob": round(p, 4),
    }
