"""Video understanding con Qwen2.5-VL (local, GPU).

Extrae frames con moviepy (evita depender de decord/av) y pide al VLM un análisis
estructurado en JSON: escenas, objetos, personajes, elementos de marca, ritmo,
tono emocional y temas — la base de nodos del grafo de contenidos.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT = (
    "Eres un analista de publicidad. Analiza este anuncio (fotogramas en orden "
    "temporal) y responde SOLO con JSON válido, sin texto adicional, con esta forma:\n"
    '{"resumen": "una frase", '
    '"escenas": [{"orden": 1, "descripcion": "..."}], '
    '"objetos": ["..."], '
    '"personajes": ["..."], '
    '"marca_elementos": ["logo, colores, producto..."], '
    '"hay_caras": true, '
    '"hay_texto_en_pantalla": true, '
    '"ritmo": "lento|medio|rapido", '
    '"tono_emocional": "una o dos palabras", '
    '"temas": ["..."]}\n'
    "Usa español. No inventes marcas que no veas."
)


@lru_cache(maxsize=1)
def _load(model_id: str = MODEL_ID):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def unload() -> None:
    """Suelta el VLM y libera VRAM (fin de la fase de comprensión)."""
    _load.cache_clear()
    from discovery.gpu import free_gpu

    free_gpu()


def extract_frames(video_path: str | Path, n: int = 8, max_dim: int = 448) -> List[str]:
    from moviepy import VideoFileClip
    from PIL import Image

    clip = VideoFileClip(str(video_path))
    try:
        dur = float(clip.duration or 0.0)
        ts = np.linspace(0, max(dur - 0.1, 0.0), n)
        paths: List[str] = []
        for i, t in enumerate(ts):
            frame = clip.get_frame(float(t))
            img = Image.fromarray(frame.astype("uint8"))
            # redimensionar (lado mayor <= max_dim) para acotar el coste de tokens del VLM
            img.thumbnail((max_dim, max_dim), Image.BILINEAR)
            p = Path(tempfile.gettempdir()) / f"qwenframe_{os.getpid()}_{i}.png"
            img.save(p)
            paths.append(str(p))
    finally:
        clip.close()
    return paths


def _parse_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"_raw": raw}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"_raw": raw}


def understand(
    video_path: str | Path,
    n_frames: int = 8,
    model_id: str = MODEL_ID,
    max_new_tokens: int = 768,
) -> dict:
    """Análisis estructurado del contenido del anuncio (dict JSON)."""
    import torch
    from qwen_vl_utils import process_vision_info

    model, processor = _load(model_id)
    frames = extract_frames(video_path, n=n_frames)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": [f"file://{p}" for p in frames]},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs.input_ids.shape[1]:]
    raw = processor.batch_decode(gen, skip_special_tokens=True)[0]
    for p in frames:
        try:
            os.unlink(p)
        except OSError:
            pass
    return _parse_json(raw)
