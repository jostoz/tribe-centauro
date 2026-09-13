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

    # Dir único por llamada: si no, en un batch los frames de un anuncio
    # sobrescriben los del anterior (mismo nombre) y todas las conversaciones
    # acaban leyendo los frames del último anuncio.
    out_dir = Path(tempfile.mkdtemp(prefix="qwenframes_"))
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
            p = out_dir / f"frame_{i:03d}.png"
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


_MAX_LIST = 40


def looks_degenerate(res: dict) -> bool:
    """Heurística de respuesta degenerada: JSON no parseado o listas con bucles.

    El batch cambia la trayectoria de la decodificación greedy; en un caso se
    observó una lista de marca repitiendo elementos. Detectarlo permite reintentar
    ese anuncio solo (batch=1) en vez de guardar una respuesta degradada.
    """
    if not isinstance(res, dict) or "_raw" in res or "_error" in res:
        return True
    for key in ("objetos", "personajes", "temas", "marca_elementos"):
        items = res.get(key)
        if not isinstance(items, list):
            continue
        if len(items) > _MAX_LIST:
            return True
        norm = [str(i).strip().lower() for i in items if str(i).strip()]
        if norm and 1 - len(set(norm)) / len(norm) > 0.3:
            return True
    return False


def _conversation(frames: List[str]) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": [f"file://{p}" for p in frames]},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]


def _cleanup(frames_per_ad: List[List[str]]) -> None:
    for frames in frames_per_ad:
        for p in frames:
            try:
                os.unlink(p)
            except OSError:
                pass
        if frames:
            try:
                Path(frames[0]).parent.rmdir()  # dir temporal único por llamada
            except OSError:
                pass


def _run_batch(model, processor, paths: List, n_frames: int, max_new_tokens: int) -> List[dict]:
    """Un solo forward para varios anuncios (requiere padding_side='left')."""
    import torch
    from qwen_vl_utils import process_vision_info

    frames_per_ad = [extract_frames(p, n=n_frames) for p in paths]
    try:
        conversations = [_conversation(fr) for fr in frames_per_ad]
        texts = [
            processor.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in conversations
        ]
        image_inputs, video_inputs = process_vision_info(conversations)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[:, inputs.input_ids.shape[1]:]
        raws = processor.batch_decode(gen, skip_special_tokens=True)
    finally:
        _cleanup(frames_per_ad)
    return [_parse_json(r) for r in raws]


def understand_batch(
    video_paths: List,
    n_frames: int = 8,
    model_id: str = MODEL_ID,
    max_new_tokens: int = 768,
    batch_size: int = 2,
) -> List[dict]:
    """Analiza N anuncios con **batch adaptativo a la VRAM**.

    Procesa en trozos de ``batch_size``; ante un OOM reduce el trozo a la mitad y
    reintenta (hasta 1). Así un batch agresivo degrada en vez de fallar la fase.
    """
    if not video_paths:
        return []
    model, processor = _load(model_id)
    # En generación batcheada el padding debe ser por la IZQUIERDA: con padding a la
    # derecha las secuencias generadas quedan desalineadas respecto a
    # `input_ids.shape[1]` y se mezclan las respuestas entre anuncios.
    processor.tokenizer.padding_side = "left"

    results: List[dict] = []
    i = 0
    bs = max(1, batch_size)
    while i < len(video_paths):
        chunk = video_paths[i:i + bs]
        try:
            outs = _run_batch(model, processor, chunk, n_frames, max_new_tokens)
        except Exception as exc:  # noqa: BLE001 - OOM u otro -> reducir y reintentar
            from discovery.gpu import free_gpu

            free_gpu()
            if bs > 1:
                bs = max(1, bs // 2)
                print(f"      [understand] batch reducido a {bs} ({type(exc).__name__})")
                continue  # reintenta el MISMO trozo con batch menor
            results.append({"_error": f"{type(exc).__name__}: {exc}"})
            i += 1
            continue
        # Guardia de calidad: el batch puede degenerar en algún anuncio -> reintento solo
        bad = [j for j, o in enumerate(outs) if bs > 1 and looks_degenerate(o)]
        if bad:
            print(f"      [understand] {len(bad)} respuesta(s) degenerada(s) en batch={bs}; reintento individual")
            for j in bad:
                try:
                    outs[j] = _run_batch(model, processor, [chunk[j]], n_frames, max_new_tokens)[0]
                except Exception as exc:  # noqa: BLE001
                    outs[j] = {"_error": f"{type(exc).__name__}: {exc}"}
        results.extend(outs)
        i += len(chunk)
    return results


def understand(
    video_path: str | Path,
    n_frames: int = 8,
    model_id: str = MODEL_ID,
    max_new_tokens: int = 768,
) -> dict:
    """Análisis estructurado de un anuncio (dict JSON). Envuelve ``understand_batch``."""
    return understand_batch(
        [video_path], n_frames=n_frames, model_id=model_id,
        max_new_tokens=max_new_tokens, batch_size=1,
    )[0]

