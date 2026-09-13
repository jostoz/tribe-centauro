# discovery/ — Descubrimiento y análisis de anuncios

Pipeline que descubre anuncios en YouTube, extrae su **script** (voz→texto),
**entiende su contenido** con un VLM local (Qwen2.5-VL) y arma un **grafo de
contenidos**. Opcionalmente añade el **perfil neural** por red funcional (TRIBE v2).

```
fetch → transcribe → understand → graph  [→ neural TRIBE]
```

## Requisitos
Ya instalados en el `.venv`: `yt-dlp`, `transformers`, `qwen-vl-utils`, `accelerate`,
`networkx`, `moviepy` (+ ffmpeg empaquetado por `imageio-ffmpeg`). GPU recomendada.
Aplica la skill `centauro-gpu-inference` (num_workers=0, una inferencia a la vez).

## Uso

```bash
# Por búsqueda
.venv/Scripts/python.exe -m discovery.pipeline --search "Telcel comercial" --n 8

# Por canal
.venv/Scripts/python.exe -m discovery.pipeline --channel @Telcel --n 12

# URLs concretas + perfil neural TRIBE (lento)
.venv/Scripts/python.exe -m discovery.pipeline --urls https://youtu.be/XXXX --neural

# Reconstruir solo el grafo desde registros ya guardados
.venv/Scripts/python.exe -m discovery.pipeline --graph-only
```

Flags: `--no-transcribe`, `--no-understand`, `--frames N` (def. 8),
`--qwen-model Qwen/Qwen2.5-VL-7B-Instruct` (mejor calidad, más VRAM), `--neural`.

## Salidas
- `data/ads/<id>.mp4` — video descargado (progresivo o fusionado con audio).
- `data/discovery/records/<id>.json` — registro por anuncio: metadata, transcript,
  understanding (escenas/objetos/personajes/marca/ritmo/tono/temas) y neural (si `--neural`).
- `data/discovery/graph.{json,graphml,html}` — grafo de contenidos.
  Abre `graph.html` en el navegador (interactivo, sin dependencias).

## Grafo
- Nodos: `ad` (anuncio) + `obj`/`pers`/`tema`/`marca` (atributos extraídos).
- Aristas: anuncio→atributo, y anuncio↔anuncio con peso = atributos compartidos
  (línea punteada). Así se ven clusters de creativos por contenido/tema/personaje.

## Notas de rendimiento
- Primera corrida descarga los modelos (Whisper-small ~0.5GB, Qwen2.5-VL-3B ~7GB); luego caché.
- Frames redimensionados a ≤448px para acotar el costo de tokens del VLM.
- El 3B es rápido pero superficial; usa `--qwen-model ...-7B-Instruct` o `--frames 16`
  para descripciones más ricas.
- `--neural` carga TRIBE + V-JEPA (varios minutos/anuncio); no lo combines con el VLM en
  paralelo (memoria de commit en Windows).

## Módulos
- `fetch.py` — yt-dlp (search / channel / urls), fuerza pista de audio.
- `transcribe.py` — Whisper (transformers) en GPU.
- `understand.py` — Qwen2.5-VL sobre frames muestreados.
- `graph.py` — networkx + export JSON/GraphML/HTML.
- `pipeline.py` — orquestación + CLI.
