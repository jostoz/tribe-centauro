# discovery/ — Descubrimiento y análisis de anuncios

Pipeline que descubre anuncios en YouTube, extrae su **script** (voz→texto),
**entiende su contenido** con un VLM local (Qwen2.5-VL) y arma un **grafo de
contenidos**. Opcionalmente añade el **perfil neural** por red funcional (TRIBE v2).

```text
fetch → transcribe → understand → graph  [→ neural TRIBE]
```

## Arquitectura: ejecución por FASES DE MODELO

Regla de oro en 1 GPU (`docs/PLAN_ESCALABILIDAD.md`): **procesar por fase de modelo, no
por anuncio**. Cada modelo se carga UNA vez por corrida, se procesan todos los anuncios
de esa fase, y luego se libera VRAM. Así el modelo se carga 3 veces en total, no 3×N.

```text
Fase 0  fetch       (CPU/red, descargas en paralelo)
Fase 1  transcribe  (Whisper residente)  → unload
Fase 2  understand  (Qwen2.5-VL residente) → unload
Fase 3  neural      (TRIBE + V-JEPA residente) → unload   [opt-in]
        → store SQLite → grafo de contenidos
```

## Idempotencia y caché

- **Caché content-addressed** (`cache.py`): clave = `sha256(bytes del video ‖ etapa ‖
  params)`. Si ya existe el resultado, no se recalcula ni se carga el modelo.
- **Store SQLite** (`store.py`, `data/discovery/centauro.db`): persiste metadata,
  transcript, understanding, neural y las **entidades canónicas**. Re-correr es resume.
- `--no-cache` fuerza el recálculo.

## Requisitos

`.venv` con `yt-dlp`, `transformers`, `qwen-vl-utils`, `accelerate`, `networkx`,
`moviepy` (+ ffmpeg empaquetado por `imageio-ffmpeg`). GPU recomendada.
Aplica la skill `centauro-gpu-inference` (num_workers=0, una inferencia a la vez).

## Uso

```bash
# Canal completo (por defecto: Whisper + Qwen 7B, 12 frames)
.venv/Scripts/python.exe -m discovery.pipeline --channel @Telcel --n 12

# Búsqueda, con perfil neural TRIBE (lento, opt-in)
.venv/Scripts/python.exe -m discovery.pipeline --search "Telcel comercial" --n 8 --neural

# URLs concretas, solo contenido (sin transcripción)
.venv/Scripts/python.exe -m discovery.pipeline --urls https://youtu.be/XXXX --no-transcribe

# Reconstruir el grafo desde el store (sin reprocesar)
.venv/Scripts/python.exe -m discovery.pipeline --graph-only

# Mantenimiento de caché
.venv/Scripts/python.exe -m discovery.pipeline --cache-stats
.venv/Scripts/python.exe -m discovery.pipeline --prune-cache 30
```

Flags: `--workers N` (descargas en paralelo, def. 4), `--frames N` (def. 12),
`--qwen-model`, `--whisper-model`, `--language`, `--no-transcribe`, `--no-understand`,
`--neural`, `--no-cache`.

Mover la caché de modelos a otro disco (evita llenar C:):
```bash
set CENTAURO_HF_HOME=D:\hf-cache
```

## Salidas

- `data/ads/<id>.mp4` — videos descargados (progresivo o fusionado con audio).
- `data/discovery/centauro.db` — store SQLite (anuncios + entidades canónicas).
- `data/discovery/cache/<etapa>/<hash>.json` — caché por etapa.
- `data/discovery/graph.{json,graphml,html}` — grafo de contenidos
  (abre `graph.html` en el navegador).

## Grafo

- Nodos: `ad` + `obj`/`pers`/`tema`/`marca` (entidades **canonicalizadas** por
  `entities.py`: minúsculas, sin acentos/artículos, sinónimos → una sola forma).
- Aristas: anuncio→entidad, y anuncio↔anuncio con peso = entidades compartidas
  (línea punteada). Así aparecen clusters de creativos por contenido/tema/personaje.

## Módulos

| Módulo | Función |
|---|---|
| `fetch.py` | yt-dlp: search/channel/urls, descarga en paralelo, fuerza pista de audio |
| `transcribe.py` | Whisper (transformers) en GPU; `unload()` para liberar VRAM |
| `understand.py` | Qwen2.5-VL sobre frames muestreados (≤448px); `unload()` |
| `neural.py` | `NeuralAnalyzer`: TRIBE residente, perfil por red funcional |
| `entities.py` | canonicalización/dedup de entidades |
| `cache.py` | caché content-addressed + `stats()`/`prune()` |
| `store.py` | SQLite de anuncios + entidades + `top_entities()` |
| `graph.py` | networkx + export JSON/GraphML/HTML |
| `gpu.py` | `free_gpu()`, `vram_report()` |
| `pipeline.py` | orquestación por fases + CLI + métricas s/anuncio |

## Rendimiento medido (RTX 4090, esta máquina)

| Etapa | Coste | Nota |
|---|---|---|
| transcribe | ~14.5 s/anuncio | Whisper residente (antes: recarga por anuncio) |
| understand (7B, 12 frames) | ~19.5 s/anuncio | antes ~600 s/anuncio sin cap de frames ni residencia |
| neural (TRIBE) | ~176 s/anuncio en frío | **dominado por V-JEPA**; 2 s si las features están cacheadas |
| resume desde store | ~0.1 s | corrida completa ya procesada |
| cache hit | ~0.00 s/anuncio | ni carga el modelo |

El cuello real es la **codificación de video (V-JEPA)**, no la inferencia de TRIBE.
