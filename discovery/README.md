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
Fase 3  neural      (TRIBE + V-JEPA residente, 2 pasadas) → unload   [opt-in]
        3a  features : V-JEPA/audio de TODOS los anuncios (lo caro, minutos/anuncio) → cacheado
        3b  forwards : forward de TRIBE de TODOS (~1 s/anuncio con features cacheadas)
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

Flags: `--workers N` (descargas en paralelo, def. 4), `--vlm-batch N` (anuncios por
forward del VLM, def. 2), `--frames N` (def. 12), `--qwen-model`, `--whisper-model`,
`--language`, `--no-transcribe`, `--no-understand`, `--neural`, `--no-cache`.

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

## Batch del VLM (fase `understand`)

Varios anuncios por forward (`--vlm-batch`, def. 2) amortiza el coste del VLM. Dos
salvaguardas obligatorias, ambas verificadas:

1. **Padding a la izquierda.** En generación batcheada el padding debe ser por la
   izquierda, o las secuencias generadas se desalinean respecto a
   `input_ids.shape[1]` y se mezclan las respuestas. `understand_batch` lo fija.
2. **Guardia anti-degeneración + reintento individual.** El batch cambia la
   trayectoria de la decodificación greedy y en algún anuncio degenera (p. ej. una
   lista repite elementos). `looks_degenerate()` lo detecta y ese anuncio se recalcula
   con `batch=1` en vez de guardar una respuesta degradada.
3. **Adaptativo a VRAM:** ante OOM reduce el trozo a la mitad y reintenta (hasta 1).

Medido (7B, 12 frames, 3 anuncios, mismo proceso):

| Config | Total | s/anuncio | vs secuencial |
|---|---|---|---|
| secuencial (3×1) | 57.7 s | 19.2 | — |
| `--vlm-batch 3` (run 1) | 22.5 s | **7.5** | x2.6 |
| `--vlm-batch 3` (run 2) | 36.2 s | **12.1** | x1.6 |

En ambas corridas **3/3 respuestas idénticas** a la inferencia individual. La guardia
disparó 1 vez (anuncio `BsMrRFH390k`, un montaje antiguo propenso a bucle): el coste del
fallback es un forward extra, que es lo que separa x2.6 de x1.6.

> **Nota: TRIBE no se batchea, se separa en dos pasadas.** El coste lo domina la codificación
> V-JEPA (~1.8 s/frame-batch), no el forward de TRIBE (~1 s). En vez de batchear anuncios (que
> exigiría fusionar sus *events* con atribución frágil de segmentos), la fase neural se ejecuta
> en **dos pasadas**: primero **todas** las features (la GPU trabaja seguida en lo caro, sin
> alternar encode→forward→encode) y después **todos** los forwards, que con las features
> cacheadas cuestan ~1 s/anuncio y son repetibles sin re-codificar.

## Métricas públicas (proxy de resultado, sin depender del cliente)

El pipeline captura **vistas, likes y comentarios públicos** de YouTube sin descargar el
vídeo y los persiste en el store (columna `stats_updated_at`, TTL **24 h**).

```bash
# Solo métricas públicas (red, sin GPU)
.venv/Scripts/python.exe -m discovery.pipeline --stats-only
.venv/Scripts/python.exe -m discovery.pipeline --stats-only --refresh-stats   # ignora el TTL
.venv/Scripts/python.exe -m discovery.pipeline --all --no-stats               # saltarlas
```

Se guardan **crudas**; las derivadas se calculan al leer con `store.stats_table()`:
`views_per_day`, `like_rate`, `comment_rate`.

### Por qué NO sirve para calibrar (medido en el corpus Telcel, 30 anuncios)

| observación | dato real | consecuencia |
|---|---|---|
| Las vistas las domina la **inversión en pauta** | 53.0M y 47.7M vistas/día 424K y 382K en dos anuncios; el resto 0–791 | sin saber el spend, no es comparable |
| Los **likes están ocultos o ausentes** en parte del corpus | un anuncio devuelve `None`; los dos virales dan 0,001 % | `like_rate` no es fiable |
| Hay **re-subidas de terceros** | canal ≠ marca en 11/30 | no son métricas de la marca |

**Conclusión honesta:** sirven para ordenar **dentro de un mismo canal** y para detectar
casos extremos; **no** calibran contra resultados de campaña. Para eso hace falta ranking del
cliente o verdad comprada (test de pauta propio).

## Convención temporal (cerrada en Fase 0.5)

`preds[k]` es la respuesta al **segundo `k`** del anuncio
(`core.ordering.ALIGNMENT_CONVENTION == "stimulus-aligned"`). El retardo hemodinámico de
5 s **ya lo aplica el checkpoint** al construir su objetivo de entrenamiento
(`estímulo(t) → BOLD(t+5)`): **no hay que restarlo** al leer `preds`.

Precisión de la localización absoluta: **≈ ±1.5 s** — una rejilla de 1 TR con un sesgo de
~1.5 s; no autoriza a afirmar precisión sub-segundo. Verificado empíricamente sobre
4 anuncios reales con dos métodos independientes: [`docs/FASE_0.5_ALINEACION.md`](../docs/FASE_0.5_ALINEACION.md).

## Rendimiento medido (RTX 4090, esta máquina)

| Etapa | Coste | Nota |
|---|---|---|
| transcribe | ~14.5 s/anuncio | Whisper residente (antes: recarga por anuncio) |
| understand (7B, 12 frames) | ~19.2 s/anuncio secuencial · **7.5–12.1 s/anuncio con `--vlm-batch 3`** (x1.6–2.6) | antes ~600 s/anuncio sin cap de frames ni residencia |
| neural `features` (V-JEPA/audio) | **~125 s por anuncio de 20 s**; ~269 s/anuncio en el corpus 20–60 s | lo caro; contiguo, sin alternar con forwards |
| neural `forward` (TRIBE) | **~0.9 s/anuncio** con features cacheadas | repetible sin re-codificar |
| resume desde store | ~0.1 s | corrida completa ya procesada |
| cache hit | ~0.00 s/anuncio | ni carga el modelo |

El cuello real es la **codificación de video (V-JEPA)**, no la inferencia de TRIBE.
