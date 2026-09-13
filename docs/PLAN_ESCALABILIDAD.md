# Plan de escalabilidad — Pipeline de descubrimiento y análisis (Centauro)

**Fecha:** 2026-09-13 · **Hardware base:** 1× RTX 4090 (24 GB), Windows, disco C: (presión de espacio)
**Alcance:** escalar `discovery/` (fetch → transcribe → understand → graph → neural) de 1 anuncio a
cientos/miles, y preparar el salto a servicio.

> El "siguiente paso" concreto era poblar el corpus Telcel (Grupo A, ~12 anuncios) y ver el grafo.
> Este documento planifica cómo hacerlo de forma que **no se rompa** al crecer 10×–100×.

---

## 1. Costos reales medidos (esta sesión)

| Etapa | Costo observado | Cuello de botella |
|---|---|---|
| fetch (yt-dlp, 1 clip 30 s) | ~3 s | red |
| transcribe (Whisper-small, GPU) | ~78 s **incluye carga del modelo**; ASR en sí segundos | carga de modelo por proceso |
| understand (Qwen2.5-VL-3B, 8 frames, sin cap) | ~600 s | **tokens de visión sin cap + recarga de modelo** |
| understand (Qwen2.5-VL-7B, 16 frames) | dominado por descarga única; inferencia ~decenas de s | carga de modelo por proceso |
| neural TRIBE (ruta video, V-JEPA cold) | ~468 s | codificación V-JEPA ViT-g (~105 s) + descargas |
| neural TRIBE (V-JEPA features **cacheadas**) | **1.1 s** | — |

**Conclusión dura:** el costo dominante NO es el cómputo, es (a) **recargar modelos en cada proceso**
y (b) **descargas/decodificación sin caché**. Con caché de features y modelos residentes, el costo
marginal por anuncio cae de minutos a segundos.

---

## 2. Problemas que impiden escalar hoy

1. **Recarga de modelos por invocación.** Cada script (`understand`, `transcribe`, `neural`) carga su
   modelo desde cero. A 100 anuncios eso es 100× la carga. → **Worker residente**.
2. **Una sola GPU, memoria de commit frágil (WinError 1455).** No se pueden tener Qwen-7B + V-JEPA +
   TRIBE en VRAM a la vez de forma fiable. → **Fases secuenciales por modelo**, no por anuncio.
3. **Disco.** La caché HF llegó a 212 GB y llenó C: (0.06 GB libres) → fallos de descarga.
   → **Política de almacenamiento y disco dedicado**.
4. **Sin idempotencia real.** Re-correr reprocesa todo. → **Cache content-addressed por hash**.
5. **Grafo desde JSON sueltos.** No dedup de entidades, se reconstruye entero cada vez.
   → **Store (SQLite) + normalización de entidades**.
6. **Serial y monolítico.** fetch (CPU/red), decode (CPU), inferencia (GPU) corren en el mismo hilo.
   → **Pipeline por etapas con solapamiento CPU/GPU**.

---

## 3. Arquitectura objetivo (single-box 4090)

```
                   ┌───────────── CPU pool (paralelo) ─────────────┐
   lista de URLs → │ fetch (yt-dlp)  →  extract_frames + audio (ffmpeg) │ → cola de "listos"
                   └───────────────────────────────────────────────┘
                                          │  (content hash = sha256(bytes+params+model_ver))
                                          ▼
                   ┌────────── GPU worker (1 modelo residente a la vez) ──────────┐
                   │  FASE A: Whisper (todos)  →  FASE B: Qwen2.5-VL (todos)       │
                   │  FASE C: TRIBE+V-JEPA (todos)   [cada fase carga 1 modelo]    │
                   └──────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                        store (SQLite)  →  build_graph incremental  →  graph.{json,html}
```

**Regla de oro en 1 GPU:** procesar **por fase de modelo, no por anuncio**. Cargar Whisper una vez,
transcribir los N; liberar; cargar Qwen una vez, entender los N; liberar; cargar TRIBE, inferir los N.
Así el modelo se carga 3 veces en total, no 3×N.

---

## 4. Cambios de ingeniería concretos (orden de implementación)

### Fase 1 — Residencia + fases (mayor ganancia, bajo esfuerzo)
- [ ] Refactor `pipeline.py` a **ejecución por fases de modelo** (batch por etapa), no por anuncio.
- [ ] Mantener cada modelo cargado durante toda su fase (ya hay `lru_cache` en `understand._load`;
      extender a transcribe/neural y **liberar VRAM entre fases** con `del model; torch.cuda.empty_cache()`).
- [ ] Aplicar siempre la skill `centauro-gpu-inference` (num_workers=0, una inferencia a la vez).

### Fase 2 — Caché e idempotencia
- [ ] `cache_key = sha256(video_bytes ‖ stage ‖ model_version)`; si existe resultado, saltar.
- [ ] Persistir features intermedias (V-JEPA ya se cachea vía backend `Cached`; extender a frames y audio).
- [ ] `--resume` por defecto: no reprocesar registros existentes.

### Fase 3 — Store + grafo incremental
- [ ] Migrar `records/*.json` → **SQLite** (`service/store.py` ya previsto en el plan MCP).
- [ ] **Normalización/dedup de entidades**: lematizar + alias; opcional embeddings (mismo VLM o
      `sentence-transformers`) para fusionar "logo Telcel" ≈ "logotipo de Telcel".
- [ ] `build_graph` incremental (solo nodos/aristas nuevos).

### Fase 4 — Solapamiento CPU/GPU y throughput
- [ ] Pool de descargas/decodificación (CPU) alimentando una cola mientras la GPU infiere.
- [ ] Batch real en Qwen (varios anuncios por forward cuando la VRAM lo permita) y en TRIBE.
- [ ] Métricas por corrida: s/anuncio por etapa, VRAM pico, aciertos de caché.

### Fase 5 — Disco y datos
- [ ] Mover caché HF a disco dedicado: `HF_HOME=D:\hf-cache` (evita llenar C:).
- [ ] Política de retención: purgar modelos no usados; `pip cache purge` en CI.
- [ ] Outputs (`data/discovery/`) fuera de git (ya en `.gitignore`); si crecen, a object storage.

---

## 5. Proyección de throughput (1× 4090)

Supuestos tras Fase 1–2 (modelos residentes, features cacheadas):
- transcribe ≈ 5–10 s/anuncio · understand (7B, 12 frames) ≈ 20–40 s/anuncio · neural ≈ 100 s/anuncio (cold V-JEPA) / 2 s (cached).

| Corpus | Solo contenido (fetch+ASR+VLM) | + neural (cold) |
|---|---|---|
| 12 (Grupo A) | ~10 min | ~30 min |
| 100 | ~1.5 h | ~4–5 h |
| 1 000 | ~15 h (overnight) | multi-día en 1 GPU → **cloud** |

**Umbral de salto a cloud:** > ~500 anuncios con neural, o SLA de minutos. Entonces: contenedor +
GPU rentada (L4/A100), procesamiento offline por lotes, mismo código (models residentes por worker).

---

## 6. Salto a servicio (reusar el plan MCP existente)

Cuando el pipeline offline esté sólido, conectar con `MCP_ADS_SERVICE_PLAN.md`:
- `service/queue.py` (arq/Redis) + `service/worker.py` (worker GPU) + `service/store.py` (SQLite→Postgres).
- Idempotencia por hash (§4 Fase 2) ya es el contrato del `analyze_creative` del MCP.
- Multi-tenant: cuotas y concurrencia GPU por tenant; 1 inferencia en vuelo por worker.

---

## 7. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Disco se vuelve a llenar | `HF_HOME` a disco dedicado + retención; alerta a <20 GB |
| VRAM insuficiente al batchear | batch adaptativo por VRAM libre; fallback a batch=1 |
| WinError 1455 | num_workers=0, fases secuenciales, liberar VRAM entre fases |
| Calidad VLM variable | 7B por defecto; validar con set de contraste (Grupo A) antes de confiar |
| yt-dlp rompe/rate-limit | reintentos con backoff; cachear metadata; respetar ToS |
| Licencia CC-BY-NC | modo research hasta licencia comercial de Meta (ver META_LICENSE_REQUEST.md) |

---

## 8. Acción inmediata recomendada

1. Implementar **Fase 1** (ejecución por fases de modelo) — es la mayor ganancia con menor esfuerzo.
2. Correr el **Grupo A Telcel (12 anuncios)** con el pipeline ya por fases:
   `.venv/Scripts/python.exe -m discovery.pipeline --channel @Telcel --n 12`
3. Medir s/anuncio reales por etapa → recalibrar esta proyección con datos duros.

---

## 9. Estado de implementación (2026-09-13) — números MEDIDOS

Implementado y verificado en esta máquina (RTX 4090). Esta sección **reemplaza** las
proyecciones de §5 donde difieran.

### Qué quedó implementado

| Fase | Estado | Módulos / cambios |
|---|---|---|
| **1. Residencia + fases** | ✅ | `pipeline.py` reescrito a fases de modelo; `neural.py` (TRIBE residente); `transcribe.unload()`, `understand.unload()`; `gpu.py` |
| **2. Caché + idempotencia** | ✅ | `cache.py` (content-addressed `sha256(video‖etapa‖params)`, `stats()`, `prune()`); resume por defecto; `--no-cache` |
| **3. Store + grafo** | ✅ | `store.py` (SQLite anuncios + entidades + `top_entities()`); `entities.py` (canonicalización/dedup); `graph.py` usa entidades canónicas |
| **4. Solapamiento CPU/GPU** | ⚠️ parcial | `fetch.download_many()` en paralelo (workers=4). **Falta** batch real en Qwen/TRIBE (riesgo de VRAM; no implementado) |
| **5. Disco** | ⚠️ parcial | `CENTAURO_HF_HOME` → `HF_HOME`; `--prune-cache N`. **Falta** alerta automática de disco |

### Costos medidos vs. antes

| Etapa | Antes | Ahora (medido) | Ganancia |
|---|---|---|---|
| transcribe | recarga de modelo por anuncio | **~14.5 s/anuncio** (Whisper residente) | ~5× |
| understand (Qwen2.5-VL-7B, 12 frames) | ~600 s/anuncio | **~19.5 s/anuncio** | **~30×** |
| neural TRIBE (frío) | ~468 s/anuncio | **~176 s/anuncio** (30 s y 60 s de video) | ~2.6× |
| neural TRIBE (features V-JEPA cacheadas) | — | **~2 s** | — |
| re-corrida completa (resume) | reprocesa todo | **0.1 s** | — |
| re-corrida (cache hit, store vacío) | — | **0.00 s/anuncio, sin cargar modelos** | — |

Pruebas de verificación ejecutadas:
- 2 anuncios por fases, cache miss → 70.8 s total.
- Re-run mismo comando → **0.1 s** (resume desde store).
- Store borrado, caché intacta, 3 URLs fijas → **3/3 cache hits, 0.00 s/ad, sin `Loading weights`**.
- Fase neural: TRIBE cargado **una vez** para 2 anuncios → 352.5 s (176 s/ad).
- Grafo: 33 nodos / 39 aristas, **3 aristas anuncio↔anuncio** por entidades canónicas
  (`marca:telcel`, `obj:telefono`, `tema:tecnologia`), `red_dominante=Vis` en 2 anuncios.

### Correcciones a las proyecciones (§5)

- El cuello de botella real del pipeline **no es la inferencia de TRIBE** sino la
  **codificación de video V-JEPA** (~1.8 s/frame-batch; un anuncio de 60 s ≈ 4 min).
  Es donde más rinde cachear features y, a escala, considerar batching/aceleración.
- `neural` cold ≈ **176 s/anuncio** (no ~100 s) para anuncios de 30–60 s.
- El VLM pasó de ser el cuello (600 s) a ser barato (19.5 s) tras residencia + cap de frames.

### Pendiente para el siguiente salto

1. **Batch real** en Qwen/TRIBE (Fase 4) con batch adaptativo a VRAM.
2. **Dedup semántico** de entidades (embeddings) — hoy solo canonicaliza strings, no fusiona
   "persona con teléfono" ≈ "persona usando móvil".
3. **Alerta de disco** automática (el llenado de C: ya rompió una corrida).
4. Mover el corpus al store definitivo (SQLite → Postgres) al pasar a servicio.

