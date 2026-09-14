# PLAN — TRIBE Ads MCP: "Ads as a Service" para agentes

**Estado:** plan de ejecución · **Fecha:** 2026-09-13 · **Base:** revisión verificada de `tribe centauro` v1.0.0

---

## 1. Veredicto de la revisión

La implementación actual **no ejecuta una sola llamada al modelo**. Los cinco puntos de entrada
(`predict_from_video`, `predict_from_audio`, `predict_from_text`, `predict_multimodal`, y los
endpoints de `api/routes.py`) lanzan excepción. Además hay métricas fabricadas que un agente
consumiría como señal de optimización.

**Portar esto a MCP sin reparar el core industrializa el error: el agente optimizaría contra
números inventados, a escala y sin humano en el loop.**

### Blockers (ordenados por gravedad)

| # | Severidad | Hallazgo | Evidencia |
|---|---|---|---|
| B1 | 🔴 Crítico | `get_events_dataframe` acepta **exactamente una** fuente. `predict_multimodal` pasa hasta tres | `demo_utils.py`: `if len(provided) != 1: raise ValueError` |
| B2 | 🔴 Crítico | Kwarg inexistente: se usa `text=`, la API es `text_path` (`.txt`) | `get_events_dataframe(self, text_path=None, audio_path=None, video_path=None)` |
| B3 | 🔴 Crítico | `EngagementMetrics.get_temporal_engagement` **no existe** → `AttributeError` en cada análisis | Ejecutado: `hasattr(...) -> False`. Vive en `TribeModelWrapper` |
| B4 | 🔴 Crítico | `EngagementMetrics.compute_memorability_index` **no existe** → `AttributeError` | Ejecutado: `hasattr(...) -> False`. Vive en `PredictionAnalyzer` |
| B5 | 🔴 Crítico | ROIs fabricadas: `slice(0,2000)`=córtex visual es anatómicamente falso. El propio código lo admite | `# Esto requeriría una correspondencia real con fsaverage5` |
| B6 | 🔴 Crítico | t-test sobre vértices aplanados: N≈20484×T, autocorrelación espacial ignorada → **falsos positivos**. Con N=1.23M el error estándar es ~4e-4, así que diferencias ≈0.1-1% de la escala de ruido salen "significativas". Verificado: dos muestras independientes del mismo proceso dan p<1e-6 | `comparison.py:130` `ttest_ind(control_flat, variant_flat)`; demostrado en `tests/test_stats.py` |
| B7 | 🟠 Alto | `tribev2` **no está en PyPI**: `tribev2==0.1.0` es irresoluble → `pip install -r requirements.txt` falla | Ejecutado: `pip index versions tribev2` → `ERROR: No matching distribution found for tribev2`. Install oficial: `uv pip install "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"` |
| B8 | 🟠 Alto | Conflicto de dependencias: `numpy==1.24.3` vs `numpy==2.2.6` de tribev2; `torch>=2.0` vs `>=2.5.1,<2.7` | `pyproject.toml` de tribev2 |
| B9 | 🟠 Alto | `opencv-python==4.8.1` sin wheels para Python 3.13 (entorno local) → build desde fuente | `python -V` → 3.13.11 |
| B10 | 🟠 Alto | `platform_opt['recommendations']['status']` → `KeyError`; `status` está en `current_compliance` | `recommendations.py:255` |
| B11 | 🟠 Alto | `np.int64`/`np.bool_` en payloads → no serializables a JSON-RPC | `attention_peaks`, `passes_critical_period` |
| B12 | 🟠 Alto | Licencia: `setup.py` declara **MIT**; el modelo es **CC-BY-NC-4.0** (no comercial) | `LICENSE` de Meta + `README` |
| B13 | 🟡 Medio | `_check_compliance` asume 30 fps; la resolución real es **1 TR = 1 s** | Notebook: "1 TR = 1 second" |
| B14 | 🟡 Medio | Fuga de rutas: `open(temp_dir / file.filename)` sin sanear → path traversal | `api/routes.py` upload |
| B15 | 🟡 Medio | `allow_origins=["*"]`, cero auth, `_process_batch_job` es un stub, `get_results`/`get_heatmap` → 501 | `api/` |
| B16 | 🟡 Medio | `self.model.to(device)` sobre `TribeExperiment` (no es `nn.Module`); el device se pasa en `from_pretrained` | `demo_utils.py` ya hace `model.to(device); model.eval()` |
| B17 | 🟡 Medio | NES = `min(100, mean(|x|)*100)` sin calibración. Umbrales 60/65/50/55 son inventados | `metrics.py:24` |
| B18 | 🟡 Medio | `fuse_multimodal_embeddings` → `AttributeError` si `video_embed is None` y `audio_embed` no | `embeddings.py` `video_embed.shape[1]` |
| B19 | 🟢 Bajo | Test tautológico: `test_get_temporal_engagement` no prueba la función que nombra | `tests/test_metrics.py:65` |
| B20 | 🟢 Bajo | Deps sin usar: matplotlib, plotly, seaborn, pillow, imageio, scikit-learn (peso en imagen de contenedor) | `requirements.txt` |

### Contratos reales de TRIBE v2 (fuente: código de Meta)

```
VALID_SUFFIXES = {
  "text_path":  {".txt"},
  "audio_path": {".wav", ".mp3", ".flac", ".ogg"},
  "video_path": {".mp4", ".avi", ".mkv", ".mov", ".webm"},
}

TribeModel.from_pretrained(checkpoint_dir, checkpoint_name="best.ckpt",
                           cache_folder=None, device="auto", config_update=None)
#   → devuelve modelo ya en `.to(device)` y `.eval()`

model.get_events_dataframe(text_path | audio_path | video_path)  # exactamente uno
preds, segments = model.predict(events=df)
#   preds.shape == (n_timesteps, 20484)  ; 1 TR = 1 s
#   fsaverage5 = 10242 vértices/hemisferio, concatenados [izq | der]
#   video ⇒ el audio se extrae del propio video (no hay pista de audio paralela)
```

Restricciones operativas que impactan el servicio:

- **Texto ⇒ red**: `TextToEvents` usa `gTTS` + `langdetect` (Google TTS) en tiempo de inferencia.
- **Llama-3.2 es gated**: se requiere token HF con acceso aprobado para las features de texto.
- **`ChunkEvents(max_duration=60, min_duration=30)`**: los estímulos se trocean en bloques de 30–60 s.
  Comportamiento con anuncios de 6–15 s **verificado**: no se descartan (`docs/FASE_0.5_VEREDICTO.md`).
- **`RemoveMissing()` + `AddSentenceToWords(max_unmatched_ratio=0.05)`**: un anuncio sin voz puede
  perder eventos → **verificado**: la ruta `audio_only` produce inferencia válida en mudo.
- **Offset hemodinámico de 5 s**: **ya lo aplica el checkpoint**, no el consumidor.
  `preds[k]` es la respuesta al segundo `k` del estímulo (`alignment = "stimulus-aligned"`,
  verificado empíricamente — `docs/FASE_0.5_ALINEACION.md`). Precisión de la localización
  absoluta: ≈ ±1.5 s.

---

## 2. Decisión de arquitectura

**Un core agnóstico de protocolo, dos adaptadores de borde.**

```
                 ┌──────────────────────────────┐
   agente  ──▶   │  mcp/   (tools/resources/    │  ← protocolo MCP (stdio | streamable HTTP)
                 │          prompts)            │
                 └──────────────┬───────────────┘
   dashboard ──▶ ┌──────────────▼───────────────┐
   (REST)        │  api/  (FastAPI, thin)       │  ← se conserva para UI/webhooks internos
                 └──────────────┬───────────────┘
                                │
                 ┌──────────────▼───────────────┐
                 │  service/  AdOptimizationService │  ← ÚNICA capa con lógica
                 │  jobs · store · queue · worker   │
                 └──────────────┬───────────────┘
                                │
                 ┌──────────────▼───────────────┐
                 │  core/   (TRIBE v2 wrapper)  │  ← reparado
                 └──────────────────────────────┘
```

**Regla:** `mcp/` y `api/` no contienen lógica de dominio. Toda la marshaling
numpy→JSON, el bounding de payloads y la taxonomía de errores vive en `service/`.
Se elimina `main.py::AdOptimizer` (duplica `api/routes.py` con marshaling divergente — el origen de B3/B4/B10).

**Por qué MCP y no solo REST:** un agente necesita (a) descubrir capacidades y límites en una
llamada, (b) errores accionables por máquina, (c) resultados acotados que no inunden su contexto,
(d) plantillas de workflow reutilizables. Ninguna de las cuatro se resuelve con REST genérico.

---

## 3. Protocolo MCP — superficie de exposición

Objetivo: **spec `2026-07-28`** (core stateless, Multi Round-Trip Requests, header-based routing,
list results cacheables, framework de extensions). `[INFERENCE]` El core stateless *habilita*
réplicas MCP sin estado y por tanto exige que el estado de jobs viva fuera del servidor — es la
razón de diseñar jobs con handles opacos desde el día uno.

> Al implementar, fijar la versión del SDK Python (`mcp`) contra
> `modelcontextprotocol.io/specification/2026-07-28` y no asumir la API de revisiones previas.

### 3.1 Tools

Principio: **barato y síncrono por defecto; caro y asíncrono con handle.** Nunca bloquear el
`tools/call` durante minutos de GPU.

| Tool | Coste | Descripción |
|---|---|---|
| `get_service_capabilities` | puro | Plataformas, formatos, límites, definiciones de métricas **con advertencias de validez**, latencia esperada, cuota, licencia |
| `get_platform_spec(platform)` | puro, cacheable | Restricciones de duración/resolución/aspecto/hook y umbrales |
| `validate_creative(source, platform?)` | ≤100 ms, sin GPU | Pre-flight: formato, tamaño, duración, mudez, idioma. Devuelve ETA y coste estimado |
| `analyze_creative(source, platform, options?)` | **asíncrono** | Devuelve `{ad_id, job_id, status:"queued"}`. Idempotente por hash de contenido |
| `get_job(job_id)` | puro | Estado, progreso, error, `result_ref` |
| `cancel_job(job_id)` | puro | Cancela encolado o en ejecución |
| `get_creative_report(ad_id, detail?)` | puro | Resumen **acotado** + URIs de recursos. `detail=summary\|standard` |
| `get_attention_timeline(ad_id, max_points=120, align="stimulus")` | puro | Serie temporal diezmada, alineada al estímulo |
| `get_editing_suggestions(ad_id, platform, max_suggestions=5)` | puro | Rangos temporales `[t0,t1]` en segundos del estímulo + acción |
| `compare_creatives(ad_ids[], metric, design)` | puro / derivado | Estadística honesta (pareada + corrección por clúster) |
| `rank_creatives(items[], platform)` | asíncrono si ≥2 sin analizar | Ranking ordenado + justificación |
| `render_brain_map(ad_id, t, view, format)` | asíncrono | Devuelve **content block de imagen** |
| `get_usage(period?)` | puro | Cuota y consumo (metering AaaS) |
| `submit_feedback(ad_id, outcome)` | puro | Cierra el loop: el agente reporta el resultado real (CTR, retención) → **set de calibración** |

Contrato de ejemplo (`analyze_creative`):

```json
{
  "name": "analyze_creative",
  "description": "Encola el análisis neural de un creativo publicitario. Devuelve inmediatamente un handle; consultar con get_job. Las puntuaciones son relativas dentro del mismo pipeline y NO están calibradas contra resultados de campaña.",
  "inputSchema": {
    "type": "object",
    "required": ["source"],
    "properties": {
      "source": {
        "oneOf": [
          {"type": "object", "required": ["url"],  "properties": {"url": {"type": "string", "format": "uri"}}},
          {"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}},
          {"type": "object", "required": ["text"], "properties": {"text": {"type": "string", "maxLength": 10000}}}
        ]
      },
      "platform": {"enum": ["instagram", "tiktok", "facebook", "youtube"]},
      "options": {
        "type": "object",
        "properties": {
          "roi_atlas": {"enum": ["schaefer200", "destrieux", "none"], "default": "schaefer200"},
          "include_maps": {"type": "boolean", "default": false}
        }
      }
    }
  },
  "outputSchema": {"$ref": "tribe://schemas/job"}
}
```

### 3.2 Resources

| URI | Contenido |
|---|---|
| `tribe://capabilities` | Espejo de `get_service_capabilities` |
| `tribe://platforms/{platform}/spec` | Especificación de plataforma |
| `tribe://metrics/definitions` | Definición + **validez y límites explícitos** de cada métrica |
| `tribe://creatives/{ad_id}/report` | Reporte completo (JSON Schema versionado) |
| `tribe://creatives/{ad_id}/timeline` | CSV diezmado |
| `tribe://creatives/{ad_id}/brain/{view}/{t}.png` | Mapa cortical |
| `tribe://jobs/{job_id}` | Estado del job |
| `tribe://schemas/{report,comparison,job,error}` | JSON Schema — el agente autovalida |
| `tribe://terms` | Licencia, uso permitido, base legal |

Todas las URIs con tenant. Sin enumeración cruzada.

### 3.3 Prompts

| Prompt | Args | Salida |
|---|---|---|
| `optimize_creative_for_platform` | `source`, `platform` | Workflow validate → analyze → poll → report → editing suggestions |
| `ab_test_creatives` | `a`, `b`, `platform` | compare + interpretación de intervalos, no solo "ganador" |
| `audit_creative_portfolio` | `items[]`, `platform` | rank + clusterización por debilidad |
| `explain_report` | `ad_id` | Traduce unidades, marca incertidumbre, evita sobreinterpretación |

### 3.4 Modelo asíncrono e idempotencia

```
queued ──▶ running ──▶ succeeded
   │          │     └▶ failed
   │          └──────▶ canceled
   └─────────────────▶ expired
```

- `job_id`: opaco, tenant-scoped, no adivinable.
- `progress`: `{phase: ingest|features|transformer|postprocess, pct}`.
- Si el cliente envía `progressToken`, emitir `notifications/progress`. **No es la fuente de
  verdad**: el agente siempre puede consultar `get_job` (requisito del core stateless).
- **Idempotencia**: `idempotency_key = sha256(media_bytes ‖ params ‖ model_version)`.
  Repetir `analyze_creative` con el mismo contenido devuelve el `ad_id` existente sin recomputar.
  Un agente que reintenta no debe pagar dos veces ni duplicar artefactos.
- **TTL**: resultados persistidos; jobs expiran; URIs de artefactos firmadas con caducidad.

### 3.5 Taxonomía de errores

`isError: true` + `structuredContent.code` accionable por máquina:

`INVALID_INPUT` · `MEDIA_UNSUPPORTED` · `MEDIA_TOO_LARGE` · `MEDIA_TOO_LONG` ·
`NO_AUDIO_TRACK` · `TEXT_TO_SPEECH_UNAVAILABLE` · `HF_ACCESS_DENIED` ·
`JOB_NOT_FOUND` · `JOB_NOT_READY` · `QUOTA_EXCEEDED` · `RATE_LIMITED` ·
`MODEL_UNAVAILABLE` · `GPU_OOM` · `LICENSE_REQUIRED` · `INTERNAL`

Cada error incluye `retryable: bool` y `hint` (cómo corregir). Sin esto, un agente no puede
auto-recuperarse y degradará a reintentos ciegos.

---

## 4. Honestidad de métricas — la decisión de producto

El core actual expone `Neural Engagement Score 0–100` con umbrales por plataforma. Es
`min(100, mean(|preds|)*100)`: **la magnitud depende de la escala de salida del modelo, no del
"engagement"**, y nunca se calibró contra ningún resultado real. Los umbrales 60/65/50/55 no
tienen origen.

Un agente optimizará contra esta señal y producirá creativos peores con alta confianza.

**Plan:**

1. Renombrar a lo que es: `mean_abs_activation` con unidades crudas. Sin número 0–100.
2. Exponer siempre:
   ```json
   {
     "units": "raw_model_activation",
     "calibrated": false,
     "validity": "population-average, research-use-only",
     "model_version": "facebook/tribev2@<rev>",
     "comparability": "within-batch-only"
   }
   ```
3. `compare_creatives` y `rank_creatives` operan **en relativo** (mismo pipeline, mismo lote) —
   más defendible que cualquier score absoluto.
4. `submit_feedback` acumula `(predicción, resultado real)` → habilita calibración real
   (monótona / isotónica) y, solo entonces, un score con semántica. Ese es el foso competitivo
   del servicio.
5. Eliminar los umbrales inventados de `validator.py` o marcarlos `provisional: true`.

---

## 5. Seguridad, multi-tenencia y licencia

| Área | Control |
|---|---|
| **Licencia** | `LICENSE_MODE=research\|commercial`. En `commercial`, el server **no arranca** sin `META_LICENSE_REF`. `research` marca cada respuesta con `commercial_use: false`. "Ads as a Service" es uso comercial → requisito legal, no opcional |
| **Gated Llama-3.2** | Token HF con acceso aprobado; fallar con `HF_ACCESS_DENIED` claro, no con un stack trace |
| **gTTS** | Dependencia de red + Google. Timeout, fallback o rechazo explícito (`TEXT_TO_SPEECH_UNAVAILABLE`) |
| **Rutas** | Perfil HTTP: **el agente nunca aporta rutas de filesystem**, solo URLs o upload. Corrige B14 |
| **SSRF** | Allowlist de esquemas (https), bloqueo de rangos privados/link-local, resolución DNS validada, límite de redirecciones |
| **Tamaño** | Cap de bytes, duración, resolución y FPS antes de decodificar |
| **Prompt injection** | El creativo es **entrada no confiable**: el texto (overlay, transcripción) nunca entra en descripciones de tools ni en prompts del servidor. Se devuelve marcado como `untrusted_content`, con control chars eliminados y longitud acotada |
| **Payloads** | Límite duro de bytes por respuesta. Datos crudos → recurso, no al contexto del agente |
| **Auth (HTTP)** | OAuth 2.1 resource server / bearer; `WWW-Authenticate` en 401; scopes `ads:analyze`, `ads:read`, `ads:write`, `ads:admin` |
| **Aislamiento** | Todo artefacto y job tenant-scoped; URIs no enumerables |
| **Abuso** | Cuota por tenant, profundidad de cola, concurrencia GPU por tenant, rate limit |
| **GPU** | 1 inferencia en vuelo por worker, `inference_mode`, descarga del modelo en idle, `GPU_OOM` recuperable |
| **Auditoría** | Log inmutable por job: tenant, hash de entrada, versión de modelo, quién, cuándo |
| **CORS** | Origen explícito en `/mcp`. Nunca `*` (corrige B15) |

---

## 6. Estructura de repositorio objetivo

```
tribe-centauro/
├── service/                      # NUEVO — protocolo-agnóstico
│   ├── api.py                    # AdOptimizationService (reemplaza AdOptimizer)
│   ├── jobs.py                   # máquina de estados
│   ├── store.py                  # SQLite → Postgres
│   ├── queue.py                  # cola (Redis/arq)
│   ├── worker.py                 # worker GPU
│   ├── cache.py                  # store content-addressed
│   ├── ingest.py                 # url|path|text → media normalizada
│   ├── errors.py                 # taxonomía
│   └── metrics/
│       ├── engagement.py         # unidades crudas, honesto
│       ├── roi.py                # atlas real (Schaefer/Destrieux)
│       ├── stats.py              # comparación corregida por clúster
│       └── calibration.py        # isotónica sobre feedback
├── mcp/                          # NUEVO — borde MCP
│   ├── server.py
│   ├── tools/{capabilities,validation,analysis,jobs,reporting,comparison,rendering,feedback}.py
│   ├── resources.py
│   ├── prompts.py
│   ├── transports.py             # stdio + streamable HTTP
│   ├── auth.py
│   └── schemas/*.schema.json     # fuente única de verdad
├── core/                         # REPARADO
│   ├── tribe_model.py            # una fuente por llamada; device en from_pretrained
│   └── ordering.py               # índice de vértices fsaverage5 [izq|der]
├── api/                          # adelgazado sobre service/
├── ads/
├── tests/
│   ├── contract/                 # sesión MCP real (initialize→tools/list→tools/call)
│   ├── service/
│   └── fixtures/                 # clips de 6 s, 15 s, mudo, sin voz
├── docker/                       # CPU y CUDA
├── pyproject.toml                # deps alineadas con tribev2
└── docs/
```

`requirements.txt` se reemplaza por `pyproject.toml` con las dependencias reales de Meta
(`neuralset==0.0.2`, `neuraltrain==0.0.2`, `exca==0.5.20`, `torch>=2.5.1,<2.7`, `numpy==2.2.6`,
`x_transformers==1.27.20`, …) y `tribev2` por URL git.

---

## 7. Fases

### Fase 0 — Reparar el core 🔴 BLOQUEANTE
Arregla API (`text_path`, una sola fuente), `from_pretrained(device=)`, ROIs con atlas real,
índices `[izq|der]`, elimina NES y umbrales inventados, serialización numpy→JSON nativa,
stats corregidas por clúster, elimina `main.py`. Extrae `service/`.
**Aceptación:** `pytest` verde; un clip real de 10 s produce `preds.shape == (10, 20484)`; los 5
puntos de entrada ejecutan sin excepción; `numpy.int64` ausente de todo payload.

### Fase 0.5 — Validación de supuestos del modelo
Clip de 6 s / 15 s / mudo / sin voz / solo texto. Fijar: comportamiento de `ChunkEvents` con
anuncios cortos, alineación hemodinámica real, si un anuncio mudo pierde eventos.
**Aceptación:** convención de alineación verificada empíricamente y tamaño de chunk observado
por duración. **Entregado:** `docs/FASE_0.5_VEREDICTO.md` (chunks y mudo) y
`docs/FASE_0.5_ALINEACION.md` (alineación, con `scripts/validate_alignment.py` reproducible).

### Fase 1 — MCP local (stdio)
`get_service_capabilities`, `get_platform_spec`, `validate_creative`, resources de specs/schemas/terms, prompts.
**Aceptación:** servidor arranca en stdio; un cliente MCP in-memory lista herramientas y ejecuta
`validate_creative`; `tools/list` valida contra el contrato; sin GPU requerida.

### Fase 2 — Jobs asíncronos + análisis
Cola, worker GPU, store, `analyze_creative`/`get_job`/`cancel_job`, report/timeline/editing, `compare_creatives`, `rank_creatives`.
**Aceptación:** dos `analyze_creative` con el mismo contenido devuelven el mismo `ad_id` y un solo
cómputo; `get_job` refleja transiciones de estado; `cancel_job` sobre encolado no ejecuta GPU;
el reporte respeta el límite de bytes; un fallo de modelo produce `MODEL_UNAVAILABLE` con `retryable`.

### Fase 3 — Servicio HTTP multi-tenant
Streamable HTTP, auth OAuth, tenancy, cuota, metering, `get_usage`, renderizado de mapas.
**Aceptación:** 401 sin token; un tenant no puede leer recursos de otro; exceder cuota da
`QUOTA_EXCEEDED`; dos réplicas del servidor MCP sin estado atienden el mismo job desde el store.

### Fase 4 — Loop de calibración
`submit_feedback`, almacén de outcomes, calibración isotónica, `calibrated: true` solo cuando la
muestra supera el umbral mínimo.
**Aceptación:** con N≥umbral de `(pred, outcome)` el reporte expone el mapeo calibrado y su
error (Spearman/MAE); por debajo, `calibrated: false`.

### Fase 5 — Endurecimiento y despliegue
SSRF, caps, prompt-injection, audit log, observabilidad (cola, p95, GPU), gate de licencia,
imágenes Docker CPU/CUDA, runbook.
**Aceptación:** checklist de seguridad ejecutado; `LICENSE_MODE=commercial` sin
`META_LICENSE_REF` **no arranca**; SSRF bloqueado contra casos de prueba (localhost, 169.254.169.254, redirect a privado).

### Alcance explícitamente fuera
Fine-tuning por demografía, integración con Meta/Google Ads API, dashboard React, auto-edición
de video. No se abordan hasta que la Fase 0–2 sea confiable.

---

## 8. Decisiones que necesito de ti

1. **Licencia** — CC-BY-NC-4.0 prohíbe uso comercial. ¿`research` ahora, o se tramita licencia Meta
   antes de exponer a agentes de terceros?
   *Recomendación:* arrancar en `research` con el gate activo; bloquea la Fase 3 externa.

2. **Métricas** — ¿aceptas exponer **unidades crudas sin calibración** (honesto, menos "vendible"),
   o prefieres retener un score 0–100 marcado `provisional: true`?
   *Recomendación:* crudas + relativo. Un score falso que un agente optimiza es deuda técnica y
   riesgo reputacional.

3. **Perfil de despliegue inicial** — stdio local (rápido, sin auth, para tus propios agentes)
   vs. HTTP hospedado (multi-tenant, requiere auth/cuota).
   *Recomendación:* Fase 1–2 en stdio; HTTP en Fase 3 cuando el core sea confiable.

4. **Atlas de ROI** — Schaefer 200 (robusto, bien documentado) vs. Destrieux (más fino, más ruidoso).
   *Recomendación:* Schaefer 200 por defecto, Destrieux como opción.

---

## 9. Por qué no portar directamente

| Si se porta hoy | Resultado |
|---|---|
| `predict_multimodal` | `ValueError` en el 100% de las llamadas |
| `analyze_ad` | `AttributeError` en el 100% de las llamadas |
| NES | El agente optimiza contra ruido escalado, con umbrales inventados |
| ROIs `slice(0,2000)` | Recomendaciones de "cortex visual" que son otra región anatómica |
| t-test aplanado | Falsos positivos: "significativo" para ruido correlacionado |
| `pytest tests/` | No cubre ninguno de los caminos rotos |
| Licencia MIT declarada | Riesgo legal en un servicio comercial |

**Orden correcto: reparar → extraer `service/` → exponer MCP → escalar.**
