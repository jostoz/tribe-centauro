# TRIBE Centauro

Análisis de activación neural de creativos publicitarios usando **TRIBE v2** (Meta FAIR).

Predice la respuesta fMRI de un anuncio (video, audio o texto) sobre la malla cortical
**fsaverage5** sin escáner, y expone las magnitudes resultantes por red funcional.

## Estado

| Componente | Estado |
|---|---|
| Inferencia TRIBE v2 (una fuente por llamada) | ✅ |
| ROIs anatómicas reales (Schaefer 2018, 7 redes) | ✅ |
| Alineación temporal verificada (Fase 0.5) | ✅ `docs/FASE_0.5_ALINEACION.md` |
| Métricas en unidades crudas, con procedencia | ✅ |
| Comparación A/B por permutación de bloques | ✅ |
| Edición guiada por feedback TRIBE | ⏳ evaluado — `docs/EVALUACION_EDICION_POR_FEEDBACK.md` |
| API REST (`/health`, `/validate`, `/analyze`, `/compare`, `/upload`) | ✅ |
| Servidor MCP para agentes | ⏳ Fase 1-2, ver `MCP_ADS_SERVICE_PLAN.md` |
| Calibración contra resultados de campaña | ⏳ Fase 4 |
| Jobs asíncronos / worker GPU | ⏳ Fase 2 |

## Aviso sobre las métricas

**No existe un "score de engagement" 0-100.** Todos los valores están en **unidades
crudas del modelo**, no están calibrados contra ningún resultado de campaña, y solo son
comparables **dentro del mismo lote** procesado por el mismo pipeline.

Cada respuesta incluye un bloque `provenance`:

```json
{
  "units": "raw_model_activation",
  "calibrated": false,
  "validity": "population-average, research-use-only",
  "comparability": "within-batch-only",
  "tr_seconds": 1.0,
  "hemodynamic_offset_seconds": 5.0
}
```

Un valor más alto **no implica mejor CTR**. Para comparar creativos, usa el endpoint de
comparación, que opera en relativo sobre las mismas condiciones.

### ROIs: por qué importan

Un `slice(0, 2000)` **no es una región cerebral**. El índice de ROI se construye con
máscaras booleanas del atlas **Schaefer 2018 (200 parcelas, 7 redes)** sobre la misma
malla que predice el modelo. Verificado: ese slice atraviesa las 7 redes y 100 de las
200 parcelas, y está mayoritariamente en la red Default, no en la visual.

Redes expuestas: `Vis`, `SomMot`, `DorsAttn`, `SalVentAttn`, `Limbic`, `Cont`, `Default`.

## Instalación

`tribev2` **no está en PyPI**; se instala desde git. Y fija `torch>=2.5.1,<2.7`.

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -e .
```

> ⚠️ Si tu entorno ya tiene `torch>=2.7` instalado, `tribev2` lo rechazará. Usa un
> entorno virtual dedicado para este proyecto.

Requiere además:
- **Token de HuggingFace con acceso a Llama-3.2** (gated) para las features de texto.
- **Red en tiempo de inferencia** si la entrada es texto: `gTTS` usa Google TTS.
- El atlas ya está incluido en `data/atlas/schaefer200/`.

## Uso

### Python

```python
from core.tribe_model import TribePredictor
from service.metrics.roi import RoiIndex
from service.metrics.engagement import summarize_activation, temporal_profile
from service.metrics.stats import compare_parcels

predictor = TribePredictor(device="auto")
preds, meta = predictor.predict(video_path="anuncio.mp4")
# preds.shape == (n_timesteps, 20484); 1 timestep = 1 s

roi = RoiIndex.from_schaefer("data/atlas/schaefer200")
print(summarize_activation(preds).as_dict())
print(roi.timeseries(preds, "Default").shape)   # (n_timesteps,)
```

Una sola fuente por llamada: `video_path` **o** `audio_path` **o** `text`.
El video ya incluye su pista de audio.

### API REST

```bash
python -m api.app
# http://127.0.0.1:8000/docs
```

```bash
curl -X POST http://127.0.0.1:8000/api/ads/analyze \
  -H "Content-Type: application/json" \
  -d '{"video_path": "anuncio.mp4", "platform": "instagram"}'
```

## Tests

```bash
python -m pytest tests/ -q
```

66 tests (65 pasan; 1 se salta si el `config.yaml` del checkpoint ya está normalizado). Cubren:
máscaras de ROI contra el atlas real, contrato de fuentes del modelo (con un doble, sin
GPU), contrato de la convención de alineación temporal, serialización JSON-safe, y una
demostración del falso positivo del t-test aplanado frente a la permutación por bloques.

## Convenciones del modelo (verificadas)

`TR = 1 s`. Orden de vértices `[hemisferio izquierdo 0..10241 | derecho 0..10241]`.

**Alineación temporal — cerrada.** `metadata.alignment = "stimulus-aligned"`:
`preds[k]` es la respuesta al **segundo `k`** del anuncio. El retardo hemodinámico de
5 s ya lo aplica el checkpoint al construir su objetivo de entrenamiento
(`estímulo(t) → BOLD(t+5)`); **no hay que restarlo** al leer `preds`. Verificado sobre 4
anuncios reales con dos métodos independientes (`edge = -1.48 + 1.023·m`, y
`envelope tracking` con lag mediano +0.5 TR). Precisión de la localización absoluta:
**≈ ±1.5 s**, no sub-segundo. Evidencia y reproducibilidad en
[`docs/FASE_0.5_ALINEACION.md`](docs/FASE_0.5_ALINEACION.md).

**También verificado** (ver [`docs/FASE_0.5_VEREDICTO.md`](docs/FASE_0.5_VEREDICTO.md)):
clips de 6–15 s frente a `ChunkEvents(min_duration=30)` no se descartan, y un anuncio mudo
produce inferencia válida.

**Sin verificar:** la potencia discriminante de la ruta de **video** (V-JEPA) y la de
**texto** (requiere token HF con Llama-3.2 gated + `gTTS`). La convención de alineación es
independiente de la ruta, porque el `offset` vive en el objetivo de fMRI compartido.

## Licencia

El código de este repositorio y los pesos de TRIBE v2 están bajo **CC-BY-NC-4.0**
(no comercial). El uso comercial requiere licencia explícita de Meta.

`LICENSE_MODE=commercial` exige `META_LICENSE_REF` y el servicio **no arranca** sin él.

## Estructura

```
core/     Inferencia TRIBE v2 y convenciones de la malla fsaverage5
service/  Lógica de dominio: ROIs, métricas, estadística, errores, serialización
api/      Adaptador REST (sin lógica de dominio)
ads/      Validación contra especificaciones de plataforma
mcp/      Servidor MCP para agentes — pendiente, ver MCP_ADS_SERVICE_PLAN.md
```

Ver [`MCP_ADS_SERVICE_PLAN.md`](MCP_ADS_SERVICE_PLAN.md) para el plan de exposición
como servicio MCP para agentes.
