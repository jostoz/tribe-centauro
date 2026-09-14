# Fase 0.5 — Veredicto go/no-go (validación empírica del producto)

**Fecha:** 2026-09-13 · **Ejecutado en:** GPU local (RTX, CUDA 12.4, torch 2.6.0+cu124)
**Script:** `scripts/validate_short_and_mute.py` (reproducible)

## Pregunta

El plan (`MCP_ADS_SERVICE_PLAN.md`) marcaba como **no verificado** el comportamiento del
pipeline TRIBE v2 con:

1. Anuncios cortos (6–15 s) frente a `ChunkEvents(max_duration=60, min_duration=30)`.
2. Anuncios mudos (sin voz) frente a `RemoveMissing()`.

Si el modelo descartaba clips cortos o degeneraba con silencio, **no había producto
publicitario**. Esto era el go/no-go.

## Hallazgo de código

`tribev2/demo_utils.py` invoca `ChunkEvents(event_type_to_chunk="Audio",
max_duration=60, min_duration=30)` **sin `event_type_to_use`**. En esa rama
(`neuralset/events/transforms/utils.py:370`), los eventos se trocean por
`max_duration`; `min_duration` solo evita crear sub-chunks internos menores a 30 s,
**nunca descarta un evento corto completo**. `RemoveMissing()` solo actúa en la ruta
de texto (`audio_only=False`), que depende de Llama-3.2 gated.

## Resultado empírico (ruta `audio_only`)

| caso       | preds        | n_segments | esperado | mean_abs | estado |
|------------|--------------|-----------:|---------:|---------:|--------|
| tono 6 s   | (6, 20484)   | 6          | 6        | 0.06580  | OK     |
| tono 10 s  | (10, 20484)  | 10         | 10       | 0.07121  | OK     |
| tono 15 s  | (15, 20484)  | 15         | 15       | 0.08667  | OK     |
| mudo 6 s   | (6, 20484)   | 6          | 6        | 0.06476  | OK     |
| mudo 10 s  | (10, 20484)  | 10         | 10       | 0.06778  | OK     |
| mudo 15 s  | (15, 20484)  | 15         | 15       | 0.07604  | OK     |

## Veredicto: **GO** (ruta audio)

- Clips cortos 6–15 s producen predicciones válidas; `n_segments` = duración exacta (1 TR/s).
- Anuncio mudo produce inferencia válida (no vacía, sin excepción).
- El tono se distingue del silencio por `mean_abs`.

## Caveats (obligan a más validación antes de vender)

1. **Discriminación audio-only modesta:** la diferencia tono↔mudo es de 1.5 % (6 s) a
   14 % (15 s) de la magnitud. La señal discriminante de la ruta solo-audio es limitada.
2. **Ruta de video sin validar:** los anuncios visuales llevan su señal en el extractor
   de video, no probado aquí. Es la siguiente validación crítica.
3. **Ruta de texto:** requiere token HuggingFace con acceso a Llama-3.2 (gated) + `gTTS`.
4. **Alineación hemodinámica: CERRADA.** `metadata.alignment = "stimulus-aligned"` —
   `preds[k]` es la respuesta al segundo `k` del estímulo; el offset de 5 s ya lo aplica
   el checkpoint. Verificado empíricamente sobre 4 anuncios reales con dos métodos
   independientes: `docs/FASE_0.5_ALINEACION.md`. Precisión de la localización absoluta:
   ≈ ±1.5 s (no sub-segundo).

## Siguiente paso

Validar la ruta de **video** con un `.mp4` real (extractor visual) antes de comprometer
el pitch de "screening de creativos visuales".
