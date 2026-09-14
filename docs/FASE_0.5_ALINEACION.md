# Fase 0.5 — Alineación temporal (cierre)

**Fecha:** 2026-09-13 · **Ejecutado en:** GPU local (RTX) · **Script:** `scripts/validate_alignment.py`
(reproducible, 26 s de punta a punta) · **Artefactos:** `data/discovery/alignment/`

---

## 1. La pregunta

Si un evento del estímulo ocurre en el segundo `t` del anuncio, ¿en qué índice `k` de
`preds` aparece su respuesta?

| hipótesis | significado | índice de la respuesta |
|---|---|---|
| **H-A** alineado al estímulo | `preds[k]` es la respuesta al segundo `k` | `k = t` |
| **H-B** BOLD crudo | `preds[k]` es el BOLD del instante `k` | `k = t + 5` |

## 2. Por qué el test obvio no sirve

Anteponer 5 s de silencio al anuncio desplaza la respuesta 5 s bajo **las dos**
hipótesis: `preds_B[k] ≈ preds_A[k-5]` tanto si el índice es tiempo de estímulo como si
es tiempo de BOLD. Es un test invariante a la convención. (El control `shift_5s` de
`docs/EVALUACION_EDICION_POR_FEEDBACK.md`, r = 0.898, demuestra estacionariedad — no
convención.)

La convención solo se puede fijar **refiriendo el índice a un onset del estímulo**, es
decir midiendo el retardo efectivo del sistema completo (features → BOLD predicho).

## 3. Prueba de código (determinista)

El `offset` no se aplica en inferencia: se aplica al construir el **objetivo de
entrenamiento**.

1. `cache/checkpoint/facebook__tribev2/config.yaml` → `data.neuro.offset = 5.0`,
   `frequency = 1.0`, `name = FmriExtractor`.
2. `neuralset/extractors/neuro.py:1300` — `FmriExtractor._get_timed_arrays` re-declara la
   serie de BOLD con `start = event.start − offset`. Con un registro que empieza en 0,
   la serie de BOLD crudo queda declarada como empezando en **−5 s**.
3. Leer de ahí la ventana de estímulo `[s, s+D]` devuelve las muestras de BOLD crudo
   `[s+5, s+5+D]`. Verificado con el propio accesor de la librería
   (`TimedArray.overlap`): `[0,3) → índices 5,6,7`; `[7,11) → índices 12,13,14,15`.

Por tanto el modelo aprende `estímulo(t) → BOLD(t+5)`. Como el BOLD en `t+5` es la
respuesta al evento neural del segundo `t` (la HRF pica ~5 s después), **`preds[k]` es la
respuesta al estímulo del segundo `k`**: H-A, con el retardo ya absorbido.

Fijado como test de contrato en `tests/test_alignment_convention.py` (no requiere GPU).

## 4. Medición empírica

### Método A — escalón "mutear desde `m`"

El operador deja el prefijo `[0, m)` byte-idéntico y silencia el resto. El cambio de
respuesta aparece como un escalón en `Δ(k) = mean_v |preds_variante(k,v) − preds_base(k,v)|`,
localizado por **máxima pendiente** (media post − media pre, ventana ±3 TR). Se ajusta
`edge = α + β·m`.

El estimador se **calibra** sobre las dos hipótesis simuladas (escalón convolucionado con
la HRF doble-gamma canónica): sobre datos sintéticos devuelve α = **0.00 TR** para H-A y
α = **+5.00 TR** para H-B. El veredicto no depende de umbrales elegidos a mano.

| anuncio | duración | `m` → `edge` |
|---|---|---|
| `ads/comercial.wav` | 20.1 s | 7→6, 10→9, 13→12 |
| `data/ads/-BrHpr8R9Yc.wav` | 30.6 s | 10→9, 15→14, 20→18 |
| `data/ads/BsMrRFH390k.wav` | 30.0 s | 10→9, 15→13, 20→19 |
| `data/ads/GH0unR4JF04.wav` | 60.1 s | 20→19, 30→29, 40→40 |

```
edge = -1.48 + 1.023·m      (n = 12, residuo RMS = 0.45 TR)
α observada  = -1.48 TR      α simulada H-A = 0.00   α simulada H-B = +5.00
distancia observada:  H-A = 1.48 TR   ·   H-B = 6.48 TR
```

**Trampa descartada.** El primer intento usó el punto medio acumulado (`cumsum` al 50 %).
Con una meseta que dura hasta el final del clip, ese estadístico queda sesgado hacia el
centro de la meseta y devolvía α = **+5.43** — es decir, el sesgo del estimador imita
exactamente la hipótesis H-B. Se descartó tras medirlo. Queda escrito aquí para que nadie
lo reintroduzca.

### Método B — `envelope tracking` (independiente)

Correlación cruzada entre la envolvente de audio (RMS por segundo) y el perfil temporal
de los 1000 vértices de mayor varianza, con desfase barrido en `[−3, +10]` TR.

| anuncio | mejor lag | r |
|---|---|---|
| `comercial` | +1 TR | 0.552 |
| `-BrHpr8R9Yc` | +2 TR | 0.493 |
| `BsMrRFH390k` | 0 TR | 0.567 |
| `GH0unR4JF04` | −2 TR | 0.613 |
| **mediana** | **+0.5 TR** | **0.559** |

H-A predice ≈ 0 TR; H-B predice ≈ +5 TR.

## 5. Veredicto

**`alignment = "stimulus-aligned"`** — H-A. Los dos métodos, uno calibrado y otro
independiente, coinciden. `preds[k]` se puede leer como el segundo `k` del anuncio.

**Precisión real: ≈ ±1.5 s.** El borde del escalón aparece ~1.5 TR *antes* de `m` en los
12 pares medidos (rango `edge − m` ∈ [−2, 0]). Causa no determinada; el candidato es la
ventana de contexto del extractor de audio, que ve audio posterior al TR predicho. Es un
sesgo de ~1.5 s sobre una rejilla de 1 TR: no cambia la conclusión (separa 0 s de 5 s),
pero **no** autoriza a afirmar precisión sub-segundo.

Consecuencia operativa: ya se puede localizar el segundo de un pico o un valle del perfil
y editar ese segundo. Lo que sigue sin estar validado es que ese pico signifique "mejor"
(`docs/EVALUACION_EDICION_POR_FEEDBACK.md`, P4).

## 6. Alcance

- **Ruta de audio** (`audio_only=True`): medida.
- **Rutas de video y texto:** no re-medidas. La convención es **independiente de la
  ruta**, porque el `offset` se aplica al objetivo de fMRI (compartido), no al extractor
  de estímulo. Si se quiere confirmar con video, el mismo script sirve cambiando el tipo
  de evento (el coste es la codificación V-JEPA, ~176 s/anuncio).
- El `offset` de 5 s **no se resta** en ningún consumidor. `metadata.alignment` lo declara;
  `api/routes.py` solo avisa si el valor deja de ser el verificado.

## 7. Reproducir

```bash
.venv/Scripts/python.exe scripts/validate_alignment.py
```

Requiere GPU (el script aborta sin ella: no hay verificación sin inferencia real). Escribe
`data/discovery/alignment/alignment.json` con los pares `(m, edge)`, las curvas Δ por TR,
la calibración simulada y los resultados de `envelope tracking`.
