# Grupo A2 — Contrastes a escala con control de duración

**Fecha:** 2026-09-13 · **Corpus:** 30 anuncios Telcel (29 con perfil neural) · **GPU:** RTX 4090
**Reproducible:** `.venv/Scripts/python.exe scripts/analyze_contrasts.py`
**Antecede:** [`GRUPO_A_RESULTADOS.md`](GRUPO_A_RESULTADOS.md) (n=15, contrastes infrapotenciados)

> Exploratorio. **Sin test de significancia y sin corrección por comparaciones múltiples**
> (7 redes × 2 contrastes). Shares relativos en unidades crudas sin calibrar. Son marcadores
> de **dirección**, no de tamaño de efecto.

---

## 1. Qué cambió respecto a A1

| | A1 | A2 |
|---|---|---|
| Anuncios con perfil neural | 14 | **29** |
| Ritmo rápido / lento | 4 / 9 | **8 / 20** |
| Sin caras | **2** (ininterpretable) | **8** |
| Confusor ritmo↔duración | **presente** (los rápidos eran los cortos) | **controlado** (rápidos ahora 20–60 s; medianas 36 s vs 30 s) |
| Contraste voz/música | planeado | **descartado con medición** (ver §4) |

## 2. Método

Composición relativa de las 7 redes Schaefer por anuncio (activación media absoluta por red,
normalizada al total del anuncio). Como la duración puede confundir, además de las medias por
grupo se reporta el **coeficiente del grupo ajustado por duración** (regresión `share ~ dummy + duración`
por mínimos cuadrados, con la duración como covariable).

## 3. Resultados

### Ritmo: rápido (n=8) vs lento (n=20) — **la señal que se sostiene**

| red | rápido | lento | Δ |
|---|---|---|---|
| **DorsAttn** | **21.6 ± 2.1** | **16.2 ± 4.3** | **+5.4** |
| Vis | 40.3 | 37.3 | +3.0 |
| SalVentAttn | 12.0 | 11.4 | +0.6 |
| Cont | 9.1 | 10.2 | −1.1 |
| Limbic | 2.3 | 4.1 | −1.7 |
| Default | 6.2 | 9.0 | −2.8 |
| SomMot | 8.4 | 11.8 | −3.4 |

- **Sobrevive al ajuste por duración: DorsAttn +5.3 pp** (vs +5.4 sin ajustar) → el efecto **no** lo
  produce la duración.
- **Separación real:** Δ 5.4 pp frente a una desviación de **2.1 pp** en el grupo rápido. Los grupos
  no se solapan por media ± SD.
- **Dirección esperada:** más atención **dorsal** en anuncios de corte rápido, y menos red
  por defecto/sensorimotora. Coherente con la neurociencia de la atención.

### Caras: con caras (n=21) vs sin caras (n=8) — **Δ grande pero no concluyente**

| red | con caras | sin caras | Δ |
|---|---|---|---|
| Vis | 40.4 ± 6.5 | 31.3 ± 12.7 | +9.1 |
| DorsAttn | 18.3 | 16.3 | +2.0 |
| SalVentAttn | 11.2 | 12.6 | −1.4 |
| Limbic | 3.0 | 4.9 | −1.9 |
| SomMot | 10.3 | 12.5 | −2.2 |
| Cont | 9.3 | 12.0 | −2.7 |
| Default | 7.4 | 10.4 | −3.0 |

- Ajustado por duración: **Vis +8.6 pp** (se mantiene).
- **Pero el grupo sin caras es muy disperso: SD 12.7 pp con n=8.** El Δ (9.1) es **menor que una
  desviación** de ese grupo → **no concluyente**. Necesita más anuncios sin caras antes de afirmar nada.
- Nota: los anuncios sin caras son además **más largos** (mediana 43 s vs 31 s), en dirección
  **opuesta** al efecto; el ajuste lo compensa, pero el grupo sigue siendo pequeño y heterogéneo.

## 4. Contraste descartado con medición: voz vs música

Se implementó un detector basado en el token `<|nospeech|>` de Whisper
(`discovery/transcribe.py`). Resultado sobre los **30/30** anuncios: `no_speech_prob ≈ 0.000`
en todos → **todos tienen locución**. No es que el ASR alucine sobre música: **no hay anuncios
sin voz en este corpus**, así que el contraste no existe. El atributo queda medido y disponible
para corpus futuros (requiere fuentes sin locución).

## 5. Veredicto honesto

- **Un contraste pasa a ser defendible:** rápido vs lento → **DorsAttn +5.4 pp**, robusto al
  control de duración y con separación mayor que la dispersión del grupo. Sigue siendo
  **exploratorio** (sin test formal, sin replicar en otra marca).
- **El otro queda abierto:** caras → Vis tiene un Δ atractivo pero su grupo control es demasiado
  disperso y pequeño.
- Lo que **no** cambia: `Vis` domina siempre (30–49 %), así que el análisis útil es la
  composición relativa; y **ningún** resultado autoriza a decir que un anuncio "funciona mejor".

## 6. Límites (además de los de A1)

1. **Predictor con error de medida:** `ritmo` y `hay_caras` son juicios del VLM sobre 12 frames,
   no anotación humana. Un predictor ruidoso **atenúa** los efectos (sesgo hacia cero) — los Δ
   observados son, si acaso, conservadores.
2. **Una sola marca (Telcel):** todos comparten identidad de marca, logo y tono; falta replicar
   en otra categoría.
3. **Sin test de significancia ni multiplicidad:** 14 comparaciones; el "mejor" hallazgo podría
   ser el más favorecido por el azar. Es el siguiente paso formal.
4. **Shares, no magnitudes:** un cambio de share puede deberse a cambios en las otras redes.

## 7. Siguiente paso

1. **Test formal con corrección por multiplicidad** (permutación por bloques ya disponible en
   `service/metrics/stats.py`) y **n≥20 por lado** en el contraste de caras.
2. **Replicar en una segunda marca/categoría** para descartar que sea un artefacto de Telcel.
3. Solo entonces tiene sentido hablar de señal de producto — y la **calibración** (resultado real)
   sigue siendo el paso que convierte esto en algo vendible.
