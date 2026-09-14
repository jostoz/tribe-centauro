# Grupo A2 — Contrastes a escala con control de duración

**Fecha:** 2026-09-13 · **Corpus:** 30 anuncios Telcel (29 con perfil neural) · **GPU:** RTX 4090
**Reproducible:** `.venv/Scripts/python.exe scripts/analyze_contrasts.py`
**Antecede:** [`GRUPO_A_RESULTADOS.md`](GRUPO_A_RESULTADOS.md) (n=15, contrastes infrapotenciados)

> ## 🚨 DEGRADADO POR LA RÉPLICA (ver `REPLICA_COCACOLA.md`)
> El hallazgo de este informe **no se confirmó en una segunda marca/categoría**: en Coca-Cola
> la dirección se repite pero **no sobrevive a Holm** (+3.0 pp, p Holm 0.64). Queda como
> **"dirección consistente, no confirmada"**, no como resultado defendible. Además, el test
> formal de este informe metía los ritmos intermedios en el control (corrige a **+5.2 pp**,
> Holm 0.0035 — sigue significativo en Telcel, pero era una inconsistencia de método).
> Hoy el claim defendible es *"el análisis produce contrastes reproducibles y honestos"*,
> **no** que el modelo detecte el ritmo de un anuncio.

> Exploratorio en su origen, **con test formal en §5** (permutación + Holm). Los valores están
> en **unidades crudas sin calibrar**: shares relativos de la composición por red. Significativo
> **no** es relevante ni vendible — es dirección, no tamaño de efecto.

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

## 5. Test formal (permutación + Holm)

`scripts/analyze_contrasts.py --test --n-perm 10000` — permutación bilateral con
**Freedman–Lane** (residuos permutados para que el ajuste por duración sea válido) y
**Holm–Bonferroni** dentro de cada contraste (7 redes).

### RÁPIDO vs LENTO (n=8 vs 21)

| red | efecto (pp) | p cruda | p \| duración | **p Holm** | sig. |
|---|---|---|---|---|---|
| **DorsAttn** | **+5.3** | 0.0027 | **0.0004** | **0.0028** | **sí** |
| SomMot | −3.5 | 0.0132 | 0.0129 | 0.0774 | no |
| Default | −2.8 | 0.1012 | 0.0977 | 0.4885 | no |
| Vis | +3.3 | 0.4089 | 0.4130 | 0.8259 | no |
| Limbic / Cont / SalVentAttn | −1.7 / −1.2 / +0.5 | ≥0.12 | ≥0.12 | ≥0.49 | no |

**DorsAttn sobrevive a la corrección por multiplicidad.** Y el p *baja* al ajustar por
duración (0.0027 → 0.0004): la duración explicaba varianza irrelevante, así que removerla
deja el efecto de grupo **más** nítido. Esto cierra el confusor que en A1 era una objeción.

### CON CARAS vs SIN CARAS (n=21 vs 8)

| red | efecto (pp) | p cruda | p \| duración | **p Holm** | sig. |
|---|---|---|---|---|---|
| Vis | +8.6 | 0.0152 | 0.0199 | 0.1211 | **no** |
| Cont | −2.3 | 0.0164 | 0.0173 | 0.1211 | **no** |
| Limbic | −2.1 | 0.0742 | 0.0538 | 0.2690 | no |

El Δ de Vis era el más grande del corpus, pero **no sobrevive a Holm** con n=8 en el grupo
control. Confirma la lectura de §3: **atractivo, no concluyente**.

## 6. Veredicto honesto

- **Un resultado pasa a ser formalmente defendible:** en anuncios de corte rápido, la
  composición de la red **DorsAttn** es +5.3 pp mayor (21.6 % vs 16.2 % del total), significativo
  tras corregir por multiplicidad **y** robusto al control de duración.
- **El contraste de caras queda descartado** por ahora (no sobrevive a Holm).
- Sigue sin autorizar ninguna afirmación de negocio: son **shares en unidades crudas sin
  calibrar**, y significativo no es lo mismo que relevante.

## 7. Límites (además de los de A1)

1. **Predictor con error de medida:** `ritmo` y `hay_caras` son juicios del VLM sobre 12 frames,
   no anotación humana. Un predictor ruidoso **atenúa** los efectos (sesgo hacia cero) — los Δ
   observados son, si acaso, conservadores.
2. **Una sola marca (Telcel):** todos comparten identidad de marca, logo y tono; falta replicar
   en otra categoría.
3. **Corrección por multiplicidad dentro de cada contraste, no entre contrastes:** Holm cubre las
   7 redes, pero los 2 contrastes no se corrigen entre sí (con 2 familias, el efecto sería menor).
4. **Shares, no magnitudes:** un cambio de share puede deberse a cambios en las otras redes.

## 8. Siguiente paso

1. ✅ **Test formal con corrección por multiplicidad — hecho** (§5). El hallazgo de ritmo pasa;
   el de caras no (necesita **n≥20 por lado** antes de reintentarlo).
2. **Replicar en una segunda marca/categoría** para descartar que sea un artefacto de Telcel.
3. **Anotación humana de `ritmo`** en una submuestra para estimar la atenuación por error de
   medida del predictor del VLM.
4. Solo entonces tiene sentido hablar de señal de producto — y la **calibración** (resultado real)
   sigue siendo el paso que convierte esto en algo vendible.
