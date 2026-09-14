# Grupo A — Resultados y límites (validación de constructo)

**Fecha:** 2026-09-13 · **Corpus:** 15 anuncios Telcel (14 con perfil neural) · **Hardware:** RTX 4090
**Datos:** `data/discovery/centauro.db` (reproducible con `python -m discovery.pipeline`)

> ## ⚠️ Superado en parte por A2
> Los contrastes de este informe estaban **infrapotenciados** (n=2 y n=3) y con el confusor
> ritmo↔duración. [`GRUPO_A2_RESULTADOS.md`](GRUPO_A2_RESULTADOS.md) los repite con 29 anuncios
> y control de duración: el contraste de **ritmo** pasa a ser robusto (DorsAttn +5.4 pp) y el
> **voz/música queda descartado con medición**. Las limitaciones de método de aquí siguen vigentes.

> Este informe es **exploratorio**. No hay test de significancia y los valores están en
> **unidades crudas del modelo, sin calibrar**. Se reportan diferencias descriptivas, no
> conclusiones.
>
> **Nota temporal:** este análisis **no depende de la alineación**. Todas las métricas son
> medias sobre los timesteps del anuncio (composición por red), no valores de un segundo
> concreto. La convención temporal del modelo ya está cerrada
> (`alignment = "stimulus-aligned"`, precisión ≈ ±1.5 s — `docs/FASE_0.5_ALINEACION.md`),
> así que cualquier lectura por segundo que se añada después debe citar esa precisión.

---

## 1. Corpus

Curado desde el canal oficial `@Telcel` y búsqueda "Telcel comercial anuncio", para cubrir los
4 contrastes del plan (caras, voz, emoción, ritmo). Duraciones 20–60 s.

| id | dur | caras | ritmo | tono | temas |
|---|---|---|---|---|---|
| `-BrHpr8R9Yc` | 31 | sí | lento | amistoso | interacción humana, ambiente urbano |
| `4i-a-DDFwXw` | 56 | — | rápido | urgente | realidad virtual, misión |
| `5Dr9yh_UoXQ` | 60 | — | rápido | energético | conexión, velocidad |
| `8p8BMaB61D0` | 31 | sí | lento | positivo | tecnología, cámara, zoom |
| `BsMrRFH390k` | 30 | sí | lento | positivo | comunicación, libertad |
| `Cb1Us4GC7Eg` | 20 | sí | rápido | positivo | conectividad, eficiencia |
| `GH0unR4JF04` | 60 | — | medio | energético | conectividad, tecnología |
| `HUWXb1g_ctY` | 20 | sí | rápido | positivo | (football) |
| `KSIa_IXLYY4` | 20 | sí | lento | positivo | comunicación, tecnología |
| `KgpstDcG-vM` | 22 | sí | rápido | energético | tecnología, movilidad |
| `KnffURmCqZU` | 53 | sí | lento | positivo/feliz | tecnología, fotografía, estilo |
| `Kumo70zjoa4` | 54 | sí | lento | positivo | tecnología, dispositivos, privacidad |
| `M0RLLJyPyZg` | 20 | sí | lento | feliz/alegre | Navidad, celebración, ofertas |
| `XJIt1wZKzzg` | 21 | sí | lento | feliz | navidad, celebración, dinero |
| `yiOqSxfr2Hs` | 30 | sí | lento | positivo | comunicación, vida cotidiana |

## 2. Perfil por red funcional (Schaefer 2018, 7 redes)

Activación media absoluta por red, normalizada a **% del total del anuncio** (composición relativa;
la magnitud absoluta no es comparable entre anuncios).

| id | Vis | SomMot | DorsAttn | SalVentAttn | Limbic | Cont | Default |
|---|---|---|---|---|---|---|---|
| `-BrHpr8R9Yc` | 29.9 | 17.7 | 13.8 | 10.6 | 5.9 | 8.8 | 13.3 |
| `4i-a-DDFwXw` | 37.8 | 8.6 | 21.0 | 12.0 | 3.6 | 10.6 | 6.5 |
| `8p8BMaB61D0` | 41.1 | 7.9 | 15.8 | 10.3 | 2.9 | 10.5 | 11.5 |
| `BsMrRFH390k` | 47.6 | 9.5 | 11.9 | 6.0 | 6.1 | 6.9 | 12.0 |
| `Cb1Us4GC7Eg` | 44.7 | 6.6 | 25.5 | 12.7 | 1.3 | 5.7 | 3.5 |
| `GH0unR4JF04` | 30.8 | 12.9 | 18.4 | 12.2 | 3.0 | 13.8 | 8.9 |
| `HUWXb1g_ctY` | 37.1 | 10.5 | 20.2 | 16.0 | 1.5 | 8.7 | 6.1 |
| `KSIa_IXLYY4` | 43.0 | 13.5 | 12.6 | 13.9 | 2.3 | 10.9 | 3.8 |
| `KgpstDcG-vM` | 35.7 | 7.8 | 22.2 | 14.5 | 2.5 | 8.7 | 8.5 |
| `KnffURmCqZU` | 34.1 | 9.8 | 20.7 | 10.0 | 2.5 | 12.4 | 10.5 |
| `Kumo70zjoa4` | 31.8 | 12.4 | 22.1 | 13.3 | 1.9 | 13.9 | 4.7 |
| `M0RLLJyPyZg` | 45.9 | 11.4 | 14.4 | 10.0 | 1.9 | 9.2 | 7.3 |
| `XJIt1wZKzzg` | 48.6 | 9.4 | 12.5 | 8.4 | 5.2 | 7.0 | 8.9 |
| `yiOqSxfr2Hs` | 42.3 | 11.7 | 12.3 | 10.3 | 6.5 | 9.1 | 7.8 |

**Observación estructural:** `Vis` domina en **todos** (30–49 %). El argmax de red **no
discrimina** entre creativos; el análisis útil es sobre el **perfil relativo**.

## 3. Contrastes (diferencias en puntos porcentuales)

### Ritmo: rápido (n=4) vs lento (n=9) — la señal más plausible

| red | rápido | lento | Δ |
|---|---|---|---|
| DorsAttn | 22.2 | 15.1 | **+7.1** |
| SalVentAttn | 13.8 | 10.3 | **+3.5** |
| SomMot | 8.4 | 11.5 | −3.1 |
| Default | 6.1 | 8.9 | −2.7 |
| Vis | 38.8 | 40.5 | −1.6 |
| Limbic | 2.2 | 3.9 | −1.7 |
| Cont | 8.4 | 9.9 | −1.4 |

**Direccionalmente consistente con la neurociencia esperada:** cortes rápidos → más atención
**dorsal** (orientación espacial) y de **saliencia**, y menos red **por defecto** (narrativa) y
sensorimotora. Es la única señal del lote con n razonable (4 vs 9) y dirección teórica previa.

### Caras (n=12) vs sin caras (n=2)

| red | con caras | sin caras | Δ |
|---|---|---|---|
| Vis | 40.1 | 34.3 | +5.9 |
| DorsAttn | 17.0 | 19.7 | −2.7 |
| Cont | 9.3 | 12.2 | −2.9 |

→ **No interpretable:** n=2 en el grupo sin caras.

### Emocional (n=3) vs no-emocional (n=11)

| red | emocional | no | Δ |
|---|---|---|---|
| Vis | 42.9 | 38.3 | +4.5 |
| SalVentAttn | 9.4 | 12.0 | −2.6 |

→ **Resultado en contra de lo esperado:** la red **Limbic** (valencia afectiva) queda **plana**
(3.2 vs 3.4). n=3 es insuficiente y la red Limbic pesa ~3 % del total, así que es ruidosa.

## 4. Veredicto honesto

- **NO queda establecida la validez de constructo.** Un contraste (ritmo) muestra una señal
  direccionalmente correcta; los otros dos están infrapotenciados o van en contra.
- Con n=15 y sin control de confusores (duración, marca, época del anuncio), **no se puede
  afirmar** que el modelo distinga creativos por su contenido.
- Lo que **sí** queda demostrado: el pipeline produce este análisis de forma reproducible y a
  bajo coste, y deja una hipótesis concreta y falsable que merece una prueba a escala.

## 5. Limitaciones detectadas (accionables)

1. **El contraste voz/música no existe en este corpus (medido, no supuesto).** Los 14 anuncios
   produjeron texto, y al medirlo con `no_speech_prob` de Whisper (probabilidad del token
   `<|nospeech|>` en el primer paso) **todos dan ≈ 0.000**: tienen locución. No es que Whisper
   alucine sobre música — es que **no hay anuncios sin voz** en esta muestra (los spots de marca
   casi siempre llevan VO). `discovery/transcribe.py` ya reporta `no_speech_prob` por anuncio,
   así que el atributo queda medido y disponible, pero **no sirve como contraste aquí**.
2. **`Vis` domina siempre:** hay que trabajar con perfiles normalizados, nunca con la red dominante.
3. **Red `Limbic` ~3 %** del total → cualquier métrica sobre ella será de alta varianza.
4. **Acentos en los campos del VLM** (`rápido` vs `rapido`): normalizar antes de agrupar.
5. **Confusor de duración:** los anuncios rápidos del lote son también los más cortos (20–22 s);
   ritmo y duración están confundidos. Hay que balancear ambas variables.

## 6. Siguiente paso para validar de verdad

- **≥10–15 anuncios por lado** en cada contraste (no 2 ni 3).
- **Controlar duración** dentro de cada grupo (o incluirla como covariable).
- Priorizar el contraste **ritmo** (el que mostró señal) y el de **caras**, que es el más limpio
  conceptualmente.
- **Descartar el contraste voz/música** en corpus de marca: medido `no_speech_prob ≈ 0` en
  todos. Solo tendría sentido con fuentes sin locución (demos de producto, contenido local).
- Correr A2 sobre los dos contrastes **con varianza real**: **ritmo** (el que mostró señal) y
  **personajes/caras**, controlando la duración como covariable.
- Reportar siempre el perfil normalizado de las 7 redes + `n`, nunca un "ganador".
