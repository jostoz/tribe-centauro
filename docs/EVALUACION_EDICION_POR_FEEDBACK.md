# Evaluación — editar contenido a partir del feedback de TRIBE v2

**Fecha:** 2026-09-13 · **GPU:** RTX 4090 local (torch 2.6.0+cu124) · **Ruta:** `audio_only=True`
**Script reproducible:** `scripts/probe_edit_feedback.py` (P1–P7)
**Complementos:** `scripts/probe_wait_corpus.py` (corpus real), `scripts/probe_video_prefix.py` (canal video)
**Artefactos:** `data/discovery/probe/probe.json` + `preds_*.npy` (11 variantes), `probe/video/video_prefix.json`, `probe/wait_corpus.json`
**Fuente:** `ads/comercial.wav` (20.12 s, 44.1 kHz → 16 kHz mono)

---

## 1. Pregunta

¿Se puede construir un lazo **editar → medir → reeditar** con TRIBE v2 como señal de
control? Eso exige cuatro cosas que el repo **no tenía verificadas**:

| # | Requisito | Por qué es bloqueante |
|---|---|---|
| P1 | La misma entrada da la misma salida (suelo de ruido) | sin esto ningún delta pequeño es interpretable |
| P2 | Los operadores de edición (cortar, mutear) mueven la señal por encima del ruido | Fase 0.5 avisó: ruta audio discrimina poco (1.5–14 %) |
| P3 | El perfil es **local en el tiempo**: cortar [t1,t2) desplaza el resto | si el perfil está globalmente acoplado, "corta el segundo malo" no existe |
| P4 | Existe un objetivo **no explotable** | si un edit degenerado lo maximiza, el lazo optimiza basura |

## 2. Veredicto en una línea

**P1 ✅ exacto · P2 ✅ de sobra · P3 ✅ local · P4 ❌ ningún objetivo naive aguanta.**
→ El lazo es viable como **asistente de edición con humano en el loop y comparación
relativa intra-lote**; **no** es viable como maximizador automático de una métrica.
El retardo absoluto estímulo↔TR quedó **resuelto** el 2026-09-13: `alignment =
"stimulus-aligned"`, `preds[k]` ↔ segundo `k` del estímulo, con precisión ≈ ±1.5 s
(`docs/FASE_0.5_ALINEACION.md`). La localización del segundo a cortar está habilitada
con esa tolerancia.

---

## 3. Resultados medidos

### P1 · Suelo de ruido — determinismo exacto

| comparación | max \|Δ\| | mean \|Δ\| | Δ relativo | r(perfil) |
|---|---|---|---|---|
| `base` vs `base` (2ª corrida) | **0.000e+00** | **0.000e+00** | 0.000 % | **1.0000** |

Dos corridas del mismo archivo dan predicciones **bit a bit idénticas** (y la 2ª tarda
0.24 s por caché de features). Consecuencia: **todo delta observado es señal, no ruido**.
El `snr_vs_noise` del JSON es ∞; no lo reportes como número.

### P2 · Sensibilidad a operadores

| variante | dur | shape | mean\|a\| | Δrel vs base | r(perfil) vs base | t |
|---|---|---|---|---|---|---|
| `base` | 20.12 s | (21, 20484) | 0.103592 | — | 1.0000 | 0.70 s |
| `mute` (silencio total) | 20.12 s | (21, 20484) | 0.077208 | **114 %** | 0.5999 | 0.38 s |
| `cut_8_12` (quita 4 s) | 16.12 s | (17, 20484) | 0.103515 | **50 %** | 0.4267 | 0.35 s |
| `shift_5s` (antepone 5 s de silencio) | 25.12 s | (26, 20484) | 0.139312 | **123 %** | −0.1850 | 0.34 s |

`n_seg` = duración redondeada en todas las variantes: **1 TR = 1 s se sostiene al editar**.

### P3 · Atribución temporal — el perfil es local

| prueba | r | lectura |
|---|---|---|
| prefijo antes del corte (`t<8` vs base) | **0.8825** | el corte no altera lo anterior |
| cola tras el corte, **desplazada** 4 s | **0.9068** | hipótesis local ✅ |
| cola tras el corte, **sin desplazar** | −0.0355 | hipótesis global ❌ |
| control: cuerpo de `shift_5s` desplazado 5 s vs base | 0.8979 | el desplazamiento puro también se conserva |

Quitar 4 s de contenido quita 4 timesteps del perfil y **el resto se corre hacia atrás
manteniendo su forma** (r=0.91). Esto es exactamente el requisito de "corta el segundo
malo": la línea de tiempo se conserva y es direccionable.

### P4 · Explotabilidad del objetivo — el hallazgo duro

Variantes degeneradas: `strobe` (1 s de contenido + 1 s de silencio en bucle: **no mejora
el creativo, solo multiplica los onsets**), `shift_5s` (antepone silencio), `repeat_peak`
(tilea el segundo de mayor activación, t\*=5 s).

| objetivo | base | mute | cut_8_12 | shift_5s | **strobe** | repeat_peak | ¿explotado por? |
|---|---|---|---|---|---|---|---|
| `mean_abs_all` | 0.1036 | 0.0772 | 0.1035 | 0.1393 | **0.1917** | 0.0743 | **strobe** |
| `mean_abs_body` | 0.1022 | 0.0785 | 0.0968 | 0.1223 | **0.1955** | 0.0799 | **strobe** |
| `SalVentAttn_body` | 0.0586 | 0.0238 | 0.0569 | 0.0618 | **0.0831** | 0.0417 | **strobe** |
| `DorsAttn_body` | **0.1087** | 0.0087 | 0.0943 | 0.1072 | 0.0960 | 0.0296 | no |
| `Default_body` | 0.0215 | 0.1260 | 0.0334 | 0.0536 | **0.2539** | 0.1242 | **strobe** |
| `Vis_body` | 0.1922 | 0.0221 | 0.1648 | **0.1974** | 0.1498 | 0.0416 | **shift_5s** |
| `peak_count_body` | 1 | 0 | 1 | 0 | **3** | 0 | **strobe** |

6 de 7 objetivos los gana un edit que **destruye el creativo**. Detalles que lo explican:

- **Transitorio de onset.** Anteponer 5 s de silencio sube la activación de esos
  timesteps iniciales a **0.2106** vs **0.1223** del cuerpo, y los picos del perfil caen
  en t=0,1,2. El clip "empieza" con activación alta pase lo que pase → `strobe` con 10
  onsets gana trivialmente.
- **Mutear sube `Default` ×5.9** (0.0215 → 0.1260) mientras `Vis` (0.1922 → 0.0221) y
  `DorsAttn` (0.1087 → 0.0087) colapsan: la composición por redes cambia de forma
  no trivial con el silencio. Un objetivo de una sola red es trivialmente movible.
- **Tilerar el pico no ayuda** (0.0743 < 0.1036): el exploit no es repetir contenido, es
  **fabricar onsets**.

### P5 · Retardo absoluto estímulo↔TR — resuelto el 2026-09-13

1. **Barrido periódico (`strobe`), ground truth contenido-par/silencio-impar:** la
   correlación **alterna signo en cada TR** (+0.78 … +0.80 en lags impares, −0.79 en
   pares). Prueba que la respuesta está **anclada al estímulo local dentro de ~1 TR**
   (no hay mezcla temporal), pero un diseño periódico **aliasa**: no puede dar el lag.
   El `mejor_lag = 7` del JSON es artefacto de borde (solo 14 muestras), no un resultado.
2. **Pulso aislado** (2 s de contenido en t=10 sobre silencio): perfil
   `0.157 0.189 0.222 0.213 0.226 0.210 0.208 0.177 0.131 0.156 0.099 0.124 0.201 0.205 0.223 0.200 0.213 0.195 0.172 0.176 0.155`
   → **no hay bump limpio**: el pico global está en t=4 (prior de onset, 0.226), hay un
   valle en t=10–11 (0.099/0.124, justo en el pulso) y un levantamiento en t=12–15
   (0.201–0.223), compatible con un retardo de **+2 a +4 s**, no concluyente.
3. **Dos pulsos en el mismo clip** (t=5 y t=15): `r(pulso@15, pulso@5) = −0.090`, y la
   diferencia muestra lóbulos de ±0.09 separados 10 s (las dos posiciones), con
   `r(diff, perfil_base) = 0.774`. Es decir: la señal contiene un componente **clavado al
   contenido** y otro **grande dependiente de la posición/onset**; el diseño no los separa.

**Conclusión P5 (cerrada el 2026-09-13):** el offset declarado de 5 s **ya está
compensado** por el checkpoint: `preds[k]` es la respuesta al segundo `k` del estímulo
(`alignment = "stimulus-aligned"`). La calibración dedicada de §7 Fase A se ejecutó
(`scripts/validate_alignment.py`; evidencia en `docs/FASE_0.5_ALINEACION.md`): sobre 4
anuncios y 12 escalones, `edge = -1.48 + 1.023·m`, frente a α = 0 TR (alineado) y α = +5 TR
(BOLD crudo) que devuelve el mismo estimador sobre las dos hipótesis simuladas — la
observada queda a 1.5 TR de la primera y a 6.5 TR de la segunda. Precisión absoluta
≈ **±1.5 s**, no ±1 TR.

Los tres diseños de esta sección quedaron cortos por el mismo motivo: miran la **forma** del
perfil (pico/bump) en vez de la **posición de un borde**, y `mean|a|` responde con polaridad
no obvia (silencio puede dar más `mean|a|` que contenido). Dato que ya estaba en la tabla del
pulso aislado y apunta a lo mismo: el perfil tiene su mínimo en t=10–11, **co-localizado con
los 2 s de pulso de t=10**, no desplazado 5 s — bajo H-B el mínimo habría caído en t=15–16.
El pico en t=4 (prior de onset) **no** se usa como evidencia: es un artefacto de inicio de
clip de convención temporal no establecida.

---

## 4. Eje A — Herramientas de edición

| herramienta | estado | operadores que habilita | veredicto |
|---|---|---|---|
| **moviepy 2.1.2** | **ya es dependencia** | `subclipped`, `without_audio`, `concatenate`, `with_speed_scaled`, `resized`/crop, `TextClip` | ✅ base del lazo; cero instalaciones nuevas |
| **imageio-ffmpeg 7.1** (bundle de moviepy) | ya instalada | mux/demux exacto, `-ss/-t`, streams, `-vf` | ✅ para cortes sin recodificar y para evitar el overhead de moviepy en lotes |
| **PySceneDetect** | nueva dep (CPU, sin GPU) | fronteras de plano (`ContentDetector`) | ✅ **necesaria**: un corte que no caiga en frontera de plano es un salto visible; los candidatos deben ser uniones de planos |
| `auto-editor` | nueva dep (CLI) | corte por silencio | ⚠️ utilidad marginal; produce justo el patrón de onsets que explota los objetivos |
| difusión de video (Runway/AnimateDiff/…), Blender, Remotion | — | generación / composición pesada | ❌ **descartadas**: compiten por VRAM con TRIBE (riesgo WinError 1455) y no son "editar", son "generar" |

**Consecuencia de arquitectura (lo importante):** el costo del lazo **no** está en editar,
está en **re-extraer features**. En la ruta de video, la codificación V-JEPA domina
(~1.8 s/frame-batch; anuncio de 60 s ≈ 4 min ≈ los 176 s/anuncio medidos en
`PLAN_ESCALABILIDAD.md`). La caché actual de neuralset
(`cache/neuralset.extractors.video.HuggingFaceVideo._get_data,release/`) está **indexada
por estímulo completo** → cada variante editada es un *cache miss* y cuesta ~176 s.
Los operadores **cortar / reordenar / mutear no cambian los frames**: con una caché de
features **por frame** (clave = hash del frame + parámetros del extractor) el costo
marginal de un candidato cae de ~176 s a **segundos** (en la ruta audio medido: 0.34–4.7 s
por variante). **Esa caché por frame es el habilitador económico del lazo, no la GPU.**

## 5. Eje B — Skills aplicables (mapeo al lazo)

| etapa del lazo | skill existente | qué aporta exactamente |
|---|---|---|
| toda inferencia TRIBE | `centauro-gpu-inference` | `num_workers=0`, una inferencia a la vez, verificación por VRAM (no por util%) |
| barrido de N variantes | `llm-eval-runner` | progreso por caso, resultados reanudables, log durable: el barrido de variantes es exactamente un eval runner |
| inferencia larga / muchos candidatos | `gpu-evaluation-safety` | PID objetivo, piso de VRAM, watchdog que solo mata ese PID, smoke test antes del lote |
| editor-VLM (M3) propone edits | `structured-output-contracts` | plan de edición como JSON validado; separar fallo de esquema de fallo de generación |
| adjudicación final | `eval-gate-review` | revisión humana **ciega** de pares: obligatoria, porque las métricas no están calibradas |

**Hueco detectado:** falta una skill `centauro-edit-loop` que codifique las reglas duras
descubiertas aquí (descartar los primeros ~5 TR, prohibido usar medias globales como
objetivo, todo candidato pasa por `paired_block_permutation`, nunca dos inferencias
concurrentes). Sin ella cada agente vuelve a caer en `mean_abs`.

## 6. Eje C — Métodos

| método | cómo | costo por candidato | veredicto |
|---|---|---|---|
| **M1 · búsqueda sobre operadores discretos** (cortar plano, reordenar, mutear tramo, cambiar velocidad, recortar encuadre) con features cacheadas y **objetivo restringido** (body, red única, sin onsets) | enumerar/greedy sobre un presupuesto de ~50–200 candidatos | audio 0.4 s; video con caché por frame ~segundos | ✅ **recomendado**: interpretable, cae en edits reales |
| **M2 · gradiente sobre píxeles/muestra** (TRIBE es torch: `∂pred/∂frame`) | optimizar entrada con pérdida = objetivo | alto (backward por frame + V-JEPA) | ❌ **descartado**: produce **adversariales** (ruido imperceptible que sube la métrica), no ediciones; Goodhart garantizado; además choca con la VRAM |
| **M3 · editor-VLM en el loop** (Qwen2.5-VL-7B ya residente en `discovery/understand.py`) que recibe el informe TRIBE (perfil + picos + valles + redes) y **propone un plan de edición** en JSON | structured output + validación | 1 inferencia VLM ≈ 2–3 s | ✅ **recomendado como generador de candidatos**, nunca como juez |
| **M4 · recombinación de planos** (pool de candidatos entre variantes/creativos, elegir las ventanas que mejor rankean) | retrieval + reensamblado | ídem M1 | ⚠️ bueno para A/B de variantes ya existentes; riesgo de "collage" incoherente sin VLM que valide narrativa |
| **M5 · ranking A/B honesto** con `compare_parcels` (permutación por bloques + IC bootstrap, **ya implementado**) + adjudicación humana ciega | par a par | ~0 (reusa preds) | ✅ **obligatorio como juez**: es el único juez que el repo ya declara válido (`within-batch-only`, `calibrated: false`) |

**Regla de oro resultante:** el VLM **propone**, TRIBE **mide**, la permutación por bloques
**compara**, el humano **decide**. Nada de "maximizar el score".

## 7. Plan por fases con criterios de aceptación

**Fase A — calibración del retardo (bloqueante, barata). ✅ HECHA (2026-09-13).**
El diseño aquí planificado (3 pulsos del mismo contenido) es **invariante a la convención**:
desplaza la respuesta 5 s bajo las dos hipótesis, así que no puede decidirla. Se sustituyó
por escalones "mutear desde `m`" con detector de borde por máxima pendiente y **calibración
del propio estimador** sobre las dos hipótesis simuladas (`scripts/validate_alignment.py`).
**Resultado:** `alignment = "stimulus-aligned"`; α observada −1.48 TR contra 0 (H-A) y +5
(H-B) simuladas; `envelope tracking` independiente con lag mediano +0.5 TR.
"corta el segundo k" queda habilitado con tolerancia ≈ **±1.5 s** (no ±1 TR).
Detalle en `docs/FASE_0.5_ALINEACION.md`.

**Fase B — operadores + validación de explotabilidad (media).**
Sobre 1 creativo real: 12–20 candidatos generados con M1 (solo operadores que preservan
narrativa), medidos con la ruta de video, ranqueados con M5. Criterios: (i) el ranking
sobrevive a excluir los primeros 5 TR y al control `strobe` como candidato **negativo**
(debe quedar último, no primero); (ii) `n_seg` = duración en el 100 % de los candidatos;
(iii) acuerdo humano↔ranking ≥ 60 % en 20 pares ciegos.

**Fase C — editor-VLM (media).** M3 emitiendo `{operator, target, rationale}` validado por
esquema; se mide si sus propuestas baten a candidatos aleatorios sobre el mismo
presupuesto. Criterio: ≥ 1.5× la tasa de "mejora aceptada por humano" de la línea base
aleatoria.

**Fase D — lazo completo con puerta humana.** Integrar en `MCP_ADS_SERVICE_PLAN.md`
(`analyze_creative` + `propose_edits` + `compare_creatives`), con la skill
`centauro-edit-loop` y la puerta `eval-gate-review` antes de cualquier entrega al cliente.

## 8. Riesgos

| riesgo | evidencia | mitigación |
|---|---|---|
| **Goodhart**: optimizar la métrica destruye el creativo | 6/7 objetivos los gana `strobe` (§P4) | objetivo restringido + candidato negativo obligatorio + humano en la puerta |
| Objetivo trivialmente movible por silencio | `mute` sube `Default` ×5.9 y hunde `Vis` ×0.11 | nunca una sola red; nunca `mean_abs` global; reportar composición, no un escalar |
| Prior de onset contamina cualquier métrica de "hook" | silencio inicial = 0.2106 vs cuerpo 0.1223 | descartar primeros ~5 TR; normalizar por nº de onsets |
| Localización absoluta errónea | P5 no resuelve el lag | Fase A antes de prometer "corta el segundo k" |
| Coste del lazo en video | ~176 s por variante sin caché por frame | caché de features por frame (habilitador, §Eje A) |
| Licencia CC-BY-NC | `README.md` | solo investigación; sin cobro hasta licencia de Meta |
| VRAM / WinError 1455 | `PLAN_ESCALABILIDAD.md` §7 | una inferencia a la vez; nunca generación de video en la misma GPU |

## 9. Qué NO se puede afirmar todavía

- Que subir un objetivo de TRIBE **mejore el CTR**: nada está calibrado (§4 de Fase 4).
- Que el segundo k del perfil sea el segundo k del anuncio: **resuelto** —
  `alignment = "stimulus-aligned"` con precisión ≈ ±1.5 s (`docs/FASE_0.5_ALINEACION.md`).
  Lo que sigue abierto es que un pico signifique "mejor" (P4/P7), no dónde cae.
- Qué operador concreto gana en un creativo real: P4 se midió sobre **1** creativo y la
  ruta **audio**; la ruta video sigue sin este barrido (Fase B lo cierra).

---

# Addendum (misma sesión) — qué premia exactamente el modelo

Refina §3-P4. El "transitorio de onset" describía mal el fenómeno.

## 10.1 La métrica premia el **prefijo sin información**, no el contenido

### Canal audio — barrido de espera (P6, `probe_edit_feedback.py`)

| espera antes del contenido | activación t=0–2 | media global |
|---|---|---|
| 0 s (original) | 0.096 | 0.1036 |
| 1 s | 0.119 | 0.1168 |
| 3 s | 0.186 | 0.1268 |
| 5 s | 0.252 | 0.1393 |
| **8 s** | **0.306** | **0.1735 (+67 %)** |

Monótono y sin techo hasta 8 s.

### Canal audio — ¿basta con estar "quieto"? (P7) **No**

| apertura de 3 s | rms | activación t=0–2 | media |
|---|---|---|---|
| **ceros digitales** | 0.00000 | **0.186** | 0.127 |
| ruido de sala (−50 dBFS) | 0.00300 | 0.103 | 0.111 |
| ruido audible bajo | 0.02999 | 0.093 | 0.112 |
| (sin apertura: original) | — | 0.096 | 0.104 |

**−50 dBFS de aire bastan para matar el efecto.** No es "silencio/quietud": es
**ausencia literal de señal**.

### Canal video (`scripts/probe_video_prefix.py`, 3 s de negro + los mismos 12 s)

| variante | t=0–2 | media global | cuerpo (los mismos 12 s) |
|---|---|---|---|
| `v_base` | 0.125 | 0.128 | 0.128 |
| `v_negro` (+3 s negro y silencio) | 0.264 | 0.231 | **0.222 (+74 %)** |
| `v_negro_ruido` (+3 s negro y ruido −50 dBFS) | 0.250 | 0.228 | **0.223 (+74 %)** |

En video el efecto **no** depende del audio (ruido ≈ ceros) y **no es local**: el cuerpo
idéntico del clip sube +74 % y además **cambia de forma** (r = 0.30 entre el perfil del
cuerpo con y sin prefijo). Es una **ganancia global**: añadir relleno vacío amplifica y
recolorea la respuesta a todo el clip.

### Corpus real (14 anuncios de `data/ads/`)

`r(silencio inicial por energía, activación t0–2) = 0.790`. Dos grupos: aperturas de
0–0.3 s (t0–2 ≈ 0.05–0.12) y de ~1.8–2.0 s (t0–2 ≈ 0.13–0.15). **n=14, confundido** con
estilo de producción y duración: sugiere, no prueba.

## 10.2 Consecuencias

1. **Ninguna métrica de nivel absoluto sirve como score de creativo.** Se puede inflar
   +74 % (video) o +67 % (audio) sin tocar una sola decisión creativa, sólo metiendo
   relleno vacío al principio. Es el mecanismo exacto del exploit `strobe` de §3-P4: sus
   10 "gaps" son 10 huecos de información.
2. **Sólo son válidas** (a) comparaciones **dentro del mismo clip** y (b) comparaciones
   entre variantes con **estructura de arranque idéntica**. Cualquier ranking que mezcle
   arranques distintos mide el arranque, no el creativo.
3. **La intuición "tardar en arrancar engancha" no queda corroborada por el modelo.**
   Lo que el modelo premia es un arranque **vacío** (negro/silencio digital); la práctica
   real llena ese arranque de contenido (música, set-piece visual), que es lo contrario.
   Que la intuición sea cierta o no sólo lo puede decidir el dato externo (retención/CTR).
   La función a extraer para ese cruce ya es medible y barata: **segundos de lead-in sin
   información** (audio < −50 dBFS; video cerca-estático).
4. **Hallazgo de herramienta:** `discovery/transcribe.py` emite timestamps de **chunks de
   30 s** (`start: 0.0, end: 26.0`), no de palabra; `return_timestamps="word"` devuelve
   0.0 en los 14 anuncios incluso con `batch_size=1`. Cualquier feature a nivel de segundo
   derivada de transcripciones es **hoy poco fiable**; el onset de habla requiere la ruta
   long-form de Whisper o `faster-whisper`.
5. **Costo medido del canal video:** 15 s de clip = 30 frames V-JEPA a **2–12 s/frame**
   (≈6–17 min por variante sin caché), frente a 0.34–4.7 s por variante en audio. La caché
   de features **por frame** (§Eje A) deja de ser una optimización y pasa a ser el
   requisito para poder editar con video.

## 10.3 Siguiente experimento (una vez exista caché por frame)

Mismo clip, tres aperturas de 3 s: **negro**, **plano de máximo movimiento** y **sin
apertura**. Si el negro gana al plano con movimiento, el modelo prefiere la ausencia de
información y queda cerrado; si gana el movimiento, el modelo sí premia un arranque
eventual y el barrido completo de operadores (Fase B) tiene sentido sobre la ruta video.
