# Evaluación: apalancarnos con datasets/corpus existentes

**Fecha:** 2026-09-13 · **Motivo:** la réplica falló por **potencia** (8 rápidos por marca) y por el
**error de medida** del predictor (etiqueta del VLM). ¿Hay corpus públicos que resuelvan eso?
**Método:** búsqueda de fuentes primarias + **verificación empírica** (cargué el dataset y medí
link rot, censura y cobertura).

> ## Veredicto
> **Sí, y hay un corpus que encaja casi perfecto: LAMBDA (2 183 anuncios con `Pace` anotado por
> humanos y `recall_score` de 1 749 participantes).** Pasaríamos de **8 vs 20** anuncios en el
> contraste de ritmo a **94 vs 1 373**. Es el mayor salto de potencia disponible y no cuesta GPU
> para conseguirlo. Segundo hallazgo: **Algonauts 2025/CNeuroMod** permite validar el *modelo*
> contra **fMRI real** (mismo atlas Schaefer que usamos).

---

## 1. Los tres cuellos que había que atacar

| Cuello | Situación hoy | Qué lo resolvería |
|---|---|---|
| **Potencia** | 8 rápidos vs 20 lentos (Telcel), 8 vs 16 (Coke) | cientos de anuncios etiquetados |
| **Predictor con error de medida** | `ritmo` = juicio del VLM sobre 12 frames (atenúa el efecto) | etiqueta **humana** de ritmo |
| **Resultado** | ninguno (solo proxies públicos de YouTube, confundidos por pauta) | un outcome humano |

## 2. LAMBDA — el corpus que encaja

**Long-Term Ad Memorability** (WACV 2025, arXiv 2309.00378). Verificado cargando el parquet real.

| Dato | Valor medido |
|---|---|
| Anuncios | **2 183** (2 205 declarados; 2008–2023, media 33 s) |
| Participantes | **1 749** · marcas **276** · industrias 113 |
| Licencia | **anotaciones MIT**; los vídeos son enlaces de YouTube (de terceros) |
| Descarga | `behavior-in-the-wild/LAMBDA` en HF · 2,5 MB de anotaciones |

Campos reales (verificados): `youtube_id`, **`recall_score`** (0–1), y `ad_details` con
**`Pace`**, **`Audio`**, `Brand`, `Duration`, `Orientation`, `Title` + `Scenes[]` (Colors,
Description, Emotions, Number, Photography Style, Tags, Text Shown, Tone, Visual Complexity).

### Cómo ataca cada cuello

| Cuello | Con LAMBDA |
|---|---|
| **Potencia** | `Pace`: **high 94 · medium 716 · low 1 373** → el contraste rápido vs lento pasa de 8 vs 20 a **94 vs 1 373**. Con n=1 373, el error de una correlación de rangos baja a **±0.027** (hoy ±0.19) |
| **Predictor** | `Pace` está **anotado**, no inferido por un VLM → desaparece el error de medida que atenúa |
| **Resultado** | **`recall_score` de 1 749 participantes** → primer test real de *"¿nuestro perfil predice algo humano?"* |
| Transcripciones ya hechas | el campo `Audio` **es la transcripción** (texto hablado, ~321 caracteres) en **1 235/2 183 (57 %)** → se puede **saltar Whisper** en esos anuncios |
| Confusores | `Duration`, `Brand`, `Orientation` → emparejamiento por diseño, no covariable |
| Pacing objetivo cruzado | media de escenas: **high 9.69 · medium 9.29 · low 7.19** → valida que `Pace` significa lo que dice |

### ⚠️ Lo que medimos ANTES de gastar GPU (y obliga a corregir el diseño)

Analizando las anotaciones contra `recall_score` (gratis, sin modelo — `scripts/analyze_lambda_annotations.py`):

| anotación | efecto sobre memorabilidad |
|---|---|
| **`Brand`** | **η² = 0,226** ← el 22,6 % de la varianza (Netflix 0,887 vs Sherwin-Williams 0,431) |
| `n_scenes` | ρ = +0,155 |
| `scenes_per_s` | ρ = +0,072 |
| `photo_style` | η² = 0,044 |
| `duration` | ρ = 0,017 (ns) |
| **`Pace`** | **η² = 0,012** (y no monótono: low 0,589 · medium 0,654 · high 0,610) |
| `orientation` / `tone` / `human_present` / `complexity` / `has_transcript` | ≤ 0,009 |

**Dos consecuencias duras:**

1. **Las anotaciones humanas apenas predicen memorabilidad.** El mejor predictor de contenido es
   `n_scenes` con ρ = 0,155 (**2,4 % de varianza**). Si propiedades que un humano anota no explican
   la memorabilidad, esperar que un modelo cerebral la extraiga del video es una expectativa muy
   optimista — hay que decirlo antes de vender nada.
2. **La marca domina y era un confusor abierto.** El primer diseño de pares emparejó **solo por
   duración**: **0 de 41 pares compartían marca**. Con 22,6 % de varianza explicada por marca, ese
   diseño habría comparado estilos de marca, no ritmo. Corregido: **19 pares emparejados por marca
   Y duración (±2 s)**, 13 marcas.

### Límites medidos (no de folleto)

1. **Link rot ~30 %.** Medido en dos muestras: 11/15 (73 % vivos, muestra pequeña) y **288/414
   (70 %) sobre el diseño grande**. Corpus usable ≈ **1 500–1 600 anuncios** (26× lo actual).
   Además, de los ads que faltaban para los pares por marca, **12/31 estaban caídos**.
2. **`Scenes` está censurado a 10**: **1 151 anuncios tienen exactamente 10** escenas. Valida que
   `Pace` significa lo que dice, pero **no** sirve como medida fina de pacing.
3. **`Audio` es la transcripción, no un tipo de audio**: presente en 1 235/2 183 (57 %). Donde
   existe, **ahorra Whisper**; donde no, hay que transcribir (43 %).
4. **`recall_score` es memorabilidad, no rendimiento de campaña.** Es un outcome humano, no CTR:
   **no sustituye la calibración** de negocio.
5. **Solo 19 pares cumplen marca + duración** en la parte ya descargada. El emparejamiento estricto
   (marca y duración) es lo correcto pero **reduce mucho el n**: pagar 1,3 h de GPU da 19 pares
   válidos, no 41 confundidos.

### Coste de adopción

| Etapa | Qué | GPU | Coste |
|---|---|---|---|
| **0 (esta)** | **19 pares marca+duración** (38 ads), con patrón por vértice | **1,3 h** | **$0,5** |
| 1 | readout multivariado sobre esos mismos patrones | ~2 min | ~$0 |
| 2 | ampliar a más pares marca+duración (requiere descargar más anuncios vivos) | ~5–10 h | ~$2–4 |
| 3 | corpus completo (~1 500 usables) para la **biblioteca**, no para validar | ~77 h | ~$27 (10 h en 8 GPUs) |

## 3. Algonauts 2025 / CNeuroMod — validar el *modelo* contra cerebros reales

- **4 sujetos**, ~**65 h** de estímulo naturalista (*Friends* S1–6 + 4 largometrajes), fMRI
  **público (CC0)**, **parcelado con Schaefer** — el mismo atlas que usamos — a TR 1.5 s.
- Es la **única vía pública** para responder: *¿las redes que TRIBE predice responden como los
  cerebros reales?* Es validación de constructo de primer nivel, no la nuestra de anuncios.
- **Coste:** 65 h de vídeo ≈ 340 h de GPU completo; **un subconjunto de 1–2 episodios (≈50 min)
  ≈ 4 h de GPU** → piloto viable.
- **Límite:** valida el modelo, no el producto publicitario (son series/películas, no anuncios).

## 4. Descartados, con razón

| Candidato | Por qué no |
|---|---|
| Benchmarks de long-video QA (LongVideoBench, MLVU, Video-MME, LVBench, HourVideo) | evalúan MLLMs con preguntas; **no dan potencia** para nuestros contrastes ni etiquetas de anuncios |
| **VideoAds** (arXiv 2504.09282), **AdsQA** | útiles para *estructura/temporalidad* de anuncios; complementarios, no sustitutos |
| **Tencent AVS (TAVS)** | anota **límites de escena** en anuncios reales (pacing objetivo) — pero requiere solicitud a Tencent y LAMBDA ya trae `Pace` |
| Datasets de CTR/campaña | **no existe ninguno público a nivel de creativo** (los benchmarks industriales son agregados) → la calibración de negocio sigue necesitando piloto o pauta propia |

## 5. Plan recomendado

1. **Etapa 1 — ingesta de anotaciones y submuestra estratificada (~400 anuncios de contenido).**
   Día 1. Coste ~3 h de GPU. Verifica además el link rot real sobre el subconjunto elegido.
2. **Etapa 2 — neural de la submuestra (~20 h, ~$16) y test pre-registrado:**
   *`Pace` humano → DorsAttn (+4.2 pp) y SomMot (−3.5 pp)*, con **94 vs ~300**. Es la prueba que la
   réplica no pudo dar. Además, con `Duration` y `Brand` **emparejados por diseño**.
3. **Etapa 3 — primer test contra un outcome humano:** ¿correlaciona el perfil neural con
   `recall_score`? Aquí es donde por primera vez se puede decir algo sobre *predecir algo*.
4. **Etapa 4 — extender** a los ~1 600 y usar la submuestra como **biblioteca de referencia**
   (percentiles estables), que es el activo que discutimos.

## 6. Lo que este camino NO resuelve

- **Sigue sin haber datos de campaña.** LAMBDA da memorabilidad humana, no ventas ni CTR. La
  calibración de negocio continúa dependiendo del **piloto** o de **pauta propia**.
- **Derechos:** las anotaciones son MIT, los vídeos no. Uso interno; nada de redistribuir.
- **Riesgo de refutación:** con potencia real, el efecto puede **desaparecer**. Es exactamente lo
  que queremos saber antes de venderlo — y es el mismo aprendizaje que dio la réplica de Coca-Cola.
