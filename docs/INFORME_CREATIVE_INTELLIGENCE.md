# Informe de Creative Intelligence — muestra

**Preparado para:** `<CLIENTE>` · **Fecha:** 2026-09-14
**Fuentes:** **405 anuncios mexicanos en 23 corpus** — 61 de dos marcas propias (Telcel, Coca-Cola) y
**344 de 21 marcas de 7 categorías** (telecom, bebidas, alimentos, retail, banca, autos, tecnología),
procesados con el pipeline propio — + **benchmark de 2 183 anuncios con anotación humana** (LAMBDA,
WACV 2025).
**Reproducible:** `scripts/creative_intel_report.py` · datos en `data/reports/creative_intel.json`

---

## Qué es y qué NO es este informe (léelo primero)

**Es** un inventario y una caracterización **verificable** de creativos: qué se está publicando,
cómo está construido y cómo se compara con un benchmark de 2 183 anuncios.

**No es** una predicción de rendimiento. **No** hay CTR, **no** hay ventas y **no** afirmamos
qué anuncio "funcionará mejor". No usamos ninguna puntuación 0–100 ni umbrales inventados.

## 1. Alcance y método

| | Telcel | Coca-Cola | 21 marcas MX | Benchmark (LAMBDA) |
|---|---|---|---|---|
| Anuncios analizados | **30** | **31** | **344** | **2 183** |
| Marca de los datos | nuestra | nuestra | 21 marcas, 7 categorías | 263 marcas |
| Ventana | 2007–2026 | 2006–2026 | 2006–2026 | 2008–2023 |
| Anotación | modelo de visión (VLM) | modelo de visión (VLM) | modelo de visión (VLM) | **humana** (1 749 participantes) |

**De dónde salen los 61 anuncios**: de fuentes públicas (canal de marca y re-subidas de terceros).
Es lo **encontrable**, no el inventario completo de la marca ni su pauta. Eso condiciona la lectura
de la cadencia y del rendimiento público (§2 y §4).

**De dónde salen los 344 de las 21 marcas**: de un manifiesto multi-marca construido con el propio
pipeline (`scripts/build_brand_manifest.py` → `data/brands/manifest.json`), que busca por marca y
**verifica que el enlace siga vivo** antes de aceptarlo. De 346 URLs verificadas se descargaron 344
(las otras 2 dejaron de estar disponibles entre la verificación y la descarga). Cada anuncio se
transcribió (Whisper-small) y se analizó con un VLM local (Qwen2.5-VL-7B, 12 frames): **130 min** de
GPU para las 21 marcas. Misma salvedad que arriba: es material público, no la pauta de la marca.

## 2. Inventario creativo

| | Telcel | Coca-Cola | Benchmark |
|---|---|---|---|
| Duración mín–**mediana**–máx | 12 – **31** – 61 s | 10 – **30** – 60 s | **30 s** |
| Anuncios en 2026 | **12** | **10** | — |
| Anuncios en 2025 | 5 | 7 | — |

**Lectura:** ambas marcas publican en el formato estándar del benchmark (**~30 s**; Telcel +1 s,
Coca-Cola exacto). El grueso de lo encontrable es **reciente** (2025–2026), lo que sugiere actividad
creativa continua — con la salvedad de que refleja disponibilidad pública, no pauta.

### 2b. Las 21 marcas: duración por marca y categoría

| Marca | Categoría | n | Duración mín–mediana–máx | Ventana |
|---|---|---|---|---|
| at_t_mexico | telecom | 18 | 10 – **30** – 59 s | 2011–2026 |
| movistar_mexico | telecom | 18 | 12 – **30** – 31 s | 2007–2026 |
| pepsi_mexico | bebidas | 18 | 20 – **31** – 60 s | 2007–2026 |
| jarritos | bebidas | 16 | 10 – **20** – 40 s | 2010–2026 |
| corona | bebidas | 18 | 15 – **30** – 31 s | 2006–2026 |
| tecate | bebidas | 18 | 10 – **21** – 59 s | 2008–2026 |
| sabritas | alimentos | 17 | 10 – **28** – 41 s | 2011–2026 |
| bimbo | alimentos | 18 | 6 – **20** – 43 s | 2011–2026 |
| marinela | alimentos | 17 | 10 – **20** – 31 s | 2012–2026 |
| mcdonald_s_mexico | alimentos | 15 | 15 – **21** – 33 s | 2012–2026 |
| walmart_mexico | retail | 18 | 10 – **22** – 31 s | 2013–2026 |
| liverpool | retail | 17 | 10 – **21** – 52 s | 2009–2025 |
| elektra | retail | 14 | 15 – **30** – 40 s | 2012–2026 |
| oxxo | retail | 17 | 10 – **20** – 58 s | 2011–2026 |
| bbva_mexico | banca | 17 | 15 – **21** – 57 s | 2012–2026 |
| banorte | banca | 16 | 19 – **20** – 40 s | 2010–2026 |
| santander_mexico | banca | 16 | 10 – **21** – 48 s | 2008–2026 |
| nissan_mexico | autos | 18 | 15 – **30** – 54 s | 2007–2026 |
| kia_mexico | autos | 17 | 10 – **34** – 60 s | 2018–2026 |
| samsung_mexico | tecnología | 14 | 15 – **35** – 47 s | 2020–2026 |
| xiaomi_mexico | tecnología | 7 | 15 – **32** – 55 s | 2015–2025 |

**Lectura:** la mediana del corpus es **22 s** (11 de las 21 marcas por debajo de 23 s) y **la
duración separa categorías con bastante limpieza**: alimentos **20 s**, banca **21 s** y retail
**22 s**, frente a telecom **30 s**, autos **32 s** y tecnología **34 s**. El formato corto
(20–22 s) domina en consumo masivo; el de ~30 s sobrevive en categorías de consideración — y ahí
caen justamente Telcel y Coca-Cola (30–31 s).

## 3. Perfil de construcción creativa

| | Telcel | Coca-Cola |
|---|---|---|
| Ritmo (lento / medio / rápido) | **21 / 1 / 8** | **16 / 6 / 8** |
| Con caras | **73 %** | **74 %** |
| Con texto en pantalla | **97 %** | 81 % |
| Tono dominante | positivo (15/30) | feliz · energético · divertido |

**Términos de marca más recurrentes** (elementos visuales, no menciones):
- **Telcel:** `Telcel` ×12 · `logotipo de Telcel` ×6 · `$100` ×2 · **`fondo azul`** ×2
- **Coca-Cola:** `Coca-Cola logo` ×9 · `Coca-Cola` ×7 · `botella de Coca-Cola` ×2 · **`fondo rojo`** ×2

**Temas:**
- **Telcel:** tecnología ×15 · comunicación ×6 · celebración ×5 · conectividad ×4 · deportes ×3
- **Coca-Cola:** celebración ×10 · deportes ×6 · familia ×6 · **Navidad** ×4 · diversión ×3

**Lectura:** las dos marcas son **identidad-céntricas** — el logo y el **color corporativo**
(azul / rojo) son de los elementos más repetidos. Telcel se apoya en **producto y tecnología**;
Coca-Cola en **celebración, familia y deporte**. Coca-Cola usa **más variedad de ritmo**
(6 medios vs 1) y Telcel mucho más **texto en pantalla** (97 % vs 81 %).

### 3b. Las 21 marcas: ritmo, caras, texto y temas (VLM)

| Marca | lento/medio/rápido | caras | texto | temas top |
|---|---|---|---|---|
| at_t_mexico | 15/0/3 | 67 % | 94 % | promoción(6) · tecnología(6) · deportes(4) |
| movistar_mexico | 10/3/5 | 100 % | 89 % | comunicación(5) · tecnología(4) · movilidad(3) |
| pepsi_mexico | 4/1/13 | 78 % | 83 % | publicidad(5) · diversión(3) · celebración(3) |
| jarritos | 15/1/0 | 62 % | 88 % | bebidas(4) · festividad(4) · bebidas mexicanas(3) |
| corona | 16/0/2 | 56 % | 83 % | bebidas(9) · ocio(8) · relajación(7) |
| tecate | 8/4/6 | 94 % | 89 % | celebración(7) · bebidas(5) · socialización(2) |
| sabritas | 12/2/3 | 82 % | 88 % | comida(6) · publicidad(5) · comida rápida(3) |
| bimbo | 14/0/3 | 56 % | 89 % | comida(9) · familia(6) · salud(3) |
| marinela | 8/2/7 | 35 % | 100 % | publicidad(5) · comida(4) · diversión(3) |
| mcdonald_s_mexico | 9/0/6 | 53 % | 80 % | comida rápida(13) · promoción(5) · diversión(4) |
| walmart_mexico | 14/2/2 | 78 % | 100 % | compras(14) · promoción(7) · ofertas(6) |
| liverpool | 12/1/4 | 41 % | 88 % | compras(9) · ofertas(4) · familia(4) |
| elektra | 9/3/2 | 86 % | 100 % | promoción(4) · compras(4) · familia(3) |
| oxxo | 11/3/3 | 71 % | 88 % | compras(5) · supermercado(4) · promoción(4) |
| bbva_mexico | 12/2/3 | 76 % | 100 % | finanzas(8) · celebración(4) · vida cotidiana(4) |
| banorte | 13/0/3 | 81 % | 100 % | finanzas(6) · promoción(6) · celebración(5) |
| santander_mexico | 8/0/8 | 56 % | 94 % | finanzas(6) · facilidad(2) · rapidez(2) |
| nissan_mexico | 8/2/8 | 56 % | 100 % | automóvil(9) · movilidad(4) · movimiento(4) |
| kia_mexico | 12/0/5 | 18 % | 88 % | diseño(4) · movimiento(4) · automoción(3) |
| samsung_mexico | 8/0/6 | 50 % | 100 % | tecnología(9) · innovación(4) · entretenimiento(2) |
| xiaomi_mexico | 6/0/1 | 29 % | 86 % | tecnología(5) · comida(1) · experiencia culinaria(1) |

**Lectura:** el ritmo **lento domina** (224 de los 343 clasificados, 65 %; rápido 93, 27 %) y **Pepsi
es la excepción clara** (13 de 18 en rápido). Solo 7 de las 21 marcas dedican un tercio o más de su
material a ritmo rápido. El **texto en pantalla es casi universal** (80–100 % en todas), mientras las
**caras son el eje que más varía**: del **18 %** de Kia —coches, no personas— al **100 %** de
Movistar. Los temas se agrupan por categoría sin apenas sorpresas: `compras`/`ofertas` en retail,
`finanzas` en banca, `comida` en alimentos, `bebidas`/`ocio` en bebidas.

## 4. Rendimiento público (proxy — con salvedades duras)

| | Telcel | Coca-Cola |
|---|---|---|
| Vistas/día (mediana) | 12 | 73 |
| Vistas/día (máximo) | 424 550 | 574 575 |
| Tasa de likes (mediana) | 0,64 % (29/30 con dato) | 0,45 % (30/31) |

⚠️ **No son métricas de marca.** Mezclan **pauta pagada con orgánico**, incluyen re-subidas de
terceros, y los máximos corresponden a campañas con inversión (dos anuncios con decenas de
millones de vistas). Las medianas bajas indican que la mayoría del material público es de canales
pequeños. **Este bloque sirve para detectar casos extremos, no para comparar marcas.**

### 4b. Las 21 marcas: rendimiento público (mismo aviso)

| Marca | vistas/día mediana | máx | like rate mediana |
|---|---|---|---|
| at_t_mexico | 3 | 36 417 | 0,58 % |
| movistar_mexico | 3 | 13 | 0,31 % |
| pepsi_mexico | 17 | 39 076 | 0,57 % |
| jarritos | 2 | 97 510 | 0,60 % |
| corona | 41 | 36 692 | 0,43 % |
| tecate | 34 | 1 245 650 | 0,30 % |
| sabritas | 19 | 179 | 0,51 % |
| bimbo | 36 | 345 260 | 0,31 % |
| marinela | 20 | 299 211 | 0,61 % |
| mcdonald_s_mexico | 2 | 2 776 | 0,87 % |
| walmart_mexico | 30 | 2 370 | 0,27 % |
| liverpool | 4 | 35 | 0,35 % |
| elektra | 3 | 6 392 | 0,19 % |
| oxxo | 10 | 852 | 0,21 % |
| bbva_mexico | 4 | 87 645 | 0,28 % |
| banorte | 20 069 | 252 519 | 0,001 % |
| santander_mexico | 5 | 32 968 | 0,32 % |
| nissan_mexico | 15 | 127 687 | 0,56 % |
| kia_mexico | 171 | 15 893 | 0,11 % |
| samsung_mexico | 7 | 66 009 | 0,40 % |
| xiaomi_mexico | 0 | 2 453 | 0,85 % |

⚠️ **La dispersión demuestra sola el aviso**: la mediana de vistas/día va de **0** (Xiaomi) a
**20 069** (Banorte), y ese extremo es una re-subida con 252 519 vistas y **0,001 %** de likes — es
decir, una pieza que se viralizó fuera del canal de la marca, no una campaña mejor. Las tasas de
likes, en cambio, son **notablemente estables** (0,2–0,9 %) y ahí sí hay una lectura: las marcas de
consumo masivo (alimentos, bebidas, retail) rondan 0,3–0,6 %, coherente con audiencias amplias.

## 5. Benchmark de categoría (2 183 anuncios, anotación humana)

**Distribución de ritmo:** lento **1 373** · medio **716** · rápido **94**.

**Memorabilidad media por ritmo** (0–1, 1 749 participantes):

| ritmo | memorabilidad |
|---|---|
| medio | **0,654** |
| rápido | 0,610 |
| lento | 0,589 |

→ **La relación no es monótona**: el ritmo **medio** obtiene la mayor memorabilidad, no el rápido.
"Cuanto más rápido, mejor" **no** se sostiene en estos datos.

**Qué correlaciona con la memorabilidad** (tamaños de efecto, n=2 183):

| propiedad del anuncio | efecto | lectura |
|---|---|---|
| **Marca** | **η² = 0,226** | lo dominante: 22,6 % de la varianza es **identidad de marca** |
| Nº de escenas | ρ = +0,155 | más planos, algo más de recuerdo (**2,4 % de varianza**) |
| Escenas por segundo | ρ = +0,072 | — |
| Estilo de fotografía | η² = 0,044 | — |
| Duración | ρ = 0,017 | **sin relación** |
| **Ritmo (etiqueta humana)** | **η² = 0,012** | prácticamente nada |
| Orientación · tono · personas · complejidad visual | ≤ 0,009 | — |

## 5b. Benchmark multi-marca — **40 marcas** con anotación humana

Mismo corpus LAMBDA, agregado por marca (≥12 anuncios). **Anotación humana**, no modelo:
ritmo, duración, nº de escenas y **memorabilidad** (1 749 participantes).

| marca | n | rápido % | medio % | lento % | dur med. | escenas | memoria |
|---|---|---|---|---|---|---|---|
| Netflix | 121 | 0 | 26 | 74 | 40 s | 8,93 | **0,887** |
| Walt-Disney | 89 | 3 | 65 | 31 | 30 s | 9,73 | 0,792 |
| Ulta Beauty | 78 | 3 | 19 | 78 | 30 s | 7,94 | 0,562 |
| Viacom | 69 | 0 | 13 | 87 | 25 s | 8,84 | 0,641 |
| Adobe | 63 | 0 | 22 | 78 | 48 s | 8,75 | 0,808 |
| Costco | 53 | 0 | 13 | 87 | 44 s | 7,32 | 0,464 |
| Nvidia | 45 | 2 | 47 | 51 | 36 s | 8,24 | 0,770 |
| Sherwin-Williams | 45 | 0 | 7 | 93 | 31 s | 6,80 | **0,431** |
| Uber | 38 | 5 | 55 | 39 | 41 s | 8,74 | 0,838 |
| Amazon | 36 | 0 | 14 | 86 | 30 s | 4,97 | 0,774 |
| Esteelauder | 35 | 0 | 66 | 34 | 25 s | 8,17 | 0,414 |
| Clorox | 34 | **15** | 24 | 62 | 15 s | 7,00 | 0,492 |
| Dick's Sporting Goods | 34 | **12** | 24 | 65 | 32 s | 7,91 | 0,761 |
| Ralphlauren | 25 | **12** | 48 | 40 | 30 s | 9,12 | 0,592 |
| … (40 marcas en `data/reports/creative_intel.json`) | | | | | | | |

**Observaciones:**

1. **El ritmo rápido es minoritario en todas las marcas.** La mayoría está en **0 %**; solo Clorox
   (15 %), Dick's (12 %) y Ralphlauren (12 %) destacan. Es decir: el corpus profesional no usa el
   corte rápido como norma.
2. **Las más memorables no son las más rápidas.** Redbull (0,958) usa **0 % rápido** y solo
   3,9 escenas; Netflix (0,887) tiene **74 % lento**. Los extremos se explican mejor por
   **categoría y familiaridad de marca** que por montaje.
3. **Contraste de niveles de análisis — importante para no sobreinterpretar:**

| nivel | asociación ritmo↔memorabilidad |
|---|---|
| **ad a ad** (n=2 183) | η² = **0,012** → nada |
| **marca a marca** (n=40) | ρ = **+0,367**, p = 0,020 → pero **no sobrevive a Holm** (~0,08) y está confundido por categoría |

→ La asociación existe **solo al agregar por marca**, es **débil**, **no pasa la corrección por
multiplicidad** y probablemente refleja que las marcas de categorías más "entretenidas" usan algo
más de ritmo rápido **y** son más recordadas. **No es una palanca que podamos vender.**

## 6. Lectura cruzada

1. **Ambas marcas están en el formato correcto** (≈30 s, ritmo mayoritariamente lento-medio) y son
   **fuertemente identitarias** (logo + color). En el benchmark, la marca es justamente el factor
   que más pesa en la memorabilidad — coherente con que las dos marcas **ya inviertan en
   reconocimiento**.
2. **Coca-Cola parece optimizar hacia el rango de mayor memorabilidad** (más anuncios "medio"),
   mientras **Telcel está más cargado a "lento"** (21 de 30) con solo 1 "medio".
   *Si* el patrón del benchmark se sostiene, ahí hay una hipótesis de trabajo concreta.
3. **El ritmo no es una palanca grande** (η²=0,012). Prometer mejoras por acelerar el montaje sería
   exactamente el tipo de afirmación que este informe evita.
4. **La duración es el eje más estable del corpus**: su mediana por marca se ordena por categoría
   (alimentos 20 s → tecnología 34 s) y las dos marcas propias caen en el extremo largo (30–31 s).
   Es el único eje comparable que no depende ni de la anotación automática (§3) ni del proxy público
   (§4), y por eso es sobre el que podemos afirmar sin reservas.
5. **El ritmo rápido es minoritario en México**: 65 % de los 344 anuncios son lentos y solo 7 de 21
   marcas lo usan en un tercio o más de su material; Pepsi (13/18) es el caso extremo. Esto es
   coherente con el hallazgo de LAMBDA (§5b): el ritmo es un eje **débil**, no una palanca de
   memorabilidad.

## 7. Límites (explícitos)

- **61 anuncios propios**: suficiente para caracterizar, insuficiente para inferencia fuerte.
- **Las 21 marcas son lo públicamente encontrable**, no la pauta de cada marca: incluye re-subidas de
  terceros y la ventana no es uniforme entre marcas (de 2006 a 2026). Con 344 anuncios el corpus da
  para comparar perfiles de construcción, **no** para inferencia por marca: Xiaomi tiene 7 anuncios y
  Samsung 14.
- **Calidad de la anotación automática en este corpus**: 341 de 344 anuncios con JSON utilizable. De
  los 3 restantes, dos son **falsos positivos del guardián de calidad** (listas cortas con un elemento
  repetido: el contenido es válido) y **uno es un fallo real del modelo**, que entra en bucle
  repitiendo una frase hasta agotar los tokens; su salida cruda queda guardada y marcada.
- **~8 % de los vídeos** (28 de 344) llegan con la **cola truncada**: los últimos 1–2 frames no son
  legibles. No afecta la inferencia (se reutiliza el último frame válido), pero importa si se
  reutilizan los frames para otra cosa.
- **Ritmo/caras/temas son anotación automática (VLM)**, no humana. En una verificación de 3 casos
  coincidió con la etiqueta humana en **2/3**.
- **LAMBDA es otro mercado y otra época** (marcas grandes, 2008–2023): el benchmark orienta, no
  equivale.
- **Sin resultados de campaña.** No hay CTR ni conversiones. Cualquier afirmación de rendimiento
  requeriría los datos de campaña del cliente.
- **Fuera de este informe**: el análisis de respuesta neural (perfil por red cerebral), que sí
  realizamos y que se entrega por separado, está en fase de validación.

## 8. Cómo se reproduce

```bash
.venv/Scripts/python.exe scripts/creative_intel_report.py     # datos de este informe
.venv/Scripts/python.exe scripts/analyze_lambda_annotations.py # benchmark de memorabilidad
```

El corpus de las 21 marcas se construye y analiza así (una marca a la vez, sin dos inferencias
concurrentes):

```bash
.venv/Scripts/python.exe scripts/build_brand_manifest.py --per-brand 18 --workers 2
.venv/Scripts/python.exe -m discovery.pipeline --corpus <marca> \
    --urls-file data/brands/<marca>.txt --workers 1     # descarga + Whisper + VLM
```

Todo número de este informe proviene de esos comandos y de la base `data/discovery/centauro.db`.

## Siguiente paso propuesto

1. ~~Extender el inventario a las categorías y competidores que definas (10–20 marcas).~~
   ✅ **Hecho**: 21 marcas de 7 categorías, 344 anuncios (§1–§4b, corpus en `data/brands/`).
2. **Añadir análisis neural** (perfil por red + visualización cortical) sobre los creativos que
   elijas — con los límites declarados que ya conocemos.
3. **Piloto de calibración**: con los resultados de una campaña real, pasamos de caracterizar a
   *validar*, que es el único camino honesto hacia afirmaciones de rendimiento.
