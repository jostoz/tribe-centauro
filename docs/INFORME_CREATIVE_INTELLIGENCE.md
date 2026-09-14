# Informe de Creative Intelligence — muestra

**Preparado para:** `<CLIENTE>` · **Fecha:** 2026-09-14
**Fuentes:** 61 anuncios de dos marcas mexicanas (Telcel, Coca-Cola) procesados con el pipeline
propio + **benchmark de 2 183 anuncios con anotación humana** (LAMBDA, WACV 2025).
**Reproducible:** `scripts/creative_intel_report.py` · datos en `data/reports/creative_intel.json`

---

## Qué es y qué NO es este informe (léelo primero)

**Es** un inventario y una caracterización **verificable** de creativos: qué se está publicando,
cómo está construido y cómo se compara con un benchmark de 2 183 anuncios.

**No es** una predicción de rendimiento. **No** hay CTR, **no** hay ventas y **no** afirmamos
qué anuncio "funcionará mejor". No usamos ninguna puntuación 0–100 ni umbrales inventados.

## 1. Alcance y método

| | Telcel | Coca-Cola | Benchmark (LAMBDA) |
|---|---|---|---|
| Anuncios analizados | **30** | **31** | **2 183** |
| Marca de los datos | nuestra | nuestra | 263 marcas |
| Ventana | 2007–2026 | 2006–2026 | 2008–2023 |
| Anotación | modelo de visión (VLM) | modelo de visión (VLM) | **humana** (1 749 participantes) |

**De dónde salen los 61 anuncios**: de fuentes públicas (canal de marca y re-subidas de terceros).
Es lo **encontrable**, no el inventario completo de la marca ni su pauta. Eso condiciona la lectura
de la cadencia y del rendimiento público (§2 y §4).

## 2. Inventario creativo

| | Telcel | Coca-Cola | Benchmark |
|---|---|---|---|
| Duración mín–**mediana**–máx | 12 – **31** – 61 s | 10 – **30** – 60 s | **30 s** |
| Anuncios en 2026 | **12** | **10** | — |
| Anuncios en 2025 | 5 | 7 | — |

**Lectura:** ambas marcas publican en el formato estándar del benchmark (**~30 s**; Telcel +1 s,
Coca-Cola exacto). El grueso de lo encontrable es **reciente** (2025–2026), lo que sugiere actividad
creativa continua — con la salvedad de que refleja disponibilidad pública, no pauta.

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

## 7. Límites (explícitos)

- **61 anuncios propios**: suficiente para caracterizar, insuficiente para inferencia fuerte.
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

Todo número de este informe proviene de esos dos comandos y de la base `data/discovery/centauro.db`.

## Siguiente paso propuesto

1. **Extender el inventario** a las categorías y competidores que definas (10–20 marcas).
2. **Añadir análisis neural** (perfil por red + visualización cortical) sobre los creativos que
   elijas — con los límites declarados que ya conocemos.
3. **Piloto de calibración**: con los resultados de una campaña real, pasamos de caracterizar a
   *validar*, que es el único camino honesto hacia afirmaciones de rendimiento.
