# LAMBDA — Resultados: el modelo distingue ritmo, pero NO memorabilidad

**Fecha:** 2026-09-14 · **Corpus:** LAMBDA (2 183 anuncios, anotaciones humanas) · **GPU:** RTX 4090
**Diseño:** 19 pares emparejados por **marca Y duración (±2 s)** → 18 pares usables (36 anuncios)
**Reproducible:** `scripts/analyze_pairs.py`, `scripts/analyze_vertices.py`
**Coste:** 1,3 h de GPU (~$0,5 de luz)

> ## Veredicto en dos líneas
> **El modelo SÍ distingue el ritmo del anuncio** en el patrón multivariado (bal-acc 0,78, p=0,005)
> y en 4 redes (Holm). **Pero NO predice la memorabilidad humana**: cero señal, ni con 7 shares ni
> con los 20 484 vértices. Lo primero valida al modelo como modelo; lo segundo es un **no** al
> supuesto del negocio.

---

## 1. Por qué este diseño (y el error que se corrigió antes de gastar GPU)

El primer diseño emparejaba **solo por duración**: 41 pares, y **0 de 41 compartían marca**.
Analizando gratis las anotaciones se descubrió que **`Brand` explica el 22,6 % de la varianza de
memorabilidad** (η²=0,226; Netflix 0,887 vs Sherwin-Williams 0,431). Ese diseño habría comparado
**estilos de marca**, no ritmo — el mismo tipo de confusor que hundió A2 (ritmo↔duración).

**Corregido:** 19 pares con **misma marca y duración dentro de ±2 s** (13 marcas), 18 usables.
Emparejamiento por diseño ⇒ permutación de signos intra-par, sin covariables.

## 2. Prueba 1 — `Pace` humano → composición por red (7 shares)

| red | delta (pp) | p cruda | p Holm | |
|---|---|---|---|---|
| **Vis** | **+11,98** | 0,0002 | **0,0012** | ✅ |
| **SomMot** | **−5,33** | 0,0001 | **0,0007** | ✅ |
| **Cont** | −3,12 | 0,0007 | **0,0035** | ✅ |
| **SalVentAttn** | −3,37 | 0,0041 | **0,0164** | ✅ |
| DorsAttn | +3,58 | 0,0627 | 0,1881 | ❌ |
| Limbic | −1,24 | 0,0987 | 0,1974 | ❌ |
| Default | −2,50 | 0,2070 | 0,2070 | ❌ |

**Cuatro redes sobreviven a Holm** — más que en Telcel (donde solo DorsAttn lo hizo). Pero **la red
protagonista cambia**: aquí es **Vis** (+12 pp) y **SomMot** (−5,3), no DorsAttn. Con marca y
duración controladas, el efecto de ritmo es claro pero **no es "el efecto de DorsAttn"**: la
hipótesis de A2 no se sostiene como mecanismo específico.

## 3. Prueba 2 — readout multivariado (20 484 vértices, no 7 shares)

Motivación: reducir a 7 shares tira el 99,99 % de la señal. Se guardó el patrón completo
(`--vertices`, 41 KB/ad en float16) y se analizó con PCA + permutación intra-par + validación
cruzada dejando un par fuera (con PCA reajustado en cada fold).

| prueba | resultado |
|---|---|
| **Clasificación high/low** (15 componentes, CV) | **bal-acc 0,78** vs nulo **0,50 ± 0,12** → **p ≈ 0,005** |
| PC0 (40,1 % de varianza) | delta −6,52 · p cruda 0,0010 · **p Holm 0,0150** ✅ |
| PC1–PC14 | ninguna significativa tras Holm |

**El patrón completo discrimina el ritmo mucho mejor que las 7 medias.** Confirma que parte de
nuestros resultados negativos previos eran **problema de lectura**, no del modelo.

Límite: n=18 pares es pequeño; el nulo del clasificador (0,50 ± 0,12) muestra la varianza del
estimador, así que 0,78 puede encogerse fuera de muestra. El p por permutación ya paga el coste de
la selección de k, pero **no** sustituye a una réplica con más pares.

## 4. Prueba 3 — ¿predice `recall_score` (memorabilidad humana)?

| readout | resultado |
|---|---|
| 7 shares de red (Spearman) | todas \|ρ\| ≤ 0,195, p ≥ 0,25 → **nada** |
| 15 componentes principales | **1 de 15** con p<0,05 sin corregir (esperado por azar ≈ 0,8) → **nada tras Holm** (mín p Holm 0,462) |

**Cero señal.** El perfil neural no predice la memorabilidad humana en esta muestra.

**Límite honesto:** n=36 (18 pares). Con n=36, un ρ=0,2 tiene un IC ≈ [−0,14, +0,50] — el nulo
**no es concluyente**, es "sin evidencia". Para descartar ρ≥0,2 harían falta ~85–100 anuncios.

## 5. Qué significa (y qué no)

**A favor del modelo:**
- Distingue ritmo de forma robusta y con confusores controlados (marca + duración) en dos readouts
  independientes (shares y patrón completo).
- El patrón completo es notablemente más informativo que los promedios por red.

**En contra del supuesto de negocio:**
- **No hay señal de memorabilidad.** Y el análisis de anotaciones (gratis) ya avisaba: propiedades
  que un humano anota explican muy poco (`n_scenes` ρ=0,155 → 2,4 % de varianza; `Pace` η²=0,012),
  salvo `Brand` (22,6 %), que es identidad de marca, no calidad creativa.
- Es decir: **el modelo captura "cómo es el anuncio", no "cuánto funciona"** — al menos con este
  outcome y este n.

## 6. Siguiente paso, con criterio de parada

| Opción | Coste | Qué decidiría |
|---|---|---|
| **Ampliar pares marca+duración** (requiere bajar más anuncios vivos) | 5–10 h | si el 0,78 del clasificador aguanta con n mayor |
| **Test de recall con n suficiente** (~100 anuncios) | ~4 h | si la ausencia de señal es real o falta potencia |
| **Corpus completo (~1 500)** | 77 h / $27 | **biblioteca** (percentiles), no validación |

**Recomendación:** antes de cualquier escalada, hacer el **test de recall con n≈100**. Si sigue
nulo con potencia adecuada, el producto no puede apoyarse en "predice efectividad" y hay que
replantearlo (o ceñirlo a lo que sí hace: caracterizar y comparar creativos de forma reproducible).
