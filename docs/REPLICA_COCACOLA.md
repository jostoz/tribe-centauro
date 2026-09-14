# Réplica en segunda marca — Coca-Cola (¿es artefacto de Telcel?)

**Fecha:** 2026-09-13 · **Réplica:** 31 anuncios Coca-Cola (bebidas, MX) · **Descubrimiento:** 30 Telcel (telecom, MX)
**Script:** `scripts/analyze_contrasts.py --test --corpus <c>`
**Antecede:** [`GRUPO_A2_RESULTADOS.md`](GRUPO_A2_RESULTADOS.md) (hallazgo original)

> ## ⚠️ Veredicto corto
> **La dirección se replica; la significancia NO.** El efecto de DorsAttn apareció con **la misma
> dirección** en la marca independiente pero **no sobrevivió a Holm** (n=8 rápidos). El hallazgo
> de A2 queda **degradado**: de "formalmente defendible" a **"dirección consistente, no confirmada"**.
> Y aparece una segunda señal (**SomMot**) notablemente estable entre marcas, **post-hoc**, que es
> una hipótesis para el próximo test pre-registrado — no un resultado.

---

## 1. Qué se probó y cómo

Hipótesis **pre-registrada** (de A2): en anuncios de corte rápido la composición de **DorsAttn** es
mayor (+5.2 pp en Telcel, Holm 0.0035).

Réplica: **marca y categoría distintas** (Coca-Cola México, bebidas), **mismo pipeline, mismo
modelo, mismo análisis** (shares de las 7 redes Schaefer; permutación bilateral con
Freedman–Lane ajustando por duración; Holm sobre las 7 redes).

Corpus Coca-Cola: 31 anuncios (961 s de video, fase neural 2 h 11 m supervisada).
Ritmo: **8 rápido / 16 lento / 6 medio / 1 sin etiquetar**. Caras: 23 / 8.

## 2. Resultados

### Ritmo — rápido vs lento, por marca

| red | Telcel (8 vs 20) | p Holm | **Coca-Cola (8 vs 16)** | p Holm |
|---|---|---|---|---|
| **DorsAttn** | **+5.2 pp** | **0.0035 ✅** | **+3.0 pp** | **0.6443 ❌** |
| **SomMot** | −3.4 pp | 0.1014 | **−3.5 pp** | 0.0763 |
| Vis | +3.1 | 0.8765 | −0.3 | 1.0000 |
| Default | −2.7 | 0.5754 | +0.7 | 1.0000 |
| Limbic | −1.7 | 0.5754 | −0.8 | 1.0000 |
| SalVentAttn / Cont | +0.6 / −1.1 | ≥0.83 | +1.2 / −0.2 | 1.0000 |

**DorsAttn: mismo signo en las dos marcas, magnitud ~40 % menor, y NO significativo en la réplica.**
Con n=8 en el grupo rápido, el test de Coca-Cola **no tiene potencia** para un efecto de 3 pp
(las SD por grupo son de 2–4 pp).

### Análisis agrupado (sensibilidad, **post-hoc**)

Juntando ambas marcas (16 rápido vs 36 lento, 60 anuncios; duración mediana idéntica: 30 s vs 30 s):

| red | efecto | p cruda | p \| duración | **p Holm** | sig |
|---|---|---|---|---|---|
| **DorsAttn** | **+4.2 pp** | 0.0007 | 0.0011 | **0.0066** | **sí** |
| **SomMot** | **−3.5 pp** | 0.0026 | 0.0009 | **0.0063** | **sí** |
| Vis | +1.5 | 0.7207 | 0.5617 | 1.0000 | no |
| Limbic | −1.3 | 0.0313 | 0.0398 | 0.1990 | no |

**Esto es un análisis agrupado post-hoc, no una confirmación**: mezcla marcas y se decidió después
de ver los datos. Vale como cota de potencia, no como evidencia nueva.

## 3. La señal que sí es estable entre marcas: SomMot

| | Telcel | Coca-Cola | Agrupado |
|---|---|---|---|
| SomMot | −3.4 pp | **−3.5 pp** | −3.5 pp |

Dos corpora **independientes, de marcas y categorías distintas**, dan **−3.4 y −3.5 pp** (menos
sensorimotor en anuncios de corte rápido). Es más estable que DorsAttn (que fue 5.2 vs 3.0).

**Pero: no era la hipótesis pre-registrada.** Elegirla ahora como "el hallazgo" sería HARKing.
Queda registrada como **hipótesis a pre-registrar** y probar en datos nuevos.

## 4. Corrección de método encontrada en este análisis

El test formal etiquetaba como control **todo lo que no era rápido** (incluía los `medio`),
mientras la parte descriptiva comparaba **rápido vs lento**. Meter ritmos intermedios en el control
**atenúa** el efecto. Corregido en `scripts/analyze_contrasts.py` (grupos explícitos, también en el
ajuste descriptivo). Los números de Telcel cambiaron ligeramente (DorsAttn +5.4 → **+5.2 pp**).

Consecuencia: la corrección **no cambia el veredicto** (Telcel sigue significativo, Coca-Cola no),
pero el reporte anterior de A2 tenía esa inconsistencia.

## 5. Límites

1. **Potencia:** 8 anuncios rápidos por marca. Para un efecto de 3–5 pp con SD 2–4 pp hacen falta
   **≥15–20 por lado**. Es el cuello de botella, no el modelo.
2. **Predictor = etiqueta del VLM** (`ritmo`, `hay_caras`) sobre 12 frames → error de medida que
   **atenúa**; los efectos reales podrían ser mayores (o las etiquetas estar mal en algunos casos).
3. **Dos marcas mexicanas.** No dice nada sobre otras categorías, idiomas o mercados.
4. **Sin calibración.** Shares en unidades crudas; significativo ≠ relevante.
5. **Una sola pipeline/configuración.** No se probó sensibilidad a `frames`, resolución o versión del VLM.

## 6. Conclusión honesta

- **No podemos afirmar** que el efecto de DorsAttn sea un fenómeno general: en la única marca
  independiente probada, **no se confirmó estadísticamente**, aunque la dirección coincide.
- **Sí podemos afirmar** que el pipeline detecta diferencias pequeñas cuando hay potencia: con 60
  anuncios agrupados, DorsAttn y SomMot salen significativos con la duración controlada.
- **Lo que esto cambia en el producto:** el hallazgo de A2 **no puede usarse como claim**. Hoy el
  claim defendible es "el análisis produce contrastes reproducibles y honestos", no "el modelo
  detecta el ritmo de un anuncio".
- **Lo que esto hizo bien:** la réplica costó 2 h de GPU y **evitó** que vendiéramos un efecto que
  no aguanta una segunda marca. Es exactamente para esto que se replica.

## 7. Siguiente paso (para confirmar o descartar)

1. **Pre-registrar** la hipótesis de SomMot (−3.5 pp) y probarla en un tercer corpus **nuevo**.
2. Subir a **≥15–20 anuncios rápidos por marca** (o muchas marcas) para potencia.
3. **Sustituir la etiqueta del VLM** por la representación universal del modelo (los 1152 dims
   pre-`SubjectLayers`, capturables con un hook) para emparejar creativos en vez de usar tags.
4. Repetir con **duración emparejada** por diseño, no por covariable.
