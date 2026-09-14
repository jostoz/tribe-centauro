# Campaña para atraer pilotos (ellos pagan la pauta, nosotros obtenemos los datos)

**Fecha:** 2026-09-13 · **Objetivo:** ≥3 pilotos activos en 6 semanas · **Coste de pauta para
nosotros: $0**
**Documentos relacionados:** [`PROPUESTA_PILOTO.md`](PROPUESTA_PILOTO.md) (one-pager para el
cliente) · [`META_LICENSE_REQUEST.md`](META_LICENSE_REQUEST.md) (licencia) ·
[`GRUPO_A2_RESULTADOS.md`](GRUPO_A2_RESULTADOS.md) (evidencia de honestidad metodológica)

---

## 1. La tesis

No podemos validar ni calibrar sin **creativos reales con su resultado real**. Comprar nuestra
propia pauta (~$1 000 USD, ver §9) valida el *método* pero no reproduce las condiciones de la
marca: cuenta sin historial ni píxel, audiencia fría, y no tenemos derechos sobre anuncios ajenos.

**El piloto resuelve las tres cosas a la vez:** el cliente pone la pauta (que ya paga), nosotros
obtenemos **creativos + métricas + tracking real** en su cuenta y su audiencia.

```
   nosotros aportamos                     el cliente aporta
   ─────────────────                      ─────────────────
   análisis neural (GPU nuestra)          la pauta (que ya paga igual)
   informe por creativo                   sus creativos
   metodología honesta y reproducible     sus métricas de desempeño
   $0 de media                            el A/B que ya iba a correr
```

## 2. Requisito legal — no negociable

1. El modelo TRIBE v2 es **CC-BY-NC-4.0**: no se puede cobrar por el servicio **todavía**.
   Por eso el piloto es **gratuito** y se enmarca como **evaluación/investigación**.
2. Toda salida va marcada `commercial_use: false` y `calibrated: false`.
3. La licencia comercial de Meta (`META_LICENSE_REQUEST.md`) se cierra **antes** de facturar
   nada. El piloto no depende de ella; la monetización sí.
4. Confidencialidad: NDA si el cliente lo pide. Los creativos y métricas se usan **solo** para
   este piloto; se guardan en `data/discovery/centauro.db` aislados por `corpus`.

## 3. Qué le pedimos al cliente (y en qué orden)

El error sería pedir métricas de entrada: es lo que más incomoda y frena la conversación.
**Secuencia de dos pasos:**

### Paso 1 — sin métricas (fricción casi cero)
- **5–10 creativos en video** (`.mp4` con audio, 10–45 s).
- Nada más. Ni CTR, ni accesos, ni NDA.

### Paso 2 — con métricas (solo si el paso 1 les pareció útil)
| Dato | Formato | Mínimo útil |
|---|---|---|
| Creativos | `.mp4`, uno por archivo, nombre con identificador de campaña | 30 para intentar calibración |
| Métricas por creativo | export de su dashboard (CSV/XLSX) o pantallazo | CTR y/o hook rate (3 s); VTR/retención si existe |
| Estructura del test | qué variantes eran A/B del mismo anuncio | ≥5 pares A/B |
| Ventana y presupuesto | fechas y gasto por creativo | para controlar el confusor de inversión |
| Autorización | correo bastando | uso para evaluación |

**Alternativa si no quieren dar números:** el **ranking** ("estos 3 fueron los mejores, estos 3 los
peores") es suficiente para validar por rangos y no expone cifras competitivas. **Pídelo siempre
como primera opción.**

## 4. Piloto ideal (a quién apuntamos)

| Criterio | Por qué |
|---|---|
| Agencia de performance o equipo in-house que **testea variantes** | ya producen ≥10 creativos/semana y ya corren A/B → los datos existen sin trabajo extra |
| Marca con **volumen** (retail, telco, bebidas, DTC) | n suficiente para correlacionar |
| Ciclo de campaña corto (2–4 semanas) | resultado en el horizonte del piloto |
| Alguien con **dolor medible**: gasta en producir creativos que fracasan | el pitch es ahorro, no "neurociencia" |

**Entrada caliente:** tu contacto con clientes de campañas (§ el que ya teníamos). **Entrada
fría:** LinkedIn a responsables de performance/creative de esas categorías.

## 5. Oferta concreta (el gancho)

> **"Mándanos 5 de tus creativos. En 48 h te devolvemos un informe de respuesta neural, gratis,
> y sin pedirte ni una métrica."**

- Coste marginal para nosotros: **~20 min de GPU** (VLM ~11 s/anuncio + neural ~2–4 min/anuncio).
- Entrega: perfil por red funcional, línea de atención por segundo (±1.5 s), y comparación
  **relativa** entre sus 5 creativos.
- Con la honestidad por delante: **no prometemos CTR**, no hay score 0–100, y decimos
  explícitamente que está **sin calibrar** y que la validación contra resultado real está en curso.
- Puerta al paso 2: *"si te sirve, con tus métricas pasamos de 'señal' a 'calibrado' — y eso es
  lo que aún no podemos afirmar."*

## 6. Activos de la campaña (lo que ya tenemos)

| Activo | Estado | Uso |
|---|---|---|
| Cerebro 3D animado (`data/discovery/brain_movie_3d.html`) | ✅ | demo visual que se comparte por enlace |
| Grafo de contenidos (`data/discovery/graph.html`) | ✅ | muestra el análisis de un corpus real |
| Informe de método con límites declarados (`GRUPO_A2_RESULTADOS.md`) | ✅ | **el activo de credibilidad**: publicamos lo que no funciona |
| Pipeline reprocesable (5 creativos < 1 h) | ✅ | cumplir la promesa de 48 h |
| One-pager del piloto (`PROPUESTA_PILOTO.md`) | ✅ | para reenviar dentro del cliente |

**Falta crear:** landing simple de una página con el formulario de 5 creativos, y una secuencia
de correo de 3 toques.

## 7. Secuencia de captación (3 toques, 3 semanas)

| Toque | Canal | Mensaje nuclear |
|---|---|---|
| 1 | WhatsApp/LinkedIn al contacto caliente | la oferta del §5 en 4 líneas + demo 3D |
| 2 (día +4) | correo con el one-pager | "esto es lo que pides, esto es lo que NO prometemos" |
| 3 (día +9) | caso concreto | informe anonimizado de un corpus real (Telecom MX, 30 anuncios) como muestra del formato |
| Frío | LinkedIn/email a performance leads | mismo guion, con la demo como prueba |

Regla de honestidad en todos los toques: **liderar con los límites**, no con la promesa. En este
mercado el "neuro-humo" es el default; la credibilidad es el diferenciador.

## 8. Instrumentación del piloto

1. Alta del cliente como `corpus` propio en el store (`--corpus <cliente>`): nada se mezcla.
2. `discovery.pipeline --all --corpus <cliente> --neural` → transcripción, comprensión y perfil neural.
3. `scripts/analyze_contrasts.py --corpus <cliente>` → contrastes internos de sus creativos.
4. Métricas del cliente → tabla de contraste `(predicción, resultado real)` para el futuro
   `submit_feedback` (el foso del servicio, ver `MCP_ADS_SERVICE_PLAN.md` §4).
5. Cierre: informe con **rangos**, no con cifras absolutas, y declaración de límites.

## 9. Coste y por qué la pauta propia es solo el plan B

| Opción | Quién paga la pauta | Qué se obtiene | Límite |
|---|---|---|---|
| **Piloto (este plan)** | **el cliente** | creativos + métricas + tracking real + derechos | requiere acceso al cliente |
| Pauta propia | nosotros (~$700–1 500 para 30 creativos × 10k impresiones) | solo validación de **método** | cuenta sin historial ni píxel, audiencia fría, y sin derechos sobre creativos ajenos |
| Ranking del cliente | nadie | validación por rangos | solo orden, sin magnitudes |

Con **N creativos, el error de una correlación de rangos es ≈ 1/√(N−1)**: 30 creativos → ±0.19
(solo detecta acuerdo **fuerte**, r ≳ 0.5); 100 creativos → ±0.10. Por eso el piloto apunta a
**≥30 creativos con métricas**, no a 5.

## 10. Criterios de éxito del programa

| Métrica | Objetivo |
|---|---|
| Contactos calientes convertidos a "mándame 5 creativos" | ≥5 |
| Informes gratuitos entregados en ≤48 h | ≥5 |
| Clientes que pasan al paso 2 (métricas) | ≥3 |
| Creativos con métrica acumulados | ≥90 (3 pilotos × 30) |
| Resultado publicable de calibración (rangos) | ≥1 |
| Coste de adquisición | $0 de pauta nuestra |

## 11. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Piden CTR/ventas garantizadas | Decirlo por escrito desde el primer toque: pre-filtrado, no oráculo |
| No quieren compartir métricas | Ofrecer el **ranking**; y usar la vía de cuenta propia solo para método |
| Fuga de creativos no lanzados | NDA + store aislado por `corpus` + no publicar nada identificable |
| Licencia CC-BY-NC | No facturar; `commercial_use: false`; cerrar licencia con Meta en paralelo |
| Expectativa rota si la correlación sale baja | Criterio de éxito = **aprender y publicar**, no confirmar. Un «no» honesto también es un resultado |
| Mezclar corpora en el análisis | Ya resuelto: `--corpus` obligatorio por cliente |

## 12. Calendario

- **Semana 1:** landing + secuencia de correo + informe de muestra anonimizado.
- **Semana 2:** toque 1 y 2 a la lista caliente; primeros informes gratuitos.
- **Semana 3–4:** toques fríos; paso 2 con quien haya respondido.
- **Semana 5–6:** acumular ≥30 creativos con métricas por cliente; primer análisis de rangos.

## 13. Lo que este plan NO resuelve

- **Sigue haciendo falta la licencia de Meta** para cobrar (§2).
- **No sustituye** a un test controlado (cuenta propia) para validar el método si ningún cliente
  comparte métricas: la pauta propia queda como plan B explícito.
- **No garantiza señal:** si la correlación predicción↔resultado sale nula en varios pilotos, el
  producto cambia (pre-filtrado por coste, no por rendimiento) o se descarta. Está escrito aquí
  para que no se descubra después.
