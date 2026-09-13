# Solicitud de licencia comercial — TRIBE v2 (Meta FAIR)

> Estado: BORRADOR listo para enviar. Rellena los campos `<...>` antes de mandarlo.
> Idioma: el email va en inglés (destinatario Meta). Las notas en español son guía interna.

---

## 0. Contexto legal (por qué esto es obligatorio)

Los pesos de TRIBE v2 y el código de referencia de Meta están bajo **CC-BY-NC-4.0
(no comercial)**. Un servicio de pago ("Ads as a Service") es **uso comercial** →
requiere autorización comercial explícita de Meta. Sin ella, cobrar por el servicio
infringe la licencia. Referencia interna: `README.md:124-129`,
`MCP_ADS_SERVICE_PLAN.md` §5 (`LICENSE_MODE=commercial` exige `META_LICENSE_REF`).

## 1. A quién dirigirlo

- Canal primario: el formulario/lista de contacto del repositorio oficial
  `github.com/facebookresearch/tribev2` (buscar `LICENSE`, `CONTACID`, o issues sobre
  uso comercial).
- Canal legal: el equipo de licencias de Meta AI / FAIR (partnerships / model licensing).
- Adjuntar: este brief + el registro de validación `docs/FASE_0.5_VEREDICTO.md`.

## 2. Email (inglés — copiar/pegar)

**Subject:** Commercial license request — TRIBE v2 model weights for a neuro-marketing screening service

Dear Meta AI / FAIR Model Licensing team,

We are `<COMPANY LEGAL NAME>` (`<COUNTRY>`, `<COMPANY REG. NUMBER>`), building a
creative pre-screening service called **Centauro** that uses the TRIBE v2 model to
estimate population-average cortical responses to advertising creatives.

TRIBE v2 is released under CC-BY-NC-4.0. Our intended use is **commercial**, so we are
formally requesting a commercial license (or written permission / commercial terms) for
the model weights and reference code.

We would like to share how we intend to use it responsibly and ask about available terms:

- **Product:** an API / MCP service that runs TRIBE v2 inference on advertising audio,
  video, and (optionally) text, returning per-cortical-network activation summaries and
  relative A/B comparisons.
- **Positioning integrity:** we do **not** sell a fabricated 0–100 "engagement score".
  All outputs are reported in raw model units with explicit provenance
  (`calibrated: false`, `within-batch-only`, `research/population-average`), with a
  human in the loop. We treat over-claiming predictive accuracy as a product red line.
- **Deployment:** `<cloud region / on-prem>`; single tenant isolation; per-tenant
  quotas; audit log per job (tenant, input hash, model version).
- **Attribution & compliance:** we will honor attribution requirements and any usage
  restrictions you specify, and gate the service so it will not start in commercial mode
  without a recorded license reference.
- **Scale (initial):** `<N>` analyses/month, `<N>` pilot customers.

Could you let us know:

1. Whether a commercial license for TRIBE v2 is available, and under what terms
   (fees, royalties, field-of-use, term length, territory).
2. Any restrictions on advertising / marketing use cases.
3. Attribution, audit, or reporting obligations.
4. Whether the license extends to the gated Llama-3.2 text features used by the
   pipeline, or if that requires a separate agreement.

We are happy to sign an NDA and provide any due-diligence materials. Attached is a
one-page technical brief and our empirical validation record.

Thank you for your time.

Best regards,
`<NAME>`, `<TITLE>`
`<COMPANY>` · `<EMAIL>` · `<PHONE>` · `<WEBSITE>`

## 3. Brief técnico de una página (adjuntar)

**Producto:** Centauro — pre-screening neural de creativos publicitarios.
**Modelo:** `facebook/tribev2` (fMRI predicho sobre malla `fsaverage5`, 20484 vértices, 1 TR = 1 s).
**Uso comercial pretendido:** API + servidor MCP para agentes; cobro por uso (metering).
**Integridad de métricas:**
- Unidades crudas del modelo; sin score 0–100 inventado.
- Bloque `provenance` en cada respuesta (`calibrated:false`, `within-batch-only`).
- ROIs anatómicas reales (Schaefer 2018, 7 redes), no slices arbitrarios.
- A/B por permutación de bloques (corrección de autocorrelación espacial).
- Loop de feedback `(predicción, resultado real)` para futura calibración isotónica.

**Validación empírica ya realizada** (ver `FASE_0.5_VEREDICTO.md`):
- Inferencia real en GPU verificada.
- Clips cortos 6–15 s producen predicciones válidas.
- Anuncios mudos no rompen el pipeline.

**Controles responsables:** tenant isolation, cuotas, audit log inmutable por job,
tratamiento del creativo como entrada no confiable (anti prompt-injection),
`commercial_use:false` marcado mientras no exista licencia.

## 4. Qué NO hacer mientras la licencia esté pendiente

- No cobrar por el servicio.
- Correr solo pilotos de evaluación/investigación con `LICENSE_MODE=research`
  (respuestas marcadas `commercial_use: false`).
- Usar ese periodo para acumular el set de calibración (feedback), que además
  refuerza la solicitud.

## 5. Plan B si Meta no concede licencia comercial

Sustituir el encoder por un modelo entrenado con datos abiertos de neuroimagen
(Algonauts, NSD, BOLD5000) bajo licencia permisiva. Es más costoso pero es el único
camino a un SaaS de pago sin dependencia legal de Meta, y se convierte en tu activo propio.
