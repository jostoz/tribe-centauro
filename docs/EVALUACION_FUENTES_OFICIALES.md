# Evaluación de fuentes oficiales de anuncios (alternativa al scraping de YouTube)

**Fecha:** 2026-09-14 · **Motivo:** YouTube bloqueó la ingesta por anti-bot a nivel de IP, y para
"Creative Intelligence" hace falta una fuente de creativos **sostenible**.
**Método:** revisión de las condiciones de acceso publicadas por cada plataforma (fuentes al final).

> ## Veredicto
> **No existe ninguna fuente oficial, programática y legítima de creativos comerciales para
> México.** Las que hay: (a) **excluyen el uso comercial** (TikTok), (b) cubren **solo anuncios
> servidos en la UE/UK** (Meta API, Google EEA, repositorios DSA), o (c) exponen los creativos
> **sin API** (Google Ads Transparency Center).
>
> **La salida práctica para MX:** el **Google Ads Transparency Center vía scraper de pago**
> (~**$0,99 por 1 000 anuncios**) da **creativos + fechas de emisión + formato + plataforma**.
> Y la fuente que **sí** es limpia y completa sigue siendo **el creativo que aporta el cliente**
> (que además trae las métricas de campaña → la calibración).

---

## 1. Comparativa

| Fuente | ¿Cubre anuncios comerciales de MX? | Acceso | Coste | Veredicto |
|---|---|---|---|---|
| **YouTube** (yt-dlp) | Sí (el creativo) | scraping | $0 | ❌ **bloqueado por anti-bot** (IP) |
| **Meta Ad Library API** | ❌ No: solo **política/social** + lo servido en **UE/UK** | token + verificación de identidad | $0 | ⚠️ útil solo para UE o política |
| Meta Ad Library (web) | Sí (visible por anunciante) | UI; scraping | vía 3.º | ⚠️ zona gris de ToS |
| **Google Ads Transparency (scraper de pago)** | ✅ **Sí: creativo + fechas + formato** | **API de pago** | **$0,99 / 1 000** | ✅ **la mejor opción para MX** |
| Google Ads Transparency (oficial) | Sí (EEA vía API; política vía BigQuery) | API oficial solo EEA / política | $0 | ⚠️ no cubre comercial MX |
| **TikTok Commercial Content API** | — | **solo academia/ONG**; uso comercial **excluido** | $0 | ❌ **no elegibles** |
| Repositorios DSA (UE) | Solo anuncios servidos en la UE | descarga/API | $0 | ❌ no aplica a MX |
| **LAMBDA** | Benchmark anotado (2 183), no creativos al día | HuggingFace | $0 | ✅ **benchmark**, no ingesta |

## 2. Qué da exactamente cada opción viable

### 2.1 Google Ads Transparency Center vía scraper de pago — **la recomendación**
Devuelve, por anuncio: `advertiserName`, `creativeId`, **`firstShown` / `lastShown` /
`servedDays`** (→ **duración de campaña**, señal real), `format` (TEXT/IMAGE/VIDEO),
`platforms` (Search, **YouTube**, Shopping, Maps, Play), `renderings[].mediaUrl` (el creativo),
`servedCountries`, e **impresiones/gasto por rango** donde la ley lo obliga (política y EEA/US).
- **Coste: $0,99 por 1 000 resultados completos** (~$0,001 por anuncio).
- 1 000 anuncios de 20 marcas ≈ **$1**. Las 21 marcas que intenté raspar serían **~$1–2**.
- Los filtros permiten: por anunciante, dominio, **región (244)**, plataforma, **formato vídeo**,
  rango de fechas, rango de impresiones y de gasto, y ordenar por gasto/impresiones.
- **Lo que NO da para comercial fuera de EEA/US**: impresiones y gasto (Google no los publica).
  Sí da **creativo, fechas, formato y plataformas** — suficiente para un inventario creativo.

### 2.2 Meta Ad Library API — gratis, pero **no para MX comercial**
- Cobertura: anuncios de **temas sociales/elecciones/política** (mundial) y **cualquier tipo de
  anuncio servido en la UE/UK** (por DSA).
- **Los anuncios comerciales que solo corren en México NO están** (ni archivados, en la mayoría de
  países fuera de UE/UK). Es la decepción más común: la UI los muestra, la API no.
- Acceso: verificación de identidad con documento oficial (1–3 días), app de desarrollador, token
  que **caduca cada 60 días**.
- Campos UE: `eu_total_reach`, `target_ages/gender/locations`, `beneficiary_payers`,
  desglose por país/edad/género. **Para comercial, sin gasto**.
- **Conclusión:** vale la pena tenerlo (gratis) **si vendemos a clientes que pautan en la UE**.

### 2.3 TikTok Commercial Content API — **no elegibles**
- Elegibilidad: **instituciones académicas y ONG** de US/EEA/UK/CH. **"Commercial users,
  creators, and advertisers are explicitly ineligible."**
- Límite ~1 000 peticiones/día, ~50 anuncios por llamada, compromiso de **uso no comercial**.
- **Conclusión:** descartado como fuente para un servicio comercial.

## 3. El problema de fondo (y por qué el piloto es la respuesta limpia)

Ninguna fuente oficial cubre **creativos comerciales de MX a escala**. Solo queda:
1. **Scraping de pago** (Google ATC ~$1/1 000; Meta Ad Library vía terceros) — rápido y barato,
   pero es **zona gris de ToS**: comprar el resultado no elimina la cuestión de cumplimiento, la
   traslada al proveedor.
2. **El creativo que aporta el cliente** — ✅ limpio, con derechos, **y trae las métricas de
   campaña**, que es justo lo que falta para calibrar.

→ La conclusión operativa es que **el piloto no es solo el camino a la calibración: es la única
fuente de creativos que es a la vez legal, completa y con resultados.**

## 4. Plan recomendado

| Fase | Qué | Coste |
|---|---|---|
| **Ahora** | Entregar el informe con lo que ya hay: 2 marcas propias + **40 marcas con anotación humana** (LAMBDA) | $0 |
| **Corto** | **Google Ads ATC vía scraper de pago** para el inventario multi-marca de MX: creativos + fechas + formato de las 21 marcas objetivo | **~$1–2** |
| **Corto** | Alta en la **Meta Ad Library API** (gratis, requiere verificación de identidad): habilita clientes de UE y anuncios políticos | $0 + verificación |
| **Medio** | **Piloto**: el cliente aporta creativos + métricas → única vía a la calibración | $0 (él paga su pauta) |
| **Si apuntamos a la UE** | Meta Ad Library API + repositorios DSA dan **todo** (todos los tipos de anuncio, alcance, targeting) | $0 |

## 5. Límites de esta evaluación

- Las condiciones de acceso cambian con frecuencia; **verificar antes de contratar**.
- Los scrapers de terceros son **no oficiales**: pueden romperse y su cumplimiento es suyo, no
  nuestro. Tratarlos como dependencia frágil, no como cimiento.
- Para **impresiones/gasto de comerciales no-EEA no hay fuente**: eso solo lo tiene el cliente.
- Esta evaluación **no** sustituye asesoría legal sobre ToS y scraping.

## Fuentes

- Meta Ad Library API y condiciones: `transparency.meta.com/researchtools/ad-library-tools/` y
  `facebook.com/ads/library/api`.
- Comparativa de librerías de transparencia y repositorios DSA (UE):
  `adlibrary.com/posts/ad-transparency-data-landscape`, `.../eu-dsa-ad-repositories-developers`.
- TikTok Commercial Content API (elegibilidad y límites): `developers.tiktok.com/products/commercial-content-api`.
- Google Ads Transparency Center sin API oficial para comercial:
  `adlibrary.com/posts/google-ads-transparency-center-api`.
- Scraper de Google ATC (campos, filtros y precio $0,99/1 000):
  `apify.com/clearpath/google-ads-transparency-scraper`.
