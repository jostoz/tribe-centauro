"""
Cierre de la Fase 0.5 — ALINEACIÓN TEMPORAL de TRIBE v2.

Pregunta exacta
---------------
Si un evento del estímulo ocurre en el segundo ``t`` del anuncio, ¿en qué índice
``k`` de ``preds`` aparece su respuesta?

  H-A (alineado):   preds[k] es la respuesta al estímulo del segundo k   -> k = t
  H-B (BOLD crudo): preds[k] es el BOLD del instante k                   -> k = t + 5

Un desplazamiento constante de 5 s **no** se puede medir comparando una variante
desplazada contra la base: ambas hipótesis predicen el mismo desplazamiento. Hay
que referir el índice a un ONSET del estímulo, es decir, medir el retardo efectivo
del sistema respecto al contenido.

Método A (escalón, primario)
    El operador "mutear desde m" deja el prefijo [0, m) byte-idéntico y silencia el
    resto, así que el cambio de respuesta aparece como un escalón en
    Δ(k) = mean_v |preds_variante(k, v) − preds_base(k, v)|.
    ``edge_index`` localiza el escalón por máxima pendiente (post−pre en ventana w).
    Se ajusta edge = α + β·m. β ≈ 1 confirma que el índice sigue al estímulo; α es
    la convención. El estimador se calibra sobre las DOS hipótesis simuladas
    (escalón convolucionado con la HRF canónica), de modo que el veredicto no
    depende de umbrales elegidos a mano: gana la hipótesis cuya α simulada esté
    más cerca de la observada.

    NOTA: el punto medio acumulado (cumsum al 50 %) NO sirve aquí — con una meseta
    que dura hasta el final del clip, ese estadístico queda sesgado hacia el centro
    de la meseta (~m+5), que es justo el valor de H-B. Se descartó tras medirlo.

Método B (independiente)
    ``envelope tracking``: correlación cruzada entre la envolvente de audio (RMS por
    segundo) y el perfil temporal de la ROI más variable. El desfase que maximiza r
    es el retardo efectivo. H-A -> ~0 TR, H-B -> ~+5 TR.

Ruta: ``audio_only=True``. Requiere GPU (se verifica; sin GPU no hay verificación).

Uso:
    .venv/Scripts/python.exe scripts/validate_alignment.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ordering import (  # noqa: E402
    FSAAVERAGE5_VERTICES,
    HEMODYNAMIC_OFFSET_SECONDS,
    TR_SECONDS,
)

OUT_DIR = Path("data/discovery/alignment")
TARGET_SR = 16000
MUTE_FRACTIONS = (0.33, 0.50, 0.66)
MIN_TAIL_S = 5.0
EDGE_W = 3  # semiancho del detector de escalón, en TR

# Fuentes FIJAS (no glob: data/ads/ lo reescribe el pipeline de discovery).
SOURCES = [
    Path("ads/comercial.wav"),
    Path("data/ads/-BrHpr8R9Yc.wav"),
    Path("data/ads/BsMrRFH390k.wav"),
    Path("data/ads/GH0unR4JF04.wav"),
]


def load_mono_16k(path: Path) -> np.ndarray:
    """Lee cualquier wav a mono 16 kHz float32 (el extractor de audio espera voz)."""
    x, sr = sf.read(str(path), always_2d=True)
    x = x.mean(axis=1).astype(np.float32)
    if sr != TARGET_SR:
        g = np.gcd(int(sr), TARGET_SR)
        x = resample_poly(x, TARGET_SR // g, int(sr) // g).astype(np.float32)
    return x


def envelope_1hz(x: np.ndarray) -> np.ndarray:
    """RMS por segundo: la envolvente de energía en la misma rejilla que los TR."""
    n = len(x) // TARGET_SR
    return np.array(
        [float(np.sqrt(np.mean(x[i * TARGET_SR : (i + 1) * TARGET_SR] ** 2))) for i in range(n)]
    )


def delta_l1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cambio medio absoluto por TR entre dos corridas (deterministas, P1)."""
    n = min(len(a), len(b))
    return np.abs(a[:n] - b[:n]).mean(axis=1)


def edge_index(delta: np.ndarray, w: int = EDGE_W) -> int | None:
    """Índice del escalón: máximo de (media post − media pre) en ventana ±w."""
    d = np.clip(np.asarray(delta, dtype=np.float64), 0.0, None)
    best, best_k = -np.inf, None
    for k in range(w, len(d) - w):
        slope = d[k : k + w].mean() - d[k - w : k].mean()
        if slope > best:
            best, best_k = slope, k
    return best_k


def hrf_canonical(n: int, tr: float = 1.0) -> np.ndarray:
    """HRF doble gamma (pico en 5 s), muestreada a 1 muestra por TR."""
    t = np.arange(n) * tr
    a1, b1, a2, b2, c = 6.0, 1.0, 16.0, 1.0, 0.35
    h = (
        t ** (a1 - 1) * b1**a1 * np.exp(-b1 * t) / math.gamma(a1)
        - c * t ** (a2 - 1) * b2**a2 * np.exp(-b2 * t) / math.gamma(a2)
    )
    return h / np.abs(h).max()


def simulated_delta(m: int, n: int, hypothesis: str) -> np.ndarray:
    """Δ sintética de un escalón del estímulo en m bajo cada hipótesis."""
    step = np.zeros(n)
    step[m:] = 1.0
    bold = np.convolve(step, hrf_canonical(n))[:n]  # BOLD en tiempo crudo
    if hypothesis == "A":  # índice = tiempo de estímulo (5 s ya compensados)
        preds = np.concatenate([bold[int(HEMODYNAMIC_OFFSET_SECONDS) :], np.zeros(int(HEMODYNAMIC_OFFSET_SECONDS))])
    else:  # índice = BOLD crudo
        preds = bold
    return np.abs(preds)


def calibrate(ms: list[int], n: int, w: int = EDGE_W) -> dict:
    """α que el propio estimador devuelve sobre cada hipótesis simulada."""
    out = {}
    for hyp in ("A", "B"):
        pairs = [(m, edge_index(simulated_delta(m, n, hyp), w)) for m in ms]
        pairs = [(m, e) for m, e in pairs if e is not None]
        beta, alpha = np.polyfit([p[0] for p in pairs], [p[1] for p in pairs], 1)
        out[hyp] = {"alpha": float(alpha), "beta": float(beta), "pares": [[int(a), int(b)] for a, b in pairs]}
    return out


def fit_edge(pairs: list[tuple[int, int]]) -> dict:
    xs = np.array([p[0] for p in pairs], dtype=np.float64)
    ys = np.array([p[1] for p in pairs], dtype=np.float64)
    beta, alpha = np.polyfit(xs, ys, 1)
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "n": len(pairs),
        "residuo_rms_tr": float(np.std(ys - (alpha + beta * xs))),
        "rango_edge_menos_m": [int((ys - xs).min()), int((ys - xs).max())],
    }


def best_lag(prof: np.ndarray, env: np.ndarray, lo: int = -3, hi: int = 10) -> dict:
    """Desfase (TR) que maximiza la correlación entre perfil y envolvente."""
    a = (prof - prof.mean()) / (prof.std() + 1e-12)
    b = (env - env.mean()) / (env.std() + 1e-12)
    scores = {}
    for L in range(lo, hi + 1):
        if L >= 0:
            u, v = a[L:], b[: len(b) - L]
        else:
            u, v = a[: len(a) + L], b[-L:]
        n = min(len(u), len(v))
        if n >= 6:
            scores[L] = float(np.corrcoef(u[:n], v[:n])[0, 1])
    best = max(scores, key=lambda k: scores[k])
    return {
        "mejor_lag": int(best),
        "r_mejor": round(scores[best], 4),
        "r_por_lag": {str(k): round(v, 4) for k, v in scores.items()},
    }


def main() -> int:
    import pandas as pd
    import torch

    from tribev2.demo_utils import TribeModel, get_audio_and_text_events

    from core.tribe_model import resolve_checkpoint_dir

    if not torch.cuda.is_available():
        print("FALLA: la verificación de alineación exige GPU.")
        return 1
    print(f"torch {torch.__version__} | cuda={torch.cuda.get_device_name(0)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = [p for p in SOURCES if p.is_file()]
    if not sources:
        print("No hay fuentes de audio")
        return 1

    local_dir = resolve_checkpoint_dir("facebook/tribev2", "./cache")
    t0 = time.time()
    model = TribeModel.from_pretrained(local_dir, cache_folder="./cache", device="auto")
    model.data.num_workers = 0  # Windows: evita WinError 1455
    print(f"modelo cargado en {time.time() - t0:.1f}s")

    def predict(wav: Path) -> np.ndarray:
        event = {
            "type": "Audio",
            "filepath": str(wav),
            "start": 0,
            "timeline": "default",
            "subject": "default",
        }
        events = get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
        preds, _ = model.predict(events=events, verbose=False)
        preds = np.asarray(preds)
        assert preds.shape[1] == FSAAVERAGE5_VERTICES, preds.shape
        return preds

    report: dict = {
        "tr_seconds": TR_SECONDS,
        "declared_offset_seconds": HEMODYNAMIC_OFFSET_SECONDS,
        "edge_window_tr": EDGE_W,
        "metodo": "A: escalon 'mutear desde m' + detector de maxima pendiente; B: envelope tracking",
        "fuentes": [],
    }
    pairs: list[tuple[int, int]] = []
    lags: list[int] = []
    rs: list[float] = []
    max_tr = 0

    for src in sources:
        base_x = load_mono_16k(src)
        dur = len(base_x) / TARGET_SR
        if dur < 12:
            print(f"[skip] {src.name}: {dur:.1f}s < 12s")
            continue
        onsets = sorted(
            {int(round(dur * f)) for f in MUTE_FRACTIONS if dur - round(dur * f) >= MIN_TAIL_S}
        )
        print(f"\n=== {src.name} ({dur:.2f}s) — mute desde {onsets} ===")

        base_path = OUT_DIR / f"{src.stem}_base.wav"
        sf.write(str(base_path), base_x, TARGET_SR)
        preds_base = predict(base_path)
        np.save(OUT_DIR / f"preds_{src.stem}_base.npy", preds_base)
        max_tr = max(max_tr, int(preds_base.shape[0]))
        print(f"  base  {preds_base.shape}")

        entry: dict = {"fuente": str(src), "dur_s": round(dur, 2), "n_tr": int(preds_base.shape[0]), "mutes": []}
        for m in onsets:
            var = base_x.copy()
            var[m * TARGET_SR :] = 0.0
            vpath = OUT_DIR / f"{src.stem}_mute{m}.wav"
            sf.write(str(vpath), var, TARGET_SR)
            preds_v = predict(vpath)
            np.save(OUT_DIR / f"preds_{src.stem}_mute{m}.npy", preds_v)
            if preds_v.shape != preds_base.shape:
                print(f"  [skip] m={m}: shape {preds_v.shape} != {preds_base.shape}")
                continue
            d = delta_l1(preds_v, preds_base)
            e = edge_index(d)
            pairs.append((m, e))
            entry["mutes"].append(
                {
                    "m_s": m,
                    "edge_tr": int(e),
                    "edge_menos_m": int(e - m),
                    "delta_por_tr": [round(float(x), 4) for x in d],
                }
            )
            print(f"  mute m={m:2d}s  edge={e:2d}  (edge-m={e - m:+d})")

        var_roi = preds_base.var(axis=0)
        roi = np.argsort(var_roi)[::-1][:1000]
        prof = preds_base[:, roi].mean(axis=1)
        env = envelope_1hz(base_x)
        n = min(len(env), len(prof))
        lag = best_lag(prof[:n], env[:n])
        lags.append(lag["mejor_lag"])
        rs.append(lag["r_mejor"])
        entry["envelope_tracking"] = lag
        print(f"  envolvente: lag={lag['mejor_lag']:+d} TR  r={lag['r_mejor']:+.3f}")
        report["fuentes"].append(entry)

    fit = fit_edge(pairs)
    ms = sorted({m for m, _ in pairs})
    sim = calibrate(ms, max(max_tr, max(ms) + 20))
    obs = fit["alpha"]
    dist = {h: abs(obs - sim[h]["alpha"]) for h in ("A", "B")}
    verdict = "ALINEADO_AL_ESTIMULO" if dist["A"] < dist["B"] else "BOLD_CRUDO_SIN_COMPENSAR"

    report["metodo_A_escalon"] = {
        "pares_m_edge": [[int(m), int(e)] for m, e in pairs],
        "ajuste_edge_igual_alpha_mas_beta_m": fit,
        "calibracion_hipotesis_simuladas": sim,
        "distancia_a_cada_hipotesis": dist,
    }
    report["metodo_B_envolvente"] = {
        "mejores_lags": lags, "r_de_cada_mejor_lag": rs,
        "lag_mediano": float(np.median(lags)) if lags else None,
        "r_mediana": float(np.median(rs)) if rs else None,
    }
    report["veredicto"] = verdict
    (OUT_DIR / "alignment.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== MÉTODO A · ESCALÓN ===")
    print(f"  pares (m, edge): {report['metodo_A_escalon']['pares_m_edge']}")
    print(f"  edge = {fit['alpha']:+.2f} + {fit['beta']:.3f}·m   residuo RMS={fit['residuo_rms_tr']:.2f} TR")
    print(f"  calibración simulada:  H-A α={sim['A']['alpha']:+.2f}   H-B α={sim['B']['alpha']:+.2f}")
    print(f"  distancia observada:   A={dist['A']:.2f}   B={dist['B']:.2f}")
    print("\n=== MÉTODO B · ENVOLVENTE ===")
    print(f"  lags: {lags}  mediana={report['metodo_B_envolvente']['lag_mediano']}"
          f"  (r mediana={report['metodo_B_envolvente']['r_mediana']:+.3f})")
    print(f"\n=== VEREDICTO: {verdict} ===")
    print(f"artefactos: {OUT_DIR}/alignment.json + preds_*.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
