"""
Probe: ¿es la predicción de TRIBE v2 una señal de control usable para EDITAR?

Antes de construir cualquier lazo "editar contenido -> medir -> reeditar" hay que
medir tres cosas que hoy no están verificadas en este repo:

  P1. Determinismo / suelo de ruido: dos corridas del MISMO archivo deben dar
      predicciones idénticas. Si no, ningún delta pequeño es interpretable.
  P2. Sensibilidad a operadores de edición: mutear y cortar deben mover la señal
      por encima del suelo de ruido. (Fase 0.5 ya avisó: la ruta audio discrimina
      poco — 1.5% a 14% — así que esto NO es obvio.)
  P3. Atribución temporal: si corto una ventana [t1,t2), el perfil posterior debe
      parecerse al perfil original DESPLAZADO por el largo del corte. Si el perfil
      es globalmente acoplado, "corta el segundo malo" no tiene sentido.

La convención de alineación (el "caveat" que este probe dejaba abierto) quedó cerrada
el 2026-09-13 en ``scripts/validate_alignment.py``: ``alignment = "stimulus-aligned"``
(``preds[k]`` ↔ segundo k del estímulo), con precisión ≈ ±1.5 s. Ver
``docs/FASE_0.5_ALINEACION.md``.

Controles: el mismo audio con 5 s de silencio al inicio (desplazamiento puro, sin
quitar contenido) separa "atribución temporal" de "el modelo mira el clip entero".

Ruta: audio_only=True (la ruta limpia, sin Llama-3.2 gated ni gTTS), igual que
``scripts/validate_short_and_mute.py``.

Uso:
    .venv/Scripts/python.exe scripts/probe_edit_feedback.py
"""

from __future__ import annotations

import json
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
from service.metrics.engagement import find_dips, find_peaks, temporal_profile  # noqa: E402
from service.metrics.roi import RoiIndex  # noqa: E402

SRC = Path("ads/comercial.wav")
OUT_DIR = Path("data/discovery/probe")
TARGET_SR = 16000
CUT = (8.0, 12.0)  # ventana eliminada en el operador "cut"
SHIFT = 5.0  # silencio antepuesto en el control de desplazamiento


def load_mono_16k(path: Path) -> np.ndarray:
    """Lee cualquier wav a mono 16 kHz float32 (el extractor de audio espera voz)."""
    x, sr = sf.read(str(path), always_2d=True)
    x = x.mean(axis=1).astype(np.float32)
    if sr != TARGET_SR:
        g = np.gcd(int(sr), TARGET_SR)
        x = resample_poly(x, TARGET_SR // g, int(sr) // g).astype(np.float32)
    return x


def write_variant(name: str, x: np.ndarray) -> Path:
    path = OUT_DIR / f"{name}.wav"
    sf.write(str(path), x, TARGET_SR)
    return path


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    if n < 3:
        return float("nan")
    a, b = a[:n], b[:n]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def main() -> int:
    import pandas as pd
    import torch

    from tribev2.demo_utils import TribeModel, get_audio_and_text_events

    from core.tribe_model import resolve_checkpoint_dir

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SRC.is_file():
        print(f"No existe {SRC}")
        return 1

    print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()}")
    print(f"fuente: {SRC}")

    base = load_mono_16k(SRC)
    dur = len(base) / TARGET_SR
    print(f"duración base: {dur:.2f}s @ {TARGET_SR} Hz mono\n")

    # --- variantes -------------------------------------------------------
    a, b = int(CUT[0] * TARGET_SR), int(CUT[1] * TARGET_SR)
    # "strobe": 1 s de contenido + 1 s de silencio en bucle. No mejora nada del
    # creativo; multiplica los ONSETS. Si con eso sube el objetivo, el objetivo
    # es basura (exploit degenerado).
    seg = TARGET_SR
    n_blocks = int(np.ceil(len(base) / (2 * seg)))
    strobe = np.zeros(n_blocks * 2 * seg, dtype=np.float32)
    for i in range(n_blocks):
        src = base[i * seg : i * seg + seg]
        strobe[i * 2 * seg : i * 2 * seg + len(src)] = src
    strobe = strobe[: len(base)]

    variants = {
        "base": base,
        "mute": np.zeros_like(base),
        "cut_8_12": np.concatenate([base[:a], base[b:]]),
        "shift_5s": np.concatenate([np.zeros(int(SHIFT * TARGET_SR), np.float32), base]),
        "strobe": strobe,
    }
    paths = {name: write_variant(name, x) for name, x in variants.items()}
    for name, p in paths.items():
        print(f"  {name:10s} {len(variants[name]) / TARGET_SR:6.2f}s  {p}")

    # --- modelo residente -------------------------------------------------
    local_dir = resolve_checkpoint_dir("facebook/tribev2", "./cache")
    t0 = time.time()
    model = TribeModel.from_pretrained(local_dir, cache_folder="./cache", device="auto")
    model.data.num_workers = 0  # Windows: evita WinError 1455
    print(f"\nmodelo cargado en {time.time() - t0:.1f}s\n")

    def predict(wav: Path):
        event = {
            "type": "Audio",
            "filepath": str(wav),
            "start": 0,
            "timeline": "default",
            "subject": "default",
        }
        events = get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
        t = time.time()
        preds, segments = model.predict(events=events, verbose=False)
        preds = np.asarray(preds)
        assert preds.shape[1] == FSAAVERAGE5_VERTICES, preds.shape
        return preds, segments, time.time() - t

    runs: dict[str, np.ndarray] = {}
    rows = []
    for name, wav in paths.items():
        preds, segments, elapsed = predict(wav)
        runs[name] = preds
        np.save(OUT_DIR / f"preds_{name}.npy", preds)
        rows.append(
            {
                "variante": name,
                "dur_s": round(len(variants[name]) / TARGET_SR, 2),
                "n_seg": len(segments),
                "seg_esperados": int(round(len(variants[name]) / TARGET_SR)),
                "shape": str(preds.shape),
                "mean_abs": round(float(np.mean(np.abs(preds))), 6),
                "t_s": round(elapsed, 2),
            }
        )

    # Segunda corrida del mismo archivo -> suelo de ruido (P1)
    preds_a2, _, t_a2 = predict(paths["base"])
    rows.append(
        {
            "variante": "base_rerun",
            "dur_s": round(dur, 2),
            "n_seg": len(preds_a2),
            "seg_esperados": int(round(dur)),
            "shape": str(preds_a2.shape),
            "mean_abs": round(float(np.mean(np.abs(preds_a2))), 6),
            "t_s": round(t_a2, 2),
        }
    )

    pd.set_option("display.width", 200)
    print("\n=== VARIANTES ===")
    print(pd.DataFrame(rows).to_string(index=False))

    base_p = runs["base"]
    prof = {name: temporal_profile(p) for name, p in runs.items()}
    prof["base_rerun"] = temporal_profile(preds_a2)

    # --- P1: suelo de ruido ----------------------------------------------
    d = np.abs(base_p - preds_a2)
    noise = {
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "rel_mean_abs_diff": float(d.mean() / max(np.mean(np.abs(base_p)), 1e-12)),
        "profile_pearson": pearson(prof["base"], prof["base_rerun"]),
    }

    # --- P2: sensibilidad --------------------------------------------------
    def effect(name: str) -> dict:
        p = runs[name]
        n = min(p.shape[0], base_p.shape[0])
        d = np.abs(p[:n] - base_p[:n])
        return {
            "mean_abs_model": float(np.mean(np.abs(p))),
            "delta_vs_base_mean": float(d.mean()),
            "delta_vs_base_max": float(d.max()),
            "delta_rel": float(d.mean() / max(np.mean(np.abs(base_p)), 1e-12)),
            "snr_vs_noise": float(d.mean() / max(noise["mean_abs_diff"], 1e-12)),
            "profile_pearson": pearson(prof[name], prof["base"]),
        }

    sensitivity = {name: effect(name) for name in ("mute", "cut_8_12", "shift_5s")}

    # --- P3: atribución temporal ------------------------------------------
    k = int(CUT[1] - CUT[0])  # desplazamiento esperado si el perfil es local
    pc, pb = prof["cut_8_12"], prof["base"]
    lo = int(CUT[0])
    attribution = {
        "cut_shift_ts": k,
        "corr_prefix_before_cut": pearson(pc[:lo], pb[:lo]),
        "corr_after_cut_shifted": pearson(pc[lo:], pb[lo + k:]),  # hipótesis LOCAL
        "corr_after_cut_unshifted": pearson(pc[lo:], pb[lo:]),  # hipótesis GLOBAL
        "n_ts_prefix": lo,
        "n_ts_after": int(len(pc) - lo),
    }
    shift_ts = int(SHIFT)
    attribution["corr_shift_control_shifted"] = pearson(
        prof["shift_5s"][shift_ts:], pb[: len(prof["shift_5s"]) - shift_ts]
    )
    # ¿el silencio antepuesto se ve como activación propia o como nada?
    attribution["shift_control_leading_mean"] = float(
        np.mean(np.abs(runs["shift_5s"][:shift_ts]))
    )
    attribution["shift_control_body_mean"] = float(
        np.mean(np.abs(runs["shift_5s"][shift_ts:]))
    )

    # --- picos/valles: ¿el feedback sirve para localizar la edición? ------
    peaks_dips = {
        name: {
            "peaks": [p["timestep"] for p in find_peaks(prof[name], 1.3)],
            "dips": find_dips(prof[name], 0.7),
        }
        for name in runs
    }

    # --- P4: ¿cuánto se puede explotar el objetivo? (Goodhart) -------------
    # "repeat_peak": tilea el segundo de mayor activación hasta llenar el clip.
    # No cuenta nada; maximiza lo que sea que midamos.
    body = int(HEMODYNAMIC_OFFSET_SECONDS)
    t_star = int(np.argmax(prof["base"][body:])) + body
    peak_seg = base[t_star * TARGET_SR : (t_star + 1) * TARGET_SR]
    reps = int(np.ceil(len(base) / len(peak_seg)))
    repeat_peak = np.tile(peak_seg, reps)[: len(base)].astype(np.float32)
    rp_path = write_variant("repeat_peak", repeat_peak)
    preds_rp, segs_rp, t_rp = predict(rp_path)
    runs["repeat_peak"] = preds_rp
    prof["repeat_peak"] = temporal_profile(preds_rp)
    np.save(OUT_DIR / "preds_repeat_peak.npy", preds_rp)

    roi = RoiIndex.from_schaefer("data/atlas/schaefer200")

    def body_net(p, net):
        return float(np.mean(np.abs(roi.timeseries(p, net)[body:])))

    objectives = {
        "mean_abs_all": lambda p: float(np.mean(np.abs(p))),
        "mean_abs_body": lambda p: float(np.mean(np.abs(p[body:]))),
        "SalVentAttn_body": lambda p: body_net(p, "SalVentAttn"),
        "DorsAttn_body": lambda p: body_net(p, "DorsAttn"),
        "Default_body": lambda p: body_net(p, "Default"),
        "Vis_body": lambda p: body_net(p, "Vis"),
        "peak_count_body": lambda p: float(
            len(find_peaks(temporal_profile(p)[body:], 1.3))
        ),
    }
    table = {
        obj: {name: round(fn(runs[name]), 6) for name in runs}
        for obj, fn in objectives.items()
    }
    # Un objetivo es explotable si una variante degenerada lo maximiza.
    degenerate = ("strobe", "repeat_peak", "shift_5s")
    exploitable = {}
    for obj in objectives:
        best_deg = max(degenerate, key=lambda d: table[obj][d])
        best_honest = max(table[obj][k] for k in ("base", "mute", "cut_8_12"))
        exploitable[obj] = best_deg if table[obj][best_deg] > best_honest else None

    # --- P5: lag efectivo (segundo <-> TR). El patrón strobe es ground truth:
    # contenido en t par, silencio en t impar. Barremos el desfase.
    # SUPERSEDIDO (2026-09-13): un diseño periódico aliasa —el lag solo se resuelve
    # módulo 2 TR— y el perfil mean|a| responde con polaridad no obvia. La convención
    # ya está medida en scripts/validate_alignment.py (edge = -1.48 + 1.023·m).
    # El número de aquí es un diagnóstico, no el resultado.
    n_ts = len(prof["strobe"])
    indicator = np.array([1.0 if t % 2 == 0 else 0.0 for t in range(n_ts)])
    lags = {}
    for L in range(-4, 8):
        if L >= 0:
            r = pearson(prof["strobe"][L:], indicator[: n_ts - L])
        else:
            r = pearson(prof["strobe"][: n_ts + L], indicator[-L:])
        lags[L] = None if not np.isfinite(r) else round(float(r), 4)
    best_lag = max(lags, key=lambda k: lags[k] if lags[k] is not None else -2)

    # P5b: pulso aislado -> retardo absoluto (el diseño periódico aliasa).
    burst_len = 2.0
    burst_start = 10.0
    burst = np.zeros_like(base)
    bs = int(burst_start * TARGET_SR)
    burst[bs : bs + int(burst_len * TARGET_SR)] = base[bs : bs + int(burst_len * TARGET_SR)]
    bp_path = write_variant("burst_10_2s", burst)
    preds_burst, segs_burst, t_burst = predict(bp_path)
    prof_burst = temporal_profile(preds_burst)
    np.save(OUT_DIR / "preds_burst_10_2s.npy", preds_burst)
    b_peak = int(np.argmax(prof_burst))
    w = np.maximum(prof_burst - np.median(prof_burst), 0.0)
    b_centroid = float((w * np.arange(len(w))).sum() / max(w.sum(), 1e-12))
    lag_burst = {
        "burst_start_s": burst_start,
        "burst_len_s": burst_len,
        "t_pico": b_peak,
        "retardo_pico_s": b_peak - burst_start,
        "centroide_s": round(b_centroid, 2),
        "retardo_centroide_s": round(b_centroid - burst_start, 2),
        "offset_declarado_s": HEMODYNAMIC_OFFSET_SECONDS,
        "perfil": [round(float(v), 4) for v in prof_burst],
        "t_s": round(t_burst, 2),
    }

    # P5c: dos pulsos en el MISMO clip (t=5 y t=15). La diferencia de perfiles
    # aísla lo que se mueve con el contenido de lo que está clavado a la posición.
    sweep = {}
    for start in (5.0, 15.0):
        x = np.zeros_like(base)
        s = int(start * TARGET_SR)
        x[s : s + int(burst_len * TARGET_SR)] = base[s : s + int(burst_len * TARGET_SR)]
        p = write_variant(f"burst_{int(start)}", x)
        pr, _, dt = predict(p)
        sweep[int(start)] = temporal_profile(pr)
        np.save(OUT_DIR / f"preds_burst_{int(start)}.npy", pr)
        sweep[f"t_{int(start)}"] = round(dt, 2)

    diff = sweep[15] - sweep[5]
    n = min(len(diff), len(prof_burst))
    lag_sweep = {
        "perfil_pulso_5": [round(float(v), 4) for v in sweep[5]],
        "perfil_pulso_15": [round(float(v), 4) for v in sweep[15]],
        "diff_15_menos_5": [round(float(v), 4) for v in diff],
        "argmax_diff": int(np.argmax(diff)),
        "argmin_diff": int(np.argmin(diff)),
        "retardo_estimado_max_s": int(np.argmax(diff)) - 15,
        "retardo_estimado_min_s": int(np.argmin(diff)) - 5,
        "corr_pulso_15_con_pulso_5": pearson(sweep[15], sweep[5]),
        "corr_diff_con_perfil_base": pearson(diff[:n], prof["base"][:n]),
    }

    # P6: barrido de ESPERA. ¿La elevación del inicio escala con cuánto se
    # tarda en arrancar el contenido (hipótesis "el silencio inicial engancha"),
    # o queda clavada en el arranque del clip sea cual sea la espera (artefacto)?
    wait_sweep = {}
    for w in (0.0, 1.0, 3.0, 5.0, 8.0):
        if w == 0.0:
            prof_w, dt_w = prof["base"], 0.0
        else:
            x = np.concatenate([np.zeros(int(w * TARGET_SR), np.float32), base])
            pr, _, dt_w = predict(write_variant(f"wait_{int(w)}", x))
            np.save(OUT_DIR / f"preds_wait_{int(w)}.npy", pr)
            prof_w = temporal_profile(pr)
        wait_sweep[int(w)] = {
            "transitorio_t0_2": round(float(prof_w[:3].mean()), 5),
            "media_global": round(float(prof_w.mean()), 5),
            "pico_t": int(np.argmax(prof_w)),
            "pico_valor": round(float(prof_w.max()), 5),
            "perfil_t0_9": [round(float(v), 4) for v in prof_w[:10]],
            "t_s": round(dt_w, 2),
        }

    # P7: ¿se premia el SILENCIO digital o cualquier arranque QUIETO? Tres
    # aperturas de 3 s antes del mismo contenido: ceros, ruido de sala a nivel
    # bajo y ruido audible bajo. Si solo gana "ceros", el efecto es del silencio
    # digital (artefacto); si ganan los tres, es del nivel/expectación.
    rng = np.random.default_rng(0)
    n3 = int(3.0 * TARGET_SR)
    quiet = {
        "quiet_ceros": np.zeros(n3, dtype=np.float32),
        "quiet_sala": rng.normal(0, 0.003, n3).astype(np.float32),
        "quiet_ruido": rng.normal(0, 0.03, n3).astype(np.float32),
    }
    quiet_sweep = {}
    for name, opener in quiet.items():
        pr, _, dt_q = predict(write_variant(name, np.concatenate([opener, base])))
        pf = temporal_profile(pr)
        np.save(OUT_DIR / f"preds_{name}.npy", pr)
        quiet_sweep[name] = {
            "rms_apertura": round(float(np.sqrt((opener**2).mean())), 5),
            "transitorio_t0_2": round(float(pf[:3].mean()), 5),
            "media_global": round(float(pf.mean()), 5),
            "pico_t": int(np.argmax(pf)),
            "perfil_t0_9": [round(float(v), 4) for v in pf[:10]],
            "t_s": round(dt_q, 2),
        }

    result = {
        "fuente": str(SRC),
        "dur_s": round(dur, 2),
        "tr_seconds": TR_SECONDS,
        "cut_window_s": list(CUT),
        "shift_s": SHIFT,
        "variantas": rows,
        "P1_ruido_repeticion": noise,
        "P2_sensibilidad": sensitivity,
        "P3_atribucion": attribution,
        "picos_valles": peaks_dips,
        "P4_objetivos": table,
        "P4_objetivo_explotado_por": exploitable,
        "P4_repeat_peak": {
            "t_star": t_star,
            "seg_s": 1.0,
            "n_seg": len(segs_rp),
            "t_s": round(t_rp, 2),
            "mean_abs": round(float(np.mean(np.abs(preds_rp))), 6),
        },
        "P5_lag_efectivo_tr": {
            "por_lag": lags,
            "mejor_lag": best_lag,
            "r_en_offset_declarado": lags.get(int(HEMODYNAMIC_OFFSET_SECONDS)),
        },
        "P5b_retardo_pulso_aislado": lag_burst,
        "P5c_barrido_pulsos": lag_sweep,
        "P6_barrido_espera": wait_sweep,
        "P7_apertura_quieta": quiet_sweep,
    }
    (OUT_DIR / "probe.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== P1 · SUELO DE RUIDO (mismo archivo, 2 corridas) ===")
    print(f"  max |Δ| = {noise['max_abs_diff']:.3e}   mean |Δ| = {noise['mean_abs_diff']:.3e}")
    print(f"  Δ relativo = {noise['rel_mean_abs_diff']:.3%}   r(perfil) = {noise['profile_pearson']:.4f}")

    print("\n=== P2 · SENSIBILIDAD A OPERADORES ===")
    print(pd.DataFrame(sensitivity).T.round(6).to_string())

    print("\n=== P3 · ATRIBUCIÓN TEMPORAL ===")
    print(f"  antes del corte  r = {attribution['corr_prefix_before_cut']:.4f}  (n={attribution['n_ts_prefix']})")
    print(f"  tras corte, DESPLAZADO r = {attribution['corr_after_cut_shifted']:.4f}  <- hipótesis local")
    print(f"  tras corte, SIN despla r = {attribution['corr_after_cut_unshifted']:.4f}  <- hipótesis global")
    print(f"  control shift: cuerpo desplazado r = {attribution['corr_shift_control_shifted']:.4f}")
    print(f"  control shift: silencio inicial mean|a| = {attribution['shift_control_leading_mean']:.5f}"
          f"  vs cuerpo {attribution['shift_control_body_mean']:.5f}")

    print("\n=== PICOS / VALLES (t) ===")
    for name, pd_ in peaks_dips.items():
        print(f"  {name:10s} picos={pd_['peaks']}  valles={pd_['dips']}")

    print("\n=== P4 · OBJETIVOS vs EXPLOIT DEGENERADO ===")
    print(pd.DataFrame(table).T.to_string())
    print(f"\n  repeat_peak: t*={t_star}s (segundo de mayor activación, body), mean|a|={np.mean(np.abs(preds_rp)):.5f}, {t_rp:.1f}s")
    print("  objetivo explotado por variante degenerada (strobe/repeat_peak/shift_5s):")
    for obj, who in exploitable.items():
        print(f"    {obj:18s} -> {who if who else 'no (aguanta)'}")

    print("\n=== P5 · LAG EFECTIVO (strobe: contenido t par / silencio t impar) ===")
    print("  r por lag (TR): " + "  ".join(f"{k}:{v}" for k, v in lags.items()))
    print(f"  mejor lag = {best_lag} TR   ·   r en offset declarado ({int(HEMODYNAMIC_OFFSET_SECONDS)} s) = {lags.get(int(HEMODYNAMIC_OFFSET_SECONDS))}")
    print("  perfil strobe [t0..t11] = " + " ".join(f"{v:.3f}" for v in prof["strobe"][:12]))
    print("  perfil base   [t0..t11] = " + " ".join(f"{v:.3f}" for v in prof["base"][:12]))

    print("\n=== P5b · RETARDO ABSOLUTO (pulso 2 s en t=10 sobre silencio) ===")
    print(f"  pico en t={lag_burst['t_pico']}s -> retardo {lag_burst['retardo_pico_s']}s"
          f"   ·   centroide {lag_burst['centroide_s']}s -> retardo {lag_burst['retardo_centroide_s']}s")
    print(f"  offset declarado por el modelo = {lag_burst['offset_declarado_s']}s")
    print("  perfil = " + " ".join(f"{v:.3f}" for v in lag_burst["perfil"]))

    print("\n=== P5c · ¿EL BUMP SE MUEVE CON EL PULSO O ESTÁ CLAVADO A LA POSICIÓN? ===")
    print(f"  r(pulso@15 vs pulso@5) = {lag_sweep['corr_pulso_15_con_pulso_5']:.4f}  (1.0 = perfil idéntico, o sea posición pura)")
    print(f"  argmax diff  = t={lag_sweep['argmax_diff']} -> retardo {lag_sweep['retardo_estimado_max_s']}s respecto al pulso@15")
    print(f"  argmin diff  = t={lag_sweep['argmin_diff']} -> retardo {lag_sweep['retardo_estimado_min_s']}s respecto al pulso@5")
    print("  diff (15-5) = " + " ".join(f"{v:+.3f}" for v in lag_sweep["diff_15_menos_5"]))

    print("\n=== P6 · BARRIDO DE ESPERA (silencio inicial + el anuncio) ===")
    print(f"  {'espera':>6} {'t0-2':>7} {'media':>7} {'pico@t':>7} {'pico':>7}   perfil t0..t9")
    for w, row in wait_sweep.items():
        print(f"  {w:>5}s {row['transitorio_t0_2']:>7.4f} {row['media_global']:>7.4f}"
              f" {row['pico_t']:>7d} {row['pico_valor']:>7.4f}   "
              + " ".join(f"{v:.3f}" for v in row["perfil_t0_9"]))

    print("\n=== P7 · APERTURA DE 3 s: ¿SILENCIO DIGITAL O CUALQUIER COSA QUIETA? ===")
    print(f"  {'apertura':>12} {'rms':>8} {'t0-2':>7} {'media':>7} {'pico@t':>7}   perfil t0..t9")
    for name, row in quiet_sweep.items():
        print(f"  {name:>12} {row['rms_apertura']:>8.5f} {row['transitorio_t0_2']:>7.4f}"
              f" {row['media_global']:>7.4f} {row['pico_t']:>7d}   "
              + " ".join(f"{v:.3f}" for v in row["perfil_t0_9"]))

    print(f"\nartefactos: {OUT_DIR}/probe.json + preds_*.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
