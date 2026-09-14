"""Análisis del diseño de pares emparejados (LAMBDA) + correlación con outcome humano.

Dos pruebas sobre la misma submuestra:

1. **Pareado por duración** (`pairs.json`): `Pace` humano (high vs low) → composición de cada red
   Schaefer. Emparejado por diseño ⇒ permutación de signos intra-par y sin covariables.
2. **Correlación con `recall_score`** (memorabilidad humana de 1 749 participantes): ¿el perfil
   neural predice *algo* humano? Es un outcome humano, **no** CTR ni ventas.

Uso:
    .venv/Scripts/python.exe scripts/analyze_pairs.py --pairs data/lambda/pairs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from discovery import store  # noqa: E402
from service.metrics.stats import paired_permutation_test  # noqa: E402

NETS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


def _shares(con) -> tuple:
    """``({id: share}, {id: corpus})`` para todos los anuncios con perfil neural."""
    shares, corpus = {}, {}
    for a in store.all_ads(con):
        neu = a.get("neural") or {}
        n = neu.get("networks")
        if not n:
            continue
        tot = sum(n[k] for k in NETS if k in n)
        shares[a["id"]] = np.array([n[k] / tot for k in NETS if k in n])
        corpus[a["id"]] = a.get("corpus")
    return shares, corpus


def main() -> int:
    ap = argparse.ArgumentParser(description="Análisis del diseño pareado + recall.")
    ap.add_argument("--pairs", default="data/lambda/pairs.json")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    con = store.connect()
    store.init(con)
    shares, _ = _shares(con)

    hi, lo, rec_hi, rec_lo, durs, usados = [], [], [], [], [], []
    for p in pairs:
        hi_id, lo_id = p["high"]["id"], p["low"]["id"]
        if hi_id in shares and lo_id in shares:
            hi.append(shares[hi_id])
            lo.append(shares[lo_id])
            rec_hi.append(float(p["high"].get("recall") or 0.0))
            rec_lo.append(float(p["low"].get("recall") or 0.0))
            durs.append(int(p.get("duration") or p.get("duration_high") or 0))
            usados += [hi_id, lo_id]

    if len(hi) < 4:
        print(f"pares con perfil neural: {len(hi)} de {len(pairs)} — aún no hay datos suficientes.")
        return 1

    A, B = np.array(hi), np.array(lo)
    print(f"pares usados: {len(A)} (de {len(pairs)}) | duración idéntica por par: {sorted(set(durs))} s")
    print("media high: " + " ".join(f"{k}={A[:,i].mean()*100:.1f}" for i, k in enumerate(NETS)))
    print("media low : " + " ".join(f"{k}={B[:,i].mean()*100:.1f}" for i, k in enumerate(NETS)))

    res = paired_permutation_test(A, B, n_permutations=args.n_perm, seed=args.seed)
    p_holm = res.holm()
    print(f"\nPACE humano (high − low), test PAREADO (n_perm={args.n_perm})")
    print(f"  {'red':11} {'delta(pp)':>10} {'p cruda':>9} {'p Holm':>9}  sig")
    for i, k in enumerate(NETS):
        sig = "sí" if p_holm[i] < 0.05 else "no"
        print(f"  {k:11} {res.statistic[i]*100:+10.2f} {res.p_values[i]:9.4f} {p_holm[i]:9.4f}  {sig}")

    # Correlación con el outcome humano (memorabilidad)
    try:
        from scipy import stats as sps

        all_ids = usados
        X = np.array([shares[i] for i in all_ids])
        r = np.array(rec_hi + rec_lo)
        print(f"\nMEMORABILIDAD humana (recall_score) vs perfil neural — Spearman (n={len(r)})")
        for i, k in enumerate(NETS):
            rho, p = sps.spearmanr(X[:, i], r)
            marca = "  <<<" if p < 0.05 else ""
            print(f"  {k:11} rho={rho:+.3f}  p={p:.4f}{marca}")
        print(
            "\n  Sin corrección por multiplicidad en este bloque (exploratorio). recall_score es\n"
            "  memorabilidad humana, NO CTR ni ventas: no es calibración de negocio."
        )
    except ImportError:
        print("\n(scipy no disponible: se omite la correlación con recall_score)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
