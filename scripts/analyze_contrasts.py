"""Análisis de contrastes del corpus (Grupo A/A2) sobre el perfil neural.

Compara la composición relativa de las 7 redes Schaefer entre grupos definidos por
atributos del VLM (ritmo, caras) y reporta el efecto **ajustado por duración**, porque en
el corpus el ritmo y la duración pueden estar confundidos.

Sin test de significancia y sin corrección por comparaciones múltiples (7 redes × k
contrastes): son tamaños de efecto descriptivos sobre unidades crudas sin calibrar.

Uso:
    .venv/Scripts/python.exe scripts/analyze_contrasts.py
"""

from __future__ import annotations

import statistics as st
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from discovery import store  # noqa: E402
from service.metrics.stats import group_permutation_test  # noqa: E402

NETS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


def _plain(s) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", str(s).lower()) if not unicodedata.combining(c)
    )


def load_rows(con) -> list:
    rows = []
    for a in store.all_ads(con):
        neu = a.get("neural") or {}
        if not neu.get("networks"):
            continue
        u = a.get("understanding") or {}
        n = neu["networks"]
        tot = sum(n[k] for k in NETS if k in n)
        rows.append(
            {
                "id": a["id"],
                "dur": float(a.get("duration") or 0),
                "ritmo": _plain(u.get("ritmo")),
                "caras": bool(u.get("hay_caras")),
                "share": np.array([n[k] / tot for k in NETS if k in n]),
            }
        )
    return rows


def _mean(rows) -> np.ndarray:
    return np.mean([r["share"] for r in rows], axis=0)


def _contrast(label: str, a: list, b: list, name_a: str, name_b: str) -> None:
    ma, mb = _mean(a), _mean(b)
    print(f"\n{label}  n={len(a)} ({name_a}) vs {len(b)} ({name_b})")
    print(f"  {'red':11} {name_a[:8]:>8} {name_b[:8]:>8} {'delta':>7}")
    for i, k in enumerate(NETS):
        print(f"  {k:11} {ma[i]*100:8.1f} {mb[i]*100:8.1f} {(ma[i]-mb[i])*100:+7.1f}")
    med_a = st.median([r["dur"] for r in a]) if a else float("nan")
    med_b = st.median([r["dur"] for r in b]) if b else float("nan")
    print(f"  duración mediana: {med_a:.0f}s vs {med_b:.0f}s")


def _adjusted(rows: list, label: str, is_a) -> None:
    """Coeficiente de la dummy del grupo, con la duración como covariable."""
    x = np.array([[1.0, 1.0 if is_a(r) else 0.0, r["dur"]] for r in rows])
    y = np.array([r["share"] for r in rows])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    print(f"\n{label}: efecto ajustado por duración (pp)")
    for i, k in enumerate(NETS):
        print(f"  {k:11} {beta[1][i]*100:+6.1f}")


def _formal(rows: list, label: str, is_a, n_perm: int, seed: int) -> None:
    """Test de permutación bilateral por red, con y sin ajuste por duración, + Holm."""
    values = np.array([r["share"] for r in rows])
    labels = np.array([bool(is_a(r)) for r in rows])
    dur = np.array([r["dur"] for r in rows])

    unadj = group_permutation_test(values, labels, n_permutations=n_perm, seed=seed)
    adj = group_permutation_test(
        values, labels, covariates=dur[:, None],
        n_permutations=n_perm, seed=seed, covariate_name="duración",
    )
    p_holm = adj.holm()

    print(f"\n{label} — test formal (n_perm={n_perm}, semilla={seed})")
    print(f"  n = {adj.n_a} vs {adj.n_b}")
    print(f"  {'red':11} {'efecto(pp)':>10} {'p cruda':>9} {'p|dur':>9} {'p Holm':>9}  sig")
    for i, k in enumerate(NETS):
        sig = "sí" if p_holm[i] < 0.05 else "no"
        print(
            f"  {k:11} {adj.statistic[i]*100:+10.1f} {unadj.p_values[i]:9.4f} "
            f"{adj.p_values[i]:9.4f} {p_holm[i]:9.4f}  {sig}"
        )
    print(
        "  'p|dur' ajusta por duración (Freedman–Lane); 'p Holm' controla las 7 comparaciones.\n"
        "  Efecto = coeficiente del grupo en puntos porcentuales de share."
    )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Contrastes del corpus sobre el perfil neural.")
    ap.add_argument("--test", action="store_true", help="test formal de permutación + Holm")
    ap.add_argument("--n-perm", type=int, default=10000, help="permutaciones (def. 10000)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    con = store.connect()
    store.init(con)
    rows = load_rows(con)
    if not rows:
        print("No hay anuncios con perfil neural. Corre el pipeline con --neural.")
        return 1

    print(f"anuncios con perfil neural: {len(rows)}")
    print("ritmo:", dict(Counter(r["ritmo"] for r in rows)))
    print("caras:", dict(Counter(r["caras"] for r in rows)))

    fast = [r for r in rows if r["ritmo"] == "rapido"]
    slow = [r for r in rows if r["ritmo"] == "lento"]
    if len(fast) >= 2 and len(slow) >= 2:
        _contrast("RÁPIDO vs LENTO", fast, slow, "rápido", "lento")
        _adjusted(rows, "RÁPIDO vs LENTO", lambda r: r["ritmo"] == "rapido")
        if args.test:
            _formal(rows, "RÁPIDO vs LENTO", lambda r: r["ritmo"] == "rapido",
                    args.n_perm, args.seed)

    with_f = [r for r in rows if r["caras"]]
    without_f = [r for r in rows if not r["caras"]]
    if len(with_f) >= 2 and len(without_f) >= 2:
        _contrast("CON CARAS vs SIN CARAS", with_f, without_f, "con caras", "sin caras")
        _adjusted(rows, "CON CARAS vs SIN CARAS", lambda r: r["caras"])
        if args.test:
            _formal(rows, "CON CARAS vs SIN CARAS", lambda r: r["caras"],
                    args.n_perm, args.seed)

    if args.test:
        print(
            "\nRecordatorio: shares relativos (%) en unidades crudas SIN CALIBRAR. El test es de\n"
            "permutación con Holm dentro de cada contraste (7 redes); los dos contrastes no se\n"
            "corrigen entre sí. Significativo ≠ relevante: es dirección, no tamaño de efecto."
        )
    else:
        print(
            "\nRecordatorio: shares relativos (%), unidades crudas sin calibrar, sin test de\n"
            "significancia ni corrección por comparaciones múltiples. Marcador de dirección,\n"
            "no de tamaño de efecto."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
