"""Readout MULTIVARIADO del modelo: ¿discrimina usando el patrón completo?

Motivación: todo lo medido hasta ahora reduce los 20 484 valores del modelo a **7 shares de red**
— tira el 99.99 % de la señal. Si el poder discriminante existe, es más probable que esté en el
**patrón por vértice** (o en la representación universal de 1152 dims) que en 7 promedios.

Dos pruebas sobre el diseño de pares emparejados por duración (LAMBDA):

1. **Pace**: ¿el patrón distingue high de low? Con el emparejamiento, el test válido es permutar
   **qué miembro de cada par** es high/low (permutación intra-par), no las etiquetas globales.
   Se evalúa con validación cruzada dejando un par fuera (clasificador de centroide/LDA sobre las
   componentes principales) → balanced accuracy vs su nulo por permutación.
2. **recall_score** (memorabilidad humana): correlación de Spearman de cada componente principal.

Requiere `data/discovery/vertices/<id>.npy` (generados con `--neural --vertices --no-cache`).

Uso:
    .venv/Scripts/python.exe scripts/analyze_vertices.py --pairs data/lambda/pairs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.metrics.stats import holm_bonferroni  # noqa: E402

VDIR = Path("data/discovery/vertices")


def load_pairs(pairs: list) -> tuple:
    """Carga por grupos, conservando **solo pares completos**. Devuelve matrices alineadas.

    Cargar `hi_ids + lo_ids` de golpe y partir por la mitad se desalinea en cuanto falta un
    patrón (caso real: 37 de 38), así que se empareja explícitamente.
    """
    hi, lo, rec_hi, rec_lo = [], [], [], []
    for p in pairs:
        fh = VDIR / f"{p['high']['id']}.npy"
        fl = VDIR / f"{p['low']['id']}.npy"
        if fh.exists() and fl.exists():
            hi.append(np.load(fh).astype(np.float32))
            lo.append(np.load(fl).astype(np.float32))
            rec_hi.append(float(p["high"].get("recall") or 0.0))
            rec_lo.append(float(p["low"].get("recall") or 0.0))
    if not hi:
        return np.empty((0, 0)), np.empty((0, 0)), np.array([])
    return np.vstack(hi), np.vstack(lo), np.array(rec_hi + rec_lo)


def pca(X: np.ndarray, k: int) -> tuple:
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    # SVD sobre la matriz centrada (n x 20484); componentes en las filas de Vt
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Xc @ Vt[:k].T
    return comps, S[:k]


def paired_intra_permutation(comps: np.ndarray, n_pairs: int, n_perm: int, seed: int) -> np.ndarray:
    """p bilateral por componente permutando qué miembro del par es 'high'."""
    d = comps[:n_pairs] - comps[n_pairs:]          # (n_pairs, k) si los pares van primero
    stat = d.mean(axis=0)
    rng = np.random.default_rng(seed)
    count = np.zeros(d.shape[1], dtype=int)
    for _ in range(n_perm):
        s = rng.choice((-1.0, 1.0), size=(n_pairs, 1))
        count += np.abs((d * s).mean(axis=0)) >= np.abs(stat)
    return (1.0 + count) / (1.0 + n_perm)


def cv_accuracy(X: np.ndarray, y: np.ndarray, n_pairs: int, k: int, seed: int) -> float:
    """Balanced accuracy dejando un par fuera, con clasificador de centroide sobre PCA (k comps).

    El PCA se reajusta en cada fold para no filtrar información del test.
    """
    n = X.shape[0]
    correct = 0
    for left_out in range(n_pairs):
        test_idx = [left_out, left_out + n_pairs]
        train_idx = [i for i in range(n) if i not in test_idx]
        Xtr, ytr = X[train_idx], y[train_idx]
        mu = Xtr.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
        W = Vt[:k].T
        Ztr, Zte = (Xtr - mu) @ W, (X[test_idx] - mu) @ W
        c1, c0 = Ztr[ytr == 1].mean(axis=0), Ztr[ytr == 0].mean(axis=0)
        for j, idx in enumerate(test_idx):
            pred = 1 if np.linalg.norm(Zte[j] - c1) < np.linalg.norm(Zte[j] - c0) else 0
            correct += int(pred == y[idx])
    return correct / n


def main() -> int:
    ap = argparse.ArgumentParser(description="Readout multivariado (patrón por vértice).")
    ap.add_argument("--pairs", default="data/lambda/pairs.json")
    ap.add_argument("--k", type=int, default=20, help="componentes principales")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    X_hi, X_lo, r = load_pairs(pairs)
    n_pairs = X_hi.shape[0]
    if n_pairs < 4:
        print(f"pares con AMBOS patrones por vértice: {n_pairs} — corre antes "
              f"`--neural --vertices --no-cache` para generarlos.")
        return 1
    X = np.vstack([X_hi, X_lo])
    y = np.array([1] * n_pairs + [0] * n_pairs)
    k = min(args.k, X.shape[0] - 1)
    print(f"pares: {n_pairs} ({X.shape[0]} ads) · {X.shape[1]} vértices · usando {k} componentes")
    if k < args.k:
        print(f"  (k limitado a n-1 = {k}: con {X.shape[0]} muestras no hay más componentes reales)")

    comps, S = pca(X, k)
    var = (S**2) / (S**2).sum()

    p = paired_intra_permutation(comps, n_pairs, args.n_perm, args.seed)
    p_holm = holm_bonferroni(p)
    print(f"\nPACE — test intra-par por componente (n_perm={args.n_perm})")
    print(f"  {'comp':>4} {'%var':>6} {'delta':>10} {'p cruda':>9} {'p Holm':>9}  sig")
    for i in range(k):
        sig = "sí" if p_holm[i] < 0.05 else "no"
        print(f"  {i:>4} {var[i]*100:6.2f} {comps[:n_pairs,i].mean()-comps[n_pairs:,i].mean():10.4f} "
              f"{p[i]:9.4f} {p_holm[i]:9.4f}  {sig}")

    acc = cv_accuracy(X, y, n_pairs, k, args.seed)
    rng = np.random.default_rng(args.seed)
    null = []
    for _ in range(200):  # nulo por permutación intra-par
        y_perm = y.copy()
        flip = rng.random(n_pairs) < 0.5
        for j in np.flatnonzero(flip):
            y_perm[j], y_perm[j + n_pairs] = 0, 1
        null.append(cv_accuracy(X, y_perm, n_pairs, k, args.seed))
    p_acc = (1 + sum(a >= acc for a in null)) / (1 + len(null))
    print(f"\nCLASIFICACIÓN high/low (CV dejando un par fuera, {k} comps): "
          f"bal-acc {acc:.2f} vs nulo {np.mean(null):.2f} ± {np.std(null):.2f} (p≈{p_acc:.3f})")

    try:
        from scipy import stats as sps

        print(f"\nRECALL (memorabilidad humana) vs componentes — Spearman (n={len(r)})")
        pv = []
        for i in range(k):
            rho, pval = sps.spearmanr(comps[:, i], r)
            pv.append(pval)
            if pval < 0.05:
                print(f"  comp {i:>3} (%var {var[i]*100:5.1f}) rho={rho:+.3f} p={pval:.4f}")
        print(f"  componentes con p<0.05 sin corregir: {sum(1 for x in pv if x < 0.05)} de {len(pv)}"
              f" (esperado por azar ≈ {len(pv)*0.05:.1f}) → p Holm mín = {holm_bonferroni(pv).min():.3f}")
    except ImportError:
        print("\n(scipy no disponible: se omite recall)")

    print("\nRecordatorio: recall_score es memorabilidad humana, NO CTR ni ventas. Sin calibrar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
