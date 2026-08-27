"""scratch_02_step.py -- STEP family: locate s*(phi;B,L) on a fine grid (vectorized)."""
from __future__ import annotations
import math, sys
import numpy as np
from scratch_core_vec import gen_batch_step, run_vec, rmse_ratio

SS = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.2, 1.5, 2.0])


def sweep(phi, B, L, S=48, n=2500, calm_frac=0.4, nodes=7):
    rng = np.random.default_rng(12345)
    theta, x, eta, prefix = gen_batch_step(rng, S, n, B, L, calm_frac)
    burn = 150
    bmask = np.zeros(n, bool); bmask[prefix:prefix + L] = True
    wmask = np.zeros(n, bool); wmask[burn:] = True
    bm = np.zeros(len(SS)); bse = np.zeros_like(bm)
    wm = np.zeros_like(bm); wse = np.zeros_like(bm); reach = np.zeros_like(bm)
    for i, s in enumerate(SS):
        em, eo, mus = run_vec(theta, x, eta, phi, s, nodes)
        bm[i], bse[i] = rmse_ratio(em, eo, bmask)
        wm[i], wse[i] = rmse_ratio(em, eo, wmask)
        reach[i] = mus[prefix + L // 2:prefix + L].mean()
    return bm, bse, wm, wse, reach, prefix


def report(name, bm, bse, wm, wse, reach, B):
    ib = int(np.argmin(bm)); iw = int(np.argmin(wm))
    print(f"\n=== {name} ===")
    print("  s     burst-ratio        whole-ratio     reach   f_B")
    for i, s in enumerate(SS):
        tag = (" b*" if i == ib else "") + (" w*" if i == iw else "")
        print(f" {s:4.2f}  {bm[i]:.3f}+-{bse[i]:.3f}  {wm[i]:.3f}+-{wse[i]:.3f}"
              f"  {reach[i]:5.2f}  {reach[i]/B if B>0 else 0:.2f}{tag}")
    print(f"  argmin: burst s*={SS[ib]:.2f}  whole s*={SS[iw]:.2f}")
    sys.stdout.flush()
    return SS[ib], SS[iw]


if __name__ == "__main__":
    phi = 0.85
    print("############ Q1: STEP family, phi=0.85, nodes=7, S=48 seeds ############")
    res = {}
    Bs = [2.0, 3.5, 5.0, 6.5]; Ls = [50, 150, 300, 600]
    for B in Bs:
        for L in Ls:
            bm, bse, wm, wse, reach, pfx = sweep(phi, B, L)
            res[(B, L)] = report(f"B={B} L={L}", bm, bse, wm, wse, reach, B)
    print("\n===== s* summary (burst* / whole*) =====")
    print("          " + "   ".join(f"L={L}" for L in Ls))
    for B in Bs:
        row = "   ".join(f"{res[(B,L)][0]:.2f}/{res[(B,L)][1]:.2f}" for L in Ls)
        print(f"  B={B:3.1f}   {row}")
