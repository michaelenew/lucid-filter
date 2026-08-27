"""Probe 0017 -- the 2-D coupling-grid de-mix, built cleanly and hammered.

Structure (from the 0011-0016 diagnosis):
  xi_k = muP + gP + delta_k   (process eigenmodes)     delta zero-mean over active
  eta_i = muM + gM + eps_i     (sensors)                 eps  zero-mean over active
  * a 2-D GRID over the globals (gP, gM), centres (muP, muM) walked -- a *distribution*
    over the process-vs-measurement split (the hedge; = VectorFilter's faithful 2-channel).
    KF per node, GPB1-collapsed -> de-mixed state; its marginal means walk (muP, muM).
  * the within-block deviations (delta, eps) walk by the caltrop axial score (decoupled
    per 0003), then are re-centred to zero mean (the mean is carried by the global).
De-mix: a hot process mode keeps E[gM]~0, so muM holds and sensors read ~0 -- no leak.
Cost: |grid|(=order^2) * (1 + within-block axial) = sub-exponential.  Reduces to the
2-channel VectorFilter at one process mode + one sensor.

This file is the correctness harness (static / hot-mode / hot-sensor / seeds / state RMSE)
before productionising.
"""
import os
import sys
import math
import importlib.util

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter.core import _chain, _LOG2PI  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "p13", os.path.join(os.path.dirname(__file__), "0013_caltrop_walker.py"))
p13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p13)
D, N, M, PHI, SS, LAM, RHO, H, HV = p13.D, p13.N, p13.M, p13.PHI, p13.SS, p13.LAM, p13.RHO, p13.H, p13.HV
Q_of, R_of, exact_grid, gen, Ichar = p13.Q_of, p13.R_of, p13.exact_grid, p13.gen, p13.Ichar
V = p13.p6.V

np.set_printoptions(precision=3, suppress=True)
_GAP, _SPAN, _RIDGE = 1.5, 3.0, 1e-4


def _scales(muP, gP, delta, muM, gM, eps, ap, asens):
    """Full (xi over process eigenmodes, eta over sensors) from globals + deviations."""
    xi = np.zeros(N); eta = np.zeros(M)
    for k in ap:
        xi[k] = muP + gP + delta[k]
    for i in asens:
        eta[i] = muM + gM + eps[i]
    return xi, eta


def coupling_demix(Y, ng=5):
    Ich = Ichar(); Ifloor = (1 - PHI[0]) / (4 * (_SPAN * SS[0]) ** 2)
    ap = [k for k in range(N) if Ich[k] >= Ifloor]          # active process eigenmodes
    asens = [i for i in range(M) if Ich[N + i] >= Ifloor]   # active sensors
    s = SS[0]; phi = PHI[0]; gap = _GAP * s
    # 2-D global grid (gP, gM): per-axis stationary chain, tensor product
    lam1, w1, T1 = _chain(phi, s, ng)
    GP, GM = [a.ravel() for a in np.meshgrid(lam1, lam1, indexing="ij")]
    pi0 = np.kron(w1, w1); T2 = np.kron(T1, T1); G = pi0.size
    # within-block axial window
    aoff = gap * np.arange(-2, 3); an = aoff.size
    aw = np.exp(-0.5 * (aoff / s) ** 2); aw /= aw.sum()
    Kstar = (1 - phi) / 4.0
    IcP = max((Ich[k] for k in ap), default=1.0); IcM = max((Ich[N + i] for i in asens), default=1.0)
    qP = Kstar ** 2 / (IcP * (1 - Kstar)); qM = Kstar ** 2 / (IcM * (1 - Kstar))

    muP = muM = 0.0; PmuP = PmuM = s * s
    delta = np.zeros(N); eps = np.zeros(M); Pd = np.full(N, s * s); Pe = np.full(M, s * s)
    m = np.zeros(N); P = np.eye(N) * (LAM.max() + RHO.max()) * N
    pig = pi0.copy()
    out = np.zeros((len(Y), D)); state = np.zeros((len(Y), N))

    def QR(gP, gM):
        xi, eta = _scales(muP, gP, delta, muM, gM, eps, ap, asens)
        Q = V @ np.diag(LAM * np.exp(np.clip(xi, -60, 60))) @ V.T
        R = np.diag(RHO * np.exp(np.clip(eta, -60, 60)))
        return Q, R

    for t, y in enumerate(Y):
        pig = pig @ T2
        e = y - H @ m
        # --- 2-D coupling grid: KF per node, collapse (the hedge)
        Qg = np.empty((G, N, N)); Rg = np.empty((G, M, M))
        for g in range(G):
            Qg[g], Rg[g] = QR(GP[g], GM[g])
        Pp = P[None] + Qg; PHt = np.einsum("gij,kj->gik", Pp, H)
        S = np.einsum("ij,gjk->gik", H, PHt) + Rg + 1e-9 * np.eye(M); Si = np.linalg.inv(S)
        sg, ld = np.linalg.slogdet(S); mh = np.einsum("i,gij,j->g", e, Si, e)
        lg = -0.5 * (M * _LOG2PI + ld + mh); w = pig * np.exp(lg - lg.max()); pig = w / w.sum()
        Kk = np.einsum("gik,gkl->gil", PHt, Si); Kb = np.einsum("g,gil->il", pig, Kk)
        m_new = m + Kb @ e; mp = m[None] + np.einsum("gil,l->gi", Kk, e) - (Kb @ e)[None]
        KH = np.einsum("gil,lj->gij", Kk, H); Ppost = Pp - np.einsum("gij,gjk->gik", KH, Pp)
        P = np.einsum("g,gij->ij", pig, Ppost) + np.einsum("g,gi,gj->ij", pig, mp, mp); P = 0.5 * (P + P.T); m = m_new
        egP = float(pig @ GP); egM = float(pig @ GM)
        # --- walk the global centres toward the grid marginal means (unbounded reach)
        if ap:
            K = PmuP / (PmuP + 1.0 / (IcP + _RIDGE)); muP += float(np.clip(K * egP, -gap, gap)); PmuP = (1 - K) * PmuP + qP
        if asens:
            K = PmuM / (PmuM + 1.0 / (IcM + _RIDGE)); muM += float(np.clip(K * egM, -gap, gap)); PmuM = (1 - K) * PmuM + qM
        # --- within-block deviations by axial score (relative to the walked global)
        def dev_walk(kind, idxs, dev, Pdev, ic_get):
            if len(idxs) <= 1:
                for k in idxs:
                    dev[k] = 0.0
                return
            for k in idxs:
                prof = np.empty(an)
                for j in range(an):
                    dd = dev.copy(); dd[k] = dev[k] + aoff[j]
                    if kind == "P":
                        xi, eta = _scales(muP, egP, dd, muM, egM, eps, ap, asens)
                    else:
                        xi, eta = _scales(muP, egP, delta, muM, egM, dd, ap, asens)
                    Q = V @ np.diag(LAM * np.exp(np.clip(xi, -60, 60))) @ V.T
                    R = np.diag(RHO * np.exp(np.clip(eta, -60, 60)))
                    Sr = H @ (P + Q) @ H.T + R + 1e-9 * np.eye(M)
                    prof[j] = -0.5 * (np.linalg.slogdet(Sr)[1] + float(e @ np.linalg.inv(Sr) @ e))
                pw = aw * np.exp(prof - prof.max()); pw /= pw.sum()
                ic = ic_get(k)
                Kd = Pdev[k] / (Pdev[k] + 1.0 / (ic + _RIDGE))
                dev[k] += float(np.clip(Kd * float(pw @ aoff), -gap, gap))
                Pdev[k] = (1 - Kd) * Pdev[k] + Kstar ** 2 / (ic * (1 - Kstar))
            mean = np.mean([dev[k] for k in idxs]);
            for k in idxs:
                dev[k] -= mean
        dev_walk("P", ap, delta, Pd, lambda k: Ich[k])
        dev_walk("M", asens, eps, Pe, lambda i: Ich[N + i])
        # --- report
        sc = np.zeros(D)
        for k in ap:
            sc[k] = muP + egP + delta[k]
        for i in asens:
            sc[N + i] = muM + egM + eps[i]
        out[t] = sc; state[t] = m
    return out, state, len(ap) + len(asens), G


def hammer():
    T = 600; b = slice(T // 3 + 40, 2 * T // 3 - 40); lab = ["xi1", "xi2", "eta1", "eta2"]
    _, _, r, G = coupling_demix(gen(None, 0.0, 60))
    print(f"active r={r}; 2-D coupling grid nodes={G}; + within-block axial")
    leaks = []; reach = []; drift = []
    for seed in range(6):
        st = coupling_demix(gen(None, 0.0, T, seed))[0][150:].mean(0)
        drift.append(np.abs(st[[1, 2, 3]]).max())
        Yx = gen(1, 1.4, T, seed); rx = exact_grid(Yx, 5); wx = coupling_demix(Yx)[0]
        Ye = gen(3, 1.4, T, seed); re = exact_grid(Ye, 5); we = coupling_demix(Ye)[0]
        leak = wx[b, 3].mean() - rx[b, 3].mean()          # sensor read when a process mode is hot, vs grid
        leaks.append(leak); reach.append((we[b, 3].mean(), re[b, 3].mean(), we[b, 2].mean()))
        print(f" s{seed}: static|max|={drift[-1]:.2f} | xi2hot eta2={wx[b,3].mean():.2f}(g{rx[b,3].mean():.2f}) "
              f"| eta2hot eta2={we[b,3].mean():.2f}(g{re[b,3].mean():.2f}) eta1={we[b,2].mean():.2f}")
    print(f"\n  static drift max over seeds: {max(drift):.2f}")
    print(f"  xi2-hot->eta2 leak vs grid: mean {np.mean(leaks):+.2f}, worst {max(leaks, key=abs):+.2f}")


if __name__ == "__main__":
    hammer()
