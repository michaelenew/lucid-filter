"""Probe 0016 -- de-mix with a 2-D coupling GRID (hedged) + caltrop within-block.

0015's lesson: a Gaussian/point walk commits to a process-vs-measurement attribution and
so fails on AMBIGUOUS data realisations (seed1), where the exact grid HEDGES with
posterior spread.  De-mixing the coupling needs a *distribution* over the split -- which
is exactly what the shipped 2-channel VectorFilter already does faithfully (#6).

Structure: put ONLY the process<->measurement coupling in a small 2-D grid over the
global scales (gP, gM) -- hedged -- and use the caltrop axial walk for the WITHIN-block
detail (which eigenmode, which sensor; decoupled per 0003):
    xi_k = muP + gP  + delta_k        eta_i = muM + gM + eps_i
The 2-D (gP,gM) grid (walked centres muP,muM) represents the coupling posterior; the KF
is collapsed over it (GPB1 -> de-mixed state); delta_k, eps_i are per-component axial
deviations.  Cost: |2-D grid| * (1 + within-block axial) -- constant coupling grid times
linear within-block = sub-exponential.  When there is one process mode and one sensor it
reduces to VectorFilter's 2-channel grid.  Tested: does eta stay ~grid when xi2 hot,
across seeds (the de-mix), and does it reach when a sensor is hot.
"""
import os
import sys
import math
import importlib.util

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter.core import _chain  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "p13", os.path.join(os.path.dirname(__file__), "0013_caltrop_walker.py"))
p13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p13)
D, N, M, PHI, SS, LAM, RHO, H, HV = p13.D, p13.N, p13.M, p13.PHI, p13.SS, p13.LAM, p13.RHO, p13.H, p13.HV
Q_of, R_of, exact_grid, gen, Ichar = p13.Q_of, p13.R_of, p13.exact_grid, p13.gen, p13.Ichar

np.set_printoptions(precision=3, suppress=True)
_GAP, _SPAN, _RIDGE = 1.5, 3.0, 1e-4
_KG = 2                          # 2-D coupling grid half-extent (5x5)


def Qfull(muP, gP, delta):       # process modes: shared gP + per-mode deviation
    xi = muP + gP + delta
    return HV, LAM * np.exp(np.clip(xi, -60, 60))     # eigenvalues; V=HV cols implicit below


def _Q_from_eig(vals):
    return (HV_full := p13.p6.V) @ np.diag(vals) @ (HV_full).T if False else None


def coupling_demix(Y, sweeps=2):
    V = p13.p6.V
    Ich = Ichar(); Ifloor = (1 - PHI[0]) / (4 * (_SPAN * SS[0]) ** 2)
    proc = [k for k in range(N) if Ich[k] >= Ifloor]                 # active process eigenmodes
    sens = [i for i in range(M) if Ich[N + i] >= Ifloor]             # active sensors
    gap = _GAP * SS[0]
    goff = gap * np.arange(-_KG, _KG + 1); ng = goff.size            # global grid offsets
    lam1, w1, T1 = _chain(PHI[0], SS[0], ng)                          # per-axis stationary chain (for the 2-D grid)
    w2 = np.kron(w1, w1); T2 = np.kron(T1, T1)
    GP, GM = np.meshgrid(lam1, lam1, indexing="ij"); GP = GP.ravel(); GM = GM.ravel()
    aoff = gap * np.arange(-2, 3); an = aoff.size                     # within-block axial offsets
    aw0 = np.exp(-0.5 * (aoff / SS[0]) ** 2); aw0 /= aw0.sum()
    Kstar = (1 - PHI[0]) / 4.0
    muP = 0.0; muM = 0.0; PmuP = SS[0] ** 2; PmuM = SS[0] ** 2
    qmuP = Kstar ** 2 / ((Ich[proc[0]] if proc else 1) * (1 - Kstar))
    qmuM = Kstar ** 2 / ((max(Ich[N + i] for i in sens) if sens else 1) * (1 - Kstar))
    delta = np.zeros(N); eps = np.zeros(M)
    Pd = np.full(N, SS[0] ** 2); Pe = np.full(M, SS[0] ** 2)
    m = np.zeros(N); P = np.eye(N) * (LAM.max() + RHO.max()) * N
    pig = w2.copy(); out = np.zeros((len(Y), D)); state = np.zeros((len(Y), N))

    def build_QR(gp, gm):
        xi = np.zeros(N); xi[:] = delta;
        for k in proc: xi[k] = muP + gp + delta[k]
        Q = V @ np.diag(LAM * np.exp(np.clip(xi, -60, 60))) @ V.T
        eta = np.array([muM + gm + eps[i] if i in sens else eps[i] for i in range(M)])
        R = np.diag(RHO * np.exp(np.clip(eta, -60, 60)))
        return Q, R

    for t, y in enumerate(Y):
        pig = pig @ T2
        e = y - H @ m
        # ---- 2-D coupling grid: KF per (gP,gM) node, collapse
        Qg = np.stack([build_QR(GP[g], GM[g])[0] for g in range(pig.size)])
        Rg = np.stack([build_QR(GP[g], GM[g])[1] for g in range(pig.size)])
        Pp = P[None] + Qg; PHt = np.einsum("gij,kj->gik", Pp, H)
        S = np.einsum("ij,gjk->gik", H, PHt) + Rg + 1e-9 * np.eye(M); Si = np.linalg.inv(S)
        sg, ld = np.linalg.slogdet(S); mh = np.einsum("i,gij,j->g", e, Si, e)
        lg = -0.5 * (ld + mh); w = pig * np.exp(lg - lg.max()); pig = w / w.sum()
        K = np.einsum("gik,gkl->gil", PHt, Si); Kb = np.einsum("g,gil->il", pig, K)
        mnew = m + Kb @ e; mp = m[None] + np.einsum("gil,l->gi", K, e) - (Kb @ e)[None]
        KH = np.einsum("gil,lj->gij", K, H); Ppost = Pp - np.einsum("gij,gjk->gik", KH, Pp)
        P = np.einsum("g,gij->ij", pig, Ppost) + np.einsum("g,gi,gj->ij", pig, mp, mp); P = 0.5 * (P + P.T); m = mnew
        # ---- walk the global centres muP, muM toward the 2-D grid marginal means
        egP = float(pig @ GP); egM = float(pig @ GM)
        for (mu_name, eg, Pmu, qmu, act) in (("P", egP, PmuP, qmuP, proc), ("M", egM, PmuM, qmuM, sens)):
            if not act:
                continue
            info = 1.0 / max(SS[0] ** 2, 1e-6); Kmu = Pmu / (Pmu + 1.0 / (Ich[act[0] if mu_name == "P" else N + act[0]] + _RIDGE))
            if mu_name == "P":
                muP += float(np.clip(Kmu * eg, -gap, gap)); PmuP = (1 - Kmu) * PmuP + qmuP
            else:
                muM += float(np.clip(Kmu * eg, -gap, gap)); PmuM = (1 - Kmu) * PmuM + qmuM
        # ---- within-block: per-mode delta_k, per-sensor eps_i axial (relative to the global)
        gpm, gmm = egP, egM
        for k in proc:                    # process eigenmode deviation (usually ~0)
            pass                          # (single active mode collapses to gP; skip for r_p=1)
        for i in sens:
            prof = np.empty(an)
            for j in range(an):
                ep = eps.copy(); ep[i] = eps[i] + aoff[j]
                xi = np.array([muP + gpm + delta[k] for k in range(N)])
                Q = V @ np.diag(LAM * np.exp(xi)) @ V.T
                eta = np.array([muM + gmm + (ep[q] if q in sens else eps[q]) for q in range(M)])
                Rr = np.diag(RHO * np.exp(eta)); Sr = H @ (P + Q) @ H.T + Rr + 1e-9 * np.eye(M)
                prof[j] = -0.5 * (np.linalg.slogdet(Sr)[1] + float(e @ np.linalg.inv(Sr) @ e))
            pe = aw0 * np.exp(prof - prof.max()); pe /= pe.sum()
            # centre the sensor deviations (gM already carries the mean): subtract the mean shift
            eps[i] += float(np.clip(0.5 * (pe @ aoff), -gap, gap))
        eps[sens] -= np.mean(eps[sens])   # keep eps zero-mean; the mean lives in gM
        # report
        sc = np.zeros(D)
        for k in range(N):
            sc[k] = (muP + egP + delta[k]) if k in proc else 0.0
        for i in range(M):
            sc[N + i] = (muM + egM + eps[i]) if i in sens else 0.0
        out[t] = sc; state[t] = m
    return out, state, len(proc) + len(sens)


if __name__ == "__main__":
    T = 600; b = slice(T // 3 + 40, 2 * T // 3 - 40)
    print("2-D coupling grid (5x5) + within-block axial; hedged de-mix")
    for seed in (1, 2, 3):
        st = coupling_demix(gen(None, 0.0, T, seed))[0][150:].mean(0)
        Y = gen(1, 1.4, T, seed); ref = exact_grid(Y, 5); wx = coupling_demix(Y)[0]
        Y2 = gen(3, 1.4, T, seed); ref2 = exact_grid(Y2, 5); we = coupling_demix(Y2)[0]
        print(f"s{seed}: STATIC {st[[1,2,3]]} | xi2hot eta2={wx[b,3].mean():.2f}(g{ref[b,3].mean():.2f}) "
              f"| eta2hot eta2={we[b,3].mean():.2f}(g{ref2[b,3].mean():.2f}) eta1={we[b,2].mean():.2f}")
