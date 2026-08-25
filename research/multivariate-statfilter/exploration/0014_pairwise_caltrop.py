"""Probe 0014 -- the PAIRWISE caltrop: two-hot arms de-mix the coupling at quadratic cost.

0013's caltrop (axial arms only) tracks state at linear cost but leaks on the *scale
attribution* (a hot process mode reads high on a sensor) -- the process<->measurement
de-mixing lives in the CORNERS the axial cross skips.  User's fix: add the two-hot arms.
Because 0003 measured only the process<->measurement block as coupled (process
eigenmodes decouple, sensors decouple), we need only the PROCESS x MEASUREMENT pairs --
r_p * r_m two-hot 2-D arms -- not all C(r,2).  That is QUADRATIC, and it should recover
the corner information that de-mixes the attribution.  Also: drop to 3 nodes/axis
(origin +/- 1) per the walker's needs.

Construction: coordinate-descent sweeps.  Within-block axes (process-process,
sensor-sensor) walk on their AXIAL score (decoupled, cheap).  Each (process p, sensor m)
pair walks on its 2-D-joint score -- the 2-D profile sees the coupling ridge, so the
score average points to the de-mixed joint direction, not the leaky marginal.
Benchmarked vs the exact grid; measures the leak and the arm count.
"""
import os
import sys
import math
import importlib.util

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
_spec = importlib.util.spec_from_file_location(
    "p13", os.path.join(os.path.dirname(__file__), "0013_caltrop_walker.py"))
p13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p13)
D, N, M, PHI, SS, LAM, RHO, H, HV = p13.D, p13.N, p13.M, p13.PHI, p13.SS, p13.LAM, p13.RHO, p13.H, p13.HV
Q_of, R_of, exact_grid, gen, Ichar = p13.Q_of, p13.R_of, p13.exact_grid, p13.gen, p13.Ichar
score_info_at = p13.score_info_at

np.set_printoptions(precision=3, suppress=True)
_GAP, _SPAN, _RIDGE = 1.5, 3.0, 1e-4
_K = 2                         # 3 nodes/axis (origin +/- 1) -- the reduced caltrop


def pairwise_caltrop(Y, sweeps=3):
    Ich = Ichar(); Ifloor = (1 - PHI[0]) / (4 * (_SPAN * SS[0]) ** 2); active = np.where(Ich >= Ifloor)[0]
    proc = [k for k in active if k < N]; meas = [k for k in active if k >= N]
    gap = _GAP * SS[0]; offs = gap * np.arange(-_K, _K + 1); nn = offs.size
    w0 = np.exp(-0.5 * (offs / SS[0]) ** 2); w0 /= w0.sum()
    Kstar = (1 - PHI) / 4.0; qmu = Kstar ** 2 / (Ich * (1 - Kstar))
    mu = np.zeros(D); Pmu = SS ** 2; m = np.zeros(N); P = np.eye(N) * (LAM.max() + RHO.max()) * N
    out = np.zeros((len(Y), D)); state = np.zeros((len(Y), N)); arms = 0

    def axial_step(k, e):
        base = mu.copy(); sco = np.empty(nn); inf = np.empty(nn); prof = np.empty(nn)
        for j in range(nn):
            base[k] = mu[k] + offs[j]
            Sm = H @ (P + Q_of(base[:N])) @ H.T + R_of(base[N:]) + 1e-9 * np.eye(M); Si = np.linalg.inv(Sm)
            prof[j] = -0.5 * (np.linalg.slogdet(Sm)[1] + float(e @ Si @ e))
            sco[j], inf[j] = score_info_at(base, k, P, e)
        pi = w0 * np.exp(prof - prof.max()); pi /= pi.sum()
        grad = float(pi @ sco); info = float(pi @ inf) + _RIDGE
        K_mu = Pmu[k] / (Pmu[k] + 1.0 / info); mu[k] += float(np.clip(K_mu * (grad / info), -gap, gap))

    def pair_step(p, q, e):
        base = mu.copy(); lg = np.empty((nn, nn)); scp = np.empty((nn, nn)); scq = np.empty((nn, nn))
        infp = np.empty((nn, nn)); infq = np.empty((nn, nn))
        for a in range(nn):
            for bnode in range(nn):
                base[p] = mu[p] + offs[a]; base[q] = mu[q] + offs[bnode]
                Sm = H @ (P + Q_of(base[:N])) @ H.T + R_of(base[N:]) + 1e-9 * np.eye(M); Si = np.linalg.inv(Sm)
                lg[a, bnode] = -0.5 * (np.linalg.slogdet(Sm)[1] + float(e @ Si @ e))
                scp[a, bnode], infp[a, bnode] = score_info_at(base, p, P, e)
                scq[a, bnode], infq[a, bnode] = score_info_at(base, q, P, e)
        pi = (w0[:, None] * w0[None, :]) * np.exp(lg - lg.max()); pi /= pi.sum()
        for k, sc, inf in ((p, scp, infp), (q, scq, infq)):
            grad = float((pi * sc).sum()); info = float((pi * inf).sum()) + _RIDGE
            K_mu = Pmu[k] / (Pmu[k] + 1.0 / info); mu[k] += float(np.clip(K_mu * (grad / info), -gap, gap))
        return nn * nn

    def pair_scores(p, q, e):
        """2-D joint over the coupled pair -> de-mixed (grad,info) for BOTH axes."""
        base = mu.copy(); lg = np.empty((nn, nn)); scp = np.empty((nn, nn)); scq = np.empty((nn, nn))
        infp = np.empty((nn, nn)); infq = np.empty((nn, nn))
        for a in range(nn):
            for bnode in range(nn):
                base[p] = mu[p] + offs[a]; base[q] = mu[q] + offs[bnode]
                Sm = H @ (P + Q_of(base[:N])) @ H.T + R_of(base[N:]) + 1e-9 * np.eye(M); Si = np.linalg.inv(Sm)
                lg[a, bnode] = -0.5 * (np.linalg.slogdet(Sm)[1] + float(e @ Si @ e))
                scp[a, bnode], infp[a, bnode] = score_info_at(base, p, P, e)
                scq[a, bnode], infq[a, bnode] = score_info_at(base, q, P, e)
        pi = (w0[:, None] * w0[None, :]) * np.exp(lg - lg.max()); pi /= pi.sum()
        return ((p, float((pi * scp).sum()), float((pi * infp).sum())),
                (q, float((pi * scq).sum()), float((pi * infq).sum())))

    def axial_scores(k, e):
        base = mu.copy(); sco = np.empty(nn); inf = np.empty(nn); prof = np.empty(nn)
        for j in range(nn):
            base[k] = mu[k] + offs[j]
            Sm = H @ (P + Q_of(base[:N])) @ H.T + R_of(base[N:]) + 1e-9 * np.eye(M); Si = np.linalg.inv(Sm)
            prof[j] = -0.5 * (np.linalg.slogdet(Sm)[1] + float(e @ Si @ e))
            sco[j], inf[j] = score_info_at(base, k, P, e)
        pi = w0 * np.exp(prof - prof.max()); pi /= pi.sum()
        return float(pi @ sco), float(pi @ inf)

    for t, y in enumerate(Y):
        e = y - H @ m
        for _sw in range(sweeps):
            gsum = {int(k): 0.0 for k in active}; isum = {int(k): 0.0 for k in active}; cnt = {int(k): 0 for k in active}
            for p in proc:                          # process x measurement two-hot arms (de-mix)
                for q in meas:
                    for (k, g, iv) in pair_scores(p, q, e):
                        gsum[k] += g; isum[k] += iv; cnt[k] += 1; arms += nn * nn
            for k in active:                        # within-block (decoupled) axial contribution
                if cnt[k] == 0:
                    g, iv = axial_scores(k, e); gsum[k] += g; isum[k] += iv; cnt[k] += 1; arms += nn
            for k in active:                        # ONE update per axis: averaged de-mixed score
                grad = gsum[k] / cnt[k]; info = isum[k] / cnt[k] + _RIDGE
                K_mu = Pmu[k] / (Pmu[k] + 1.0 / info); mu[k] += float(np.clip(K_mu * (grad / info), -gap, gap))
        for k in active:
            info = float(Ich[k]) + _RIDGE; K_mu = Pmu[k] / (Pmu[k] + 1.0 / info); Pmu[k] = (1 - K_mu) * Pmu[k] + qmu[k]
        Pp = P + Q_of(mu[:N]); Sb = H @ Pp @ H.T + R_of(mu[N:]) + 1e-9 * np.eye(M)
        Kk = Pp @ H.T @ np.linalg.inv(Sb); m = m + Kk @ e; P = Pp - Kk @ H @ Pp; P = 0.5 * (P + P.T)
        out[t] = mu; state[t] = m
    return out, state, len(active), arms // max(len(Y), 1)


if __name__ == "__main__":
    T = 600; b = slice(T // 3 + 40, 2 * T // 3 - 40); lab = ["xi1", "xi2", "eta1", "eta2"]
    _, _, r, armct = pairwise_caltrop(gen(None, 0.0, 60))
    print(f"active r={r}; nodes/axis={2*_K+1}; arms/step~{armct} (quadratic in the coupled block) vs grid {5**r}")
    print("STATIC:", pairwise_caltrop(gen(None, 0.0, T))[0][150:].mean(0))
    for ax, nm in [(1, "xi2 hot"), (3, "eta2 hot")]:
        Y = gen(ax, 1.4, T); ref = exact_grid(Y, 5); w, *_ = pairwise_caltrop(Y)
        cr = [np.corrcoef(w[30:, k], ref[30:, k])[0, 1] if w[30:, k].std() > 1e-9 else np.nan for k in range(D)]
        print(f"{nm}: GRID {ref[b].mean(0)}  PAIRWISE {w[b].mean(0)}  corr {np.array(cr)}")
