"""Probe 0023 -- does route 1 (the spectrally-floored full-Fisher walk, 0019) hold and stay
sub-exponential as the number of active axes grows?  The whole robotics-practicality goal is
"linear/quadratic in the active axes, sub-exponential" -- 0013-0022 only tested D=4 (r=3).

Self-contained, parametrised model of arbitrary N process modes + M mixing sensors:
    theta_t = theta_{t-1} + w,  w ~ N(0, Q),  Q = V diag(LAM e^xi) V^T   (V fixed random ortho)
    y_t = H theta_t + v,        v ~ N(0, R),  R = diag(RHO e^eta)        (H random MIXING)
    scales xi_k, eta_i each walk as finding-18 AR(1) (phi, s).
Algorithm = 0019 verbatim, generalised: analytic score+Fisher over the active axes, floored
pseudo-inverse (freeze sub-floor eigendirections), smooth matrix finding-18 gain.

Reports, at D = 6, 12, 24, 48:
  * per-step wall time and its growth exponent (cost ~ D^p) -- must be polynomial, not 2^D;
  * DE-MIX separation: with one process mode and one sensor hot, does the estimate put the
    heat on the RIGHT axis (hot value) and keep every other axis low (max leak)?
"""
import time

import numpy as np

np.set_printoptions(precision=3, suppress=True)
PHI, SS, SPAN = 0.9, 0.5, 3.0


def make_model(N, M, seed=0):
    rng = np.random.default_rng(seed)
    V, _ = np.linalg.qr(rng.standard_normal((N, N)))                 # fixed process eigenbasis
    LAM = np.geomspace(1.0, 0.15, N)                                 # spread process eigenvalues
    RHO = np.full(M, 1.0)
    H = rng.standard_normal((M, N)) / np.sqrt(N)                     # random MIXING observation
    H += np.eye(M, N)                                                # keep it well-conditioned/observable
    return dict(N=N, M=M, D=N + M, V=V, LAM=LAM, RHO=RHO, H=H, HV=H @ V)


def Q_of(mdl, xi):
    return mdl["V"] @ np.diag(mdl["LAM"] * np.exp(np.clip(xi, -60, 60))) @ mdl["V"].T


def R_of(mdl, eta):
    return np.diag(mdl["RHO"] * np.exp(np.clip(eta, -60, 60)))


def dS_k(mdl, sc, k):
    N = mdl["N"]
    if k < N:
        hv = mdl["HV"][:, k]; return mdl["LAM"][k] * np.exp(min(sc[k], 60)) * np.outer(hv, hv)
    E = np.zeros((mdl["M"], mdl["M"])); i = k - N; E[i, i] = mdl["RHO"][i] * np.exp(min(sc[k], 60)); return E


def Ichar(mdl):
    N, M, H = mdl["N"], mdl["M"], mdl["H"]
    P = np.eye(N) * (mdl["LAM"].max() + mdl["RHO"].max()); Q0 = Q_of(mdl, np.zeros(N)); R0 = R_of(mdl, np.zeros(M))
    for _ in range(400):
        Pp = P + Q0; S = H @ Pp @ H.T + R0; K = Pp @ H.T @ np.linalg.inv(S); P = Pp - K @ H @ Pp
    Pp = P + Q0; Si = np.linalg.inv(H @ Pp @ H.T + R0)
    o = []
    for k in range(N + M):
        d = dS_k(mdl, np.zeros(N + M), k); o.append(0.5 * np.trace(Si @ d @ Si @ d))
    return np.array(o)


def gen(mdl, hots, amp, T, seed):
    N, M, D, H = mdl["N"], mdl["M"], mdl["D"], mdl["H"]
    rng = np.random.default_rng(seed); psi = np.zeros((T, D))
    for h in hots:
        psi[T // 3: 2 * T // 3, h] = amp
    th = np.zeros(N); Y = np.zeros((T, M)); TH = np.zeros((T, N))
    for t in range(T):
        th = th + np.linalg.cholesky(Q_of(mdl, psi[t, :N]) + 1e-12 * np.eye(N)) @ rng.standard_normal(N)
        Y[t] = H @ th + np.sqrt(np.diag(R_of(mdl, psi[t, N:]))) * rng.standard_normal(M); TH[t] = th
    return Y, TH


def floored_block(mdl, Y):
    N, M, D, H = mdl["N"], mdl["M"], mdl["D"], mdl["H"]
    Ich = Ichar(mdl); floor = (1 - PHI) / (4 * (SPAN * SS) ** 2)
    act = np.where(Ich >= floor)[0]; r = len(act)
    gap = 1.5 * SS; Kstar = (1 - PHI) / 4.0
    qmu = np.array([Kstar ** 2 / (Ich[int(k)] * (1 - Kstar)) for k in act])
    mu = np.zeros(D); Pmu = np.eye(r) * SS ** 2
    m = np.zeros(N); P = np.eye(N) * (mdl["LAM"].max() + mdl["RHO"].max()) * N
    out = np.zeros((len(Y), D)); I_r = np.eye(r)
    for t, y in enumerate(Y):
        e = y - H @ m
        S = H @ (P + Q_of(mdl, mu[:N])) @ H.T + R_of(mdl, mu[N:]) + 1e-9 * np.eye(M)
        Si = np.linalg.inv(S); Sie = Si @ e
        dS = [dS_k(mdl, mu, int(k)) for k in act]; SidS = [Si @ d for d in dS]
        grad = np.array([0.5 * (Sie @ dS[a] @ Sie - np.trace(SidS[a])) for a in range(r)])
        F = np.array([[0.5 * np.trace(SidS[a] @ SidS[b]) for b in range(r)] for a in range(r)]); F = 0.5 * (F + F.T)
        lam, U = np.linalg.eigh(F)
        inv = np.where(lam >= floor, 1.0 / np.maximum(lam, 1e-30), 0.0)
        Rmu = U @ np.diag(inv) @ U.T
        K = Pmu @ np.linalg.inv(Pmu + Rmu)
        mu[act] += np.clip(K @ (Rmu @ grad), -gap, gap)
        Pmu = (I_r - K) @ Pmu + np.diag(qmu); Pmu = 0.5 * (Pmu + Pmu.T)
        Pp = P + Q_of(mdl, mu[:N]); Sb = H @ Pp @ H.T + R_of(mdl, mu[N:]) + 1e-9 * np.eye(M)
        Kk = Pp @ H.T @ np.linalg.inv(Sb); m = m + Kk @ e; P = Pp - Kk @ H @ Pp; P = 0.5 * (P + P.T)
        out[t] = mu
    return out, r


def _loglik(mdl, mu, m, P, y):
    H, M = mdl["H"], mdl["M"]
    S = H @ (P + Q_of(mdl, mu[:mdl["N"]])) @ H.T + R_of(mdl, mu[mdl["N"]:]) + 1e-9 * np.eye(M)
    return -0.5 * (np.linalg.slogdet(S)[1] + float((y - H @ m) @ np.linalg.inv(S) @ (y - H @ m)))


def adf_eigen(mdl, Y):
    N, M, D, H = mdl["N"], mdl["M"], mdl["D"], mdl["H"]
    Ich = Ichar(mdl); floor = (1 - PHI) / (4 * (SPAN * SS) ** 2)
    act = np.where(Ich >= floor)[0]; r = len(act)
    gap = 1.5 * SS; Kstar = (1 - PHI) / 4.0
    qmu = np.array([Kstar ** 2 / (Ich[int(k)] * (1 - Kstar)) for k in act])
    off = gap * np.arange(-3, 4); vfloor = (gap / 3.0) ** 2
    mu = np.zeros(D); Pmu = np.eye(r) * SS ** 2
    m = np.zeros(N); P = np.eye(N) * (mdl["LAM"].max() + mdl["RHO"].max()) * N
    out = np.zeros((len(Y), D)); I_r = np.eye(r)
    for t, y in enumerate(Y):
        e = y - H @ m
        S = H @ (P + Q_of(mdl, mu[:N])) @ H.T + R_of(mdl, mu[N:]) + 1e-9 * np.eye(M)
        Si = np.linalg.inv(S)
        dS = [dS_k(mdl, mu, int(k)) for k in act]; SidS = [Si @ d for d in dS]
        F = np.array([[0.5 * np.trace(SidS[a] @ SidS[b]) for b in range(r)] for a in range(r)]); F = 0.5 * (F + F.T)
        lam, U = np.linalg.eigh(F)
        ostar = np.zeros(r); vj = np.full(r, np.inf)
        for j in range(r):
            if lam[j] < floor:
                continue
            uj = np.zeros(D); uj[act] = U[:, j]
            prof = np.array([_loglik(mdl, mu + o * uj, m, P, y) for o in off])
            w = np.exp(prof - prof.max()); w /= w.sum()
            ostar[j] = float(w @ off); vj[j] = float(w @ (off - ostar[j]) ** 2) + vfloor
        fin = np.isfinite(vj)
        Rmu = U[:, fin] @ np.diag(vj[fin]) @ U[:, fin].T + 1e6 * (I_r - U[:, fin] @ U[:, fin].T)
        K = Pmu @ np.linalg.inv(Pmu + Rmu)
        mu[act] += np.clip(K @ (U[:, fin] @ ostar[fin]), -gap, gap)
        Pmu = (I_r - K) @ Pmu + np.diag(qmu); Pmu = 0.5 * (Pmu + Pmu.T)
        Pp = P + Q_of(mdl, mu[:N]); Sb = H @ Pp @ H.T + R_of(mdl, mu[N:]) + 1e-9 * np.eye(M)
        Kk = Pp @ H.T @ np.linalg.inv(Sb); m = m + Kk @ e; P = Pp - Kk @ H @ Pp; P = 0.5 * (P + P.T)
        out[t] = mu
    return out, r


def run():
    T = 300; b = slice(T // 3 + 30, 2 * T // 3 - 30)
    sizes = [(3, 3), (6, 6), (12, 12), (24, 24)]
    for name, fn in [("0019 floored-block", floored_block), ("0022 ADF-profile ", adf_eigen)]:
        print(f"\n=== {name} ===")
        print(f"{'D':>4} {'r':>4} {'ms/step':>9} {'exp':>6} | hot-axis  mean-leak  static-floor  SNR(hot/floor)")
        prev = None
        for N, M in sizes:
            mdl = make_model(N, M, seed=1)
            hp, hs = 1, N + 1
            Y, _ = gen(mdl, [hp, hs], 1.4, T, seed=2)
            t0 = time.time(); out, r = fn(mdl, Y); dt = (time.time() - t0) / T * 1e3
            est = out[b].mean(0)
            others = [k for k in range(mdl["D"]) if k not in (hp, hs)]
            mean_leak = np.mean([abs(est[k]) for k in others]); hotval = 0.5 * (est[hp] + est[hs])
            Ys, _ = gen(mdl, [], 0.0, T, seed=2); floor_lvl = np.max(np.abs(fn(mdl, Ys)[0][b].mean(0)))
            exp = "" if prev is None else f"{np.log(dt / prev[1]) / np.log(mdl['D'] / prev[0]):.1f}"
            print(f"{mdl['D']:>4} {r:>4} {dt:>9.2f} {exp:>6} | {hotval:+.2f}     {mean_leak:.2f}       {floor_lvl:.2f}        {hotval/max(floor_lvl,1e-6):.1f}x")
            prev = (mdl["D"], dt)


if __name__ == "__main__":
    run()
