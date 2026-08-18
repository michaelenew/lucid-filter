"""0015 -- Is the drift shape estimable?  Profile it and count nats.

0014 concluded that no invariance principle fixes the shape of the dynamics
drift, so it has to be learned.  That is only a plan if the data carries
evidence about it.  The parent workstream is the warning: it measured s_P at
0.0017 nats/point and had to tell callers not to read the fitted value.

Parameterise the drift covariance, determinant-normalised so that scale and
shape are separate coordinates:

    Sigma(nu, tau, psi) = nu^2 R(psi) diag(tau, 1/tau) R(psi)'
        det = nu^4 for every (tau, psi);  eigenvalues nu^2 tau, nu^2/tau
        tau = 1 is isotropic;  psi is the major axis

and profile the marginal likelihood over (tau, psi) with nu maximised out.  The
question is how many nats per point separate the truth from the isotropic
member, and how sharply psi is pinned.

Two generating conditions, matched in determinant so they differ ONLY in shape:

    anisotropic   tau = 4 (axis SDs 0.10 and 0.025), psi = 0.9
    isotropic     tau = 1 (both SDs 0.05)

The second is the control: a profile that finds anisotropy there is finding
noise.

Base point (rho, theta) = (0.75, 0.60), chosen deep enough inside the
stationarity triangle that a drift of this size rarely reaches the boundary --
otherwise the boundary, not the shape, would drive the likelihood.

The generating drift is a stationary OU in alpha (phi = 0.95) while the filter's
kernel is a random walk.  That mismatch is deliberate and is the same for every
shape tested, so it cannot favour one; it does mean the fitted nu is not
comparable to the generating SD.
"""
import json
import os
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from importlib import import_module  # noqa: E402
_m = import_module("0008_anisotropy_at_p2")

BASE = (0.75, 0.60)
PHI_GEN = 0.95
S_GEN = 0.05
COND = {"anisotropic": (4.0, 0.9), "isotropic": (1.0, 0.0)}

TAUS = [1.0, 2.0, 4.0, 8.0]
PSIS = np.linspace(0.0, np.pi, 6, endpoint=False)
NUS = [0.010, 0.025, 0.060, 0.140]
H = 5


def shape(tau, psi):
    R = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
    return R @ np.diag([tau, 1.0 / tau]) @ R.T


def build_T_const(al, nu, tau, psi):
    """Sparse random-walk kernel with a single constant covariance."""
    S = nu * nu * shape(tau, psi)
    det = max(S[0, 0] * S[1, 1] - S[0, 1] ** 2, 1e-300)
    tr = S[0, 0] + S[1, 1]
    lam = 0.5 * (tr + np.sqrt(max(tr * tr - 4 * det, 0.0)))
    rad = 4.0 * np.sqrt(lam)
    G = len(al)
    rows, cols, vals = [], [], []
    for i in range(G):
        d = al - al[i]
        near = np.nonzero((np.abs(d[:, 0]) <= rad) & (np.abs(d[:, 1]) <= rad))[0]
        dd = d[near]
        q = (S[1, 1] * dd[:, 0] ** 2 - 2 * S[0, 1] * dd[:, 0] * dd[:, 1]
             + S[0, 0] * dd[:, 1] ** 2) / det
        rows.append(np.full(len(near), i))
        cols.append(near)
        vals.append(np.exp(-0.5 * np.minimum(q, 200.0)))
    T = sp.csr_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))), shape=(G, G))
    return sp.diags(1.0 / np.asarray(T.sum(1)).ravel()) @ T


def stationary(a):
    return abs(a[1]) < 0.995 and a[0] + a[1] < 0.995 and a[1] - a[0] < 0.995


def gen_path(a0, tau, psi, n, rng):
    """Stationary OU in alpha with the prescribed shape, rejected at the edge."""
    Sig = S_GEN * S_GEN * shape(tau, psi)
    L = np.linalg.cholesky(Sig)
    Ln = np.sqrt(1.0 - PHI_GEN ** 2) * L
    d = L @ rng.standard_normal(2)
    out = np.empty((n, 2))
    for t in range(n):
        prop = PHI_GEN * d + Ln @ rng.standard_normal(2)
        if stationary(a0 + prop):
            d = prop
        out[t] = a0 + d
    return out


def simulate_path(a_path, Q, S2, rng):
    n = len(a_path)
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = (a_path[t, 0] * x[t - 1] + a_path[t, 1] * x[t - 2]
                + np.sqrt(Q) * rng.standard_normal())
    return x, x + np.sqrt(S2) * rng.standard_normal(n)


def main():
    n, R, kappa, Q = 1200, 6, 0.35, 1.0
    al = _m.build_grid()
    g = _m.Grid2(al)
    a0 = _m.alpha_osc(*BASE)
    print(f"grid {len(al)} nodes, step {_m.STEP};  base alpha = {np.round(a0,4)}")
    print(f"generating: OU phi={PHI_GEN}, s={S_GEN}; axis SDs "
          f"{S_GEN*2:.3f}/{S_GEN/2:.3f} (tau=4) vs {S_GEN:.3f}/{S_GEN:.3f} (tau=1)\n")

    print("building kernels...", end=" ", flush=True)
    Ts = {}
    for nu in NUS:
        Ts[(nu, 1.0, 0.0)] = build_T_const(al, nu, 1.0, 0.0)
        for tau in TAUS[1:]:
            for psi in PSIS:
                Ts[(nu, tau, float(psi))] = build_T_const(al, nu, tau, float(psi))
    print(f"{len(Ts)} kernels")

    master = np.random.default_rng(271828)
    out = {}
    for cname, (tau_t, psi_t) in COND.items():
        rngref = np.random.default_rng(9)
        ap = gen_path(a0, tau_t, psi_t, 20000, rngref)
        xr, _ = simulate_path(ap, Q, 0.0, rngref)
        S2 = (kappa * np.std(np.diff(xr))) ** 2
        data = []
        for _ in range(R):
            rng = np.random.default_rng(master.integers(2 ** 63))
            ap = gen_path(a0, tau_t, psi_t, n, rng)
            data.append(simulate_path(ap, Q, S2, rng))

        prof = {}
        for key, T in Ts.items():
            nu, tau, psi = key
            lls, sc = [], []
            for x, y in data:
                ll, fc, _ = _m.run(y, g, T, Q, S2)
                lls.append(ll / n)
                sc.append(_m.score(fc, x)[H])
            k = (tau, psi if tau > 1.0 else 0.0)
            v = (float(np.mean(lls)), float(np.mean(sc)))
            if k not in prof or v[0] > prof[k][0]:
                prof[k] = (v[0], v[1], nu)

        best = max(prof, key=lambda k: prof[k][0])
        iso = prof[(1.0, 0.0)]
        print(f"--- generated {cname} (truth tau={tau_t}, psi={psi_t}) ---")
        print(f"  profile argmax: tau={best[0]:.1f}  psi={best[1]:.3f}  "
              f"nu*={prof[best][2]:.3f}")
        print(f"  evidence over isotropic: "
              f"{prof[best][0] - iso[0]:.5f} nats/point "
              f"({(prof[best][0] - iso[0]) * n:.2f} nats total)")
        print(f"  h={H} forecast MSE: shape-fitted {prof[best][1]:.3f}  "
              f"isotropic {iso[1]:.3f}  ratio {prof[best][1]/iso[1]:.4f}")
        # how sharply is psi pinned, at the best tau?
        if best[0] > 1.0:
            row = [(p, prof[(best[0], p)][0]) for p in
                   sorted(k[1] for k in prof if k[0] == best[0])]
            span = max(v for _, v in row) - min(v for _, v in row)
            print(f"  psi profile at tau={best[0]:.1f}: span {span:.5f} nats/point"
                  f"  [{'  '.join(f'{p:.2f}:{v:+.4f}' for p, v in row)}]")
        print()
        out[cname] = {f"{k[0]}|{k[1]:.4f}": list(v) for k, v in prof.items()}

    print("benchmark: the parent measured s_P at 0.0017 nats/point and "
          "declared it unreadable.")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), sharey=True)
    for ax, cname in zip(axes, COND):
        prof = {tuple(float(t) for t in k.split("|")): v
                for k, v in out[cname].items()}
        iso = prof[(1.0, 0.0)][0]
        for i, tau in enumerate(TAUS[1:]):
            ps = sorted(k[1] for k in prof if k[0] == tau)
            ax.plot(ps, [(prof[(tau, p)][0] - iso) * 1000 for p in ps],
                    marker="o", color=ts.SERIES[i], label=fr"$\tau$={tau:.0f}")
        ax.axhline(0.0, color=ts.INK, lw=1.0, ls="--", zorder=0,
                   label=r"isotropic ($\tau$=1)")
        if COND[cname][0] > 1:
            ax.axvline(COND[cname][1], color=ts.SERIES[7], lw=1.2, ls=":",
                       label=r"true $\psi$")
        ax.set_xlabel(r"major axis $\psi$")
        ax.set_title(f"generated {cname}")
        ts.tidy(ax)
    axes[0].set_ylabel("millinats/point over isotropic")
    axes[0].legend(fontsize=8)
    ts.save(fig, os.path.join(HERE, "figures", "fig11-drift-shape-profile.png"))

    with open(os.path.join(HERE, "figures", "ode015.json"), "w") as f:
        json.dump(dict(profiles=out, base=BASE, phi_gen=PHI_GEN, s_gen=S_GEN,
                       cond={k: list(v) for k, v in COND.items()},
                       taus=TAUS, psis=PSIS.tolist(), nus=NUS,
                       n=n, R=R, kappa=kappa, horizon=H), f, indent=1)


if __name__ == "__main__":
    main()
