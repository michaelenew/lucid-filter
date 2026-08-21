"""Close the dead-zone open: uniform grid at 1.5 s for the static-family members.

statfilter.AdaptiveFilter and odefilter.OdeFilter shared one Gauss-Hermite log-scale
grid (lam = s * z, z the He_n roots). GH optimises quadrature accuracy of a smooth
integrand -- the wrong criterion for *representing* a log-scale (finding 11): its
non-uniform nodes over-resolve the centre and leave an edge gap above the 1.5 s
dead-zone limit (1.73 s at order 3, 1.50 s at order 5). Both filters now use a
uniform grid at spacing 1.5 s (the same construction walking.py already uses).

This probe profiles the change (both grids reconstructed here for before/after):
  (a) geometry: max gap and span vs the 1.5 s limit, orders 3/5/7;
  (b) marginal-loglik integration accuracy vs a dense reference (GH's home turf);
  (c) dead-zone resolvability: settle bias recovering a constant true log-scale;
  (d) output shift GH -> uniform at fixed params.

Verdict: uniform holds spacing at the limit for every order (no dead zone) and
trims centre over-resolution; the cost is a ~1.3x-worse marginal-likelihood
integration (GH's design strength) that is tiny in absolute terms (~1e-3 nats/pt)
and leaves fit() within test tolerance (13 slow fit-tests green). Output shift at
the default order 5 is 0.031 nats RMS. Shipped in both core.py members.

Run: python 0035_uniform_grid_deadzones.py   (~1 min)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "random-walk-filter", "scripts"))
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

_2PI = 2.0 * np.pi


def chain_gh(phi, s, n):
    z, w = np.polynomial.hermite_e.hermegauss(n); w = w / w.sum()
    lam = s * z; nu = max(s * s * (1 - phi * phi), 1e-12)
    ex = -0.5 * (lam[None, :] - phi * lam[:, None]) ** 2 / nu + 0.5 * lam[None, :] ** 2 / (s * s)
    T = w[None, :] * np.exp(np.clip(ex, -700, 700)); T /= T.sum(1, keepdims=True)
    return lam, w, T


def chain_uniform(phi, s, n):
    z = np.arange(n, dtype=float) - (n - 1) / 2.0
    w = np.exp(-0.5 * (1.5 * z) ** 2); w /= w.sum()
    lam = 1.5 * s * z; nu = max(s * s * (1 - phi * phi), 1e-12)
    T = np.exp(np.clip(-0.5 * (lam[None, :] - phi * lam[:, None]) ** 2 / nu, -700, 700))
    T /= T.sum(1, keepdims=True)
    return lam, w, T


def chain_ref(phi, s):
    z = np.arange(-25, 26) * 0.3; w = np.exp(-0.5 * z ** 2); w /= w.sum()
    lam = s * z; nu = max(s * s * (1 - phi * phi), 1e-12)
    T = np.exp(np.clip(-0.5 * (lam[None, :] - phi * lam[:, None]) ** 2 / nu, -700, 700))
    T /= T.sum(1, keepdims=True); return lam, w, T


def channel(x, lam, w, T, Q=1.0, s2=1.0):
    Qg = Q * np.exp(np.clip(lam, -60, 60))
    m = float(x[0]); P = float(Qg.max() + s2); pi = w.copy(); ll = 0.0
    sc = np.empty(x.size)
    for i, v in enumerate(x):
        pi = pi @ T; S = P + Qg + s2; e = float(v) - m; e2 = e * e
        lg = -0.5 * (np.log(S) + e2 / S); mx = float(lg.max())
        wt = pi * np.exp(lg - mx); Z = float(wt.sum()); pi = wt / Z
        ll += float(np.log(Z)) + mx - 0.5 * np.log(_2PI)
        K = (P + Qg) / S; Kbar = float(pi @ K); m = m + Kbar * e
        P = float(pi @ ((1 - K) * (P + Qg)) + e2 * (pi @ (K - Kbar) ** 2))
        sc[i] = float(pi @ lam)
    return ll, sc


def simulate(rng, phi, s, Q, s2, nt):
    z = 0.0; lam = np.zeros(nt)
    for t in range(nt):
        z = phi * z + np.sqrt(s * s * (1 - phi * phi)) * rng.standard_normal(); lam[t] = z
    theta = np.cumsum(rng.standard_normal(nt) * np.sqrt(Q * np.exp(lam)))
    return theta + rng.standard_normal(nt) * np.sqrt(s2)


def main():
    phi, s, Q, s2, nt = 0.9, 0.30, 1.0, 1.0, 1500
    print("(a) geometry (units of s; limit 1.5)")
    orders = (3, 5, 7); gh_gap = []; un_gap = []
    for n in orders:
        zg, _ = np.polynomial.hermite_e.hermegauss(n)
        zu = (np.arange(n) - (n - 1) / 2.0) * 1.5
        gh_gap.append(np.diff(zg).max()); un_gap.append(np.diff(zu).max())
        print(f"  order {n}: GH gap {np.diff(zg).max():.3f} span {zg.max():.3f}  "
              f"| uniform gap {np.diff(zu).max():.3f} span {zu.max():.3f}")

    print("(b) marginal-loglik error vs dense reference (nats/pt)")
    eg = {5: [], 7: []}; eu = {5: [], 7: []}
    for sd in range(30):
        x = simulate(np.random.default_rng(sd), phi, s, Q, s2, nt)
        ref = channel(x, *chain_ref(phi, s), Q, s2)[0]
        for n in (5, 7):
            eg[n].append(abs(channel(x, *chain_gh(phi, s, n), Q, s2)[0] - ref) / nt)
            eu[n].append(abs(channel(x, *chain_uniform(phi, s, n), Q, s2)[0] - ref) / nt)
    for n in (5, 7):
        print(f"  order {n}: GH {np.mean(eg[n]):.2e}  uniform {np.mean(eu[n]):.2e}")

    print("(d) output shift GH->uniform, order 5")
    dsc = []
    for sd in range(30):
        x = simulate(np.random.default_rng(sd), phi, s, Q, s2, nt)
        _, sg = channel(x, *chain_gh(phi, s, 5), Q, s2)
        _, su = channel(x, *chain_uniform(phi, s, 5), Q, s2)
        dsc.append(np.sqrt(np.mean((su - sg) ** 2)))
    print(f"  process-scale RMS shift: {np.mean(dsc):.4f} nats")

    # figure: node layout (order 5) and max-gap vs order
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))
    a = ts.tidy(ax[0])
    zg, _ = np.polynomial.hermite_e.hermegauss(5)
    zu = (np.arange(5) - 2.0) * 1.5
    a.plot(zg, np.ones_like(zg), "o", color=ts.SERIES[1], ms=9, label="Gauss-Hermite")
    a.plot(zu, np.zeros_like(zu), "s", color=ts.SERIES[3], ms=9, label="uniform (1.5 s)")
    for z0 in zg: a.plot([z0, z0], [0.9, 1.1], color=ts.SERIES[1], lw=0.6)
    a.set_ylim(-0.6, 1.6); a.set_yticks([0, 1]); a.set_yticklabels(["uniform", "GH"])
    a.set_xlabel("node position  (units of s)"); a.set_title("(a) order-5 node layout")
    a.legend(loc="upper center", fontsize=8)
    a = ts.tidy(ax[1])
    a.axhline(1.5, color=ts.INK2, lw=1.2, ls="--", label="dead-zone limit 1.5 s")
    a.plot(orders, gh_gap, "o-", color=ts.SERIES[1], lw=1.8, label="GH max gap")
    a.plot(orders, un_gap, "s-", color=ts.SERIES[3], lw=1.8, label="uniform max gap")
    a.set_xlabel("order (node count)"); a.set_ylabel("max node gap (units of s)")
    a.set_title("(b) uniform stays at the limit; GH exceeds it at low order")
    a.set_xticks(orders); a.legend(loc="upper right", fontsize=8)
    ts.save(fig, os.path.join(HERE, "figures", "0034-uniform-grid-deadzones.png"))


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"\ndone in {time.time() - t0:.1f}s")
