"""Stress battery: shipped WalkingFilter vs an oracle with the nonlinearity handled.

The question (finding 20): how much is left on the table by the stiffening-well
nonlinearity?  We bound it with an ORACLE that is the shipped filter in every
respect -- same grid, level filter, drift variance, cap -- EXCEPT the walk step
uses the TRUE offset e = lam_true - mu instead of the grid-estimated Newton step
grad/info.  That is a perfect force-linearization (log(I/I0) on uncorrupted
inputs), so the gap shipped - oracle is exactly the cost of the nonlinearity, with
everything else held fixed.  A second oracle drops the cap too, to separate the
reach ceiling from the estimation ceiling.

If the gap is marginal across a punishing battery, handling the nonlinearity is a
theory-only revisit item (like `forget`), not a practical win.

Battery (each a log-scale path lam(t), data x_t = cumsum(N(0,e^lam)) + N(0,1)):
  extreme up / down jumps, a staircase, fast square-wave regime switching, slow /
  resonant / fast sinusoids, and an AR(2) (oscillatory, out-of-class) log-scale.

Run: python 0034_stress_battery.py   (~4-6 min)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "random-walk-filter", "scripts"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lucid"))

from statfilter import WalkingFilter                 # noqa: E402
import theory_style as ts                            # noqa: E402
import matplotlib.pyplot as plt                      # noqa: E402

_RIDGE = 1e-4


# ------------------------------------------------------------------ the arms
def _grid(phi, s, Q, s2, nodes):
    K = nodes // 2
    gap = 1.5 * s
    lam = gap * np.arange(-K, K + 1, dtype=float)
    w0 = np.exp(-0.5 * (lam / s) ** 2); w0 /= w0.sum()
    nu = max(s * s * (1 - phi * phi), 1e-12)
    T = np.exp(np.clip(-0.5 * (lam[None, :] - phi * lam[:, None]) ** 2 / nu, -700, 700))
    T /= T.sum(1, keepdims=True)
    return lam, w0, T, gap


def run(x, lam_true=None, phi=0.9, s=0.30, Q=1.0, s2=1.0, nodes=7, uncapped=False,
        nl=False):
    """Shipped (lam_true=None); or an oracle stepping on the true offset -- linearly
    (nl=False) or through the well's Newton nonlinearity 1-e^-off (nl=True).  The
    oracle-linear vs oracle-nl gap isolates the pure cost of the stiffening, since
    both have perfect estimation (finding 20)."""
    lam, w0, T, gap = _grid(phi, s, Q, s2, nodes); cap = gap
    f0 = WalkingFilter(Q, s2, phi=phi, s=s); qmu = f0._qmu
    mu = 0.0; pi = None; m = None; P = None; Pmu = s * s
    out = np.empty(x.size)
    for i, v in enumerate(x):
        Qg = Q * np.exp(np.clip(lam + mu, -60, 60))
        if pi is None:
            pi = w0.copy(); m = float(v) if m is None else m
            P = float(Qg.max() + s2) if P is None else P
        pi = pi @ T
        S = P + Qg + s2; e = float(v) - m; e2 = e * e
        lg = -0.5 * (np.log(S) + e2 / S); mx = float(lg.max())
        w = pi * np.exp(lg - mx); Z = float(w.sum()); pi = w / Z
        Kk = (P + Qg) / S; Kbar = float(pi @ Kk); m = m + Kbar * e
        P = float(pi @ ((1 - Kk) * (P + Qg)) + e2 * (pi @ (Kk - Kbar) ** 2))
        gS = Qg / S
        info = float(pi @ (0.5 * gS * gS)) + _RIDGE
        if lam_true is None:                               # shipped: grid Newton step
            e_est = float(pi @ (0.5 * gS * (e2 / S - 1.0))) / info
        else:                                              # oracle: exact offset
            off = float(lam_true[i]) - mu
            e_est = (1.0 - float(np.exp(-min(off, 60.0)))) if nl else off
        R_mu = 1.0 / info; K_mu = Pmu / (Pmu + R_mu)
        step = K_mu * e_est
        mu += step if uncapped else float(np.clip(step, -cap, cap))
        Pmu = (1.0 - K_mu) * Pmu + qmu
        out[i] = mu + float(pi @ lam)
    return out


# ------------------------------------------------------------- the scenarios
def _paths(phi=0.9):
    NT = 1200
    t = np.arange(NT)
    P = {}
    P["extreme jump up (+6)"] = np.where(t < 400, 0.0, 6.0)
    P["jump down (-3, quiet)"] = np.where(t < 400, 0.0, -3.0)
    st = np.zeros(NT)
    for k, lv in enumerate([0, 2, 4, 1, -1]):
        st[k * 240:(k + 1) * 240] = lv
    P["staircase"] = st
    P["fast square (0/+3, 30)"] = np.where((t // 30) % 2 == 0, 0.0, 3.0)
    P["slow sine (period 300)"] = 2.0 * np.sin(2 * np.pi * t / 300)
    per = 2 * np.pi / ((1 - phi) / 2)                        # loop natural period ~ 2/(1-phi)
    P[f"resonant sine (~{per:.0f})"] = 2.0 * np.sin(2 * np.pi * t / per)
    P["fast sine (period 12)"] = 2.0 * np.sin(2 * np.pi * t / 12)
    return P, NT


def _ar2(rng, NT, r=0.985, period=80, sd_target=1.4):
    """An oscillatory AR(2) log-scale (out of the AR(1) class)."""
    w = 2 * np.pi / period
    a1 = 2 * r * np.cos(w); a2 = -r * r
    z = np.zeros(NT)
    for k in range(2, NT):
        z[k] = a1 * z[k - 1] + a2 * z[k - 2] + rng.standard_normal()
    z *= sd_target / z.std()
    return z


def _data(rng, lam):
    theta = np.cumsum(rng.standard_normal(lam.size) * np.sqrt(np.exp(lam)))
    return theta + rng.standard_normal(lam.size)


def _rmse(est, lam, burn=200):
    return float(np.sqrt(np.mean((est[burn:] - lam[burn:]) ** 2)))


def main():
    paths, NT = _paths()
    # add the AR(2) path (regenerated per seed inside the loop)
    names = list(paths.keys()) + ["AR(2) oscillatory"]
    NSEED = 30

    rms = {n: {"ship": [], "orac": [], "orac_nl": []} for n in names}
    demo = {}                                               # one seed for the trajectory plot
    for sd in range(NSEED):
        rng = np.random.default_rng(4000 + sd)
        for n in names:
            lam = _ar2(rng, NT) if n == "AR(2) oscillatory" else paths[n]
            x = _data(rng, lam)
            es = run(x); eo = run(x, lam_true=lam); enl = run(x, lam_true=lam, nl=True)
            rms[n]["ship"].append(_rmse(es, lam)); rms[n]["orac"].append(_rmse(eo, lam))
            rms[n]["orac_nl"].append(_rmse(enl, lam))
            if sd == 0:
                demo[n] = (x, lam, es, eo)

    # shipped vs oracle CONFOUNDS the nonlinearity with estimation noise; the fair
    # isolation is oracle-linear vs oracle-with-NL (both perfect offset, finding 20).
    print(f"{'scenario':26s} {'shipped':>8s} {'orac-lin':>8s} {'orac-NL':>8s} "
          f"{'NL cost(var%)':>13s}")
    for n in names:
        sh = np.mean(rms[n]["ship"]); li = np.mean(rms[n]["orac"]); nl = np.mean(rms[n]["orac_nl"])
        cost = 100 * (1 - (li / nl) ** 2) if nl > li else 0.0
        print(f"{n:26s} {sh:8.3f} {li:8.3f} {nl:8.3f} {cost:12.0f}%")

    # -------- trajectory figure: grey data-scale proxy, black truth, colored ests
    fig, ax = plt.subplots(2, 4, figsize=(21, 8.5))
    for a, n in zip(ax.ravel(), names):
        x, lam, es, eo = demo[n]
        tt = np.arange(lam.size)
        aa = ts.tidy(a)
        proxy = np.log(np.maximum(np.diff(x, prepend=x[0]) ** 2, 1e-3))   # noisy per-step scale evidence
        aa.scatter(tt, np.clip(proxy, -4, 8), s=2, color=ts.INK2, alpha=0.16, linewidths=0)
        aa.plot(tt, lam, color="black", lw=1.7, label="true log-scale")
        aa.plot(tt, es, color=ts.SERIES[1], lw=1.5, label="shipped")
        aa.plot(tt, eo, color=ts.SERIES[3], lw=1.3, ls="--", label="oracle (nl handled)")
        sh = np.mean(rms[n]["ship"]); orc = np.mean(rms[n]["orac"])
        aa.set_title(f"{n}\nRMSE ship {sh:.2f} / oracle {orc:.2f} ({100*(sh-orc)/sh:+.0f}%)", fontsize=9)
        aa.set_ylim(-5, 8)
        if n == names[0]:
            aa.legend(loc="upper left", fontsize=7)
    fig.suptitle("Stress battery: shipped vs oracle with the stiffening nonlinearity handled",
                 fontsize=12, y=1.0)
    ts.save(fig, os.path.join(HERE, "figures", "0033-stress-battery.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
