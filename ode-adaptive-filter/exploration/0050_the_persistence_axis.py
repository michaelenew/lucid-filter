"""0050 -- The persistence axis of the tau kernel: the kinetic (tau, taudot)
grid, and whether the likelihood endorses it.

0047 section 3 diagnosed the ramp failure: the hyper family spanned impulse
(restart) and undirected diffusion, and a deterministic drift is atypical
under both -- the tracker staircases and undercovers (0.61 on the ramp)
while staying prequentially near-optimal.  The missing structure is the
parent's persistence coordinate, which for a time-valued nuisance is a
VELOCITY: nodes (tau_j, taudot_r) whose kernel ADVECTS mass along tau at each
node's velocity (fractional shifts split between bracketing nodes -- first-
order upwind transport), plus a small velocity-switching mass and the restart
channel.

Three members, Bayes-mixed online by prequential likelihood (regret <= log 3):
  FLAT     : (s_tau, eps) = (0, 0)          -- "the offset does not move"
  DIFFUSE  : (0.03, 1e-2), 0046's best     -- impulse + undirected diffusion
  KINETIC  : (tau, taudot) advection, velocity switch kappa = 0.01, eps = 1e-3

Questions, each measured on the jump+ramp path and on a static path:
  1. does KINETIC take the posterior mass during the ramp, and only there?
  2. does ramp coverage recover (0.61 -> ~0.9) and the staircase RMS drop?
  3. is taudot itself readable -- does its posterior find the true rate
     -0.00367/step?
  4. does the static run still put its mass on FLAT (no hallucinated drift)?

The KINETIC member's velocity grid and kappa are structural constants here;
in a full treatment they join the hyper-grid like everything else.

Outputs: figures/fig36-persistence-axis.png, figures/ode050.json
"""
import json
import numpy as np

import theory_style as ts
import matplotlib.pyplot as plt

from importlib import import_module
d43 = import_module("0043_the_delay_row")
d46 = import_module("0046_online_learned_offset")

FINE, N, K = d46.FINE, d46.N, d46.K
SIG1 = SIG2 = 0.3
TAUS, CS = d46.TAUS, d46.CS                    # 31 tau nodes, 5 gains
VDOTS = np.array([-0.006, -0.003, 0.0, 0.003, 0.006])
KAPPA = 0.01                                   # velocity switch mass / step
EPS_K = 1e-3                                   # restart mass / step
RAMP_RATE = -(1.9 - 0.8) / 299                 # truth during the ramp


def kinetic_kernel():
    """(tau, taudot) transition: advect, then switch velocity, then restart.
    tau-major, taudot-minor ordering; row = from, column = to."""
    nt, nv = len(TAUS), len(VDOTS)
    n = nt * nv
    dlt = TAUS[1] - TAUS[0]
    adv = np.zeros((n, n))
    for j in range(nt):
        for r in range(nv):
            g = np.clip((TAUS[j] + VDOTS[r] - TAUS[0]) / dlt, 0, nt - 1)
            f = int(np.floor(g)); a = g - f
            adv[j * nv + r, f * nv + r] += 1 - a
            if a > 0:
                adv[j * nv + r, (f + 1) * nv + r] += a
    sw = np.kron(np.eye(nt), (1 - KAPPA) * np.eye(nv) + KAPPA / nv)
    res = (1 - EPS_K) * np.eye(n) + EPS_K / n
    return adv @ sw @ res


def run(kind, seed):
    sys_, y1, y2 = d46.simulate(seed, kind, True)
    G, LLt, read, F, Q, C0, h1, H2, R2, D = d46.coupled_setup()
    nt, nc, nv = len(TAUS), len(CS), len(VDOTS)
    # KINETIC nodes (tau, taudot, c): observation ignores taudot
    H2k = np.repeat(H2.reshape(nt, nc, D), nv, axis=0).reshape(-1, D)
    R2k = np.repeat(R2.reshape(nt, nc), nv, axis=0).reshape(-1)
    Tk = np.kron(kinetic_kernel(), np.eye(nc))
    members = {
        "FLAT": d46.Mixture(F, Q, C0, h1, H2, R2, D, 0.0, 0.0),
        "DIFFUSE": d46.Mixture(F, Q, C0, h1, H2, R2, D, 0.03, 1e-2),
        "KINETIC": d46.Mixture(F, Q, C0, h1, H2k, R2k, D, T=Tk),
    }
    names = list(members)
    null = d46.NullBank(G, LLt, read)
    Wlog = np.zeros(len(names))
    Wm_t = np.zeros((N, len(names)))
    post_tau = np.zeros((N, nt))
    post_v = np.zeros((N, nv))
    Lam = np.zeros(N); lam = 0.0
    for t in range(N):
        lls = np.zeros(len(names))
        pts = []
        for i, nm in enumerate(names):
            mx = members[nm]
            lls[i], _ = mx.step(y1[t], y2[t])
            if nm == "KINETIC":
                w3 = mx.w.reshape(nt, nv, nc)
                pts.append(w3.sum(axis=(1, 2)))
                post_v[t] = w3.sum(axis=(0, 2))
            else:
                pts.append(mx.w.reshape(nt, nc).sum(axis=1))
        a = Wlog + lls
        mxv = a.max()
        ll2_mix = np.log(np.mean(np.exp(a - mxv))) + mxv \
            - (np.log(np.mean(np.exp(Wlog - Wlog.max()))) + Wlog.max())
        Wlog = a; Wlog -= Wlog.max()
        Wm = np.exp(Wlog); Wm /= Wm.sum()
        Wm_t[t] = Wm
        post_tau[t] = sum(Wm[i] * pts[i] for i in range(len(names)))
        ll_null, _ = null.step(y2[t])
        lam += ll2_mix - ll_null
        Lam[t] = lam
    return dict(Wm=Wm_t, post_tau=post_tau, post_v=post_v, Lam=Lam)


if __name__ == "__main__":
    A = run("moving", 4601)                    # same seed/path as 0046 run A
    B = run("static", 4602)

    tau_true = d46.tau_path("moving") / FINE
    mean_tau = A["post_tau"] @ TAUS
    cdf = np.cumsum(A["post_tau"], axis=1)
    lo = TAUS[np.argmax(cdf >= 0.05, axis=1)]
    hi = TAUS[np.argmax(cdf >= 0.95, axis=1)]
    inb = (tau_true >= lo - 0.05) & (tau_true <= hi + 0.05)
    ramp = slice(630, 900)
    vd_mean_ramp = float(np.mean(A["post_v"][ramp] @ VDOTS))
    names = ["FLAT", "DIFFUSE", "KINETIC"]
    res = {
        "A_moving": {
            "coverage90_ramp": float(np.mean(inb[ramp])),
            "coverage90_all": float(np.mean(inb)),
            "rms_err_ramp": float(np.sqrt(np.mean(
                (mean_tau[ramp] - tau_true[ramp]) ** 2))),
            "latency_points": int(np.argmax(np.abs(mean_tau[300:] - 1.9)
                                            < 0.2)),
            "member_mass_ramp": {nm: float(np.mean(A["Wm"][ramp, i]))
                                 for i, nm in enumerate(names)},
            "member_mass_prejump": {nm: float(np.mean(A["Wm"][50:300, i]))
                                    for i, nm in enumerate(names)},
            "taudot_mean_ramp": vd_mean_ramp,
            "taudot_true_ramp": float(RAMP_RATE),
            "Lam_slope": float((A["Lam"][-1] - A["Lam"][100]) / (N - 100)),
        },
        "B_static": {
            "member_mass_final": {nm: float(B["Wm"][-1, i])
                                  for i, nm in enumerate(names)},
            "taudot_mean_final": float(B["post_v"][-1] @ VDOTS),
        },
    }
    print(json.dumps(res, indent=1))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.4),
                             gridspec_kw={"height_ratios": [1.3, 1]})
    ax = axes[0, 0]
    im = ax.pcolormesh(np.arange(N), TAUS, np.log10(A["post_tau"].T + 1e-6),
                       cmap="Blues", vmin=-4, vmax=0, shading="auto")
    ax.plot(np.arange(N), tau_true, color=ts.SERIES[1], lw=1.4, ls="--",
            label=r"true $\tau_t$")
    ax.set_title("run A with the kinetic member: ramp coverage "
                 f"{res['A_moving']['coverage90_ramp']:.2f} (was 0.61)")
    ax.set_xlabel("t"); ax.set_ylabel(r"$\tau$"); ax.legend(loc="upper left")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}$ posterior")
    ts.tidy(ax)

    ax = axes[0, 1]
    for i, (nm, col) in enumerate(zip(names, (ts.SERIES[0], ts.SERIES[2],
                                              ts.SERIES[1]))):
        ax.plot(np.arange(N), A["Wm"][:, i], color=col, label=nm)
    for x in (300, 600):
        ax.axvline(x, color=ts.INK2, lw=0.7, ls=":")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("who carries the posterior, and when")
    ax.set_xlabel("t"); ax.set_ylabel("member mass"); ax.legend()
    ts.tidy(ax)

    ax = axes[1, 0]
    im = ax.pcolormesh(np.arange(N), VDOTS, np.log10(A["post_v"].T + 1e-6),
                       cmap="Blues", vmin=-4, vmax=0, shading="auto")
    ax.axhline(RAMP_RATE, color=ts.SERIES[1], ls="--", lw=1.2,
               label="true ramp rate")
    ax.set_title(r"$\dot\tau$ posterior (KINETIC member): ramp mean "
                 f"{vd_mean_ramp:.4f} vs {RAMP_RATE:.4f}")
    ax.set_xlabel("t"); ax.set_ylabel(r"$\dot\tau$"); ax.legend(loc="lower left")
    fig.colorbar(im, ax=ax)
    ts.tidy(ax)

    ax = axes[1, 1]
    for i, (nm, col) in enumerate(zip(names, (ts.SERIES[0], ts.SERIES[2],
                                              ts.SERIES[1]))):
        ax.plot(np.arange(N), B["Wm"][:, i], color=col, label=nm)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("static run: no hallucinated drift, mass "
                 f"{res['B_static']['member_mass_final']['FLAT']:.2f} on FLAT")
    ax.set_xlabel("t"); ax.set_ylabel("member mass"); ax.legend()
    ts.tidy(ax)
    ts.save(fig, "figures/fig36-persistence-axis.png")

    with open("figures/ode050.json", "w") as f:
        json.dump(res, f, indent=1)
    print("wrote figures/ode050.json")
