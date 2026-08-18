"""0048 -- The tube grid: the saturated rung online, and the sliding-window
anchor.

Two things, both from the session's directives.

(1) THE TUBE.  The free-coupling family y2 = b' z + v is bilinear and grids
exponentially in p -- but the delay manifold organises b-space around itself.
In the delayed frame the manifold's tangent directions are `read` (the gain
direction) and `read @ G` (the derivative direction -- 0043's ridge made
flesh), so the normal complement is what "not a delay, not a derivative"
means to first order.  For this latent (level + one pair, d = 3) the tube
coordinates (c, tau, eta) with eta along the unit normal are a COMPLETE
reparameterisation of b-space; for larger p they are a genuine tube.  Nodes
(tau_j, c_k, eta_l) run as one mixture; the posterior over eta is the trust
ladder's upper rail, online:

    P(eta = 0 | data)  ==  "the coupling is within measurement distance of
                            the delay family"

Run A: pure delay truth  -> mass must concentrate on eta = 0 (Occam by
       prequential likelihood, no penalty term anywhere);
Run B: off-manifold truth (eta = 0.3) -> delay-trust must collapse WHILE
       coupling-trust vs the matched null stays saturated: the three-way
       verdict null / delay / related-but-not-delay, all online.

(2) THE ANCHOR.  The static estimator instinct -- argmin of the covariance as
the windows slide across one another -- is measured here as the closed-form
start it should be: independent measurement noises leave the CROSS-covariance
unbiased (unlike the autocovariance at lag 0), so the slide is a clean moment
identity, the analogue of the parent's variogram and the ODE filter's IV.
At fractional tau the question is the interpolant between integer lags: the
model says it is the autocovariance shape gamma(.) itself, not a parabola.
Both are scored against full ML.

Outputs: figures/fig35-tube-grid.png, figures/ode048.json
"""
import json
import numpy as np

import theory_style as ts
import matplotlib.pyplot as plt

from importlib import import_module
d43 = import_module("0043_the_delay_row")
d46 = import_module("0046_online_learned_offset")

FINE = 32
N = 700
K = 4
SIG1 = SIG2 = 0.3
C_TRUE = 0.7
TAU_TRUE = 1.2
ETA_TRUE_B = 0.3
TAUS = np.arange(0.0, 3.001, 0.1)          # 31
CS = np.geomspace(0.35, 1.4, 5)            # includes 0.7
ETAS = np.array([-0.3, -0.1, 0.0, 0.1, 0.3])
EPS = 1e-3                                  # restart mass on tau


def normal_direction(G, read):
    """Unit normal to the delay manifold's tangent span {read, read G}."""
    T = np.vstack([read, read @ G])
    _, _, V = np.linalg.svd(T)
    n = V[-1]
    return n / np.linalg.norm(n)


def build_nodes(G, LLt, read, d):
    nhat = normal_direction(G, read)
    H2, R2 = [], []
    for tau in TAUS:
        for c in CS:
            for eta in ETAS:
                f = c * (read + eta * nhat)
                row, vb = d43.delay_row(G, LLt, f, tau, K, d)
                H2.append(row); R2.append(vb + SIG2 ** 2)
    return np.array(H2), np.array(R2), nhat


def run(eta_true, seed):
    G, LLt, read = d43.blocks(d43.rw(0.5), d43.osc(0.05, 2.0))
    d = G.shape[0]
    nhat = normal_direction(G, read)
    f2 = C_TRUE * (read + eta_true * nhat)
    y1, y2 = d43.simulate(G, LLt, read, N, int(round(TAU_TRUE * FINE)),
                          SIG1, SIG2, K, seed, f2=f2)
    F, Q, C0 = d43.aug_model(G, LLt, K, diffuse=False)
    _, _, C0d = d43.aug_model(G, LLt, K, diffuse=True)
    lvl = np.array([i * d for i in range(K + 1)])
    C0[np.ix_(lvl, lvl)] = C0d[np.ix_(lvl, lvl)]
    D = d * (K + 1)
    h1 = np.zeros(D); h1[:d] = read
    H2, R2, _ = build_nodes(G, LLt, read, d)

    nT, nC, nE = len(TAUS), len(CS), len(ETAS)
    B = nT * nC * nE
    m = np.zeros((B, D))
    P = np.broadcast_to(C0, (B, D, D)).copy()
    w = np.full(B, 1.0 / B)
    null = d46.NullBank(G, LLt, read)

    post_eta = np.zeros((N, nE)); post_tau = np.zeros((N, nT))
    Lam_delay = np.zeros(N); Lam_couple = np.zeros(N)
    ld = lc = 0.0
    for t in range(N):
        # mixing: restart over tau within each (c, eta) lineage
        w3 = w.reshape(nT, nC * nE)
        S = w3.sum(axis=0)
        win3 = (1 - EPS) * w3 + (EPS / nT) * S[None, :]
        a = ((1 - EPS) * w3 / np.maximum(win3, 1e-300))     # keep-share
        b_ = ((EPS / nT) * S[None, :] / np.maximum(win3, 1e-300))
        m3 = m.reshape(nT, nC * nE, D)
        P3 = P.reshape(nT, nC * nE, D, D)
        wm3 = w3[:, :, None] * m3
        msum = wm3.sum(axis=0)                              # (nCE, D)
        MM = P3 + m3[:, :, :, None] * m3[:, :, None, :]
        wMM = (w3[:, :, None, None] * MM).sum(axis=0)       # (nCE, D, D)
        m_new = a[:, :, None] * m3 + b_[:, :, None] * (msum[None] / \
            np.maximum(S[None, :, None], 1e-300))
        # second moments: E[mm'] under the mixed distribution
        M2_new = a[:, :, None, None] * MM + b_[:, :, None, None] * (
            wMM[None] / np.maximum(S[None, :, None, None], 1e-300))
        P3 = M2_new - m_new[:, :, :, None] * m_new[:, :, None, :]
        m = m_new.reshape(B, D); P = P3.reshape(B, D, D); w = win3.reshape(B)
        w /= w.sum()

        m = m @ F.T
        P = np.matmul(np.matmul(F, P), F.T) + Q
        Ph = P @ h1
        Sv = Ph @ h1 + SIG1 ** 2
        e = y1[t] - m @ h1
        Kg = Ph / Sv[:, None]
        m = m + Kg * e[:, None]
        P = P - Kg[:, :, None] * Ph[:, None, :]

        Ph = np.einsum('bij,bj->bi', P, H2)
        Sv = np.einsum('bi,bi->b', H2, Ph) + R2
        e = y2[t] - np.einsum('bi,bi->b', H2, m)
        lp = -0.5 * (np.log(2 * np.pi * Sv) + e * e / Sv)
        mx = lp.max()
        ll2 = np.log(np.sum(w * np.exp(lp - mx))) + mx
        w = w * np.exp(lp - mx); w /= w.sum()
        Kg = Ph / Sv[:, None]
        m = m + Kg * e[:, None]
        P = P - Kg[:, :, None] * Ph[:, None, :]
        P = 0.5 * (P + np.swapaxes(P, 1, 2))

        ll_null, _ = null.step(y2[t])
        wt = w.reshape(nT, nC, nE)
        post_eta[t] = wt.sum(axis=(0, 1))
        post_tau[t] = wt.sum(axis=(1, 2))
        p0 = post_eta[t][2]
        ld = np.log(max(p0, 1e-300)) - np.log(max(1 - p0, 1e-300))
        lc += ll2 - ll_null
        Lam_delay[t] = ld; Lam_couple[t] = lc
    return post_eta, post_tau, Lam_delay, Lam_couple


# ---------------------------------------------------------------- anchor

def gamma_osc(s, gamma=0.05, wd=2.0):
    s = np.abs(s)
    return np.exp(-gamma * s) * (np.cos(wd * s) + (gamma / wd) * np.sin(wd * s))


def anchor_trial(seed, n=600):
    """Sliding cross-covariance on a stationary oscillator pair: parabola vs
    gamma-shaped interpolation vs full ML."""
    G, LLt, read = d43.osc(0.05, 2.0)
    y1, y2 = d43.simulate(G, LLt, read, n, int(round(TAU_TRUE * FINE)),
                          SIG1, SIG2, K, seed)
    ks = np.arange(0, 7)
    chat = np.array([np.mean(y1[:n - k] * y2[k:]) for k in ks])
    # parabola through the argmax and neighbours
    i = int(np.argmax(chat[1:-1])) + 1
    dl, d0, dr = chat[i - 1], chat[i], chat[i + 1]
    tau_par = ks[i] + 0.5 * (dl - dr) / (dl - 2 * d0 + dr)
    # gamma-shaped least squares over (c, tau)
    grid = np.arange(0.0, 3.001, 0.01)
    best, tau_gam = np.inf, np.nan
    for tau in grid:
        g = gamma_osc(ks - tau)
        c = (g @ chat) / (g @ g)
        r = float(np.sum((chat - c * g) ** 2))
        if r < best:
            best, tau_gam = r, tau
    # full ML on a fine grid
    taus = np.arange(0.6, 1.81, 0.02)
    ll = d43.profile_tau(d43.osc(0.05, 2.0), TAU_TRUE, n, K, seed, taus,
                         c_grid=np.geomspace(0.4, 1.2, 7))
    tau_ml = float(taus[np.argmax(ll)])
    return float(tau_par), float(tau_gam), tau_ml


if __name__ == "__main__":
    res = {}

    pA_eta, pA_tau, LdA, LcA = run(0.0, 4801)
    pB_eta, pB_tau, LdB, LcB = run(ETA_TRUE_B, 4802)

    t99 = lambda L, sgn: int(np.argmax(sgn * L >= np.log(99.0))) \
        if np.any(sgn * L >= np.log(99.0)) else -1
    res["A_pure_delay"] = {
        "P_eta0_final": float(pA_eta[-1, 2]),
        "t99_delay": t99(LdA, +1),
        "Lam_couple_slope": float((LcA[-1] - LcA[100]) / (N - 100)),
        "tau_hat": float(pA_tau[-1] @ TAUS),
    }
    res["B_off_manifold"] = {
        "P_eta0_final": float(pB_eta[-1, 2]),
        "P_eta_at_truth": float(pB_eta[-1, 4]),
        "t99_not_delay": t99(LdB, -1),
        "Lam_couple_slope": float((LcB[-1] - LcB[100]) / (N - 100)),
        "tau_hat": float(pB_tau[-1] @ TAUS),
    }

    trials = np.array([anchor_trial(s) for s in range(20, 28)])
    err = np.abs(trials - TAU_TRUE)
    res["anchor"] = {
        "rmse_parabola": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "rmse_gamma_shape": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "rmse_ml": float(np.sqrt(np.mean(err[:, 2] ** 2))),
        "n_trials": 8,
    }
    print(json.dumps(res, indent=1))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    ax = axes[0]
    for pe, lab, col in ((pA_eta, "A: pure delay", ts.SERIES[2]),
                         (pB_eta, "B: off-manifold", ts.SERIES[3])):
        ax.plot(np.arange(N), pe[:, 2], color=col, label=f"{lab}: "
                r"$P(\eta=0)$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("the upper rail: is it a pure delay?")
    ax.set_xlabel("t"); ax.set_ylabel(r"$P(\eta=0\mid\mathrm{data})$")
    ax.legend(); ts.tidy(ax)

    ax = axes[1]
    ax.plot(np.arange(N), LcA, color=ts.SERIES[2], label="A: coupling vs null")
    ax.plot(np.arange(N), LcB, color=ts.SERIES[3], label="B: coupling vs null")
    ax.set_title("coupling trust is unharmed off-manifold")
    ax.set_xlabel("t"); ax.set_ylabel(r"$\Lambda$ (nats)")
    ax.legend(); ts.tidy(ax)

    ax = axes[2]
    G, LLt, read = d43.osc(0.05, 2.0)
    y1, y2 = d43.simulate(G, LLt, read, 600, int(round(TAU_TRUE * FINE)),
                          SIG1, SIG2, K, 20)
    ks = np.arange(0, 7)
    chat = np.array([np.mean(y1[:600 - k] * y2[k:]) for k in ks])
    ss = np.linspace(0, 6, 400)
    g = gamma_osc(ss - trials[0, 1])
    c = float((gamma_osc(ks - trials[0, 1]) @ chat)
              / (gamma_osc(ks - trials[0, 1]) @ gamma_osc(ks - trials[0, 1])))
    ax.plot(ss, c * g, color=ts.SERIES[0], label=r"$c\,\gamma(k-\hat\tau)$")
    ax.plot(ks, chat, "o", color=ts.SERIES[1], label="sliding cross-cov")
    ax.axvline(TAU_TRUE, color=ts.INK2, ls="--", lw=0.8)
    ax.set_title(f"the anchor: RMSE parabola {res['anchor']['rmse_parabola']:.3f} / "
                 f"$\\gamma$-shape {res['anchor']['rmse_gamma_shape']:.3f} / "
                 f"ML {res['anchor']['rmse_ml']:.3f}")
    ax.set_xlabel("lag $k$"); ax.set_ylabel("cross-covariance")
    ax.legend(); ts.tidy(ax)
    ts.save(fig, "figures/fig35-tube-grid.png")

    with open("figures/ode048.json", "w") as f:
        json.dump(res, f, indent=1)
    print("wrote figures/ode048.json")
