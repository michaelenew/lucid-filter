"""0046 -- The offset channel with nothing fixed: kernel, gain, and null all
learned online, and trust read as directed information.

0044 fixed four things by hand and one dishonestly:
  s_tau (kernel step), eps (restart mass), c (gain)   -- fixed, not learned;
  v_null = Var(y2) over the WHOLE series              -- looks at the future.

This probe removes all four with the workstream's standing techniques:

  * (s_tau, eps) become a 12-point HYPER-GRID of complete mixture filters,
    Bayes-mixed online by prequential likelihood.  s_tau = 0 and eps = 0 are
    explicit members -- the FLAT analogue: "the offset does not move" is a
    hypothesis with a likelihood, not an absence.  The information-theoretic
    price is bounded in advance: a Bayes mixture trails the best member in
    hindsight by at most log 12 = 2.48 nats, total, ever.
  * c becomes a static log-spaced grid crossed with the tau nodes (a scale,
    so its natural coordinate is log c).
  * the null becomes a MATCHED model: an independent same-class latent read
    by y2 alone, its amplitude gridded and Bayes-mixed online.  No marginal
    statistics are taken from the future.

With the matched null, Lambda_t = sum log p(y2_t | both histories) -
log p(y2_t | y2's own history) is a prequential estimate of the DIRECTED
INFORMATION rate from series 1 to series 2 beyond self-prediction -- trust in
the coupling is a measured information flow, not a score against a strawman.
(Only y2's predictive terms are counted, so the estimate is conservative: the
coupled model's improved y1 predictions are discarded.)

Runs:
  A: tau jumps and ramps (0044's path), coupled at c = 0.7;
  B: tau static at 0.6, coupled           -- do the FLAT members win?
  C: tau path as A, y2 from an independent replica -- trust must collapse.

Outputs: figures/fig34-online-learned.png, figures/ode046.json
"""
import json
import numpy as np

import theory_style as ts
import matplotlib.pyplot as plt

from importlib import import_module
d43 = import_module("0043_the_delay_row")

FINE = 32
N = 900
K = 4
SIG1 = SIG2 = 0.3
C_TRUE = 0.7
TAUS = np.arange(0.0, 3.001, 0.1)                 # 31 nodes
CS = np.geomspace(0.35, 1.4, 5)                   # includes 0.7 exactly
HYPERS = [(st, ep) for st in (0.0, 0.01, 0.03, 0.1)
          for ep in (0.0, 1e-3, 1e-2)]            # 12 members


def tau_path(kind):
    t = np.arange(N)
    if kind == "static":
        return np.full(N, int(round(0.6 * FINE)))
    tau = np.where(t < 300, 0.6, 1.9).astype(float)
    ramp = t >= 600
    tau[ramp] = 1.9 - (1.9 - 0.8) * (t[ramp] - 600) / 299
    return np.round(tau * FINE).astype(int)


def simulate(seed, kind, coupled):
    rng = np.random.default_rng(seed)
    G, LLt, read = d43.blocks(d43.rw(0.5), d43.osc(0.05, 2.0))
    d = G.shape[0]
    Af, Qf = d43.disc(G, LLt, 1.0 / FINE)
    w, U = np.linalg.eigh(Qf)
    Lf = U @ np.diag(np.sqrt(np.clip(w, 0, None)))

    def path():
        n_fine = (N + K + 3) * FINE
        Z = np.zeros((n_fine, d)); z = np.zeros(d)
        for t in range(n_fine):
            z = Af @ z + Lf @ rng.standard_normal(d)
            Z[t] = z
        return Z

    Z = path()
    tf = tau_path(kind)
    idx = (np.arange(N) + K + 2) * FINE
    y1 = Z[idx] @ read + SIG1 * rng.standard_normal(N)
    src = Z if coupled else path()
    y2 = C_TRUE * (src[idx - tf] @ read) + SIG2 * rng.standard_normal(N)
    return (G, LLt, read), y1, y2


def coupled_setup():
    G, LLt, read = d43.blocks(d43.rw(0.5), d43.osc(0.05, 2.0))
    d = G.shape[0]
    F, Q, C0 = d43.aug_model(G, LLt, K, diffuse=False)
    _, _, C0d = d43.aug_model(G, LLt, K, diffuse=True)
    lvl = np.array([i * d for i in range(K + 1)])
    C0[np.ix_(lvl, lvl)] = C0d[np.ix_(lvl, lvl)]
    D = d * (K + 1)
    h1 = np.zeros(D); h1[:d] = read
    # (tau, c) product nodes, tau-major
    H2, R2 = [], []
    for tau in TAUS:
        row, vb = d43.delay_row(G, LLt, read, tau, K, d)
        for c in CS:
            H2.append(c * row); R2.append(c * c * vb + SIG2 ** 2)
    return G, LLt, read, F, Q, C0, h1, np.array(H2), np.array(R2), D


def kernel(s_tau, eps):
    """Per-step transition on the tau grid: the matrix exponential of a
    reflecting diffusion generator with variance rate s_tau^2, plus restart
    mass eps.  A per-step Gaussian kernel is wrong for s_tau below the node
    spacing (it rounds to the identity, so slow drift becomes untrackable);
    the generator form is faithful at every scale."""
    from scipy.linalg import expm
    nt = len(TAUS)
    if s_tau > 0:
        dlt = TAUS[1] - TAUS[0]
        L = -2.0 * np.eye(nt) + np.eye(nt, k=1) + np.eye(nt, k=-1)
        L[0, 0] = -1.0; L[-1, -1] = -1.0           # reflecting
        Kt = expm((s_tau ** 2 / (2 * dlt ** 2)) * L)
    else:
        Kt = np.eye(nt)
    Kt = (1 - eps) * Kt + eps / nt
    return np.kron(Kt, np.eye(len(CS)))            # c is static


class Mixture:
    """One (s_tau, eps) member: an IMM mixture over the (tau, c) nodes."""

    def __init__(self, F, Q, C0, h1, H2, R2, D, s_tau, eps):
        self.F, self.Q, self.h1, self.H2, self.R2 = F, Q, h1, H2, R2
        B = len(H2)
        self.m = np.zeros((B, D))
        self.P = np.broadcast_to(C0, (B, D, D)).copy()
        self.w = np.full(B, 1.0 / B)
        self.T = kernel(s_tau, eps)
        self.mix = s_tau > 0 or eps > 0
        self.B, self.D = B, D

    def step(self, y1t, y2t):
        B, D = self.B, self.D
        if self.mix:
            win = self.T.T @ self.w
            W = (self.T * self.w[:, None]).T / np.maximum(win[:, None], 1e-300)
            m_new = W @ self.m
            MM = (self.m[:, :, None] * self.m[:, None, :]).reshape(B, -1)
            P_new = (W @ self.P.reshape(B, -1) + W @ MM).reshape(B, D, D) \
                - m_new[:, :, None] * m_new[:, None, :]
            self.m, self.P, self.w = m_new, P_new, win
        self.m = self.m @ self.F.T
        self.P = np.matmul(np.matmul(self.F, self.P), self.F.T) + self.Q
        # y1 (same row every node)
        Ph = self.P @ self.h1
        S = Ph @ self.h1 + SIG1 ** 2
        e = y1t - self.m @ self.h1
        Kg = Ph / S[:, None]
        self.m = self.m + Kg * e[:, None]
        self.P = self.P - Kg[:, :, None] * Ph[:, None, :]
        # y2 (per-node rows; the arbitration point)
        Ph = np.einsum('bij,bj->bi', self.P, self.H2)
        S = np.einsum('bi,bi->b', self.H2, Ph) + self.R2
        e = y2t - np.einsum('bi,bi->b', self.H2, self.m)
        lp = -0.5 * (np.log(2 * np.pi * S) + e * e / S)
        mx = lp.max()
        ll2 = np.log(np.sum(self.w * np.exp(lp - mx))) + mx
        yhat = np.einsum('bi,bi->b', self.H2, self.m)
        pred_mean = float(np.sum(self.w * yhat))
        pred_var = float(np.sum(self.w * (S + yhat ** 2)) - pred_mean ** 2)
        self.w = self.w * np.exp(lp - mx); self.w /= self.w.sum()
        Kg = Ph / S[:, None]
        self.m = self.m + Kg * e[:, None]
        self.P = self.P - Kg[:, :, None] * Ph[:, None, :]
        self.P = 0.5 * (self.P + np.swapaxes(self.P, 1, 2))
        return ll2, pred_var


class NullBank:
    """Matched null: independent same-class latent read by y2 alone, amplitude
    gridded over CS, Bayes-mixed online.  Nothing taken from the future."""

    def __init__(self, G, LLt, read):
        d = G.shape[0]
        A1, Q1 = d43.disc(G, LLt, 1.0)
        from scipy.linalg import solve_continuous_lyapunov
        P0 = solve_continuous_lyapunov(G, -LLt)
        P0 = 0.5 * (P0 + P0.T); P0[0, 0] = 1e4     # diffuse level
        na = len(CS)
        self.A1, self.Q1 = A1, Q1
        self.H = CS[:, None] * read[None, :]
        self.m = np.zeros((na, d))
        self.P = np.broadcast_to(P0, (na, d, d)).copy()
        self.u = np.full(na, 1.0 / na)

    def step(self, y2t):
        self.m = self.m @ self.A1.T
        self.P = np.matmul(np.matmul(self.A1, self.P), self.A1.T) + self.Q1
        Ph = np.einsum('aij,aj->ai', self.P, self.H)
        S = np.einsum('ai,ai->a', self.H, Ph) + SIG2 ** 2
        e = y2t - np.einsum('ai,ai->a', self.H, self.m)
        lp = -0.5 * (np.log(2 * np.pi * S) + e * e / S)
        mx = lp.max()
        ll = np.log(np.sum(self.u * np.exp(lp - mx))) + mx
        pmean = float(np.sum(self.u * np.einsum('ai,ai->a', self.H, self.m)))
        pvar = float(np.sum(self.u * (S + np.einsum(
            'ai,ai->a', self.H, self.m) ** 2)) - pmean ** 2)
        self.u = self.u * np.exp(lp - mx); self.u /= self.u.sum()
        Kg = Ph / S[:, None]
        self.m = self.m + Kg * e[:, None]
        self.P = self.P - Kg[:, :, None] * Ph[:, None, :]
        return ll, pvar


def run(kind, coupled, seed):
    sys_, y1, y2 = simulate(seed, kind, coupled)
    G, LLt, read, F, Q, C0, h1, H2, R2, D = coupled_setup()
    mixes = [Mixture(F, Q, C0, h1, H2, R2, D, st, ep) for st, ep in HYPERS]
    null = NullBank(G, LLt, read)
    nH = len(HYPERS)
    Lh = np.zeros(nH)                               # per-hyper cumulative ll2
    Wlog = np.zeros(nH)                             # hyper log-weights
    Lam = np.zeros(N)
    post_tau = np.zeros((N, len(TAUS)))
    post_c = np.zeros((N, len(CS)))
    Wh_t = np.zeros((N, nH))
    vr = np.zeros((N, 2))                           # predictive vars (mix,null)
    lam = 0.0
    for t in range(N):
        lls = np.zeros(nH); pv = np.zeros(nH)
        for i, mx in enumerate(mixes):
            lls[i], pv[i] = mx.step(y1[t], y2[t])
        Lh += lls
        a = Wlog + lls
        m0 = a.max()
        ll2_mix = np.log(np.mean(np.exp(Wlog - m0 + lls))) + m0 \
            - (np.log(np.mean(np.exp(Wlog - m0))) + m0)
        Wlog = Wlog + lls
        Wlog -= Wlog.max()
        Wh = np.exp(Wlog); Wh /= Wh.sum()
        ll2_null, pv_null = null.step(y2[t])
        lam += ll2_mix - ll2_null
        Lam[t] = lam
        Wh_t[t] = Wh
        w_nodes = sum(Wh[i] * mixes[i].w for i in range(nH))
        post_tau[t] = w_nodes.reshape(len(TAUS), len(CS)).sum(axis=1)
        post_c[t] = w_nodes.reshape(len(TAUS), len(CS)).sum(axis=0)
        vr[t] = (np.sum(Wh * pv), pv_null)
    regret = float(Lh.max() - (np.log(np.mean(np.exp(Lh - Lh.max())))
                               + Lh.max()))
    return dict(y2=y2, Lam=Lam, post_tau=post_tau, post_c=post_c,
                Wh=Wh_t, Lh=Lh, regret=regret, vr=vr)


if __name__ == "__main__":
    A = run("moving", True, 4601)
    B = run("static", True, 4602)
    C = run("moving", False, 4603)

    tau_true = tau_path("moving") / FINE
    mean_tau = A["post_tau"] @ TAUS
    reloc = int(np.argmax(np.abs(mean_tau[300:] - 1.9) < 0.2))
    cdf = np.cumsum(A["post_tau"], axis=1)
    lo = TAUS[np.argmax(cdf >= 0.05, axis=1)]
    hi = TAUS[np.argmax(cdf >= 0.95, axis=1)]
    cover = float(np.mean((tau_true >= lo - 0.05) & (tau_true <= hi + 0.05)))

    grp = lambda W, sel: float(sum(W[-1, i] for i, (st, ep) in
                                   enumerate(HYPERS) if sel(st, ep)))
    c_mean = float(A["post_c"][-1] @ CS)
    res = {
        "A_moving": {
            "latency_points": reloc,
            "coverage90": cover,
            "rms_err_static_segs": float(np.sqrt(np.mean(
                (mean_tau[np.r_[50:300, 330:600]]
                 - tau_true[np.r_[50:300, 330:600]]) ** 2))),
            "c_posterior_mean": c_mean,
            "c_mass_on_0.7": float(A["post_c"][-1][2]),
            "hyper_mass_s0": grp(A["Wh"], lambda s, e: s == 0),
            "hyper_mass_eps0": grp(A["Wh"], lambda s, e: e == 0),
            "regret_nats": A["regret"], "regret_bound": float(np.log(12)),
            "Lam_slope": float((A["Lam"][-1] - A["Lam"][100]) / (N - 100)),
            "half_log_var_ratio": float(np.mean(
                0.5 * np.log(A["vr"][100:, 1] / A["vr"][100:, 0]))),
        },
        "B_static": {
            "hyper_mass_s0": grp(B["Wh"], lambda s, e: s == 0),
            "hyper_mass_eps0": grp(B["Wh"], lambda s, e: e == 0),
            "best_hyper": HYPERS[int(np.argmax(B["Lh"]))],
            "regret_nats": B["regret"],
        },
        "C_control": {
            "Lam_slope": float((C["Lam"][-1] - C["Lam"][100]) / (N - 100)),
            "trust_final": float(1 / (1 + np.exp(
                -np.clip(C["Lam"][-1], -500, 500)))),
        },
        "A_best_hyper": HYPERS[int(np.argmax(A["Lh"]))],
    }
    print(json.dumps(res, indent=1))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.6),
                             gridspec_kw={"height_ratios": [1.3, 1]})
    ax = axes[0, 0]
    im = ax.pcolormesh(np.arange(N), TAUS, np.log10(A["post_tau"].T + 1e-6),
                       cmap="Blues", vmin=-4, vmax=0, shading="auto")
    ax.plot(np.arange(N), tau_true, color=ts.SERIES[1], lw=1.4, ls="--",
            label=r"true $\tau_t$")
    ax.set_title("run A: $\\tau$ posterior, kernel and gain learned online")
    ax.set_xlabel("t"); ax.set_ylabel(r"$\tau$"); ax.legend(loc="upper left")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}$ posterior")
    ts.tidy(ax)

    ax = axes[0, 1]
    for W, lab, col in ((A["Wh"], "A (moving)", ts.SERIES[0]),
                        (B["Wh"], "B (static)", ts.SERIES[1])):
        mass = np.array([sum(W[t, i] for i, (st, ep) in enumerate(HYPERS)
                             if st == 0) for t in range(N)])
        ax.plot(np.arange(N), mass, color=col, label=f"{lab}: mass on "
                                                     r"$s_\tau=0$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("the FLAT analogue: 'the offset does not move'")
    ax.set_xlabel("t"); ax.set_ylabel("posterior mass"); ax.legend()
    ts.tidy(ax)

    ax = axes[1, 0]
    im = ax.pcolormesh(np.arange(N), np.log(CS), np.log10(A["post_c"].T
                                                          + 1e-6),
                       cmap="Blues", vmin=-4, vmax=0, shading="auto")
    ax.axhline(np.log(C_TRUE), color=ts.SERIES[1], ls="--", lw=1.2)
    ax.set_title(f"gain: posterior mean {res['A_moving']['c_posterior_mean']:.3f}"
                 f" (truth {C_TRUE})")
    ax.set_xlabel("t"); ax.set_ylabel(r"$\log c$")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}$ posterior")
    ts.tidy(ax)

    ax = axes[1, 1]
    ax.plot(np.arange(N), A["Lam"], color=ts.SERIES[2],
            label=f"A coupled: {res['A_moving']['Lam_slope']:.2f} nats/pt")
    ax.plot(np.arange(N), C["Lam"], color=ts.SERIES[7],
            label=f"C control: {res['C_control']['Lam_slope']:.1f} nats/pt")
    ax.set_title("directed information vs the matched null")
    ax.set_xlabel("t"); ax.set_ylabel(r"$\Lambda$ (nats)")
    ax.legend(); ts.tidy(ax)
    ts.save(fig, "figures/fig34-online-learned.png")

    with open("figures/ode046.json", "w") as f:
        json.dump(res, f, indent=1)
    print("wrote figures/ode046.json")
