"""0044 -- Tracking the offset: the gridded-tau mixture as a trusted
distribution, online.

Probes claim 7 of 0042 section 7.  A latent process with both tau channels --
a random-walk level (absolute, unaliased) and a damped oscillator (sharp,
aliased) -- is read twice, the second reading at an offset that JUMPS mid-run
and then RAMPS.  The filter grids tau, runs one conditional Kalman recursion
per node on the augmented state (IMM mixing under an in-model transition
kernel: random-walk steps of scale S_TAU plus restart mass EPS), and carries a
null member (series uncoupled, matched marginal) outside the kernel.

Reported:
  - the posterior over tau against the truth, every step (the trusted
    distribution);
  - trust sigma(Lambda) of "the series are related", for a coupled run and an
    uncoupled control;
  - relocation latency after the jump, against the parent's ledger arithmetic
    (log(1/EPS) + target) / KL-rate;
  - coverage of the central 90% posterior band.

Outputs: figures/fig33-offset-tracking.png, figures/ode044.json
"""
import json
import numpy as np

import theory_style as ts
import matplotlib.pyplot as plt

from importlib import import_module
d43 = import_module("0043_the_delay_row")

FINE = 32
N = 900
K = 4                      # stored past states -> tau window [0, 3] safe
S_TAU = 0.02               # kernel: random-walk step SD on tau, per point
EPS = 1e-3                 # kernel: restart mass, per point
SIG1 = SIG2 = 0.3
TAUS = np.arange(0.0, 3.001, 0.05)


def tau_path():
    t = np.arange(N)
    tau = np.where(t < 300, 0.6, 1.9).astype(float)
    ramp = t >= 600
    tau[ramp] = 1.9 - (1.9 - 0.8) * (t[ramp] - 600) / 299
    return np.round(tau * FINE).astype(int)


def simulate_two(seed, coupled):
    """Latent = RW level + damped oscillator; y2 reads the delayed sum along
    tau_path (or an independent replica's, if uncoupled)."""
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
    tf = tau_path()
    idx = (np.arange(N) + K + 2) * FINE
    y1 = Z[idx] @ read + SIG1 * rng.standard_normal(N)
    src = Z if coupled else path()          # control: independent replica
    y2 = src[idx - tf] @ read + SIG2 * rng.standard_normal(N)
    return (G, LLt, read), y1, y2


def run(seed=4401, coupled=True):
    sys, y1, y2 = simulate_two(seed, coupled)
    G, LLt, read = sys
    d = G.shape[0]
    F, Q, C0 = d43.aug_model(G, LLt, K, diffuse=False)
    # diffuse level: overwrite level entries of C0 with the exact
    # forward-consistent diffuse construction from aug_model
    _, _, C0d = d43.aug_model(G, LLt, K, diffuse=True)
    lvl = np.array([i * d for i in range(K + 1)])
    C0[np.ix_(lvl, lvl)] = C0d[np.ix_(lvl, lvl)]
    D = d * (K + 1)
    h1 = np.zeros(D); h1[:d] = read

    B = len(TAUS)
    H2 = np.zeros((B, D)); R2 = np.zeros(B)
    for j, tau in enumerate(TAUS):
        row, vb = d43.delay_row(G, LLt, read, tau, K, d)
        H2[j] = row; R2[j] = vb + SIG2 ** 2

    # kernel on the grid: reflected Gaussian RW + restart mass
    dist = TAUS[None, :] - TAUS[:, None]
    Kern = np.exp(-0.5 * (dist / S_TAU) ** 2)
    Kern /= Kern.sum(axis=1, keepdims=True)
    Kern = (1 - EPS) * Kern + EPS / B

    # null member: y2 iid with matched marginal (stationary var of the
    # oscillator + measurement; the RW level is shared -- the honest null for
    # "no *offset* relationship" keeps the marginal scale of y2)
    v_null = float(np.var(y2))

    m = np.zeros((B, D))
    P = np.broadcast_to(C0, (B, D, D)).copy()
    wgt = np.full(B, 1.0 / B)
    m0 = np.zeros(D); P0 = C0.copy()        # null runs y1 alone
    post = np.zeros((N, B)); Lam = np.zeros(N)
    ll2_mix_t = np.zeros(N); ll2_null_t = np.zeros(N)
    lp_nodes = np.zeros((N, B))             # per-node log-predictives, kept
    lam = 0.0
    for t in range(N):
        # IMM mixing of the coupled family under the kernel
        win = Kern.T @ wgt
        Wmix = Kern * wgt[None, :].T          # (src, tgt) joint
        Wmix = (Wmix / np.maximum(win[None, :], 1e-300)).T  # (tgt, src)
        m_new = Wmix @ m
        dm = m[None, :, :] - m_new[:, None, :]
        P_new = np.einsum('ts,sij->tij', Wmix, P) \
            + np.einsum('ts,tsi,tsj->tij', Wmix, dm, dm)
        m, P, wgt = m_new, P_new, win

        # predict
        m = m @ F.T
        P = np.einsum('ij,bjk,lk->bil', F, P, F) + Q
        m0 = F @ m0; P0 = F @ P0 @ F.T + Q

        # update y1 (identical model for every member and the null)
        for arr in ("mix",):
            Ph = np.einsum('bij,j->bi', P, h1)
            S = Ph @ h1 + SIG1 ** 2
            e = y1[t] - m @ h1
            Kg = Ph / S[:, None]
            m = m + Kg * e[:, None]
            P = P - np.einsum('bi,bj->bij', Kg, Ph)
        Ph0 = P0 @ h1; S0 = h1 @ Ph0 + SIG1 ** 2
        Kg0 = Ph0 / S0
        m0 = m0 + Kg0 * (y1[t] - h1 @ m0); P0 = P0 - np.outer(Kg0, Ph0)

        # update y2: per-node likelihoods arbitrate the mixture
        Ph = np.einsum('bij,bj->bi', P, H2)
        S = np.einsum('bi,bi->b', H2, Ph) + R2
        e = y2[t] - np.einsum('bi,bi->b', H2, m)
        lp = -0.5 * (np.log(2 * np.pi * S) + e * e / S)
        ll2_mix = np.log(np.sum(wgt * np.exp(lp - lp.max()))) + lp.max()
        ll2_null = -0.5 * (np.log(2 * np.pi * v_null) + y2[t] ** 2 / v_null)
        wgt = wgt * np.exp(lp - lp.max())
        wgt = wgt / wgt.sum()
        Kg = Ph / S[:, None]
        m = m + Kg * e[:, None]
        P = P - np.einsum('bi,bj->bij', Kg, Ph)
        P = 0.5 * (P + np.swapaxes(P, 1, 2))

        lam += ll2_mix - ll2_null
        post[t] = wgt; Lam[t] = lam
        ll2_mix_t[t] = ll2_mix; ll2_null_t[t] = ll2_null
        lp_nodes[t] = lp
    return post, Lam, ll2_mix_t, ll2_null_t, lp_nodes


def summarise(post, tau_true):
    mean = post @ TAUS
    err = mean - tau_true
    # central 90% band coverage
    cdf = np.cumsum(post, axis=1)
    lo = TAUS[np.argmax(cdf >= 0.05, axis=1)]
    hi = TAUS[np.argmax(cdf >= 0.95, axis=1)]
    cover = float(np.mean((tau_true >= lo - 0.025) & (tau_true <= hi + 0.025)))
    return mean, err, cover


if __name__ == "__main__":
    tau_true = tau_path() / FINE

    post, Lam, llm, lln, lp_nodes = run(coupled=True)
    post_c, Lam_c, _, _, _ = run(seed=4402, coupled=False)

    mean, err, cover = summarise(post, tau_true)

    seg = {"pre-jump": slice(50, 300), "post-jump": slice(330, 600),
           "ramp": slice(630, 900)}
    res = {"rms_err": {k: float(np.sqrt(np.mean(err[s] ** 2)))
                       for k, s in seg.items()},
           "coverage90": cover}

    # relocation latency after the jump at t=300, against the ledger
    # arithmetic: latency ~ (log(1/EPS) + Lambda_target) / KL-rate, with the
    # KL-rate measured as the per-point log-predictive gap between the node at
    # the new tau and the node at the old tau, just after the jump
    reloc = np.argmax(np.abs(mean[300:] - 1.9) < 0.2)
    j_old = int(np.argmin(np.abs(TAUS - 0.6)))
    j_new = int(np.argmin(np.abs(TAUS - 1.9)))
    kl_rate = float(np.mean(lp_nodes[301:307, j_new] - lp_nodes[301:307, j_old]))
    res["latency_points"] = int(reloc)
    res["kl_rate_post_jump"] = kl_rate
    res["ledger_predicted_latency"] = float(
        (np.log(1.0 / EPS) + np.log(99.0)) / kl_rate)
    res["lambda_slope_coupled"] = float((Lam[-1] - Lam[100]) / (N - 100))
    res["lambda_slope_control"] = float((Lam_c[-1] - Lam_c[100]) / (N - 100))
    sig = lambda x: float(1 / (1 + np.exp(-np.clip(x, -500, 500))))
    res["trust_at_20"] = sig(Lam[20])
    res["control_trust_final"] = sig(Lam_c[-1])

    print(json.dumps(res, indent=1))

    # ---- figure
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.4),
                             gridspec_kw={"height_ratios": [1.4, 1]})
    ax = axes[0, 0]
    im = ax.pcolormesh(np.arange(N), TAUS, np.log10(post.T + 1e-6),
                       cmap="Blues", vmin=-4, vmax=0, shading="auto")
    ax.plot(np.arange(N), tau_true, color=ts.SERIES[1], lw=1.4, ls="--",
            label=r"true $\tau_t$")
    ax.set_title("the trusted distribution: posterior over $\\tau$, every step")
    ax.set_xlabel("t"); ax.set_ylabel(r"$\tau$"); ax.legend(loc="upper left")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}$ posterior")
    ts.tidy(ax)

    ax = axes[0, 1]
    ax.plot(np.arange(N), mean, color=ts.SERIES[0], label="posterior mean")
    ax.plot(np.arange(N), tau_true, color=ts.SERIES[1], ls="--", lw=1.2,
            label="truth")
    ax.set_title(f"tracking: latency {res['latency_points']} points, "
                 f"coverage90 {cover:.2f}")
    ax.set_xlabel("t"); ax.set_ylabel(r"$\tau$"); ax.legend()
    ts.tidy(ax)

    ax = axes[1, 0]
    ax.plot(np.arange(N), 1 / (1 + np.exp(-np.clip(Lam, -500, 500))),
            color=ts.SERIES[2], label="coupled run")
    ax.plot(np.arange(N), 1 / (1 + np.exp(-np.clip(Lam_c, -500, 500))),
            color=ts.SERIES[7], label="uncoupled control")
    ax.set_title(r"trust $\sigma(\Lambda)$ that the series are related")
    ax.set_xlabel("t"); ax.set_ylabel("trust"); ax.legend()
    ts.tidy(ax)

    ax = axes[1, 1]
    ax.plot(np.arange(N), Lam, color=ts.SERIES[2], label="coupled")
    ax.plot(np.arange(N), Lam_c, color=ts.SERIES[7], label="control")
    ax.set_title(f"accumulated nats: slopes "
                 f"{res['lambda_slope_coupled']:.2f} / "
                 f"{res['lambda_slope_control']:.2f} per point")
    ax.set_xlabel("t"); ax.set_ylabel(r"$\Lambda$"); ax.legend()
    ts.tidy(ax)
    ts.save(fig, "figures/fig33-offset-tracking.png")

    with open("figures/ode044.json", "w") as f:
        json.dump(res, f, indent=1)
    print("wrote figures/ode044.json")
