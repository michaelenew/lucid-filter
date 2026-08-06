"""0054 -- Which series leads: negative tau via deferred updates.

0047 section 4 item 5, built as 0042 section 4 designed it: "a lead is a lag
in processing time."  A node with tau_j < 0 says y2 reads the latent's FUTURE;
its bracketing states first exist d_j = ceil(-tau_j) steps later, so that node
consumes the y2 stream with a processing lag of d_j samples and applies the
SAME bridge row at fractional position s = d_j + tau_j in [0, 1).  Nothing
else changes: the update ledger reorders per node, and each y2 sample is
processed by every node exactly once.  Log-likelihoods are accumulated PER
SAMPLE so the node comparison is over a common observation set (nodes differ
by at most 2 samples of information at any instant; the comparison set is
truncated to what all nodes have seen).

Runs (latent = RW level + damped oscillator, c = 0.7 known, restart kernel):
  A: y2 LEADS  (tau = -0.8) -- the sign must be detected, not assumed;
  B: y2 LAGS   (tau = +0.8) -- symmetry check;
  C: sign FLIP (+0.8 -> -0.8 at t = 450) -- relocation across zero, where the
     processing lag itself changes.

Reported: sign posterior P(tau < 0), points to 99:1 on the sign, tau RMS,
flip relocation latency.

Outputs: figures/fig38-which-leads.png, figures/ode054.json
"""
import json
import numpy as np

import theory_style as ts
import matplotlib.pyplot as plt

from importlib import import_module
d43 = import_module("0043_the_delay_row")
d46 = import_module("0046_online_learned_offset")

FINE = 32
N = 900
K = 4
SIG1 = SIG2 = 0.3
C = 0.7
TAUS = np.arange(-2.0, 2.001, 0.1)         # 41 nodes, sign free
EPS = 1e-3


def simulate(tau_path_fine, seed):
    rng = np.random.default_rng(seed)
    G, LLt, read = d43.blocks(d43.rw(0.5), d43.osc(0.05, 2.0))
    d = G.shape[0]
    Af, Qf = d43.disc(G, LLt, 1.0 / FINE)
    w, U = np.linalg.eigh(Qf)
    Lf = U @ np.diag(np.sqrt(np.clip(w, 0, None)))
    n_fine = (N + K + 6) * FINE
    Z = np.zeros((n_fine, d)); z = np.zeros(d)
    for t in range(n_fine):
        z = Af @ z + Lf @ rng.standard_normal(d)
        Z[t] = z
    idx = (np.arange(N) + K + 2) * FINE
    y1 = Z[idx] @ read + SIG1 * rng.standard_normal(N)
    y2 = C * (Z[idx - tau_path_fine] @ read) + SIG2 * rng.standard_normal(N)
    return (G, LLt, read), y1, y2


def run(tau_path_fine, seed):
    sys_, y1, y2 = simulate(tau_path_fine, seed)
    G, LLt, read = sys_
    d = G.shape[0]
    F, Q, C0 = d43.aug_model(G, LLt, K, diffuse=False)
    _, _, C0d = d43.aug_model(G, LLt, K, diffuse=True)
    lvl = np.array([i * d for i in range(K + 1)])
    C0[np.ix_(lvl, lvl)] = C0d[np.ix_(lvl, lvl)]
    D = d * (K + 1)
    h1 = np.zeros(D); h1[:d] = read

    # UNIFORM deferral: every node processes y2[k] at time k + dmax, so all
    # nodes predict under identical conditioning.  (Per-node deferral d_j =
    # ceil(-tau_j) is biased: a node processing later has seen more y1, which
    # systematically favours longer deferrals -- measured as a persistent
    # spurious band at the d = 1 -> 2 class boundary before this fix.)
    B = len(TAUS)
    dmax = int(np.ceil(-TAUS.min()))
    H2 = np.zeros((B, D)); R2 = np.zeros(B)
    DEF = np.full(B, dmax, dtype=int)
    for j, tau in enumerate(TAUS):
        s = dmax + tau                      # effective read-back, in [0, K]
        row, vb = d43.delay_row(G, LLt, read, s, K, d)
        H2[j] = C * row; R2[j] = C * C * vb + SIG2 ** 2

    m = np.zeros((B, D))
    P = np.broadcast_to(C0, (B, D, D)).copy()
    w = np.full(B, 1.0 / B)
    post = np.zeros((N, B))
    for t in range(N):
        # restart kernel over tau
        w = (1 - EPS) * w + EPS / B
        m = m @ F.T
        P = np.matmul(np.matmul(F, P), F.T) + Q
        Ph = P @ h1
        S = Ph @ h1 + SIG1 ** 2
        e = y1[t] - m @ h1
        Kg = Ph / S[:, None]
        m = m + Kg * e[:, None]
        P = P - Kg[:, :, None] * Ph[:, None, :]
        # y2: node j consumes sample t - DEF[j]
        for dj in range(dmax + 1):
            k = t - dj
            sel = DEF == dj
            if k < 0 or not sel.any():
                continue
            Hs = H2[sel]; Rs = R2[sel]
            Ps = P[sel]; ms = m[sel]
            Ph = np.einsum('bij,bj->bi', Ps, Hs)
            S = np.einsum('bi,bi->b', Hs, Ph) + Rs
            e = y2[k] - np.einsum('bi,bi->b', Hs, ms)
            lp = -0.5 * (np.log(2 * np.pi * S) + e * e / S)
            w[sel] = w[sel] * np.exp(lp - lp.max())
            Kg = Ph / S[:, None]
            m[sel] = ms + Kg * e[:, None]
            P[sel] = Ps - Kg[:, :, None] * Ph[:, None, :]
        w = w / w.sum()
        P = 0.5 * (P + np.swapaxes(P, 1, 2))
        post[t] = w
    return post


if __name__ == "__main__":
    res = {}
    paths = {
        "A_leads": np.full(N, int(round(-0.8 * FINE))),
        "B_lags": np.full(N, int(round(0.8 * FINE))),
        "C_flip": np.where(np.arange(N) < 450, int(round(0.8 * FINE)),
                           int(round(-0.8 * FINE))),
    }
    posts = {}
    for i, (nm, pth) in enumerate(paths.items()):
        post = run(pth, 5401 + i)
        posts[nm] = post
        tau_true = pth / FINE
        mean_tau = post @ TAUS
        p_neg = post[:, TAUS < -0.05].sum(axis=1)
        p_pos = post[:, TAUS > 0.05].sum(axis=1)
        sgn_true = np.sign(tau_true)
        p_right = np.where(sgn_true < 0, p_neg, p_pos)
        lo = np.log(np.maximum(p_right, 1e-300)) \
            - np.log(np.maximum(1 - p_right, 1e-300))
        t99 = int(np.argmax(lo[20:] >= np.log(99.0))) + 20 \
            if np.any(lo[20:] >= np.log(99.0)) else -1
        entry = {
            "sign_t99": t99,
            "P_sign_final": float(p_right[-1]),
            "rms_tau_err": float(np.sqrt(np.mean(
                (mean_tau[50:] - tau_true[50:]) ** 2))),
        }
        if nm == "C_flip":
            reloc = np.argmax(np.abs(mean_tau[450:] + 0.8) < 0.2)
            entry["flip_latency_points"] = int(reloc)
            entry["rms_tau_err"] = float(np.sqrt(np.mean(
                (mean_tau[np.r_[50:450, 470:N]]
                 - tau_true[np.r_[50:450, 470:N]]) ** 2)))
        res[nm] = entry
    print(json.dumps(res, indent=1))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for ax, nm in zip(axes, paths):
        im = ax.pcolormesh(np.arange(N), TAUS, np.log10(posts[nm].T + 1e-6),
                           cmap="Blues", vmin=-4, vmax=0, shading="auto")
        ax.plot(np.arange(N), paths[nm] / FINE, color=ts.SERIES[1], lw=1.3,
                ls="--", label=r"true $\tau_t$")
        ax.axhline(0, color=ts.INK2, lw=0.8)
        ttl = {"A_leads": r"A: $y_2$ leads ($\tau=-0.8$)",
               "B_lags": r"B: $y_2$ lags ($\tau=+0.8$)",
               "C_flip": "C: the lead/lag flips"}[nm]
        extra = f", flip latency {res[nm]['flip_latency_points']}" \
            if nm == "C_flip" else f", sign 99:1 at t={res[nm]['sign_t99']}"
        ax.set_title(ttl + extra)
        ax.set_xlabel("t"); ax.set_ylabel(r"$\tau$")
        ax.legend(loc="upper right")
        ts.tidy(ax)
    fig.colorbar(im, ax=axes[-1], label=r"$\log_{10}$ posterior")
    ts.save(fig, "figures/fig38-which-leads.png")

    with open("figures/ode054.json", "w") as f:
        json.dump(res, f, indent=1)
    print("wrote figures/ode054.json")
