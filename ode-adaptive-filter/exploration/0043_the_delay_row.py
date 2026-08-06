"""0043 -- The delay row: exactness, the bridge variance, aliasing, and the
derivative/lag ridge.

Probes claims 1-6 of 0042 section 7:
  (a) the delay row e^{-tau G} is exact on the noiseless solution space;
  (b) the bridge row and its variance R_b(tau) match Monte Carlo, R_b vanishes
      at integer tau, and the endpoint exponent is 2m-1 (m = integrations from
      noise entry to the reading);
  (c) likelihood profiles over tau: one oscillator gives an aliased comb at
      spacing 2*pi/omega; a random-walk latent gives one unaliased peak; two
      oscillators resolve the comb;
  (d) the (mu, tau) ridge: on one oscillator pair, fractional-derivative order
      and lag trade off along the phase line with slope arg(lambda)/omega_d
      ("a derivative is a quarter-period lead"); a second channel breaks it.

World: continuous-time linear SDEs, simulated by exact discretisation on a fine
grid (dt = 1/32), observed at unit steps.  The filter model is the exact
discrete-time model of the same SDE, so likelihood comparisons across tau are
comparisons of tau alone.

Outputs: figures/fig30-delay-bridge.png, figures/fig31-offset-aliasing.png,
figures/fig32-derivative-lag-ridge.png, figures/ode043.json
"""
import json
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

import theory_style as ts
import matplotlib.pyplot as plt

rng_global = np.random.default_rng(43)
FINE = 32          # fine steps per unit time
BURN = 30          # observations excluded from every scored likelihood


# ---------------------------------------------------------------- systems

def osc(gamma, wd, var=1.0):
    """Damped oscillator xdd + 2 gamma xd + w0^2 x = sc * white, stationary
    position variance `var`.  State (x, xd)."""
    w02 = gamma ** 2 + wd ** 2
    sc2 = 4.0 * gamma * w02 * var
    G = np.array([[0.0, 1.0], [-w02, -2.0 * gamma]])
    LLt = np.array([[0.0, 0.0], [0.0, sc2]])
    read = np.array([1.0, 0.0])
    return G, LLt, read


def rw(sig):
    """Random-walk level.  State (x)."""
    return np.array([[0.0]]), np.array([[sig ** 2]]), np.array([1.0])


def blocks(*systems):
    """Block-diagonal combination; the observation reads the sum."""
    Gs, Ls, rs = zip(*systems)
    d = sum(g.shape[0] for g in Gs)
    G = np.zeros((d, d)); LLt = np.zeros((d, d)); read = np.zeros(d)
    i = 0
    for g, l, r in zip(Gs, Ls, rs):
        k = g.shape[0]
        G[i:i + k, i:i + k] = g; LLt[i:i + k, i:i + k] = l
        read[i:i + k] = r; i += k
    return G, LLt, read


def disc(G, LLt, h):
    """Exact discretisation (A, Q) over step h, Van Loan."""
    d = G.shape[0]
    C = np.zeros((2 * d, 2 * d))
    C[:d, :d] = -G; C[:d, d:] = LLt; C[d:, d:] = G.T
    E = expm(C * h)
    A = E[d:, d:].T
    Q = A @ E[:d, d:]
    return A, 0.5 * (Q + Q.T)


# ---------------------------------------------------------------- bridge

def bridge(G, LLt, s_after):
    """Reading a linear functional at time s_after inside a unit interval with
    endpoint states a (left) and b (right): coefficients (Ca, Cb) and the
    conditional covariance Vb of the mid state given both endpoints."""
    A_s, Q_s = disc(G, LLt, s_after)
    A_r, _ = disc(G, LLt, 1.0 - s_after)
    A_1, Q_1 = disc(G, LLt, 1.0)
    X = np.linalg.solve(Q_1.T, (Q_s @ A_r.T).T).T      # Q_s A_r' Q_1^{-1}
    Ca = A_s - X @ A_1
    Cb = X
    Vb = Q_s - X @ A_r @ Q_s
    return Ca, Cb, 0.5 * (Vb + Vb.T)


def delay_row(G, LLt, f, tau, K, d):
    """Augmented-state observation row for reading functional f at offset tau
    (0 <= tau <= K).  State blocks are (z_t, z_{t-1}, ..., z_{t-K}).
    Returns (row of length d*(K+1), bridge variance)."""
    k = int(np.floor(tau + 1e-12))
    s = tau - k
    row = np.zeros(d * (K + 1))
    if s < 1e-12:
        row[k * d:(k + 1) * d] = f
        return row, 0.0
    Ca, Cb, Vb = bridge(G, LLt, 1.0 - s)               # time after LEFT endpoint
    row[(k + 1) * d:(k + 2) * d] = f @ Ca              # left endpoint z_{t-k-1}
    row[k * d:(k + 1) * d] = f @ Cb                    # right endpoint z_{t-k}
    return row, max(float(f @ Vb @ f), 0.0)


# ---------------------------------------------------------------- filter

def aug_model(G, LLt, K, diffuse):
    """Discrete unit-step model on the augmented state, with exact stationary
    (or diffuse, forward-consistent) initial covariance."""
    d = G.shape[0]
    A1, Q1 = disc(G, LLt, 1.0)
    D = d * (K + 1)
    F = np.zeros((D, D)); F[:d, :d] = A1
    for i in range(1, K + 1):
        F[i * d:(i + 1) * d, (i - 1) * d:i * d] = np.eye(d)
    Q = np.zeros((D, D)); Q[:d, :d] = Q1
    if diffuse:
        P_start = 1e4 * np.eye(d)
    else:
        P_start = solve_continuous_lyapunov(G, -LLt)
    C = np.zeros((D, D))
    C[K * d:, K * d:] = P_start
    for i in range(K - 1, -1, -1):                     # forward recursion
        for j in range(K, i, -1):
            C[i * d:(i + 1) * d, j * d:(j + 1) * d] = \
                A1 @ C[(i + 1) * d:(i + 2) * d, j * d:(j + 1) * d]
            C[j * d:(j + 1) * d, i * d:(i + 1) * d] = \
                C[i * d:(i + 1) * d, j * d:(j + 1) * d].T
        C[i * d:(i + 1) * d, i * d:(i + 1) * d] = \
            A1 @ C[(i + 1) * d:(i + 2) * d, (i + 1) * d:(i + 2) * d] @ A1.T + Q1
    return F, Q, 0.5 * (C + C.T)


def loglik_batch(y1, y2, F, Q, C0, h1, r1, H2, R2):
    """Prequential log-likelihood of (y1, y2) under B hypotheses that differ
    only in the second observation row H2 (B,D) and floor R2 (B,).
    Vectorised over hypotheses; first BURN points excluded."""
    B, D = H2.shape
    n = len(y1)
    m = np.zeros((B, D))
    P = np.broadcast_to(C0, (B, D, D)).copy()
    ll = np.zeros(B)
    for t in range(n):
        m = m @ F.T
        P = np.einsum('ij,bjk,lk->bil', F, P, F) + Q
        for h, r, y in ((np.broadcast_to(h1, (B, D)), np.full(B, r1), y1[t]),
                        (H2, R2, y2[t])):
            Ph = np.einsum('bij,bj->bi', P, h)
            S = np.einsum('bi,bi->b', h, Ph) + r
            e = y - np.einsum('bi,bi->b', h, m)
            if t >= BURN:
                ll += -0.5 * (np.log(2 * np.pi * S) + e * e / S)
            Kg = Ph / S[:, None]
            m = m + Kg * e[:, None]
            P = P - np.einsum('bi,bj->bij', Kg, Ph)
            P = 0.5 * (P + np.swapaxes(P, 1, 2))
    return ll


# ---------------------------------------------------------------- simulate

def simulate(G, LLt, read, n, tau_fine, sig1, sig2, K, seed, f2=None):
    """Fine-grid exact simulation; unit-step observations.  y2 reads functional
    f2 (default: `read`) at offset tau_fine/FINE.  Returns y1, y2."""
    rng = np.random.default_rng(seed)
    d = G.shape[0]
    Af, Qf = disc(G, LLt, 1.0 / FINE)
    w, U = np.linalg.eigh(Qf)
    Lf = U @ np.diag(np.sqrt(np.clip(w, 0, None)))
    n_fine = (n + K + 3) * FINE
    Z = np.zeros((n_fine, d))
    z = np.zeros(d)
    for t in range(n_fine):
        z = Af @ z + Lf @ rng.standard_normal(d)
        Z[t] = z
    f2 = read if f2 is None else f2
    idx = (np.arange(n) + K + 2) * FINE
    y1 = Z[idx] @ read + sig1 * rng.standard_normal(n)
    y2 = Z[idx - tau_fine] @ f2 + sig2 * rng.standard_normal(n)
    return y1, y2


# ---------------------------------------------------------------- (a)

def probe_a():
    G, LLt, read = blocks(osc(0.05, 2.0), rw(0.5))
    rng = np.random.default_rng(1)
    errs = []
    for _ in range(20):
        z0 = rng.standard_normal(3)
        for tau in (0.3, 0.77, 1.5, 2.94):
            t = 5.0
            zt = expm(G * t) @ z0
            direct = read @ expm(G * (t - tau)) @ z0
            via_row = read @ expm(-tau * G) @ zt
            errs.append(abs(direct - via_row))
    return float(np.max(errs))


# ---------------------------------------------------------------- (b)

def probe_b():
    out = {}
    curves = {}
    for name, sys, m_expected in (("osc", osc(0.05, 2.0), 2), ("rw", rw(0.5), 1)):
        G, LLt, read = sys
        d = G.shape[0]
        taus = np.linspace(0.02, 1.98, 60)
        Vb_th = np.array([delay_row(G, LLt, read, t, 2, d)[1] for t in taus])
        # Monte Carlo: residual variance of the bridge row, stationary paths
        A1, Q1 = disc(G, LLt, 1.0)
        rng = np.random.default_rng(7)
        n_rep = 4000
        emp = []
        for tau in taus[::6]:
            k = int(np.floor(tau)); s = tau - k
            Ca, Cb, _ = bridge(G, LLt, 1.0 - s)
            As, Qs = disc(G, LLt, 1.0 - s)
            Ar, Qr = disc(G, LLt, s)
            if name == "osc":
                P0 = solve_continuous_lyapunov(G, -LLt)
            else:
                P0 = np.array([[100.0]])
            La = np.linalg.cholesky(P0 + 1e-12 * np.eye(d))
            res = np.zeros(n_rep)
            for i in range(n_rep):
                a = La @ rng.standard_normal(d)
                mid = As @ a + np.linalg.cholesky(Qs + 1e-14 * np.eye(d)) \
                    @ rng.standard_normal(d)
                b = Ar @ mid + np.linalg.cholesky(Qr + 1e-14 * np.eye(d)) \
                    @ rng.standard_normal(d)
                res[i] = read @ mid - read @ (Ca @ a + Cb @ b)
            emp.append(float(np.var(res)))
        # endpoint exponent: slope of log Vb vs log s as s -> 0
        ss = np.array([0.02, 0.04, 0.08, 0.16])
        vv = np.array([delay_row(G, LLt, read, t, 2, d)[1] for t in ss])
        slope = np.polyfit(np.log(ss), np.log(vv), 1)[0]
        out[name] = {"endpoint_exponent": float(slope),
                     "expected": 2 * m_expected - 1,
                     "mc_ratio": [float(e / t) for e, t in
                                  zip(emp, Vb_th[::6])]}
        curves[name] = (taus, Vb_th, taus[::6], np.array(emp))
    return out, curves


# ---------------------------------------------------------------- (c)

def profile_tau(sys, tau_true, n, K, seed, taus, sig1=0.3, sig2=0.3,
                diffuse=False, c_grid=None):
    G, LLt, read = sys
    d = G.shape[0]
    y1, y2 = simulate(G, LLt, read, n, int(round(tau_true * FINE)),
                      sig1, sig2, K, seed)
    F, Q, C0 = aug_model(G, LLt, K, diffuse)
    D = d * (K + 1)
    h1 = np.zeros(D); h1[:d] = read
    rows, floors = [], []
    cs = np.array([1.0]) if c_grid is None else c_grid
    for tau in taus:
        row, vb = delay_row(G, LLt, read, tau, K, d)
        for c in cs:
            rows.append(c * row); floors.append(c * c * vb + sig2 ** 2)
    ll = loglik_batch(y1, y2, F, Q, C0, h1, sig1 ** 2,
                      np.array(rows), np.array(floors))
    ll = ll.reshape(len(taus), len(cs)).max(axis=1)     # profile over c
    return ll


def probe_c():
    taus = np.arange(0.0, 6.001, 0.05)
    n = 600
    res = {}
    # one oscillator, c known
    ll1 = profile_tau(osc(0.05, 2.0), 1.3, n, 6, 11, taus)
    # one oscillator, c profiled (does the amplitude channel carry the gap?)
    ll1c = profile_tau(osc(0.05, 2.0), 1.3, n, 6, 11, taus,
                       c_grid=np.geomspace(0.5, 2.0, 9))
    # random walk only
    ll_rw = profile_tau(rw(0.5), 1.3, n, 6, 12, taus, diffuse=True)
    # two oscillators
    ll2 = profile_tau(blocks(osc(0.05, 2.0), osc(0.08, 0.9)), 1.3, n, 6, 13,
                      taus)
    # one oscillator at low SNR: the comb the sweep predicts should be visible
    ll_lo = profile_tau(osc(0.05, 2.0), 1.3, n, 6, 11, taus, sig2=3.0)

    def peaks(ll):
        i = int(np.argmax(ll))
        # best local max further than half a comb spacing away
        far = np.abs(taus - taus[i]) > 0.8
        j = int(np.argmax(np.where(far, ll, -np.inf)))
        return taus[i], taus[j], float(ll[i] - ll[j])

    for name, ll in (("osc_c_known", ll1), ("osc_c_free", ll1c),
                     ("rw", ll_rw), ("two_osc", ll2), ("osc_low_snr", ll_lo)):
        t_hat, t_alias, gap = peaks(ll)
        res[name] = {"tau_hat": float(t_hat), "second_peak": float(t_alias),
                     "gap_nats": gap, "gap_per_point": gap / (600 - BURN)}
    return res, taus, {"osc_c_known": ll1, "osc_c_free": ll1c,
                       "rw": ll_rw, "two_osc": ll2, "osc_low_snr": ll_lo}


def probe_c_sweep():
    """What sets the alias separation rate?  Two-node profiles (true tau vs its
    alias tau + 2*pi/omega_d) swept over damping and over measurement noise."""
    out = {"damping": [], "noise": []}
    for gamma in (0.0125, 0.025, 0.05, 0.1, 0.2):
        ll = profile_tau(osc(gamma, 2.0), 1.3, 600, 6, 11,
                         np.array([1.3, 1.3 + np.pi]))
        out["damping"].append({"gamma": gamma,
                               "gap_per_point": float((ll[0] - ll[1]) / 570)})
    for sig2 in (0.1, 0.3, 1.0, 3.0):
        ll = profile_tau(osc(0.05, 2.0), 1.3, 600, 6, 11,
                         np.array([1.3, 1.3 + np.pi]), sig2=sig2)
        out["noise"].append({"sig2": sig2,
                             "gap_per_point": float((ll[0] - ll[1]) / 570)})
    return out


# ---------------------------------------------------------------- (d)

def frac_power(G, mu):
    w, V = np.linalg.eig(G)
    return np.real(V @ np.diag(w.astype(complex) ** mu) @ np.linalg.inv(V))


def probe_d():
    n, K = 600, 3
    mus = np.arange(0.0, 1.501, 0.1)
    taus = np.arange(0.0, 3.001, 0.05)
    cs = np.geomspace(0.4, 2.5, 9)
    out = {}
    grids = {}
    for name, sys in (("one_pair", (osc(0.05, 2.0),)),
                      ("two_pair", (osc(0.05, 2.0), osc(0.08, 0.9)))):
        G, LLt, read = blocks(*sys)
        d = G.shape[0]
        y1, y2 = simulate(G, LLt, read, n, int(round(1.2 * FINE)),
                          0.3, 0.3, K, 21)
        F, Q, C0 = aug_model(G, LLt, K, False)
        D = d * (K + 1)
        h1 = np.zeros(D); h1[:d] = read
        rows, floors = [], []
        for mu in mus:
            # functional: mu-th derivative of the observed sum
            f = np.zeros(d); i = 0
            for g, _, r in sys:
                k = g.shape[0]
                f[i:i + k] = r @ frac_power(g, mu); i += k
            for tau in taus:
                row, vb = delay_row(G, LLt, f, tau, K, d)
                for c in cs:
                    rows.append(c * row)
                    floors.append(c * c * vb + 0.3 ** 2)
        ll = loglik_batch(y1, y2, F, Q, C0, h1, 0.3 ** 2,
                          np.array(rows), np.array(floors))
        ll = ll.reshape(len(mus), len(taus), len(cs)).max(axis=2)
        grids[name] = ll
        i, j = np.unravel_index(np.argmax(ll), ll.shape)
        # ridge: best tau for each mu, and the log-lik drop along it
        ridge_tau = taus[np.argmax(ll, axis=1)]
        ridge_ll = ll.max(axis=1)
        slope = np.polyfit(mus[:8], ridge_tau[:8], 1)[0]
        lam = np.angle(-0.05 + 2.0j) / 2.0
        out[name] = {
            "argmax": [float(mus[i]), float(taus[j])],
            "ridge_slope": float(slope), "predicted_slope": float(lam),
            "drop_along_ridge_at_mu1_nats":
                float(ridge_ll[np.argmin(np.abs(mus - 1.0))] - ll[i, j]),
        }
    return out, mus, taus, grids


# ---------------------------------------------------------------- run

if __name__ == "__main__":
    results = {}

    err_a = probe_a()
    results["a_max_abs_err"] = err_a
    print(f"(a) delay row exactness: max |err| = {err_a:.3e}")

    res_b, curves_b = probe_b()
    results["b"] = res_b
    print("(b)", json.dumps(res_b, indent=1))

    res_c, taus_c, lls_c = probe_c()
    results["c"] = res_c
    print("(c)", json.dumps(res_c, indent=1))

    res_cs = probe_c_sweep()
    results["c_sweep"] = res_cs
    print("(c sweep)", json.dumps(res_cs, indent=1))

    res_d, mus_d, taus_d, grids_d = probe_d()
    results["d"] = res_d
    print("(d)", json.dumps(res_d, indent=1))

    # ---- figures
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for ax, name, m in zip(axes, ("osc", "rw"), (2, 1)):
        taus, vth, tmc, vmc = curves_b[name]
        ax.plot(taus, vth, color=ts.SERIES[0], label="closed form $R_b$")
        ax.plot(tmc, vmc, "o", color=ts.SERIES[1], label="Monte Carlo")
        ax.set_title(f"{name}: bridge variance, exponent "
                     f"{res_b[name]['endpoint_exponent']:.2f} "
                     f"(expected {2 * m - 1})")
        ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$R_b(\tau)$")
        ax.legend(); ts.tidy(ax)
    ts.save(fig, "figures/fig30-delay-bridge.png")

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.4))
    for ax, (name, label, ylim) in zip(axes, (
            ("osc_c_known", "one oscillator", -80),
            ("osc_low_snr", r"same, $\sigma_2=3$: the comb", -8),
            ("rw", "random walk", -80),
            ("two_osc", "oscillator + oscillator", -80))):
        ll = lls_c[name]
        ax.plot(taus_c, ll - ll.max(), color=ts.SERIES[0])
        ax.axvline(1.3, color=ts.INK2, lw=0.8, ls="--")
        ax.set_ylim(ylim, 0.05 * abs(ylim))
        ax.set_title(f"{label}: gap {res_c[name]['gap_nats']:.1f} nats")
        ax.set_xlabel(r"$\tau$ hypothesis"); ax.set_ylabel("log-lik $-$ max")
        ts.tidy(ax)
    ts.save(fig, "figures/fig31-offset-aliasing.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, name in zip(axes, ("one_pair", "two_pair")):
        ll = grids_d[name]
        z = np.clip(ll - ll.max(), -60, 0)
        im = ax.pcolormesh(taus_d, mus_d, z, cmap="Blues", shading="auto")
        pred = 1.2 + mus_d * res_d[name]["predicted_slope"]
        ax.plot(pred, mus_d, color=ts.SERIES[1], ls="--",
                label=r"phase line, slope $\arg\lambda/\omega_d$")
        ax.plot(1.2, 0.0, "*", color=ts.SERIES[3], ms=12, label="truth")
        ax.set_title(f"{name}: ridge slope {res_d[name]['ridge_slope']:.3f} "
                     f"(pred {res_d[name]['predicted_slope']:.3f})")
        ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$\mu$ (derivative order)")
        ax.legend(loc="upper left"); fig.colorbar(im, ax=ax)
        ts.tidy(ax)
    ts.save(fig, "figures/fig32-derivative-lag-ridge.png")

    with open("figures/ode043.json", "w") as f:
        json.dump(results, f, indent=1)
    print("wrote figures/ode043.json")
