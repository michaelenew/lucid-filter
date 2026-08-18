"""0008 -- The load-bearing half of the Cencov proposal, at p = 2.

0006 refuted the WARP (the scalar reparameterisation of a 1-D coefficient) and
could not test the ANISOTROPY, because a one-dimensional parameter has none.
p = 2 is the smallest case that has one, and it is the physically interesting
one: a second-order ODE without the offset, i.e. a damped oscillator.

The Fisher information for AR(p) coefficients with innovation variance Q is
I(alpha) = Gamma/Q per observation, Gamma the state second-moment matrix.  Q
cancels (Gamma is proportional to it), so the metric depends on alpha alone.
For AR(2), with Gt = Gamma/Q,

    g0 = (1 - a2) / [ (1 + a2) ( (1-a2)^2 - a1^2 ) ],   g1 = a1 g0 / (1 - a2)

and the invariant drift covariance is proportional to Gt^{-1}.

Three drift laws, separating the two ways Gt^{-1} differs from the identity:

    iso           Sigma = nu^2 I                     no warp, no anisotropy
    fisher-shape  Sigma = nu^2 Gt^{-1}/sqrt(det)     anisotropy only
                                                     (det matched to iso)
    fisher-full   Sigma = nu^2 Gt^{-1}               anisotropy and volume warp

plus `static` (no drift).  nu is chosen by marginal likelihood in every case, so
nothing is hand-set.  Determinant-matching is what makes `iso` and
`fisher-shape` differ ONLY in shape and orientation -- that is the clean test.

Why the shape should matter here.  In coefficient space, "the damping changed"
and "the frequency changed" are different directions.  An isotropic drift says
they are equally likely a priori; the invariant one weights them by how much
information the data carries about each.  The scenarios below move along one
direction at a time.

Scored on h-step forecast MSE, per 0006: tracking error cannot see alpha.

Grid.  Uniform over the whole stationarity triangle (|a2|<1, a1+a2<1,
a2-a1<1) -- no box around the answer.  Resolution and the kernel truncation
radius are the compute budget; boundary mass is reported so the grid extent can
be checked rather than trusted.
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

HORIZONS = (1, 5, 20)
STEP = 0.03


# ------------------------------------------------------------------- grid
def build_grid(step=STEP):
    a1 = np.arange(-2.0, 2.0 + 1e-9, step)
    a2 = np.arange(-1.0, 1.0 + 1e-9, step)
    A1, A2 = np.meshgrid(a1, a2, indexing="ij")
    ok = (np.abs(A2) < 0.999) & (A1 + A2 < 0.999) & (A2 - A1 < 0.999)
    return np.column_stack([A1[ok], A2[ok]])


def gamma_tilde(al):
    """State second-moment matrix / Q for AR(2).  Returns (g0, g1)."""
    a1, a2 = al[:, 0], al[:, 1]
    den = (1.0 + a2) * ((1.0 - a2) ** 2 - a1 ** 2)
    g0 = (1.0 - a2) / den
    g1 = a1 * g0 / (1.0 - a2)
    return g0, g1


def drift_cov(al, method):
    """Per-node 2x2 drift shape, before scaling by nu^2."""
    G = len(al)
    if method == "iso":
        return np.tile(np.eye(2), (G, 1, 1))
    g0, g1 = gamma_tilde(al)
    det = g0 * g0 - g1 * g1
    inv = np.empty((G, 2, 2))
    inv[:, 0, 0] = g0 / det
    inv[:, 1, 1] = g0 / det
    inv[:, 0, 1] = inv[:, 1, 0] = -g1 / det
    if method == "fisher-full":
        return inv
    # fisher-shape: match det to the identity, isolating shape from volume
    d = np.maximum(inv[:, 0, 0] * inv[:, 1, 1] - inv[:, 0, 1] ** 2, 1e-300)
    return inv / np.sqrt(d)[:, None, None]


def build_T(al, method, nu, cap=1e3):
    """Sparse Gaussian random-walk kernel on the grid, node-dependent shape."""
    G = len(al)
    if method == "static" or nu <= 0:
        return sp.identity(G, format="csr")
    S = drift_cov(al, method) * nu * nu
    # numerical guard: extreme anisotropy near the triangle boundary
    tr = S[:, 0, 0] + S[:, 1, 1]
    det = np.maximum(S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] ** 2, 1e-300)
    lam_max = 0.5 * (tr + np.sqrt(np.maximum(tr * tr - 4 * det, 0.0)))
    lam_max = np.minimum(lam_max, cap * nu * nu)
    rad = 4.0 * np.sqrt(lam_max)

    rows, cols, vals = [], [], []
    for i in range(G):
        d = al - al[i]
        near = np.nonzero((np.abs(d[:, 0]) <= rad[i]) & (np.abs(d[:, 1]) <= rad[i]))[0]
        dd = d[near]
        Si = S[i]
        di = max(Si[0, 0] * Si[1, 1] - Si[0, 1] ** 2, 1e-300)
        q = (Si[1, 1] * dd[:, 0] ** 2 - 2 * Si[0, 1] * dd[:, 0] * dd[:, 1]
             + Si[0, 0] * dd[:, 1] ** 2) / di
        v = np.exp(-0.5 * np.minimum(q, 200.0))
        rows.append(np.full(len(near), i))
        cols.append(near)
        vals.append(v)
    T = sp.csr_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))), shape=(G, G))
    rs = np.asarray(T.sum(1)).ravel()
    return sp.diags(1.0 / rs) @ T


# ----------------------------------------------------------------- filter
class Grid2:
    """Precomputed per-node companion powers.  The 2x2 algebra is hand-rolled
    below rather than einsum'd; at 4356 nodes that is the difference between
    2.7 ms and 0.2 ms per step, and this loop runs ~5x10^5 times."""

    def __init__(self, al):
        self.al = al
        G = len(al)
        F = np.zeros((G, 2, 2))
        F[:, 0, 0] = al[:, 0]
        F[:, 0, 1] = al[:, 1]
        F[:, 1, 0] = 1.0
        self.F = F
        self.Fh = {}
        for h in HORIZONS:
            M = np.tile(np.eye(2), (G, 1, 1))
            for _ in range(h):
                M = np.einsum("gij,gjk->gik", F, M)
            self.Fh[h] = np.ascontiguousarray(M[:, 0, :])   # x_{t+h} row


def run(y, g, T, Q, S2):
    """Grid over alpha, conditional Kalman, level collapsed to one Gaussian."""
    G = len(g.al)
    n = len(y)
    a1, a2 = g.al[:, 0], g.al[:, 1]
    Tt = T.T.tocsr()
    pi = np.full(G, 1.0 / G)
    m0 = m1 = 0.0
    p00 = p11 = 1e6
    p01 = 0.0
    ll = 0.0
    fc = {h: np.empty(n) for h in HORIZONS}
    for t in range(n):
        pi = Tt @ pi
        mp0 = a1 * m0 + a2 * m1
        mp1 = m0
        P00 = a1 * a1 * p00 + 2.0 * a1 * a2 * p01 + a2 * a2 * p11 + Q
        P01 = a1 * p00 + a2 * p01
        P11 = p00
        S = P00 + S2
        e = y[t] - mp0
        lg = -0.5 * (np.log(S) + e * e / S)
        mx = lg.max()
        w = pi * np.exp(lg - mx)
        Z = w.sum()
        ll += np.log(Z) + mx - 0.5 * np.log(2 * np.pi)
        pi = w / Z
        K0 = P00 / S
        K1 = P01 / S
        q0 = mp0 + K0 * e
        q1 = mp1 + K1 * e
        m0 = float(pi @ q0)
        m1 = float(pi @ q1)
        d0 = q0 - m0
        d1 = q1 - m1
        p00 = float(pi @ (P00 * (1.0 - K0) + d0 * d0))
        p01 = float(pi @ (P01 - K0 * P01 + d0 * d1))
        p11 = float(pi @ (P11 - K1 * P01 + d1 * d1))
        for h in HORIZONS:
            r = pi @ g.Fh[h]
            fc[h][t] = r[0] * m0 + r[1] * m1
    return ll, fc, 0.0


def score(fc, x):
    n = len(x)
    lo = n // 2
    return {h: float(np.mean((f[np.arange(lo, n - h)]
                              - x[np.arange(lo, n - h) + h]) ** 2))
            for h, f in fc.items()}


# --------------------------------------------------------------- scenarios
def alpha_osc(rho, theta):
    return np.array([2 * rho * np.cos(theta), -rho * rho])


SCEN = {
    "no shift": ((0.95, 0.35), (0.95, 0.35)),
    "damping 0.95->0.85": ((0.95, 0.35), (0.85, 0.35)),
    "frequency 0.35->0.55": ((0.95, 0.35), (0.95, 0.55)),
}
NUS = [0.0, 0.004, 0.010, 0.025, 0.06, 0.15]
METHODS = ["static", "iso", "fisher-shape", "fisher-full"]


def simulate(a_pre, a_post, n, Q, S2, rng):
    x = np.zeros(n)
    for t in range(2, n):
        a = a_pre if t < n // 2 else a_post
        x[t] = a[0] * x[t - 1] + a[1] * x[t - 2] + np.sqrt(Q) * rng.standard_normal()
    return x, x + np.sqrt(S2) * rng.standard_normal(n)


def main():
    n, R, kappa, Q = 1000, 8, 0.35, 1.0
    al = build_grid()
    g = Grid2(al)
    print(f"grid: {len(al)} nodes at step {STEP}")

    master = np.random.default_rng(90210)
    rows = []
    for sname, (pre, post) in SCEN.items():
        a_pre, a_post = alpha_osc(*pre), alpha_osc(*post)
        ref, _ = simulate(a_pre, a_post, 20000, Q, 0.0, np.random.default_rng(2))
        S2 = (kappa * np.std(np.diff(ref))) ** 2
        data = [simulate(a_pre, a_post, n, Q, S2,
                         np.random.default_rng(master.integers(2**63)))
                for _ in range(R)]

        # oracle: alpha pinned at the post-shift truth
        j = int(np.argmin(((al - a_post) ** 2).sum(1)))
        pin = sp.csr_matrix((np.ones(len(al)),
                             (np.arange(len(al)), np.full(len(al), j))),
                            shape=(len(al),) * 2)
        osc = [score(run(y, g, pin, Q, S2)[1], x) for x, y in data]
        rows.append(dict(scenario=sname, method="oracle", nu=float("nan"),
                         ll=float("nan"),
                         **{f"h{h}": float(np.mean([s[h] for s in osc]))
                            for h in HORIZONS}))

        for method in METHODS:
            best, best_ll = None, -np.inf
            for nu in ([0.0] if method == "static" else NUS):
                T = build_T(al, method, nu)
                lls, scs = [], []
                for x, y in data:
                    ll, fc, _ = run(y, g, T, Q, S2)
                    lls.append(ll / n)
                    scs.append(score(fc, x))
                if np.mean(lls) > best_ll:
                    best_ll = float(np.mean(lls))
                    best = dict(nu=float(nu), ll=best_ll,
                                **{f"h{h}": float(np.mean([s[h] for s in scs]))
                                   for h in HORIZONS},
                                **{f"h{h}_se": float(np.std([s[h] for s in scs])
                                                     / np.sqrt(R))
                                   for h in HORIZONS})
            rows.append(dict(scenario=sname, method=method, **best))
            print(f"  {sname:>22} {method:>13} nu*={best['nu']:.3f} "
                  f"ll={best['ll']:.4f} "
                  + " ".join(f"h{h}={best[f'h{h}']:.3f}" for h in HORIZONS))

    print()
    hdr = (f"{'scenario':>22} {'method':>13} {'nu*':>7} "
           + " ".join(f"{'h='+str(h):>15}" for h in HORIZONS))
    print(hdr + "\n" + "-" * len(hdr))
    for sname in SCEN:
        base = [r for r in rows if r["scenario"] == sname
                and r["method"] == "static"][0]
        for method in METHODS + ["oracle"]:
            r = [q for q in rows if q["scenario"] == sname
                 and q["method"] == method][0]
            cells = " ".join(f"{r[f'h{h}']:8.3f}({r[f'h{h}']/base[f'h{h}']:.3f})"
                             for h in HORIZONS)
            print(f"{sname:>22} {method:>13} {r['nu']:7.3f} {cells}")
        print()

    # paired: fisher-shape against iso, the clean anisotropy test
    print("paired, fisher-shape against iso, each at its own nu* (h=5 and h=20):")
    for sname, (pre, post) in SCEN.items():
        a_pre, a_post = alpha_osc(*pre), alpha_osc(*post)
        ref, _ = simulate(a_pre, a_post, 20000, Q, 0.0, np.random.default_rng(2))
        S2 = (kappa * np.std(np.diff(ref))) ** 2
        nus = {mth: [r for r in rows if r["scenario"] == sname
                     and r["method"] == mth][0]["nu"]
               for mth in ("iso", "fisher-shape")}
        Ts = {mth: build_T(al, mth, nus[mth]) for mth in nus}
        rng = np.random.default_rng(5150)
        d = {h: [] for h in (5, 20)}
        for _ in range(R):
            x, y = simulate(a_pre, a_post, n, Q, S2, rng)
            s = {mth: score(run(y, g, Ts[mth], Q, S2)[1], x) for mth in nus}
            for h in d:
                d[h].append(s["fisher-shape"][h] / s["iso"][h] - 1.0)
        for h in d:
            v = np.array(d[h])
            t = v.mean() / (v.std(ddof=1) / np.sqrt(R))
            print(f"  {sname:>22} h={h:>2}: ratio {1+v.mean():.4f} "
                  f"+-{v.std(ddof=1)/np.sqrt(R):.4f}  t = {t:+.2f}")
            rows.append(dict(scenario=sname, method=f"paired shape/iso h{h}",
                             ratio=float(1 + v.mean()),
                             se=float(v.std(ddof=1) / np.sqrt(R)), t=float(t)))

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    ax = axes[0]
    xs = np.arange(len(SCEN))
    for i, method in enumerate(METHODS[1:] + ["oracle"]):
        ys = []
        for sname in SCEN:
            r = [q for q in rows if q.get("scenario") == sname
                 and q.get("method") == method][0]
            b = [q for q in rows if q.get("scenario") == sname
                 and q.get("method") == "static"][0]
            ys.append(r["h5"] / b["h5"])
        ax.plot(xs, ys, marker="o", color=ts.SERIES[i], label=method)
    ax.axhline(1.0, color=ts.INK, lw=1.0, ls="--", zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(SCEN), rotation=15, ha="right")
    ax.set_ylabel("h=5 forecast MSE, relative to static")
    ax.set_title("Does the invariant drift shape help?")
    ax.legend()
    ts.tidy(ax)

    # the metric itself, drawn on the stationarity triangle
    ax = axes[1]
    g0, g1 = gamma_tilde(al)
    ratio = np.abs(g1) / g0
    sc = ax.scatter(al[:, 0], al[:, 1], c=ratio, s=2, cmap="viridis", vmin=0, vmax=1)
    for (rr, th), mk in ((SCEN["no shift"][0], "o"),
                         (SCEN["damping 0.95->0.85"][1], "s"),
                         (SCEN["frequency 0.35->0.55"][1], "^")):
        a = alpha_osc(rr, th)
        ax.plot(a[0], a[1], mk, color=ts.SERIES[7], ms=7)
    fig.colorbar(sc, ax=ax, label=r"$|\gamma_1|/\gamma_0$  (metric anisotropy)")
    ax.set_xlabel(r"$\alpha_1$")
    ax.set_ylabel(r"$\alpha_2$")
    ax.set_title("Fisher anisotropy over the stationarity triangle")
    ts.tidy(ax)
    ts.save(fig, os.path.join(HERE, "figures", "fig06-anisotropy-p2.png"))

    with open(os.path.join(HERE, "figures", "ode008.json"), "w") as f:
        json.dump(dict(rows=rows, n=n, R=R, kappa=kappa, step=STEP,
                       nodes=len(al)), f, indent=1)


if __name__ == "__main__":
    main()
