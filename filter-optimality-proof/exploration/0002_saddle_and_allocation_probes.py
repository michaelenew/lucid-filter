"""Three numerical probes for the claims in 0001.

A. Is the Gaussian the *worst* noise shape at fixed variance?  (i.e. is it the
   least-favourable shape, so that a filter built for it is minimax?)
B. Does the exact three-way linear allocation identity survive outside the
   Gaussian: is E[X | X+Y] linear for independent symmetric alpha-stable X, Y,
   with slope c_X^alpha / (c_X^alpha + c_Y^alpha)?
C. At fixed TOTAL variance, does log-scale mixing make estimation easier?
   If yes, a max-entropy-at-fixed-total-variance argument sets s = 0, and the
   filter's s > 0 cannot come from that argument.

All MMSEs are computed exactly (up to a 1-D outer quadrature), not by
simulation and not by binning:  conditional on a noise atom v_j, theta = x - v_j
deterministically, so the posterior is a finite mixture in closed form.

Run: python3 0002_saddle_and_allocation_probes.py
"""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss

rng = np.random.default_rng(20260729)
SQ2PI = np.sqrt(2.0 * np.pi)


# ---------------------------------------------------------------- exact MMSE
def mmse_atomic(P, v, w, half_width=14.0, ngrid=200_001):
    """E[(theta - E[theta|x])^2] for theta ~ N(0,P), noise atomic at (v, w).

    Given the atom v_j, theta = x - v_j exactly, so
        p(x)       = sum_j w_j phi(x - v_j; 0, P)
        E[theta|x] = sum_j w_j phi(x - v_j) (x - v_j) / p(x)
    and the posterior is degenerate within each atom.  Exact apart from the
    outer trapezoid over x.
    """
    sd = np.sqrt(P + float(np.sum(w * v * v)))
    x = np.linspace(-half_width * sd, half_width * sd, ngrid)
    m0 = np.zeros_like(x)
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    for vj, wj in zip(v, w):
        r = x - vj
        d = wj * np.exp(-0.5 * r * r / P) / (np.sqrt(P) * SQ2PI)
        m0 += d
        m1 += d * r
        m2 += d * r * r
    ok = m0 > 1e-300
    integrand = np.zeros_like(x)
    integrand[ok] = m2[ok] - m1[ok] ** 2 / m0[ok]
    return float(np.trapezoid(integrand, x))


def mmse_scale_mixture(P, R, s, n_lam=81, half_width=14.0, ngrid=200_001):
    """theta ~ N(0,P); v = sqrt(R e^lam) z, lam ~ N(-s^2/2, s^2) so Var(v) = R.

    Conditional on lam the pair is jointly Gaussian, so everything is closed
    form and only the outer integral over x is numerical.
    """
    if s <= 0:
        return P * R / (P + R)
    z, wq = hermegauss(n_lam)
    wq = wq / wq.sum()
    lam = -0.5 * s * s + s * z
    S = P + R * np.exp(lam)                      # predictive variance per node
    K = P / S                                    # gain per node
    V = P * R * np.exp(lam) / S                  # posterior variance per node
    sd = np.sqrt(P + R)
    x = np.linspace(-half_width * sd, half_width * sd, ngrid)
    d = (wq[:, None] / (np.sqrt(S)[:, None] * SQ2PI)
         * np.exp(-0.5 * x[None, :] ** 2 / S[:, None]))
    m0 = d.sum(0)
    m1 = (d * (K[:, None] * x[None, :])).sum(0)
    m2 = (d * (V[:, None] + (K[:, None] * x[None, :]) ** 2)).sum(0)
    ok = m0 > 1e-300
    integrand = np.zeros_like(x)
    integrand[ok] = m2[ok] - m1[ok] ** 2 / m0[ok]
    return float(np.trapezoid(integrand, x))


# ---------------------------------------------------------------- probe A
def probe_A():
    print("=" * 74)
    print("A. Gaussian as least-favourable SHAPE at fixed variance")
    print("   theta ~ N(0,P);  v mean-zero with variance R, shape varying.")
    print("   LMMSE = PR/(P+R) is attained by the linear filter for EVERY shape.")
    print("   MMSE <= LMMSE always, with equality iff jointly Gaussian.")
    P, R = 1.0, 1.0
    lmmse = P * R / (P + R)

    shapes = {}
    shapes["two-point +-sqrt(R)"] = (np.array([-1.0, 1.0]) * np.sqrt(R),
                                     np.array([0.5, 0.5]))
    u = np.linspace(-np.sqrt(3 * R), np.sqrt(3 * R), 2001)
    shapes["uniform (light tail)"] = (u, np.full(u.size, 1 / u.size))
    from scipy.stats import t as tdist
    df = 5.0
    q = (np.arange(2001) + 0.5) / 2001
    tv = tdist.ppf(q, df) * np.sqrt(R * (df - 2) / df)
    shapes["student-t5 (heavy tail)"] = (tv, np.full(tv.size, 1 / tv.size))
    zg, wg = hermegauss(201)
    shapes["gaussian (control)"] = (np.sqrt(R) * zg, wg / wg.sum())

    print(f"   LMMSE = {lmmse:.6f}")
    for name, (vn, vw) in shapes.items():
        var = float(np.sum(vw * vn ** 2))
        m = mmse_atomic(P, vn, vw)
        print(f"   {name:26s} var={var:.4f}  MMSE={m:.6f}  MMSE/LMMSE={m/lmmse:.4f}")
    m = mmse_scale_mixture(P, R, 1.0)
    print(f"   {'log-scale mixture s=1':26s} var={R:.4f}  MMSE={m:.6f}"
          f"  MMSE/LMMSE={m/lmmse:.4f}")
    print("   EXPECT: gaussian control ~= 1.0000, every other shape < 1.")


# ---------------------------------------------------------------- probe C
def probe_C():
    print("=" * 74)
    print("C. At fixed TOTAL variance R, does log-scale mixing lower MMSE?")
    print("   If yes, max-entropy AT FIXED TOTAL VARIANCE would pick s = 0,")
    print("   so the filter's s > 0 must be justified some other way.")
    P, R = 1.0, 1.0
    lmmse = P * R / (P + R)
    for s in (0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        m = mmse_scale_mixture(P, R, s)
        print(f"   s={s:.2f}  MMSE={m:.6f}  MMSE/LMMSE={m/lmmse:.4f}")
    print("   EXPECT: decreasing in s  ->  s = 0 is the hardest case.")


# ---------------------------------------------------------------- probe B
def rstable(alpha, scale, size, rng):
    """Chambers-Mallows-Stuck, symmetric alpha-stable (beta=0), scale c."""
    U = rng.uniform(-np.pi / 2, np.pi / 2, size)
    W = rng.exponential(1.0, size)
    if abs(alpha - 1.0) < 1e-9:
        return scale * np.tan(U)
    return scale * (np.sin(alpha * U) / np.cos(U) ** (1 / alpha)
                    * (np.cos(U - alpha * U) / W) ** ((1 - alpha) / alpha))


def probe_B(N=40_000_000):
    print("=" * 74)
    print("B. Is E[X | X+Y] linear for independent symmetric alpha-stable?")
    print("   Conjecture: slope = c_X^alpha / (c_X^alpha + c_Y^alpha),")
    print("   i.e. the Gaussian allocation identity with c^alpha for variance.")
    for alpha in (2.0, 1.8, 1.5, 1.2):
        cX, cY = 1.0, 2.0
        X = rstable(alpha, cX, N, rng)
        Y = rstable(alpha, cY, N, rng)
        Z = X + Y
        pred = cX ** alpha / (cX ** alpha + cY ** alpha)
        qs = np.quantile(Z, [0.20, 0.35, 0.50, 0.65, 0.80])
        edges = np.concatenate([[-np.inf], qs, [np.inf]])
        idx = np.searchsorted(edges, Z) - 1
        rows = []
        for b in range(len(edges) - 1):
            m = idx == b
            if m.sum() < 1000:
                continue
            zb, xb = Z[m].mean(), X[m].mean()
            if abs(zb) > 0.5:
                rows.append(xb / zb)
        print(f"   alpha={alpha:.1f}  predicted={pred:.4f}  binwise E[X|z]/z = "
              + ", ".join(f"{s:.4f}" for s in rows))
    print("   EXPECT if true: binwise ratios ~= predicted AND constant across")
    print("   bins (constancy is the linearity; the level is the slope).")


if __name__ == "__main__":
    probe_A()
    probe_C()
    probe_B()
