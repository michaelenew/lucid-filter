"""0008 -- The lag/lead covariance: the sum over roots becomes the integral.

The parent shipped the offset channel (odefilter/offset.py): two series, one
latent, a fractional time offset tau, all of it resting on the MODAL form --
`delay_row` reads x(t-s) through V diag(z_i^-sigma) V^-1 and `_gamma_modal`
extends the autocovariance to real lags by gamma(s) = Re sum_i b_i z_i^s.
Sums over distinct roots.  A fractional kernel has no such roots: the branch
point's truncation puts K artifact roots on a Jentzsch ring that MOVE WITH
THE BUDGET.  Section A records exactly how the modal machinery fails on
split-GL kernels -- including the worst failure mode, where whether
`delay_row` is even defined depends on K.

The replacement is this workstream's standing move (0001 s2): replace the
sum over roots with the integral over the channel density.  For the
stationary fractional part f in (0, 1/2), carrying the channel-density
continuation of the impulse response through the autocovariance sum gives

    gamma_f(s) = s2w * G(1-2f) G(s+f) / ( G(f) G(1-f) G(s+1-f) ),   s real,

i.e. THE CLASSICAL ARFIMA AUTOCOVARIANCE WITH THE INTEGER LAG CONTINUED TO
REAL s, and the two continuations -- Stieltjes (channel density) and Gamma
(analytic) -- coincide exactly.  Derivation in 0009; section B verifies the
identity three ways and checks positive definiteness on mixed real grids by
Cholesky (which is also what licenses sampling the continuation directly).

Everything the offset channel needs then falls out of ONE Schur complement
of gamma_f, with no eigendecomposition anywhere:

  read row     r(sigma) = gamma_vec(sigma)' Gamma_win^{-1}      (section C)
  bridge floor R_b(sigma) = gamma_f(0) - gamma_vec' Gamma_win^{-1} gamma_vec

exact pick-out row and R_b = 0 at integer sigma, a hump between.  The parent
needed the modal row PLUS a separate bridge argument (absorbed into s2_2 as
"the class gap ... until the class itself is continuous"); the continued
covariance supplies both at once, which is that gap closing from this side.

Section D runs the parent's `cross_anchor` construction with gamma_f as the
interpolant on pairs sampled EXACTLY from the continuation (Cholesky on the
joint integer+shifted grid), fractional tau, lead and lag sides, against the
parabola interpolant; and the nu = m + f case by differencing out the
integer part exactly (the analogue of cross_anchor's near-unit handling,
with m known rather than inspected).

Run:  python 0008_the_lag_lead_covariance.py        (~2 min)
"""
import sys
import math
import pathlib
import importlib

import numpy as np
from scipy.special import gammaln
from scipy.integrate import quad

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "lucid"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from odefilter.offset import delay_row, _gamma_modal  # noqa: E402

m2 = importlib.import_module("0002_is_nu_learnable")
m4 = importlib.import_module("0004_the_integer_part_must_be_exact")

FIG = pathlib.Path(__file__).resolve().parent / "figures"


# ------------------------------------------------- the continued covariance
def frac_gamma(f: float, s, s2w: float = 1.0):
    """Autocovariance of (1-B)^-f driven by white noise (variance s2w),
    continued from integer lags to real s.  Stationary range 0 < f < 1/2;
    f -> 0 degenerates to white noise correctly (gamma(s>0) -> 0)."""
    if not 0.0 <= f < 0.5:
        raise ValueError("stationary fractional part needs f in [0, 1/2)")
    s = np.abs(np.asarray(s, dtype=float))
    if f == 0.0:
        return s2w * (s == 0.0).astype(float)
    out = np.exp(gammaln(1.0 - 2.0 * f) + gammaln(s + f)
                 - gammaln(f) - gammaln(1.0 - f) - gammaln(s + 1.0 - f))
    return s2w * out


def read_row(f: float, sigma: float, lags: np.ndarray, s2w: float = 1.0):
    """(row, R_b): the GLS read of x(t - sigma) from the stored values at
    integer offsets `lags`, and its bridge variance.  One Schur complement of
    the continued covariance; no eigendecomposition, no branch choice."""
    lags = np.asarray(lags, dtype=float)
    G = frac_gamma(f, lags[:, None] - lags[None, :], s2w)
    g = frac_gamma(f, sigma - lags, s2w)
    row = np.linalg.solve(G, g)
    Rb = float(frac_gamma(f, 0.0, s2w) - row @ g)
    return row, Rb


def sample_joint(f: float, times: np.ndarray, s2w: float, rng):
    """Exact sample of the continued process at arbitrary real times."""
    C = frac_gamma(f, times[:, None] - times[None, :], s2w)
    L = np.linalg.cholesky(C + 1e-10 * C[0, 0] * np.eye(len(times)))
    return L @ rng.normal(size=len(times))


def frac_anchor(y1, y2, f: float, window=(-4.0, 4.0), shape="gamma"):
    """The parent's cross_anchor with gamma_f as the interpolant (or a
    parabola through the empirical peak, as the control)."""
    n = min(len(y1), len(y2))
    lo, hi = int(math.floor(window[0])), int(math.ceil(window[1]))
    ks, chat = [], []
    for k in range(lo, hi + 1):
        a = y2[max(k, 0):n + min(k, 0)]
        b = y1[max(-k, 0):n - max(k, 0)]
        mm = min(len(a), len(b))
        if mm < 30:
            continue
        ks.append(k)
        chat.append(float(np.mean(a[:mm] * b[:mm])))
    ks = np.asarray(ks, dtype=float)
    chat = np.asarray(chat)
    if shape == "parabola":
        i = int(np.argmax(np.abs(chat)))
        i = min(max(i, 1), len(ks) - 2)
        y0, y1_, y2_ = chat[i - 1], chat[i], chat[i + 1]
        den = y0 - 2 * y1_ + y2_
        return float(ks[i] + (0.5 * (y0 - y2_) / den if den != 0 else 0.0))
    taus = np.arange(window[0], window[1] + 1e-9, 0.01)
    best, tau_hat = np.inf, float(taus[0])
    for tau in taus:
        g = frac_gamma(f, ks - tau)
        gg = float(g @ g)
        if gg <= 0.0:
            continue
        r = float(chat @ chat - (g @ chat) ** 2 / gg)
        if r < best:
            best, tau_hat = r, float(tau)
    return tau_hat


def main():
    rng = np.random.default_rng(0)

    # ---- A: the modal machinery on fractional kernels ----------------------
    print("== A: the parent's modal machinery on split-GL kernels ==")
    for nu, K in ((0.7, 25), (0.7, 10), (1.3, 25), (2.3, 25)):
        a = m4.gl_split(nu, K)
        for name, fn in (("delay_row", lambda: delay_row(a, 1.5, K + 2)),
                         ("gamma_modal", lambda: _gamma_modal(a, 1.0))):
            try:
                fn()
                verdict = "ok (on truncation-artifact roots that move with K)"
            except ValueError as e:
                verdict = f"FAILS: {str(e)[:58]}"
            print(f"  nu={nu:<4} K={K:<3} {name:<12} {verdict}")

    # ---- B: the continuation identity --------------------------------------
    print("\n== B: gamma_f -- three routes, one function ==")
    print(f"{'f':>5} {'s':>5} {'Gamma form':>12} {'Beta integral':>13}"
          f" {'brute sum':>12}")
    for f in (0.15, 0.3, 0.45):
        # brute: gamma(k) = s2w * sum_j h_j h_{j+k}, tail-corrected
        J = 400_000
        h = np.empty(J)
        h[0] = 1.0
        for j in range(1, J):
            h[j] = h[j - 1] * (j - 1 + f) / j
        for s in (0.0, 1.0, 2.5):
            g1 = float(frac_gamma(f, s))
            # the endpoint singularities go into quad's algebraic weight
            g2 = (math.sin(math.pi * f) / math.pi) * quad(
                lambda r: 1.0, 0.0, 1.0, weight="alg",
                wvar=(s + f - 1.0, -2.0 * f), limit=200)[0]
            if s == int(s):
                k = int(s)
                tail = (h[-1] ** 2) * (J ** 1.0) / (1.0 - 2 * f)  # ~int j^{2f-2}
                g3 = f"{float(h[:J - k] @ h[k:]) + tail:>12.6f}"
            else:
                g3 = f"{'--':>12}"
            print(f"{f:>5} {s:>5} {g1:>12.6f} {g2:>13.6f} {g3}")

    print("\npositive definiteness on mixed real grids (min eigenvalue):")
    for f in (0.15, 0.3, 0.45):
        t = np.sort(np.concatenate([np.arange(40.0),
                                    np.arange(40.0) + 0.37,
                                    rng.uniform(0, 40, 40)]))
        C = frac_gamma(f, t[:, None] - t[None, :])
        ev = float(np.linalg.eigvalsh(C).min())
        print(f"  f={f}: n={len(t)} grid, min eig = {ev:.3e}"
              f"  (gamma(0) = {float(frac_gamma(f, 0.0)):.3f})")

    # ---- C: read row and bridge from one Schur complement ------------------
    print("\n== C: the read row and the bridge ==")
    lags = np.arange(-20.0, 21.0)
    for f in (0.15, 0.3, 0.45):
        row, Rb = read_row(f, 3.0, lags)             # integer read
        pick = np.zeros(len(lags))
        pick[np.where(lags == 3.0)[0][0]] = 1.0
        print(f"  f={f}: integer sigma=3 -> |row - pickout| ="
              f" {float(np.max(np.abs(row - pick))):.2e},  R_b = {Rb:.2e}")
    # The endpoint exponent.  gamma'(0+) = -gamma(0) * pi * cot(pi f) is
    # finite and nonzero, so the continuation is kinked at s = 0 like an
    # OU process and the bridge must open linearly: the parent's 2m-1 law
    # at m = 1, for EVERY f.  The fractional part sets the covariance TAIL
    # (long memory), not the kink (local roughness); exponents above 1 come
    # only from the exact integrations in nu = m + f.
    print(f"  {'f':>5} {'R_b(0.5)':>10} {'slope s~0.05':>13} {'slope s~0.003':>14}")
    for f in (0.15, 0.3, 0.45):
        rb5 = read_row(f, 0.5, lags)[1]
        sl = []
        for lo, hi in ((0.01, 0.1), (0.001, 0.01)):
            ss = np.geomspace(lo, hi, 6)
            rb = np.array([read_row(f, s, lags)[1] for s in ss])
            sl.append(float(np.polyfit(np.log(ss), np.log(rb), 1)[0]))
        print(f"  {f:>5} {rb5:>10.4f} {sl[0]:>13.3f} {sl[1]:>14.3f}")

    # ---- D: the anchor on exactly-sampled fractional pairs -----------------
    print("\n== D: the fractional cross anchor ==")
    print("D1: stationary f = 0.3, exact continuation pairs, c = 1,"
          " s2_meas = 0.25 * gamma(0), 4 seeds")
    n = 1200
    f = 0.3
    for tau0 in (1.3, -0.7):
        errs_g, errs_p = [], []
        for sd in range(4):
            r = np.random.default_rng(sd)
            times = np.concatenate([np.arange(n, dtype=float),
                                    np.arange(n, dtype=float) - tau0])
            x = sample_joint(f, times, 1.0, r)
            sm = math.sqrt(0.25 * float(frac_gamma(f, 0.0)))
            y1 = x[:n] + r.normal(0, sm, n)
            y2 = x[n:] + r.normal(0, sm, n)
            errs_g.append(frac_anchor(y1, y2, f) - tau0)
            errs_p.append(frac_anchor(y1, y2, f, shape="parabola") - tau0)
        print(f"  tau={tau0:>5}: gamma-interp err "
              f"{np.mean(np.abs(errs_g)):.3f} (RMS {np.std(errs_g):.3f})   "
              f"parabola err {np.mean(np.abs(errs_p)):.3f}")

    print("D2: nu = 1.3 (m=1, f=0.3), type-II data, integer tau = 2,"
          " difference out the integer part exactly")
    for sd in range(3):
        y, s2m = m2.simulate(1.3, 2000, 0.5, sd)
        r = np.random.default_rng(100 + sd)
        x = None  # y IS x + noise; build y2 as a delayed noisy read of y's latent
        # regenerate the latent to read it delayed (same seed path as simulate)
        rng2 = np.random.default_rng(sd)
        c = m2.gl_alpha(1.3, 2000)
        xl = np.zeros(2000)
        w = rng2.normal(0.0, 1.0, 2000)
        xl[0] = w[0]
        for t in range(1, 2000):
            xl[t] = c[:t] @ xl[t - 1::-1] + w[t]
        y2 = np.roll(xl, 2) + r.normal(0, math.sqrt(s2m), 2000)
        y2[:2] = np.nan
        d1 = np.diff(y)
        d2 = np.diff(y2)
        ok = np.isfinite(d1) & np.isfinite(d2)
        tau_hat = frac_anchor(d1[ok], d2[ok], 0.3)
        print(f"  seed {sd}: tau_hat = {tau_hat:.2f}  (truth 2.0)")

    # ---------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ss = np.linspace(0.0, 6.0, 301)
    for f, col in ((0.15, "C0"), (0.3, "C1"), (0.45, "C2")):
        g = frac_gamma(f, ss) / float(frac_gamma(f, 0.0))
        ax[0].plot(ss, g, col, label=f"$f={f}$")
        kk = np.arange(0, 7)
        ax[0].plot(kk, frac_gamma(f, kk) / float(frac_gamma(f, 0.0)),
                   col + "o", ms=4)
    ax[0].set_xlabel(r"real lag $s$")
    ax[0].set_ylabel(r"$\gamma_f(s)/\gamma_f(0)$")
    ax[0].set_title("the continued covariance (dots: the integer lags)")
    ax[0].legend(fontsize=8)
    lags = np.arange(-20.0, 21.0)
    sg = np.linspace(0.0, 3.0, 151)
    for f, col in ((0.15, "C0"), (0.3, "C1"), (0.45, "C2")):
        rb = np.array([read_row(f, s, lags)[1] for s in sg])
        ax[1].plot(sg, rb / float(frac_gamma(f, 0.0)), col, label=f"$f={f}$")
    ax[1].set_xlabel(r"reading point $\sigma$")
    ax[1].set_ylabel(r"$R_b(\sigma)/\gamma_f(0)$")
    ax[1].set_title("the bridge: zero at integers, humps between")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig03-lag-lead-covariance.png", dpi=130)
    print(f"\nfigure -> {FIG / 'fig03-lag-lead-covariance.png'}")


if __name__ == "__main__":
    main()
