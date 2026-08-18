"""0002 -- Is nu learnable, from both sides, with an error bar?

The integer audit (ode-filter/0030) found the order p learnable from
below and blind from above.  This probe asks whether the fractional order nu
has what p lacked: an interior likelihood maximum with curvature on BOTH
sides -- an estimate with an error bar rather than a floor.

Three measurements:

  A. Profile the marginal likelihood over nu on data of known fractional
     order (nu in {0.4, 0.7, 1.0, 1.3, 1.7}, two seeds each).  Report the
     argmax, the profile-curvature SE, and the likelihood drop 0.3 to either
     side of the truth (two-sidedness).
  B. nu = 1.0 doubles as the acid test: random-walk data must profile to
     nu ~ 1 -- the audit's recover-the-parent test, continuous edition.
  C. The truncation budget: max likelihood and hat-nu versus K.  A budget
     must converge monotonically with no interior optimum.

The likelihood is the PARENT'S: `_face_profile` from odefilter.core -- the
s = 0 face with sigma^2 concentrated out -- called with alpha pinned to the
truncated Gruenwald-Letnikov vector.  Nothing in the parent is modified or
copied; this is the inheritability principle of 0001 section 3 exercised, not
just asserted.  The face (homoscedastic noise) is the right instrument here
because the data is generated homoscedastic; the full grid would measure the
same thing at ~30x the cost.

Simulation is "type II" fractional integration: zero pre-history, the
recurrence run with the full growing kernel (no truncation in the generator,
so the model's truncation at K is a genuine approximation being tested).
Measurement noise is sigma = 0.5 * SD(diff x) per series (kappa = 0.5 in the
repo's units).

Run:  python 0002_is_nu_learnable.py        (~5 min)
"""
import sys
import time
import math
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "lucid"))
from odefilter.core import _face_profile  # noqa: E402

FIG = pathlib.Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)


# ----------------------------------------------------------------- the map
def gl_alpha(nu: float, K: int) -> np.ndarray:
    """Truncated GL coefficients: x_t = sum_k c_k x_{t-k} + w_t.

    c_1 = nu, c_k = c_{k-1} (k-1-nu)/k.  At integer nu the recurrence
    terminates: nu=1 -> (1,0,...), the parent; nu=2 -> (2,-1,0,...).
    """
    c = np.empty(K)
    c[0] = nu
    for k in range(2, K + 1):
        c[k - 1] = c[k - 2] * (k - 1 - nu) / k
    return c


def simulate(nu: float, n: int, kappa: float, seed: int):
    """Type-II fractional process (Q = 1) plus white measurement noise."""
    rng = np.random.default_rng(seed)
    c = gl_alpha(nu, n)
    x = np.zeros(n)
    w = rng.normal(0.0, 1.0, n)
    x[0] = w[0]
    for t in range(1, n):
        x[t] = c[:t] @ x[t - 1::-1] + w[t]
    sig = kappa * float(np.std(np.diff(x)))
    return x + rng.normal(0.0, sig, n), sig * sig


# ------------------------------------------------------- profile likelihood
def _ll_at(y, nu, K, lq):
    return _face_profile(y, gl_alpha(nu, K), math.exp(lq))[1]


def profile_point(y, nu, K, lq0=None):
    """max over log q of the face likelihood at this nu.  Returns (ll, lq*)."""
    from scipy.optimize import minimize_scalar
    if lq0 is None:
        grid = np.linspace(-16.0, 6.0, 12)
    else:
        grid = lq0 + np.linspace(-2.0, 2.0, 7)
    vals = [_ll_at(y, nu, K, g) for g in grid]
    g0 = float(grid[int(np.argmax(vals))])
    r = minimize_scalar(lambda lq: -_ll_at(y, nu, K, lq),
                        bounds=(g0 - 1.0, g0 + 1.0), method="bounded",
                        options=dict(xatol=1e-2, maxiter=25))
    if -r.fun >= max(vals):
        return float(-r.fun), float(r.x)
    return float(max(vals)), g0


def profile(y, nus, K):
    lls = np.empty(len(nus))
    lq = None
    for i, nu in enumerate(nus):
        lls[i], lq = profile_point(y, float(nu), K, lq)
    return lls


def summarise(nus, lls):
    """(nu_hat, SE, ok) by a local quadratic through the argmax +-2 points."""
    i = int(np.argmax(lls))
    if i == 0 or i == len(nus) - 1:
        return float(nus[i]), math.nan, False
    sl = slice(max(0, i - 2), min(len(nus), i + 3))
    a, b, c = np.polyfit(nus[sl], lls[sl], 2)
    if a >= 0:
        return float(nus[i]), math.nan, False
    return float(-b / (2 * a)), float(1.0 / math.sqrt(-2.0 * a)), True


# ------------------------------------------------------------------- run it
def main():
    n, kappa, K = 1500, 0.5, 25
    nus = np.round(np.arange(0.15, 2.40, 0.05), 4)
    truths = [0.4, 0.7, 1.0, 1.3, 1.7]
    seeds = [0, 1]

    print(f"n={n}  kappa={kappa}  K={K}  grid=[{nus[0]},{nus[-1]}] step 0.05")
    print("\n== A/B: the profile over nu ==")
    print(f"{'truth':>6} {'seed':>4} {'nu_hat':>7} {'SE':>6} "
          f"{'drop(-0.3)':>10} {'drop(+0.3)':>10}")

    curves = {}
    for nu0 in truths:
        for sd in seeds:
            y, _ = simulate(nu0, n, kappa, sd)
            t0 = time.time()
            lls = profile(y, nus, K)
            nu_hat, se, ok = summarise(nus, lls)
            # two-sidedness: total-loglik drop 0.3 to each side of the truth
            im = int(np.argmax(lls))
            iL = int(np.argmin(np.abs(nus - (nus[im] - 0.3))))
            iR = int(np.argmin(np.abs(nus - (nus[im] + 0.3))))
            dl = lls[im] - lls[iL]
            dr = lls[im] - lls[iR]
            curves[(nu0, sd)] = lls
            print(f"{nu0:>6} {sd:>4} {nu_hat:>7.3f} "
                  f"{se:>6.3f} {dl:>10.1f} {dr:>10.1f}"
                  f"   [{time.time()-t0:.0f}s]"
                  + ("" if ok else "  (edge/degenerate)"))

    print("\n== C: the truncation budget, truth nu=0.7 and 1.7, seed 0 ==")
    print(f"{'truth':>6} {'K':>4} {'max ll/n':>10} {'nu_hat':>7}")
    kcurv = {}
    for nu0 in (0.7, 1.7):
        y, _ = simulate(nu0, n, kappa, 0)
        for Kb in (5, 10, 20, 40):
            lls = profile(y, nus, Kb)
            nu_hat, _, _ = summarise(nus, lls)
            kcurv[(nu0, Kb)] = (max(lls) / n, nu_hat)
            print(f"{nu0:>6} {Kb:>4} {max(lls)/n:>10.4f} {nu_hat:>7.3f}")

    # ---------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap("viridis")
    for (nu0, sd), lls in curves.items():
        col = cmap((truths.index(nu0)) / (len(truths) - 1))
        ax[0].plot(nus, lls - lls.max(), color=col, alpha=0.9 if sd == 0 else 0.45,
                   label=f"$\\nu={nu0}$" if sd == 0 else None)
        ax[0].axvline(nu0, color=col, lw=0.6, ls=":")
    ax[0].set_ylim(-60, 2)
    ax[0].set_xlabel(r"$\nu$")
    ax[0].set_ylabel("profile loglik $-$ max (total nats)")
    ax[0].set_title("A: the profile is two-sided (dotted = truth)")
    ax[0].legend(fontsize=8)

    for nu0, mk in ((0.7, "o-"), (1.7, "s-")):
        Ks = [5, 10, 20, 40]
        ax[1].plot(Ks, [kcurv[(nu0, k)][0] for k in Ks], mk,
                   label=f"truth $\\nu={nu0}$")
    ax[1].set_xscale("log")
    ax[1].set_xticks([5, 10, 20, 40])
    ax[1].set_xticklabels([5, 10, 20, 40])
    ax[1].set_xlabel("truncation $K$ (lags kept)")
    ax[1].set_ylabel("max profile loglik / n")
    ax[1].set_title("C: $K$ is a budget iff this is monotone")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig01-nu-profiles.png", dpi=130)
    print(f"\nfigure -> {FIG / 'fig01-nu-profiles.png'}")


if __name__ == "__main__":
    main()
