"""0002 -- C2: is the identification ridge the mean-reversion direction?

The claim under test (SUMMARY C2).  `adaptive-grid` findings 13-16 measured the
class coordinates `(phi, s)` to be identified but SLOPPY -- a flat identification
ridge the bank marginalises.  The resolvable-regime reframing predicts WHICH
direction that is: per step the likelihood sees the log-scale's INCREMENT
variance `nu`; the mean-reversion rate `kappa` enters only through reversion
accumulated over many steps.  So the flat direction should be `kappa` at fixed
`nu`, and the least-committal member on it is `kappa = 0`.

Sloppiness is NOT coordinate-invariant -- any direction can be made to look flat
by reparameterising -- so the measurement here is deliberately coordinate-free in
its conclusion.  We take the smallest-eigenvalue eigenvector of the observed
information (a direction in MODEL space), and then ask what that one direction
DOES: how much does it move the per-step increment variance `nu`, and how much
does it move `kappa`?  The ratio `|dlog nu| / |dlog kappa|` along it is the
number, and it means the same thing in any chart.

Model (measurement channel only, the same scalar rig `optimality-proof` fits):

    theta_t = theta_{t-1} + w_t,    w_t ~ N(0, Q)              Q fixed, not walked
    x_t     = theta_t     + v_t,    v_t ~ N(0, sig2 e^{lam_t})
    lam_t   = phi lam_{t-1} + e_t,  e_t ~ N(0, nu),  nu = s^2 (1 - phi^2)

Observable log-likelihood by a deterministic grid filter on `lam` with a single
Gaussian collapse per step -- i.e. exactly the shipped filter's GPB1
approximation, whose cost `optimality-proof/0034` measured against an exact
particle filter as 0.006% at s=0.2 and 1.35% at s=0.55.  Deterministic in the
data, so central differences are clean.

Run: python3 0002_the_ridge_coordinates.py
"""
import math

import numpy as np

# --------------------------------------------------------------- the rig
Q_PROC = 1.0            # fixed process variance -- the measurement channel is the subject
SIG2 = 1.0              # base measurement variance
T = 8000                # steps per record
NSEED = 5
GRID_LO, GRID_HI, GRID_N = -9.0, 9.0, 241
FD_H = 0.08             # central-difference step, in log coordinates


def simulate(phi, s, seed):
    """One record from the truth.  Stationary lam start."""
    rng = np.random.default_rng(seed)
    nu = s * s * (1.0 - phi * phi)
    lam = rng.normal(0.0, s)
    th = 0.0
    xs = np.empty(T)
    for t in range(T):
        lam = phi * lam + rng.normal(0.0, math.sqrt(nu)) if nu > 0 else phi * lam
        th += rng.normal(0.0, math.sqrt(Q_PROC))
        xs[t] = th + rng.normal(0.0, math.sqrt(SIG2 * math.exp(lam)))
    return xs


# ------------------------------------------------- observable log-likelihood
_NODES = np.linspace(GRID_LO, GRID_HI, GRID_N)
_EXPN = np.exp(_NODES)


def _kernel(phi, nu):
    """AR(1) transition on the fixed lam grid, row-normalised."""
    mu = phi * _NODES[:, None]                     # (from, 1)
    d = _NODES[None, :] - mu                       # (from, to)
    k = np.exp(-0.5 * d * d / nu)
    return k / k.sum(axis=1, keepdims=True)


def loglik(xs, phi, s):
    """Prequential log-likelihood of the record under (phi, s).  GPB1 collapse."""
    nu = s * s * (1.0 - phi * phi)
    if nu <= 1e-12 or s <= 0.0:
        return -np.inf
    K = _kernel(phi, nu)
    w = np.exp(-0.5 * _NODES ** 2 / (s * s))       # stationary prior on the grid
    w /= w.sum()
    rv = SIG2 * _EXPN                              # per-node measurement variance
    m, P = 0.0, 1e4                                # diffuse state start
    total = 0.0
    for t in range(xs.size):
        P += Q_PROC
        w = w @ K
        S = P + rv
        innov = xs[t] - m
        ll = -0.5 * (np.log(2.0 * np.pi * S) + innov * innov / S)
        lw = np.log(w + 1e-300) + ll
        mx = lw.max()
        ew = np.exp(lw - mx)
        Z = ew.sum()
        total += mx + math.log(Z)
        w = ew / Z
        gain = P / S                               # per-node posterior
        mk = m + gain * innov
        Pk = (1.0 - gain) * P
        m = float(w @ mk)                          # GPB1: moment-match to one Gaussian
        P = float(w @ (Pk + (mk - m) ** 2))
    return total


# ---------------------------------------------------------- coordinates
def from_logs(u, v):
    """(u, v) = (log kappa, log sigma2)  ->  (phi, s).

    kappa is the OU mean-reversion rate per step, sigma2 the diffusion rate per
    unit time: phi = e^-kappa, s^2 = sigma2 / (2 kappa).
    """
    kappa, sig2 = math.exp(u), math.exp(v)
    phi = math.exp(-kappa)
    s = math.sqrt(sig2 / (2.0 * kappa))
    return phi, s


def nu_of(u, v):
    """Per-step increment variance at (log kappa, log sigma2) -- the observable."""
    phi, s = from_logs(u, v)
    return s * s * (1.0 - phi * phi)


def hessian(xs, u0, v0, h=FD_H):
    """Central-difference Hessian of the observable loglik in (log kappa, log sigma2)."""
    def f(du, dv):
        phi, s = from_logs(u0 + du, v0 + dv)
        return loglik(xs, phi, s)
    f00 = f(0, 0)
    fuu = f(h, 0) + f(-h, 0) - 2 * f00
    fvv = f(0, h) + f(0, -h) - 2 * f00
    fuv = (f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h)) / 4.0
    return np.array([[fuu / h ** 2, fuv / h ** 2],
                     [fuv / h ** 2, fvv / h ** 2]])


def describe(u0, v0, d, eps=1e-4):
    """What one direction DOES, read off in every chart that matters.

    Returns d log of: kappa, sigma2, nu (per-step increment variance),
    s^2 (= gamma_0, the stationary variance), phi, and gamma_1 = phi s^2.
    All per unit length along `d` in (log kappa, log sigma2).
    """
    def chart(u, v):
        kappa, sig2 = math.exp(u), math.exp(v)
        phi = math.exp(-kappa)
        s2 = sig2 / (2.0 * kappa)
        nu = s2 * (1.0 - phi * phi)
        return dict(kappa=kappa, sigma2=sig2, nu=nu, gamma0=s2,
                    phi=phi, gamma1=phi * s2)
    a = chart(u0 + eps * d[0], v0 + eps * d[1])
    b = chart(u0 - eps * d[0], v0 - eps * d[1])
    return {k: (math.log(a[k]) - math.log(b[k])) / (2 * eps) for k in a}


def run_point(phi_star, s_star, label):
    kappa = -math.log(phi_star)
    sig2 = 2.0 * kappa * s_star * s_star
    u0, v0 = math.log(kappa), math.log(sig2)
    nu0 = nu_of(u0, v0)

    rows = []
    for seed in range(NSEED):
        xs = simulate(phi_star, s_star, seed)
        J = -hessian(xs, u0, v0)                   # observed information
        evals, evecs = np.linalg.eigh(J)
        order = np.argsort(evals)
        soft = evecs[:, order[0]]                  # flattest direction
        stiff = evecs[:, order[1]]
        lo, hi = evals[order[0]], evals[order[1]]
        # what each direction DOES, in every chart
        ds = describe(u0, v0, soft)
        dt = describe(u0, v0, stiff)
        # direct check that the Hessian did not lie: profile loglik along both
        step = 0.25
        f0 = loglik(xs, *from_logs(u0, v0))
        fs = 0.5 * (loglik(xs, *from_logs(u0 + step * soft[0], v0 + step * soft[1]))
                    + loglik(xs, *from_logs(u0 - step * soft[0], v0 - step * soft[1])))
        ft = 0.5 * (loglik(xs, *from_logs(u0 + step * stiff[0], v0 + step * stiff[1]))
                    + loglik(xs, *from_logs(u0 - step * stiff[0], v0 - step * stiff[1])))
        rows.append((lo, hi, abs(soft[0]), abs(soft[1]),
                     abs(ds["nu"]), abs(ds["gamma0"]), abs(ds["gamma1"]), abs(ds["kappa"]),
                     abs(dt["nu"]), abs(dt["gamma0"]), abs(dt["gamma1"]), abs(dt["kappa"]),
                     f0 - fs, f0 - ft))
    a = np.array(rows)
    mean, sem = a.mean(axis=0), a.std(axis=0, ddof=1) / math.sqrt(NSEED)

    print(f"\n--- {label}:  phi*={phi_star}  s*={s_star}   "
          f"(kappa={kappa:.4f}, sigma2={sig2:.4f}, nu={nu0:.4f})")
    print(f"  observed information eigenvalues      {mean[0]:9.2f} (flat) "
          f"{mean[1]:9.2f} (stiff)   ratio {mean[1] / mean[0]:6.1f}")
    print(f"  flat eigenvector in (log kappa, log sigma2): "
          f"({mean[2]:.3f}, {mean[3]:.3f})")
    print(f"  what each direction moves, |dlog .| per unit length:")
    print(f"{'':24s}{'nu':>9s}{'gamma0':>9s}{'gamma1':>9s}{'kappa':>9s}")
    print(f"    flat  direction   {mean[4]:9.3f}{mean[5]:9.3f}{mean[6]:9.3f}{mean[7]:9.3f}")
    print(f"    stiff direction   {mean[8]:9.3f}{mean[9]:9.3f}{mean[10]:9.3f}{mean[11]:9.3f}")
    print(f"  Hessian check, loglik drop at step 0.25: flat {mean[12]:8.3f}  "
          f"stiff {mean[13]:8.3f}   (flat must be much the smaller)")
    print(f"  ==> C2 predicts the flat direction has |dlog nu| ~ 0 and |dlog kappa| ~ 1; "
          f"measured {mean[4]:.3f} and {mean[7]:.3f}")
    return dict(label=label, ratio_eig=mean[1] / mean[0],
                flat_nu=mean[4], flat_kappa=mean[7], flat_g0=mean[5], flat_g1=mean[6],
                drop_flat=mean[12], drop_stiff=mean[13])


if __name__ == "__main__":
    print(__doc__.split("Run:")[0])
    print(f"T={T} steps, {NSEED} seeds, grid {GRID_N} nodes on [{GRID_LO}, {GRID_HI}], "
          f"FD step {FD_H}")
    out = []
    for phi_s, s_s, lab in [(0.95, 0.55, "slow, moderate (the repo's own benchmark point)"),
                            (0.85, 0.80, "box centre"),
                            (0.70, 0.40, "fast, weak"),
                            (0.70, 1.60, "fast, strong (near the resolution ceiling)")]:
        out.append(run_point(phi_s, s_s, lab))
    print("\n================ summary ================")
    print(f"{'point':46s}{'eig ratio':>10s}{'flat dlog nu':>14s}"
          f"{'flat dlog kappa':>17s}{'flat dlog g0':>14s}")
    for r in out:
        print(f"{r['label']:46s}{r['ratio_eig']:10.1f}{r['flat_nu']:14.3f}"
              f"{r['flat_kappa']:17.3f}{r['flat_g0']:14.3f}")
    print("\nC2 (SUMMARY): the flat direction is kappa at fixed nu -- "
          "predicts flat dlog nu ~ 0, flat dlog kappa ~ 1.")
