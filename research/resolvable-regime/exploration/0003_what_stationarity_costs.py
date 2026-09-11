"""0003 -- the decisive test: does the stationarity assumption COST anything?

0002 killed C2 and, in doing so, removed this workstream's cheap motivation: the
`(phi, s)` chart turns out to be correctly oriented (`s` stiff, `phi` flat), so
the reframing cannot be sold as "better coordinates".  What is left is the only
question that ever mattered.

    If the world's log-scale does NOT revert to a mean -- if it wanders -- what
    does it cost to insist, as the shipped class does, that it does?

Two classes, fit by maximum observable likelihood on the same records:

    STATIONARY   lam_t = phi lam_{t-1} + e_t,  e_t ~ N(0, s^2 (1-phi^2)),
                 lam_0 ~ N(0, s^2)                     [two numbers: phi, s]
    WANDERING    lam_t = lam_{t-1} + e_t,      e_t ~ N(0, sigma^2),
                 lam_0 ~ flat on the grid              [one number: sigma -- kappa = 0]

against two truths -- one from each class -- and scored against an ORACLE told
the whole latent variance path.  Four cells.  The swap is justified only if the
wandering class is near-free on a stationary truth AND clearly better on a
wandering one.  If it is merely as good, the thread is a reparameterisation with
no product consequence and should say so.

Same scalar rig and the same GPB1 grid filter as 0002.

Run: python3 0003_what_stationarity_costs.py
"""
import math

import numpy as np

Q_PROC = 1.0
SIG2 = 1.0
T = 4000
NSEED = 8
GRID_LO, GRID_HI, GRID_N = -10.0, 10.0, 281

_NODES = np.linspace(GRID_LO, GRID_HI, GRID_N)
_EXPN = np.exp(_NODES)


# ------------------------------------------------------------------ truths
def sim_stationary(phi, s, seed):
    rng = np.random.default_rng(seed)
    nu = s * s * (1.0 - phi * phi)
    lam = rng.normal(0.0, s)
    lams = np.empty(T)
    for t in range(T):
        lam = phi * lam + rng.normal(0.0, math.sqrt(nu))
        lams[t] = lam
    return _emit(lams, seed)


def sim_wandering(sigma, seed):
    """Brownian log-scale, kappa = 0.  Centred so the record is not systematically
    louder or quieter than its base -- a level offset is a different channel."""
    rng = np.random.default_rng(seed)
    lams = np.cumsum(rng.normal(0.0, sigma, T))
    return _emit(lams - lams.mean(), seed)


def _emit(lams, seed):
    rng = np.random.default_rng(seed + 10_000)
    th = 0.0
    xs = np.empty(T)
    for t in range(T):
        th += rng.normal(0.0, math.sqrt(Q_PROC))
        xs[t] = th + rng.normal(0.0, math.sqrt(SIG2 * math.exp(lams[t])))
    return xs, lams, _truth_path(xs, lams)


def _truth_path(xs, lams):
    """Oracle: a Kalman filter told the true per-step measurement variance."""
    m, P = 0.0, 1e4
    out = np.empty(T)
    for t in range(T):
        P += Q_PROC
        S = P + SIG2 * math.exp(lams[t])
        g = P / S
        m = m + g * (xs[t] - m)
        P = (1.0 - g) * P
        out[t] = m
    return out


# ------------------------------------------------------- the two filters
def _kernel(phi, nu):
    d = _NODES[None, :] - phi * _NODES[:, None]
    k = np.exp(-0.5 * d * d / nu)
    return k / k.sum(axis=1, keepdims=True)


def run_filter(xs, phi, s=None, sigma=None):
    """phi < 1 with s: the stationary class.  phi == 1 with sigma: the wandering
    class (flat initial -- the max-entropy member at fixed increment variance)."""
    if sigma is not None:
        nu, w = sigma * sigma, np.ones(GRID_N) / GRID_N
    else:
        nu = s * s * (1.0 - phi * phi)
        w = np.exp(-0.5 * _NODES ** 2 / (s * s))
        w = w / w.sum()
    if nu <= 1e-12:
        return -np.inf, None
    K = _kernel(phi, nu)
    rv = SIG2 * _EXPN
    m, P, total = 0.0, 1e4, 0.0
    est = np.empty(xs.size)
    for t in range(xs.size):
        P += Q_PROC
        w = w @ K
        S = P + rv
        innov = xs[t] - m
        lw = np.log(w + 1e-300) - 0.5 * (np.log(2.0 * np.pi * S) + innov * innov / S)
        mx = lw.max()
        ew = np.exp(lw - mx)
        Z = ew.sum()
        total += mx + math.log(Z)
        w = ew / Z
        g = P / S
        mk = m + g * innov
        Pk = (1.0 - g) * P
        m = float(w @ mk)
        P = float(w @ (Pk + (mk - m) ** 2))
        est[t] = m
    return total, est


# --------------------------------------------------------------- fitting
PHIS = np.array([0.60, 0.75, 0.85, 0.92, 0.96, 0.98, 0.995])
SS = np.array([0.15, 0.25, 0.40, 0.65, 1.00, 1.60, 2.50])
SIGMAS = np.array([0.003, 0.006, 0.012, 0.025, 0.050, 0.100, 0.200, 0.400])


def fit_stationary(xs):
    best = (-np.inf, None, None, None)
    for phi in PHIS:
        for s in SS:
            ll, est = run_filter(xs, phi, s=s)
            if ll > best[0]:
                best = (ll, phi, s, est)
    return best


def fit_wandering(xs):
    best = (-np.inf, None, None)
    for sg in SIGMAS:
        ll, est = run_filter(xs, 1.0, sigma=sg)
        if ll > best[0]:
            best = (ll, sg, est)
    return best


def cell(name, maker, args):
    rows = []
    for seed in range(NSEED):
        xs, lams, oracle = maker(*args, seed)
        # the estimand: the oracle's own state estimate is the reference both
        # classes are trying to match, so score each against the oracle's track
        ll_s, phi_s, s_s, est_s = fit_stationary(xs)
        ll_w, sg_w, est_w = fit_wandering(xs)
        mse_s = float(np.mean((est_s - oracle) ** 2))
        mse_w = float(np.mean((est_w - oracle) ** 2))
        rows.append((mse_w / mse_s, (ll_w - ll_s) / T, phi_s, s_s, sg_w,
                     mse_s, mse_w))
    a = np.array(rows)
    mean = a.mean(axis=0)
    sem = a.std(axis=0, ddof=1) / math.sqrt(NSEED)
    print(f"\n--- truth: {name}   ({NSEED} seeds, T={T})")
    print(f"  fitted stationary  phi = {mean[2]:.3f}, s = {mean[3]:.3f}")
    print(f"  fitted wandering   sigma = {mean[4]:.4f}  "
          f"(per-step increment SD; ceiling sqrt(2) = 1.414)")
    print(f"  excess MSE vs the oracle's track: stationary {mean[5]:.5f}, "
          f"wandering {mean[6]:.5f}")
    print(f"  WANDERING / STATIONARY  MSE ratio = {mean[0]:.4f} +- {sem[0]:.4f}"
          f"   ( < 1 means dropping stationarity HELPS )")
    print(f"  code length advantage of wandering  = {mean[1]:+.5f} +- {sem[1]:.5f} "
          f"nat/step")
    return mean[0], sem[0], mean[1], sem[1]


if __name__ == "__main__":
    print(__doc__.split("Run:")[0])
    res = {}
    res["stationary phi=0.96, s=0.55"] = cell(
        "STATIONARY, phi=0.96 s=0.55 (slow, moderate)", sim_stationary, (0.96, 0.55))
    res["stationary phi=0.85, s=0.80"] = cell(
        "STATIONARY, phi=0.85 s=0.80 (box centre)", sim_stationary, (0.85, 0.80))
    res["wandering sigma=0.03"] = cell(
        "WANDERING, sigma=0.03 (kappa = 0, spread ~1.9 over the record)",
        sim_wandering, (0.03,))
    res["wandering sigma=0.06"] = cell(
        "WANDERING, sigma=0.06 (kappa = 0, spread ~3.8 over the record)",
        sim_wandering, (0.06,))
    print("\n================ summary ================")
    print(f"{'truth':42s}{'MSE ratio w/s':>16s}{'nat/step w-s':>16s}")
    for k, (r, rs, d, ds) in res.items():
        print(f"{k:42s}{r:10.4f} +-{rs:5.4f}{d:+10.5f} +-{ds:5.5f}")
    print("\nThe swap is justified only if wandering is ~1.0 on stationary truths")
    print("AND clearly < 1.0 on wandering ones.")
