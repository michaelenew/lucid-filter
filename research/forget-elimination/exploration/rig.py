"""Shared rig for the forget-elimination probes (extracted from 0001, unchanged).

    e_t = exp(lam_t / 2) z_t,   lam a stationary AR(1) log-variance (memory phi, spread s)
    about a known common log-level.  Bank = the shipped box _PHIS x _SS (15 members), each an
    exact-to-quadrature grid filter.  `member_loglik` returns one-step predictive
    log-likelihoods, which do not depend on the weight rule.
"""
import os
import pickle
import numpy as np

PHIS = (0.70, 0.85, 0.95)
SS = (0.20, 0.40, 0.80, 1.60, 3.20)
MEMBERS = [(p, s) for p in PHIS for s in SS]
M = len(MEMBERS)
K, G = 65, 5.0                     # quadrature nodes per member, half-range in units of s
EPS0 = 3.1e-4                      # the repository's one aliasing tolerance (resolution-criterion 0002)
C0 = float(np.log(1 / EPS0))       # 8.08 nats

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("FORGET_CACHE", os.path.join(HERE, "_cache"))
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(CACHE, exist_ok=True); os.makedirs(FIG, exist_ok=True)
NS = len(SS)

# ---------------------------------------------------------------- the bank
def build_bank(k=K):
    X = np.empty((M, k)); A = np.empty((M, k, k)); P0 = np.empty((M, k))
    for m, (phi, s) in enumerate(MEMBERS):
        x = np.linspace(-G * s, G * s, k)
        sd = s * np.sqrt(1 - phi * phi)
        a = np.exp(-0.5 * ((x[:, None] - phi * x[None, :]) / sd) ** 2)   # a[k', k]: to k' from k
        A[m] = a / a.sum(axis=0, keepdims=True)
        p0 = np.exp(-0.5 * (x / s) ** 2)
        X[m], P0[m] = x, p0 / p0.sum()
    return X, A, P0

def member_loglik(e, k=K):
    """e: (T, R) draws -> ll: (T, R, M) one-step predictive log-likelihoods."""
    X, A, P0 = build_bank(k)
    T, R = e.shape
    p = np.broadcast_to(P0, (R, M, k)).copy()
    ll = np.empty((T, R, M))
    half_log2pi = 0.5 * np.log(2 * np.pi)
    for t in range(T):
        pred = np.matmul(A[None], p[..., None])[..., 0]                  # (R, M, k)
        e2 = (e[t] ** 2)[:, None, None]
        lem = -half_log2pi - 0.5 * X[None] - 0.5 * e2 * np.exp(-X[None])
        top = lem.max(axis=2, keepdims=True)                             # log-domain: a huge draw
        joint = np.maximum(pred, 1e-300) * np.exp(lem - top)             # must not zero a narrow member
        lik = joint.sum(axis=2)
        p = joint / lik[..., None]
        ll[t] = np.log(lik) + top[..., 0]
    return ll

# ---------------------------------------------------------------- truths
def simulate(T, R, D, hop, seed):
    """Piecewise-constant member index (switching every D steps) and the draws it generates."""
    rng = np.random.default_rng(seed)
    idx = np.empty((T, R), int)
    cur = rng.integers(M, size=R)
    for t in range(T):
        if D and t and t % D == 0:
            for r in range(R):
                if hop == "any":
                    cur[r] = (cur[r] + rng.integers(1, M)) % M
                else:                                                    # neighbour on the grid
                    ip, js = divmod(cur[r], len(SS))
                    opts = [(ip + d, js) for d in (-1, 1) if 0 <= ip + d < len(PHIS)] + \
                           [(ip, js + d) for d in (-1, 1) if 0 <= js + d < len(SS)]
                    a, b = opts[rng.integers(len(opts))]
                    cur[r] = a * len(SS) + b
        idx[t] = cur
    phi = np.array([m[0] for m in MEMBERS])[idx]; s = np.array([m[1] for m in MEMBERS])[idx]
    z = np.empty((T, R)); z[0] = rng.standard_normal(R)                  # standardised level
    innov = rng.standard_normal((T, R))
    for t in range(1, T):
        z[t] = phi[t] * z[t - 1] + np.sqrt(1 - phi[t] ** 2) * innov[t]
    e = np.exp(0.5 * s * z) * rng.standard_normal((T, R))
    return idx, e

def static_truth(T, R, m, seed):
    rng = np.random.default_rng(seed)
    phi, s = MEMBERS[m]
    z = np.empty((T, R)); z[0] = rng.standard_normal(R)
    innov = rng.standard_normal((T, R))
    for t in range(1, T):
        z[t] = phi * z[t - 1] + np.sqrt(1 - phi * phi) * innov[t]
    return np.full((T, R), m), np.exp(0.5 * s * z) * rng.standard_normal((T, R))

def cached(name, fn):
    path = os.path.join(CACHE, name + ".pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    out = fn(); pickle.dump(out, open(path, "wb"), protocol=4)
    return out

def lse(a, axis=-1, keepdims=False):
    m = a.max(axis=axis, keepdims=True)
    out = m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis)


def neighbours(i):
    """Grid neighbours of member i: one rung in phi or in s."""
    ip, js = divmod(i, NS)
    out = [(ip + d) * NS + js for d in (-1, 1) if 0 <= ip + d < len(PHIS)]
    return out + [ip * NS + js + d for d in (-1, 1) if 0 <= js + d < NS]

def draws_from_path(idx, rng):
    """Draws for a given (T, R) member-index path; the standardised level is continuous."""
    T, R = idx.shape
    phi = np.array([m[0] for m in MEMBERS])[idx]; s = np.array([m[1] for m in MEMBERS])[idx]
    z = np.empty((T, R)); z[0] = rng.standard_normal(R)
    innov = rng.standard_normal((T, R))
    for t in range(1, T):
        z[t] = phi[t] * z[t - 1] + np.sqrt(1 - phi[t] ** 2) * innov[t]
    return np.exp(0.5 * s * z) * rng.standard_normal((T, R))

def lab(m): return f"({MEMBERS[m][0]:.2f},{MEMBERS[m][1]:.1f})"
