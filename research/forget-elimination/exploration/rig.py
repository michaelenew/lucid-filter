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


# ------------------------------------------------------------------------------------------
# Added for 0004: the ceiling rates, the 0003 ladder, the drifting worlds, cached likelihoods.
# (0003 carries its own copies of these; they are kept identical.)
C_RES = 1.5 ** 2 / 2                                   # 1.125 nats: spacing of 1.5 blur widths

def ceiling_rates():
    """H[i, j]: fastest resolvable drift from member i to grid neighbour j (0002)."""
    KL = pickle.load(open(os.path.join(CACHE, "partA.pkl"), "rb"))[0]
    KLs = 0.5 * (KL + KL.T)
    H = np.zeros((M, M))
    for i in range(M):
        js = neighbours(i); H[i, js] = KLs[i, js] / C_RES
        if H[i].sum() > 0.5: H[i] *= 0.5 / H[i].sum()
    return H

def world(T, R, alpha_of_t, seed, H):
    """Truth drawn from the local chain at a (possibly time-varying) fraction of the ceiling."""
    rng = np.random.default_rng(seed)
    idx = np.empty((T, R), int); cur = rng.integers(M, size=R)
    nb = [neighbours(i) for i in range(M)]
    for t in range(T):
        a = alpha_of_t(t)
        if a > 0:
            u = rng.random(R)
            for r in range(R):
                i = cur[r]; p = a * H[i, nb[i]]; c = np.cumsum(p)
                if u[r] < c[-1]: cur[r] = nb[i][np.searchsorted(c, u[r])]
        idx[t] = cur
    return idx, draws_from_path(idx, rng)

HALF = 30000
def scenarios(H, R=8):
    """name -> thunk returning (idx, ll, score_from).  Same seeds as 0003."""
    out = {}
    for hop, D, T in [("nbr", 10000, 120000), ("nbr", 1000, 60000), ("nbr", 100, 60000),
                      ("any", 10000, 120000), ("any", 1000, 60000), ("any", 100, 60000)]:
        out[f"{hop} {D}"] = (lambda hop=hop, D=D, T=T: simulate(T, R, D, hop, 5000 + D + (0 if hop == "any" else 7)), D)
    for a in (0.01, 0.14, 1.0):
        out[f"drift a={a}"] = (lambda a=a: world(60000, R, lambda t: a, 300 + int(1000 * a), H), 2000)
    for name, f in (("calm -> ceiling", lambda t: 0.0 if t < HALF else 1.0),
                    ("ceiling -> calm", lambda t: 1.0 if t < HALF else 0.0)):
        out[name + " (2nd half)"] = (lambda f=f, name=name: world(2 * HALF, R, f, 77 + len(name), H), HALF)
    return out

def scenario_ll(name, gen):
    """(idx, ll) for a scenario, cached on disk (float32)."""
    def make():
        idx, e = gen()
        return idx.astype(np.int16), member_loglik(e).astype(np.float32)
    idx, ll = cached("ll_" + name.replace(" ", "_").replace(">", "").replace("(", "").replace(")", ""), make)
    return idx.astype(int), ll.astype(float)

ALPHAS = np.array([0.0] + [2.0 ** -k for k in range(12, -1, -1)])

def ladder_rungs(ll, H):
    """0003: per-rung one-step log-predictives (T, R, 14)."""
    NR = len(ALPHAS)
    PS = np.stack([np.eye(M) + a * (H - np.diag(H.sum(axis=1))) for a in ALPHAS])
    T, R, _ = ll.shape
    w = np.full((R, NR - 1, M), 1.0 / M); l0 = np.full((R, M), -np.log(M))
    out = np.empty((T, R, NR))
    for t in range(T):
        out[t, :, 0] = lse(l0 + ll[t]); l0 = l0 + ll[t]; l0 -= lse(l0, keepdims=True)
        mx = ll[t].max(axis=1)[:, None, None]
        j = np.maximum(w, 1e-300) * np.exp(ll[t][:, None, :] - mx)
        s = np.maximum(j.sum(axis=2), 1e-300)
        out[t, :, 1:] = np.log(s) + mx[:, :, 0]
        w = np.einsum("rkm,kmn->rkn", j / s[..., None], PS[1:])
    return out

def ladder_mix(lp, clamp=None):
    T, R, NR = lp.shape
    lw = np.full((R, NR), -np.log(NR)); mix = np.empty((T, R)); am = np.empty((T, R))
    for t in range(T):
        mix[t] = lse(lw + lp[t]); am[t] = np.exp(lw) @ ALPHAS
        lw = lw + lp[t]
        if clamp: lw = np.maximum(lw, lw.max(axis=1, keepdims=True) - clamp)
        lw -= lse(lw, keepdims=True)
    return mix, am

def simple_rule(ll, rule, v):
    """`forget` (v = f) or member-level clamp (v = c nats)."""
    T, R, _ = ll.shape
    logw = np.full((R, M), -np.log(M)); mix = np.empty((T, R))
    for t in range(T):
        mix[t] = lse(logw + ll[t])
        nxt = logw + ll[t]
        logw = v * logw + ll[t] if rule == "forget" else np.maximum(nxt, nxt.max(axis=1, keepdims=True) - v)
        logw -= lse(logw, keepdims=True)
    return mix
