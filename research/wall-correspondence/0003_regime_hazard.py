"""0003 -- regime hazard vs AR(1): the sector tier, priced.

The correspondence's gap-4 experiment (sibling: superselection
sectors; here: discrete trust regimes -- the ode-filter 6.8% stratum's
question in a controlled miniature). Design gem: a symmetric 2-state
regime chain with hazard h and an AR(1) with phi = 1 - 2h have
IDENTICAL mean, variance, and autocorrelation -- the closures differ
only beyond second order. All filters run with known parameters (the
closure question isolated from fitting). Every number in
0003_regime_hazard.md is computed here. No free parameters; grids are
compute budgets (convergence shown in-file).

  s1  matched second-order design verified (ACF ratio ~ 1).
  s2  the closure matrix (gap to oracle, nats/pt, 6 seeds):
      right closure wins on its own data BOTH ways; the wrong-closure
      penalty is symmetric and modest (~ +0.003 nats/pt).
  s3  the quantization ladder: bin-mean G-state closures of the
      matched AR(1) -- monotone improvement with G on continuous
      data; on sector data the EXACT 2-state hazard closure beats
      every quantized-continuum member: structure (right states +
      jump transitions) beats resolution.
  s4  the honest negative: the sector posterior is NOT sharp online
      even when sectors are real (entropy ~ 0.42 of ln 2 on both data
      kinds): sector identity is a slow observable; the prequential
      gain comes from mixture structure, not confident classification.
"""

import numpy as np
from math import erf

A, H, Q, S2, N = 1.2, 0.01, 0.05, 0.3, 2000
PHI = 1 - 2 * H
SEEDS = range(6)


def gen(seed, kind):
    r = np.random.default_rng(seed)
    lam = np.zeros(N)
    if kind == 'regime':
        s = 1 if r.random() < 0.5 else -1
        for t in range(N):
            if r.random() < H:
                s = -s
            lam[t] = A * s
    else:
        sig = A * np.sqrt(1 - PHI * PHI)
        lam[0] = A * r.normal()
        for t in range(1, N):
            lam[t] = PHI * lam[t - 1] + sig * r.normal()
    th = np.cumsum(r.normal(size=N) * np.sqrt(Q * np.exp(lam)))
    x = th + r.normal(size=N) * np.sqrt(S2)
    return x, lam


def imm(x, grid, T, want_entropy=False):
    K = len(grid)
    m = np.zeros(K)
    P = np.ones(K) * 10.0
    w = np.ones(K) / K
    ll, ent = 0.0, []
    for xt in x:
        win = T.T @ w
        min_ = (T.T @ (w * m)) / np.maximum(win, 1e-300)
        Pin = (T.T @ (w * (P + m * m))) / np.maximum(win, 1e-300) \
            - min_ ** 2
        Pp = Pin + Q * np.exp(grid)
        S = Pp + S2
        lik = np.exp(-(xt - min_) ** 2 / (2 * S)) / np.sqrt(2 * np.pi * S)
        ll += np.log(max(float(np.sum(win * lik)), 1e-300))
        w = win * lik
        w /= w.sum()
        if want_entropy:
            ent.append(-float(np.sum(w * np.log(np.maximum(w, 1e-300)))))
        Kg = Pp / S
        m = min_ + Kg * (xt - min_)
        P = Pp * (1 - Kg)
    return (ll / len(x), float(np.mean(ent))) if want_entropy \
        else ll / len(x)


def oracle(x, lam):
    m, P, ll = 0.0, 10.0, 0.0
    for t, xt in enumerate(x):
        Pp = P + Q * np.exp(lam[t])
        S = Pp + S2
        ll += -0.5 * np.log(2 * np.pi * S) - (xt - m) ** 2 / (2 * S)
        Kg = Pp / S
        m += Kg * (xt - m)
        P = Pp * (1 - Kg)
    return ll / len(x)


REGIME_T = np.array([[1 - H, H], [H, 1 - H]])
REGIME_G = np.array([-A, A])


def linspace_closure(G, span=3.0):
    gg = np.linspace(-span * A, span * A, G)
    sig2 = A * A * (1 - PHI * PHI)
    T = np.zeros((G, G))
    for i in range(G):
        T[i] = np.exp(-(gg - PHI * gg[i]) ** 2 / (2 * sig2))
        T[i] /= T[i].sum()
    return gg, T


def binmean_closure(G):
    r = np.random.default_rng(0)
    z = r.normal(0, A, 400000)
    edges = np.quantile(z, np.arange(1, G) / G)
    states = np.array([z[(z >= lo) & (z < hi)].mean() for lo, hi in
                       zip(np.r_[-np.inf, edges], np.r_[edges, np.inf])])
    sig = A * np.sqrt(1 - PHI * PHI)

    def Phi(u):
        return 0.5 * (1 + np.vectorize(erf)(u / np.sqrt(2)))

    T = np.zeros((G, G))
    for i in range(G):
        mu = PHI * states[i]
        T[i] = Phi((np.r_[edges, np.inf] - mu) / sig) \
            - Phi((np.r_[-np.inf, edges] - mu) / sig)
        T[i] /= T[i].sum()
    return states, T


def s1_design():
    print("== s1: matched second-order design ==")
    acs = {}
    for kind in ('regime', 'ar1'):
        vals = []
        for seed in SEEDS:
            _, lam = gen(seed, kind)
            l0 = lam - lam.mean()
            ac = [float(np.dot(l0[:-k], l0[k:]) / np.dot(l0, l0))
                  for k in (10, 30)]
            vals.append(ac)
        acs[kind] = np.mean(vals, axis=0)
    ratio = acs['regime'] / acs['ar1']
    print(f"  ACF(10,30) regime/ar1 ratios: {ratio[0]:.3f}, "
          f"{ratio[1]:.3f} (design: 1)")
    assert np.all(np.abs(ratio - 1) < 0.25)
    print("  identical second-order structure; the closures differ "
          "only beyond it\n")


def s2_matrix():
    print("== s2: the closure matrix (gap to oracle, nats/pt) ==")
    gg21, T21 = linspace_closure(21)
    out = {}
    for kind in ('regime', 'ar1'):
        rows = {'regime': [], 'ar1-21': [], 'static': []}
        for seed in SEEDS:
            x, lam = gen(seed, kind)
            o = oracle(x, lam)
            rows['regime'].append(o - imm(x, REGIME_G, REGIME_T))
            rows['ar1-21'].append(o - imm(x, gg21, T21))
            rows['static'].append(
                o - imm(x, np.array([0.0]), np.array([[1.0]])))
        for name, v in rows.items():
            v = np.array(v)
            out[(kind, name)] = v.mean()
            print(f"  {kind:6s} data | {name:7s}: {v.mean():+.4f} "
                  f"+- {v.std() / np.sqrt(len(v)):.4f}")
    assert out[('regime', 'regime')] < out[('regime', 'ar1-21')]
    assert out[('ar1', 'ar1-21')] < out[('ar1', 'regime')]
    pen_r = out[('regime', 'ar1-21')] - out[('regime', 'regime')]
    pen_a = out[('ar1', 'regime')] - out[('ar1', 'ar1-21')]
    span = out[('regime', 'static')] - out[('regime', 'regime')]
    print(f"  wrong-closure penalty: {pen_r:+.4f} (ar1-on-regime), "
          f"{pen_a:+.4f} (regime-on-ar1)")
    print(f"  as share of the static-to-oracle span (regime data): "
          f"{100 * pen_r / span:.0f}%")
    assert 0.001 < pen_r < 0.01 and 0.001 < pen_a < 0.01
    print("  right closure wins both ways; the sector-vs-continuum "
          "distinction is worth ~0.003")
    print("  nats/pt here -- the 6.8% stratum's shape, in miniature\n")


def s3_ladder():
    print("== s3: the quantization ladder (bin-mean G-state "
          "closures) ==")
    res = {}
    for kind in ('regime', 'ar1'):
        row = []
        for G in (2, 3, 5, 9, 21):
            st, T = binmean_closure(G)
            gaps = []
            for s in SEEDS:
                x, lam = gen(s, kind)
                gaps.append(oracle(x, lam) - imm(x, st, T))
            row.append(float(np.mean(gaps)))
        res[kind] = row
        print(f"  {kind:6s}: " + "  ".join(
            f"G={g}: {v:+.4f}" for g, v in zip((2, 3, 5, 9, 21), row)))
    assert res['ar1'][0] > res['ar1'][-1]
    assert res['regime'][0] > res['regime'][-1]
    # exact sector closure beats every quantized-continuum member
    ex = []
    for s in SEEDS:
        x, lam = gen(s, 'regime')
        ex.append(oracle(x, lam) - imm(x, REGIME_G, REGIME_T))
    exact = float(np.mean(ex))
    print(f"  exact 2-state hazard closure on regime data: "
          f"{exact:+.4f} -- beats the whole ladder")
    assert exact < min(res['regime']) - 0.005
    print("  G is learnable from below (the p-floor logic on the "
          "sector count); and on sector")
    print("  data STRUCTURE (right states + jump transitions) beats "
          "resolution\n")


def s4_entropy():
    print("== s4: the honest negative -- sectors are slow "
          "observables ==")
    es = {}
    for kind in ('regime', 'ar1'):
        vals = [imm(gen(s, kind)[0], REGIME_G, REGIME_T,
                    want_entropy=True)[1] for s in SEEDS]
        es[kind] = (float(np.mean(vals)),
                    float(np.std(vals) / np.sqrt(len(vals))))
        print(f"  2-state posterior entropy, {kind:6s} data: "
              f"{es[kind][0]:.3f} +- {es[kind][1]:.3f}  (max ln2 = "
              f"0.693)")
    assert abs(es['regime'][0] - es['ar1'][0]) < 0.05
    print("  NOT sharper on sector data: sector identity is a slow "
          "observable; the gain is in")
    print("  the mixture's structure, not confident classification. "
          "(Sibling reading: a")
    print("  superselection charge is read by histories/loops, not "
          "local probes)\n")


if __name__ == "__main__":
    s1_design()
    s2_matrix()
    s3_ladder()
    s4_entropy()
    print("all assertions passed")
