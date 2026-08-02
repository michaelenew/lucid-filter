"""0038 -- does the fast path compute the same thing as the slow one?

Three checks, because a speedup that changes the answer is not a speedup:

  1. `_loglik_batch` against `OdeFilter.loglik`, the shipped scalar recursion,
     over random parameter vectors -- with and without missing observations,
     with and without the dynamics channel active.
  2. `_face_optimum`'s concentrated closed form against a brute-force 2-D scan
     of the same face, and its homogeneity claim (Q, S2) -> c(Q, S2) directly.
  3. `fit()` end to end: the new fit against the pre-speedup one shipped in
     `speedbench/core_baseline.py`, on likelihood attained and on wall clock.

Run:  python3 0038_speed_invariants.py [--quick]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "output"))

from odefilter import OdeFilter, Params                              # noqa: E402
from odefilter.core import (_face_optimum, _face_scan, _iv_alpha,   # noqa: E402
                            _loglik_batch, _moment_noises)

ALPHA3 = (2.6, -2.31, 0.71)          # a lightly damped oscillator plus an offset


def _baseline():
    spec = importlib.util.spec_from_file_location(
        "core_baseline", HERE / "speedbench" / "core_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["core_baseline"] = mod          # dataclasses resolves via here
    spec.loader.exec_module(mod)
    return mod


def series(n=900, seed=0, Q=1.0, s2=9.0, alpha=ALPHA3):
    rng = np.random.default_rng(seed)
    p = len(alpha)
    x = np.zeros(n + p)
    for t in range(p, n + p):
        x[t] = np.dot(alpha, x[t - p:t][::-1]) + rng.normal(0, math.sqrt(Q))
    x = x[p:]
    return x + rng.normal(0, math.sqrt(s2), n)


# ------------------------------------------------------------------- check 1
def check_batch(quick: bool) -> dict:
    rng = np.random.default_rng(7)
    y = series(600 if quick else 900)
    ygap = y.copy()
    ygap[[50, 51, 200, 401]] = np.nan

    vecs = []
    for _ in range(8 if quick else 16):
        a = np.array(ALPHA3) + rng.normal(0, 0.02, 3)
        vecs.append(np.concatenate([a, [
            math.log(np.exp(rng.normal(0, 1))), math.log(9 * np.exp(rng.normal(0, 1))),
            rng.normal(0, 2), rng.normal(0, 2),
            math.log(np.exp(rng.normal(-1, 1))), math.log(np.exp(rng.normal(-1, 1))),
            rng.normal(0, 2), math.log(np.exp(rng.normal(-1.5, 1)))]]))
    V = np.array(vecs)

    out = {}
    for name, data in (("dense", y), ("gaps", ygap)):
        want = []
        for v in V:
            f = OdeFilter(Params._from_vec(v, 3), order=5, order_A=3)
            want.append(f.loglik(data))
        got = _loglik_batch(data, V, 3, 5, 3)
        want = np.array(want)
        rel = np.abs(got - want) / np.maximum(np.abs(want), 1.0)
        out[name] = dict(max_rel=float(rel.max()),
                         max_abs=float(np.abs(got - want).max()),
                         exact=int(np.sum(got == want)), n=int(V.shape[0]))
        print(f"  {name:6s}: max |rel| = {rel.max():.3e}  "
              f"bit-identical {out[name]['exact']}/{V.shape[0]}")

    # The channel collapse.  A channel pinned at the 1e-6 floor is carried by
    # one node in the batch and by `order` nodes in the scalar path; the claim
    # is that this is a numerical, not a modelling, difference.
    off = math.log(1e-6)
    coll = []
    for name, pins in (("all off", (7, 8, 10)), ("scales off", (7, 8)),
                       ("dynamics off", (10,))):
        v = V[0].copy()
        for i in pins:
            v[i] = off
        got = float(_loglik_batch(y, v[None], 3, 5, 3)[0])
        want = OdeFilter(Params._from_vec(v, 3), order=5, order_A=3).loglik(y)
        coll.append(dict(case=name, rel=float(abs(got - want) / abs(want))))
        print(f"  collapse, {name:13s}: |rel| = {coll[-1]['rel']:.2e}")
    out["collapse"] = coll

    # Explosive alpha.  The interesting fact is that "explosive" and "the
    # recursion breaks" are different conditions: the measurement update keeps
    # the posterior variance bounded, so many explosive vectors have a
    # perfectly computable and merely terrible likelihood.  The batch must
    # agree with the scalar path on which is which.
    exp = []
    for a in ((3.0, 0.0, 0.0), (3.0, -3.0, 2.0), (1.5, 0.0, 0.0)):
        v = V[0].copy()
        v[:3] = a
        got = float(_loglik_batch(y, v[None], 3, 5, 3)[0])
        want = OdeFilter(Params._from_vec(v, 3), order=5, order_A=3).loglik(y)
        rel = abs(got - want) / max(abs(want), 1.0)
        exp.append(dict(alpha=list(a), batch=got, scalar=float(want),
                        rel=float(rel)))
        print(f"  alpha={a}: scalar {want:>12.4f}  batch {got:>12.4f}"
              f"   |rel| = {rel:.1e}")
    out["explosive"] = exp
    return out


# ------------------------------------------------------------------- check 2
def check_face(quick: bool) -> dict:
    y = series(500 if quick else 900)
    a = _iv_alpha(y, 3)

    Q, S2 = _face_optimum(y, a)
    # Brute force the same face on a grid centred on the MOMENT estimate -- the
    # quantity the closed form replaces -- so the comparison is independent of
    # the answer being checked.
    Qm, Sm = _moment_noises(y, a)
    best, bq, bs = -np.inf, None, None
    for Qc in Qm * np.logspace(-2.0, 2.0, 25):
        for Sc in Sm * np.logspace(-2.0, 2.0, 25):
            v = OdeFilter(Params(alpha=tuple(a), Q=Qc, s2=Sc),
                          order=5, order_A=3).loglik(y)
            if v > best:
                best, bq, bs = v, Qc, Sc
    ll_star = OdeFilter(Params(alpha=tuple(a), Q=Q, s2=S2),
                        order=5, order_A=3).loglik(y)

    # Homogeneity, exactly.  Scaling both noises by c leaves every gain and every
    # innovation alone and scales every predictive variance, so
    #     ll(cQ, cS2) = ll(Q, S2) - (n/2) log c + (1 - 1/c) A / 2,  A = sum e^2/S,
    # and at the concentrated optimum the first-order condition makes A = n
    # exactly.  Both halves of that are checked.
    c = 3.7
    n = int(np.isfinite(y).sum())
    ll_c = OdeFilter(Params(alpha=tuple(a), Q=c * Q, s2=c * S2),
                     order=5, order_A=3).loglik(y)
    pred = ll_star - 0.5 * n * math.log(c) + 0.5 * (1.0 - 1.0 / c) * n
    _, prof = _face_scan(y, a, np.array([Q / S2]))

    out = dict(Q=float(Q), S2=float(S2), moment_Q=float(Qm), moment_S2=float(Sm),
               grid_Q=float(bq), grid_S2=float(bs),
               ll_closed_form=float(ll_star), ll_grid_best=float(best),
               ll_moment=float(OdeFilter(Params(alpha=tuple(a), Q=Qm, s2=Sm),
                                         order=5, order_A=3).loglik(y)),
               profile_err=float(abs(prof[0] - ll_star)),
               homogeneity_err=float(abs(ll_c - pred)))
    print(f"  closed form  Q={Q:.4g} S2={S2:.4g}  ll={ll_star:.4f}")
    print(f"  moments      Q={Qm:.4g} S2={Sm:.4g}  ll={out['ll_moment']:.4f}")
    print(f"  625-pt grid  Q={bq:.4g} S2={bs:.4g}  ll={best:.4f}"
          f"   (closed form better by {ll_star - best:+.4f})")
    print(f"  profile ll matches concentrated ll to {out['profile_err']:.2e}")
    print(f"  homogeneity residual {out['homogeneity_err']:.2e}")
    return out


# ------------------------------------------------------------------- check 3
def check_fit(quick: bool) -> dict:
    base = _baseline()
    rows = []
    seeds = (0,) if quick else (0, 1, 2)
    n = 600 if quick else 900
    for seed in seeds:
        y = series(n, seed=seed)
        t0 = time.perf_counter()
        fnew = OdeFilter.fit(y, p=3)
        tnew = time.perf_counter() - t0
        t0 = time.perf_counter()
        fold = base.OdeFilter.fit(y, p=3)
        told = time.perf_counter() - t0
        # score both with the SHIPPED recursion, so the comparison is of the
        # estimates and not of two evaluators
        ll_new = fnew.loglik(y)
        ll_old = OdeFilter(Params.from_dict(fold.params.to_dict()),
                           order=5, order_A=3).loglik(y)
        rows.append(dict(seed=seed, n=n, t_new=tnew, t_old=told,
                         speedup=told / tnew,
                         ll_new=float(ll_new), ll_old=float(ll_old),
                         nats_per_point=float((ll_new - ll_old) / n),
                         alpha_new=list(np.round(fnew.params.alpha, 4)),
                         alpha_old=list(np.round(fold.params.alpha, 4))))
        r = rows[-1]
        print(f"  seed {seed}: {told:7.1f}s -> {tnew:6.1f}s  ({r['speedup']:.1f}x), "
              f"loglik {ll_old:.3f} -> {ll_new:.3f} "
              f"({r['nats_per_point']:+.5f} nats/pt)")
    return dict(rows=rows,
                geo_speedup=float(np.exp(np.mean(np.log([r["speedup"] for r in rows])))),
                worst_nats=float(min(r["nats_per_point"] for r in rows)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    res = {}
    print("1. batched evaluator vs the shipped scalar recursion")
    res["batch"] = check_batch(args.quick)
    print("2. the concentrated face optimum")
    res["face"] = check_face(args.quick)
    print("3. fit(): new vs pre-speedup baseline")
    res["fit"] = check_fit(args.quick)

    out = HERE / "figures" / "ode038.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {out}")
