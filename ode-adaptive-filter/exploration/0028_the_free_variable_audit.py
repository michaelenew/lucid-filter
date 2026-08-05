"""0028 -- Every knob in the candidate filter, and whether it is theoretically free.

The workstream's rule is that there are no theoretically relevant free
parameters, with compute budgets explicitly exempt.  Auditing that means
separating three kinds of number:

  BUDGET       moving it costs time and buys accuracy, monotonically.  Exempt by
               the stated rule.
  SCAFFOLDING  it seeds a maximum-likelihood search.  Free only if the answer
               depends on it -- which is a measurable question, not an opinion.
  COMMITMENT   it changes the model.  These are the real ones, and the honest
               response is either to derive them or to learn them.

`p` is the one known COMMITMENT.  This probe asks two things:

  A.  Do the SCAFFOLDING choices move the answer?  Vary the Q scan window, its
      resolution, the persistence start, and the IV instrument count, and
      compare the fitted parameters.  If they agree, the scaffolding is not
      free in any sense that matters; if they disagree, it is a hidden knob.

  B.  Is `p` learnable, or is it genuinely free?  Order selection by
      **prequential log-loss** -- fit on the first half, score the log
      predictive density of the second.  Out-of-sample scoring needs no
      complexity penalty, and a penalty constant (AIC's 2, BIC's log n) would
      itself be a free parameter, so this is the only order-selection rule
      available that does not import one.

Part B is item 10 of SUMMARY's list, and per `0024` it is the same question as
counting channels.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ode-adaptive-filter", "output"))
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from odefilter import OdeFilter, Params  # noqa: E402
from odefilter.core import _iv_alpha, _moment_noises, _logit  # noqa: E402

ALPHA3 = np.array([2.785218519281637, -2.6855430450862655, 0.9003245225862656])


def gen(kind, n, Q, S2, rng):
    if kind == "WALK":
        x = np.cumsum(np.sqrt(Q) * rng.standard_normal(n))
        return x, x + np.sqrt(S2) * rng.standard_normal(n)
    z = np.zeros(3)
    x = np.zeros(n)
    for t in range(n):
        xn = float(ALPHA3 @ z) + np.sqrt(Q) * rng.standard_normal()
        z = np.concatenate([[xn], z[:-1]])
        x[t] = xn
    return x, x + np.sqrt(S2) * rng.standard_normal(n)


# --------------------------------------------------------------- part A
def staged_fit(y, p, order, max_iter, qwin, qn, phi0, iv_m, scales=False):
    """core.fit_ with the scaffolding exposed.

    Mirrors `OdeFilter.fit_` exactly at the default arguments
    (qwin=(-2,1), qn=13, phi0=0.5, iv_m=None) -- checked in main() by
    comparing against the real thing.  `scales=False` throughout, because the
    scale channels are the parent's and this probe is about what this
    workstream added.
    """
    from scipy.optimize import minimize

    f = OdeFilter(order=order)
    y = np.asarray(y, dtype=float)
    a0 = _iv_alpha(y[np.isfinite(y)], p, extra=iv_m)
    Q0, s20 = _moment_noises(y[np.isfinite(y)], a0)
    n = max(y.size, 1)
    off = math.log(1e-6)

    def nll(v):
        try:
            f.params = Params._from_vec(v, p)
        except (ValueError, OverflowError):
            return np.inf
        r = f._run(y, want=False)
        return np.inf if not np.isfinite(r) else -r / n

    base = np.concatenate([a0, [math.log(Q0), math.log(s20),
                                _logit(phi0), _logit(phi0), off, off]])
    best_q, best_v = Q0, -np.inf
    for Qc in Q0 * np.logspace(qwin[0], qwin[1], qn):
        v = base.copy()
        v[p] = math.log(Qc)
        val = -nll(v)
        if val > best_v:
            best_q, best_v = Qc, val
    base[p] = math.log(best_q)

    idx = list(range(p + 2))
    full = base.copy()

    def sub(vs):
        v = full.copy()
        v[idx] = vs
        return nll(v)

    r2 = minimize(sub, full[idx], method="Nelder-Mead",
                  options=dict(maxiter=int(max_iter), xatol=1e-3, fatol=1e-5))
    full[idx] = r2.x
    f.params = Params._from_vec(full, p)
    f._built = None
    f.reset()
    return f, -r2.fun * n


def summarise(f):
    r = f.params.roots
    comp = r[np.abs(r.imag) > 1e-9]
    return dict(alpha=list(np.round(f.params.alpha, 4)),
                Q=f.params.Q, s2=f.params.s2,
                rmax=float(np.max(np.abs(r))),
                osc=float(np.abs(comp[0])) if comp.size else float("nan"),
                freq=float(abs(np.angle(comp[0]))) if comp.size else float("nan"))


def part_a(seeds, n, order, max_iter):
    print("=== A. does the scaffolding move the answer? ===", flush=True)
    truth_r = np.roots(np.concatenate([[1.0], -ALPHA3]))
    tc = truth_r[np.abs(truth_r.imag) > 1e-9][0]
    print(f"  truth: |osc| = {abs(tc):.4f}, freq = {abs(np.angle(tc)):.4f}, "
          f"Q = 1.0, S2 = 9.0\n", flush=True)

    variants = [
        ("default",        dict(qwin=(-2.0, 1.0), qn=13, phi0=0.5, iv_m=None)),
        ("Q window wider", dict(qwin=(-4.0, 2.0), qn=13, phi0=0.5, iv_m=None)),
        ("Q window tight", dict(qwin=(-1.0, 0.5), qn=13, phi0=0.5, iv_m=None)),
        ("Q grid 5",       dict(qwin=(-2.0, 1.0), qn=5,  phi0=0.5, iv_m=None)),
        ("Q grid 31",      dict(qwin=(-2.0, 1.0), qn=31, phi0=0.5, iv_m=None)),
        ("no Q scan",      dict(qwin=(0.0, 0.0),  qn=1,  phi0=0.5, iv_m=None)),
        ("phi0 = 0.05",    dict(qwin=(-2.0, 1.0), qn=13, phi0=0.05, iv_m=None)),
        ("phi0 = 0.95",    dict(qwin=(-2.0, 1.0), qn=13, phi0=0.95, iv_m=None)),
        ("IV m = p",       dict(qwin=(-2.0, 1.0), qn=13, phi0=0.5, iv_m=3)),
        ("IV m = 4p",      dict(qwin=(-2.0, 1.0), qn=13, phi0=0.5, iv_m=12)),
    ]
    rows = []
    for name, kw in variants:
        recs, lls = [], []
        for s in seeds:
            _, y = gen("ODE", n, 1.0, 9.0, np.random.default_rng(s))
            t0 = time.time()
            f, ll = staged_fit(y, 3, order, max_iter, **kw)
            recs.append(summarise(f))
            lls.append(ll)
        rec = dict(variant=name, **kw,
                   osc=float(np.mean([r["osc"] for r in recs])),
                   freq=float(np.mean([r["freq"] for r in recs])),
                   rmax=float(np.mean([r["rmax"] for r in recs])),
                   Q=float(np.mean([r["Q"] for r in recs])),
                   s2=float(np.mean([r["s2"] for r in recs])),
                   loglik=float(np.mean(lls)),
                   per_seed_ll=lls)
        rec["qwin"] = list(kw["qwin"])
        rows.append(rec)
        print(f"  {name:>16}: |osc|={rec['osc']:.4f} freq={rec['freq']:.4f} "
              f"Q={rec['Q']:7.3f} S2={rec['s2']:6.3f} "
              f"loglik={rec['loglik']:10.2f}  ({time.time()-t0:.0f}s/seed)",
              flush=True)

    d = rows[0]
    print("\n  --- deviation from the default, per variant ---")
    print(f"  {'variant':>16} {'d|osc|':>9} {'dfreq':>9} {'dQ':>9} "
          f"{'dS2':>9} {'d loglik':>10}")
    for r in rows[1:]:
        print(f"  {r['variant']:>16} {r['osc']-d['osc']:+9.4f} "
              f"{r['freq']-d['freq']:+9.4f} {r['Q']-d['Q']:+9.3f} "
              f"{r['s2']-d['s2']:+9.3f} {r['loglik']-d['loglik']:+10.2f}")
    print("\n  A positive `d loglik` means the default scaffolding found a WORSE")
    print("  optimum than the variant: that is the signature of a hidden knob.")
    return rows


# --------------------------------------------------------------- part A2
def part_a2(n, order):
    """How fast does the diffuse prior wash out?

    `update` starts P at (max Rg + max Qg) * p * I.  The `* p` is not derived
    from anything -- it is an inflation I chose.  A start that is forgotten is
    not a free parameter, so the question is how many steps it survives.
    """
    print("\n=== A2. the diffuse prior: how long does the choice survive? ===",
          flush=True)
    _, y = gen("ODE", n, 1.0, 9.0, np.random.default_rng(101))
    pr = Params(alpha=tuple(ALPHA3), Q=1.0, s2=9.0)
    ref = None
    rows = []
    for infl in (0.01, 1.0, 100.0, 10000.0):
        f = OdeFilter(pr, order=order)
        f.reset()
        f.update(y[0])
        f._P = f._P * infl                    # override the chosen inflation
        m = np.array([f.update(v).mean for v in y[1:]])
        if ref is None:
            ref = m
            rows.append(dict(infl=infl, first_gap=0.0, steps=0))
            continue
        gap = np.abs(m - ref) / np.std(y)
        far = np.where(gap > 1e-6)[0]
        steps = int(far[-1]) + 1 if far.size else 0
        rows.append(dict(infl=infl, first_gap=float(gap[0]), steps=steps))
        print(f"  prior x{infl:>8.2f}: first-step gap {gap[0]:.3e} SD, "
              f"last step differing by >1e-6 SD: t = {steps}", flush=True)
    return rows


# --------------------------------------------------------------- part B
def part_b(seeds, n, order, max_iter):
    """Order selection by prequential log-loss.  Is `p` learnable?"""
    print("\n=== B. is p learnable?  prequential log-loss, nats/point ===",
          flush=True)
    print("  fit on the first half; score the log predictive density of the")
    print("  second.  No complexity penalty, because a penalty constant would")
    print("  itself be a free parameter.\n", flush=True)
    ps = (1, 2, 3, 4, 5)
    out = {}
    for kind in ("ODE", "WALK"):
        tab = {p: [] for p in ps}
        for s in seeds:
            _, y = gen(kind, n, 1.0, 9.0, np.random.default_rng(1000 + s))
            half = n // 2
            for p in ps:
                t0 = time.time()
                f, _ = staged_fit(y[:half], p, order, max_iter,
                                  qwin=(-2.0, 1.0), qn=13, phi0=0.5, iv_m=None)
                # run from the start so the state is warm, but accumulate the
                # score only over the held-out half
                f.reset()
                tot, cnt = 0.0, 0
                for t, v in enumerate(y):
                    st = f.update(v)
                    if t >= half and np.isfinite(st.loglik):
                        tot += st.loglik
                        cnt += 1
                tab[p].append(tot / cnt)
                print(f"  {kind} seed {s} p={p}: {tot/cnt:+8.4f} nats/pt "
                      f"({time.time()-t0:.0f}s)", flush=True)
        out[kind] = {p: (float(np.mean(v)),
                         float(np.std(v) / np.sqrt(len(v)))) for p, v in tab.items()}

    print(f"\n  {'data':>6} " + " ".join(f"{'p='+str(p):>16}" for p in ps))
    for kind, tab in out.items():
        cells = [f"{tab[p][0]:+8.4f}+-{tab[p][1]:.4f}" for p in ps]
        best = max(ps, key=lambda p: tab[p][0])
        print(f"  {kind:>6} " + " ".join(cells) + f"   -> picks p = {best}")
    return {k: {str(p): v for p, v in t.items()} for k, t in out.items()}


def main():
    order, max_iter = 5, 250
    seeds_a, seeds_b = (7, 8), (7, 8, 9)
    n_a, n_b = 500, 700

    # sanity: the mirrored fit must agree with the real one at the defaults
    _, y = gen("ODE", 400, 1.0, 9.0, np.random.default_rng(77))
    fa, _ = staged_fit(y, 3, order, max_iter,
                       qwin=(-2.0, 1.0), qn=13, phi0=0.5, iv_m=None)
    fb = OdeFilter.fit(y, p=3, order=order, max_iter=max_iter, scales=False)
    da = float(np.max(np.abs(np.array(fa.params.alpha) - np.array(fb.params.alpha))))
    print(f"mirror check: max |alpha_probe - alpha_core| = {da:.2e} "
          f"({'ok' if da < 1e-6 else 'MISMATCH -- the probe has drifted from core'})\n",
          flush=True)

    rows_a = part_a(seeds_a, n_a, order, max_iter)
    rows_a2 = part_a2(600, order)
    tab_b = part_b(seeds_b, n_b, order, max_iter)

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))
    ax = axes[0]
    names = [r["variant"] for r in rows_a]
    raw = [r["loglik"] - rows_a[0]["loglik"] for r in rows_a]
    # a diverged variant is -inf, which matplotlib cannot place; peg it to the
    # left edge and label it, so the one failure stays visible rather than
    # silently vanishing from the chart
    fin = [v for v in raw if np.isfinite(v)]
    floor = min(fin + [0.0]) - 0.25 * (max(fin + [0.1]) - min(fin + [0.0]) + 0.2)
    dll = [v if np.isfinite(v) else floor for v in raw]
    ax.barh(np.arange(len(names)), dll,
            color=[ts.SERIES[7] if not np.isfinite(v) else
                   (ts.SERIES[0] if v <= 0 else ts.SERIES[1]) for v in raw])
    for i, v in enumerate(raw):
        if not np.isfinite(v):
            ax.text(0.0, i, "diverged  ", ha="right", va="center",
                    fontsize=8, color=ts.SURFACE, weight="bold")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color=ts.INK, lw=1.0)
    ax.set_xlabel("log-likelihood minus the default's (nats)")
    ax.set_title("Scaffolding: does the answer move?")
    ts.tidy(ax)

    ax = axes[1]
    ps = [1, 2, 3, 4, 5]
    for i, (kind, tab) in enumerate(tab_b.items()):
        mu = [tab[str(p)][0] for p in ps]
        se = [tab[str(p)][1] for p in ps]
        ax.errorbar(ps, mu, yerr=se, marker="o", color=ts.SERIES[i], label=kind,
                    capsize=3)
        b = ps[int(np.argmax(mu))]
        ax.scatter([b], [max(mu)], s=90, facecolors="none",
                   edgecolors=ts.SERIES[i], linewidths=1.8, zorder=5)
    ax.set_xticks(ps)
    ax.set_xlabel("recurrence order p")
    ax.set_ylabel("prequential log-loss (nats/pt)")
    ax.set_title("Order selection: is p learnable?")
    ax.legend(fontsize=8)
    ts.tidy(ax)
    ts.save(fig, os.path.join(HERE, "figures", "fig19-free-variables.png"))

    with open(os.path.join(HERE, "figures", "ode028.json"), "w") as f:
        json.dump(dict(scaffolding=rows_a, prior=rows_a2, order=tab_b,
                       n_a=n_a, n_b=n_b, order_q=order), f, indent=1)


if __name__ == "__main__":
    main()
