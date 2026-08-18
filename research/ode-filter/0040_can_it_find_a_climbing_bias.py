"""0040 -- Can the filter find a climbing or declining bias?

The model has no intercept anywhere.  A constant offset is a root at z = 1;
a LINEAR offset -- a level whose rate of change is itself part of the state,
which is what a climbing or declining bias is -- is a DOUBLE root at z = 1.
The applied workstream measured that structure worth +0.005 to +0.027
nats/bar by fitting the differenced series, and flagged first-class support
as this filter's call.  `fit(unit_roots=d)` now pins d roots at z = 1 exactly
and searches only the quotient polynomial.  This probe asks whether that was
worth building, in five sections:

  A  THE DEFICIENCY, ON THE PARENT'S GROUND.  A random walk with deterministic
     drift, observed in noise.  Where does a free fit put its roots, and what
     does an h-step forecast then do?  The mechanism under test: a maximum-
     likelihood unit root lands at 1 - eps, and a root at 1 - eps forecasts a
     drift that DECAYS with (1-eps)^h instead of one that continues.

  B  THE TARGET CLASS ONE ORDER UP.  In-class data: (z-1)^2 x oscillator,
     white process noise -- the linear-offset ODE class itself, where the
     bias's rate wanders.  Does the free p=4 fit find the double root the
     data actually carries?  Does the pinned fit recover the quotient?

  C  OUT OF CLASS TWICE.  A deterministic trend over an INTEGRATED
     oscillator.  Differencing the trend away turns white process noise into
     (1-L)w, so no candidate's class contains this data.  Who degrades
     gracefully?  (This is where a wild fitted Q was first seen, and it is
     recorded rather than hidden.)

  D  PINNING WHEN WRONG.  Data with a constant offset and no climb.  What
     does the unnecessary d=1 pin cost, and what does the WRONG d=2 pin cost?
     The decision between them is the same prequential density the filter
     uses everywhere else, so the premium/exposure asymmetry is the number
     that matters.

  E  INTERNAL PIN vs EXTERNAL DIFFERENCING.  The external recipe -- fit(p) on
     dy -- is the same hypothesis as unit_roots=1 on the level with one more
     order, except that differencing pushes iid measurement noise out of the
     model class (it becomes MA(1)) and the internal pin leaves it alone.
     One-step held-out densities are directly comparable (the Jacobian of
     y -> dy given the past is 1).

Scoring is prequential throughout: fit on the first half, stream over the
whole series, sum log predictive density over the second half only.
Forecasts are scored from rolling origins in the second half.  Every fit is
`dynamics=False`: the dynamics channel is orthogonal to what is measured
here and pass 5 is the expensive one.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "lucid"))
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from odefilter import OdeFilter  # noqa: E402
from odefilter.core import _pin_maps  # noqa: E402

FIG = os.path.join(HERE, "figures")
CACHE = os.path.join(FIG, "ode040_fits.json")
QUICK = "--quick" in sys.argv

BETA_OSC = np.array([1.7852187045, -0.9003245226])   # ALPHA3's oscillator pair
ALPHA3 = np.array([2.785218519281637, -2.6855430450862655, 0.9003245225862656])
Q0, S20 = 1.0, 9.0
SEEDS = (29, 30, 31) if not QUICK else (29,)
HS = (1, 5, 20)


def ar(n, alpha, Q, S2, rng):
    p = len(alpha)
    z = np.zeros(p)
    x = np.zeros(n)
    for t in range(n):
        xn = float(np.dot(alpha, z)) + math.sqrt(Q) * rng.standard_normal()
        z = np.concatenate([[xn], z[:-1]])
        x[t] = xn
    return x, x + math.sqrt(S2) * rng.standard_normal(n)


def prequential(f, y, half):
    """Stream over y; (nats/pt over the second half, forecast records).

    Forecast records: from every 10th origin t >= half, the h-step-ahead
    predictive mean for each h in HS, paired with the realised y[t + h].
    """
    f.reset()
    ll, cnt = 0.0, 0
    fc = {h: [] for h in HS}
    for t, v in enumerate(y):
        st = f.update(float(v))
        if t >= half:
            ll += st.loglik
            cnt += 1
        if t >= half and t % 10 == 0 and t + max(HS) < y.size:
            for h in HS:
                fc[h].append((f.predict(h)[0], y[t + h]))
    return ll / max(cnt, 1), fc


def fkey(name, seed):
    return f"{name}:{seed}"


def get_fit(cache, name, seed, y, **kw):
    """Fit with a JSON cache so re-runs are cheap; returns the filter."""
    k = fkey(name, seed)
    if k in cache:
        return OdeFilter.from_dict(cache[k])
    f = OdeFilter.fit(y, dynamics=False, max_iter=200, **kw)
    cache[k] = f.to_dict()
    return f


def root_summary(f):
    r = f.params.roots
    top = np.sort(np.abs(r))[::-1]
    return dict(alpha=list(np.round(f.params.alpha, 5)),
                absroots=list(np.round(np.abs(r), 5)),
                gap1=float(np.round(1.0 - top[0], 5)),
                gap2=float(np.round(1.0 - top[1], 5)) if top.size > 1 else None,
                Q=float(np.round(f.params.Q, 4)),
                s2=float(np.round(f.params.s2, 4)),
                unit_roots=f.params.unit_roots)


def fc_stats(fc):
    out = {}
    for h, rows in fc.items():
        a = np.asarray(rows)
        err = a[:, 1] - a[:, 0]
        out[h] = dict(bias=float(np.mean(err)), rmse=float(np.sqrt(np.mean(err ** 2))))
    return out


def main():
    os.makedirs(FIG, exist_ok=True)
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE) as fh:
            cache = json.load(fh)
    out = {}

    n = 800 if not QUICK else 500
    half = n // 2
    drift = 0.25

    # ---------------------------------------------------------------- A
    print("=== A: random walk + deterministic drift, S2 =", S20)
    A = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        walk = np.cumsum(math.sqrt(Q0) * rng.standard_normal(n))
        y = walk + drift * np.arange(n) + math.sqrt(S20) * rng.standard_normal(n)
        row = {}
        for name, kw in [("p1_free", dict(p=1)),
                         ("p2_free", dict(p=2)),
                         ("p2_pin2", dict(p=2, unit_roots=2))]:
            f = get_fit(cache, f"A_{name}", seed, y[:half], **kw)
            nats, fc = prequential(f, y, half)
            row[name] = dict(nats=float(np.round(nats, 4)), fc=fc_stats(fc),
                             roots=root_summary(f))
            print(f"  seed {seed} {name:8s} nats/pt {nats:+.4f} "
                  f"|roots| {row[name]['roots']['absroots']} "
                  f"bias(h=20) {row[name]['fc'][20]['bias']:+.2f} "
                  f"rmse(h=20) {row[name]['fc'][20]['rmse']:.2f}")
        A[seed] = row
    out["A"] = A
    print(f"  (a flat forecast on this drift is biased by r*h = "
          f"{drift * 20:.1f} at h=20)")

    # ---------------------------------------------------------------- B
    print("=== B: in-class linear offset over an oscillator, p=4 d=2 truth")
    nB = 900 if not QUICK else 500
    halfB = nB // 2
    base4, M4 = _pin_maps(4, 2)
    A4 = base4 + BETA_OSC @ M4
    B = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        x, y = ar(nB, A4, Q0, S20, rng)
        row = {}
        for name, kw in [("p3_free", dict(p=3)),
                         ("p4_free", dict(p=4)),
                         ("p4_pin1", dict(p=4, unit_roots=1)),
                         ("p4_pin2", dict(p=4, unit_roots=2))]:
            f = get_fit(cache, f"B_{name}", seed, y[:halfB], **kw)
            nats, fc = prequential(f, y, halfB)
            row[name] = dict(nats=float(np.round(nats, 4)), fc=fc_stats(fc),
                             roots=root_summary(f))
            print(f"  seed {seed} {name:8s} nats/pt {nats:+.4f} "
                  f"gap to |z|=1: {row[name]['roots']['gap1']}, "
                  f"{row[name]['roots']['gap2']} "
                  f"rmse(h=20) {row[name]['fc'][20]['rmse']:.1f}")
        B[seed] = row
    out["B"] = B

    # ---------------------------------------------------------------- C
    print("=== C: deterministic trend over an INTEGRATED oscillator "
          "(out of class for everyone)")
    C = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        x, y = ar(nB, ALPHA3, Q0, S20, rng)
        y = y + 0.4 * np.arange(nB)
        row = {}
        for name, kw in [("p3_free", dict(p=3)),
                         ("p4_free", dict(p=4)),
                         ("p4_pin2", dict(p=4, unit_roots=2))]:
            f = get_fit(cache, f"C_{name}", seed, y[:halfB], **kw)
            nats, fc = prequential(f, y, halfB)
            row[name] = dict(nats=float(np.round(nats, 4)), fc=fc_stats(fc),
                             roots=root_summary(f))
            print(f"  seed {seed} {name:8s} nats/pt {nats:+.4f} "
                  f"Q {row[name]['roots']['Q']:.2f} "
                  f"bias(h=20) {row[name]['fc'][20]['bias']:+.2f} "
                  f"rmse(h=20) {row[name]['fc'][20]['rmse']:.1f}")
        C[seed] = row
    out["C"] = C

    # ---------------------------------------------------------------- D
    print("=== D: constant offset, no climb -- what does pinning cost?")
    D = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        x, y = ar(nB, ALPHA3, Q0, S20, rng)
        row = {}
        for name, kw in [("p3_free", dict(p=3)),
                         ("p3_pin1", dict(p=3, unit_roots=1)),
                         ("p4_pin2", dict(p=4, unit_roots=2))]:
            f = get_fit(cache, f"D_{name}", seed, y[:halfB], **kw)
            nats, fc = prequential(f, y, halfB)
            row[name] = dict(nats=float(np.round(nats, 4)), fc=fc_stats(fc),
                             roots=root_summary(f))
            print(f"  seed {seed} {name:8s} nats/pt {nats:+.4f} "
                  f"rmse(h=20) {row[name]['fc'][20]['rmse']:.1f}")
        D[seed] = row
    out["D"] = D

    # ---------------------------------------------------------------- E
    print("=== E: internal pin vs external differencing, same hypothesis")
    # On the B data (linear offset + oscillator): the external recipe is
    # fit(p=3) on dy; the internal form is fit(p=4, unit_roots=1) on y.
    # Same root budget for the same job; only the measurement noise differs
    # in treatment.  One-step densities are comparable (Jacobian 1); the
    # external candidate's second-half score starts one observation later,
    # which at these lengths is noise.
    E = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        x, y = ar(nB, A4, Q0, S20, rng)
        dy = np.diff(y)
        fx = get_fit(cache, "E_ext_p3diff", seed, dy[:halfB - 1], p=3)
        nats_x, _ = prequential(fx, dy, halfB - 1)
        fi = cache.get(fkey("B_p4_pin1", seed))
        fi = OdeFilter.from_dict(fi)
        nats_i, _ = prequential(fi, y, halfB)
        E[seed] = dict(external=float(np.round(nats_x, 4)),
                       internal=float(np.round(nats_i, 4)),
                       gain=float(np.round(nats_i - nats_x, 4)),
                       ext_roots=root_summary(fx))
        print(f"  seed {seed} external {nats_x:+.4f} internal {nats_i:+.4f} "
              f"internal-external {nats_i - nats_x:+.4f} nats/pt "
              f"(ext |roots| {E[seed]['ext_roots']['absroots']})")
    out["E"] = E

    with open(CACHE, "w") as fh:
        json.dump(cache, fh)
    with open(os.path.join(FIG, "ode040.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", os.path.join(FIG, "ode040.json"))

    # ------------------------------------------------------------- figure
    seed = SEEDS[0]
    rng = np.random.default_rng(seed)
    walk = np.cumsum(math.sqrt(Q0) * rng.standard_normal(n))
    y = walk + drift * np.arange(n) + math.sqrt(S20) * rng.standard_normal(n)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))

    # panel 1: what each filter believes the next 60 steps look like
    ax = ts.tidy(axes[0])
    t0 = 640
    ax.plot(np.arange(t0 - 90, n), y[t0 - 90:], color=ts.INK2, lw=1.0,
            alpha=0.6, label="observed")
    for i, name in enumerate(("p1_free", "p2_free", "p2_pin2")):
        f = OdeFilter.from_dict(cache[fkey(f"A_{name}", seed)])
        f.reset()
        for v in y[:t0]:
            f.update(float(v))
        hh = np.arange(1, 61)
        mu = np.array([f.predict(int(h))[0] for h in hh])
        ax.plot(t0 + hh, mu, color=ts.SERIES[i], label=name.replace("_", " "))
    ax.axvline(t0, color=ts.GRID, lw=1.0)
    ax.set_title("A: forecasts on a climbing bias")
    ax.set_xlabel("t")
    ax.legend()

    # panel 2: signed forecast bias by horizon (section A, seed-pooled)
    ax = ts.tidy(axes[1])
    for i, name in enumerate(("p1_free", "p2_free", "p2_pin2")):
        b = [np.mean([A[s][name]["fc"][h]["bias"] for s in SEEDS]) for h in HS]
        ax.plot(HS, b, "o-", color=ts.SERIES[i], label=name.replace("_", " "))
    ax.plot(HS, [drift * h for h in HS], "--", color=ts.INK2, lw=1.2,
            label="flat forecast: r·h")
    ax.axhline(0.0, color=ts.GRID, lw=1.0)
    ax.set_title("A: signed forecast error vs horizon")
    ax.set_xlabel("h")
    ax.set_ylabel("mean(y - forecast)")
    ax.legend()

    # panel 3: the gap to |z| = 1 that the free fits leave (sections A + B)
    ax = ts.tidy(axes[2])
    labels, gaps, cols = [], [], []
    for sec, name, lab, c in [("A", "p2_free", "A p2 free", 0),
                              ("B", "p4_free", "B p4 free", 1)]:
        rows = out[sec]
        for s in SEEDS:
            labels.append(f"{lab}\nseed {s}")
            g2 = rows[s][name]["roots"]["gap2"]
            gaps.append(max(g2, 1e-5) if g2 is not None else np.nan)
            cols.append(ts.SERIES[c])
    ax.bar(range(len(gaps)), gaps, color=cols)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title("the second root's gap to the circle, free fits")
    ax.set_ylabel("1 - |z|  (log)")

    ts.save(fig, os.path.join(FIG, "fig29-climbing-bias.png"))


if __name__ == "__main__":
    main()
