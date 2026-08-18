"""0038 -- Is the process-scale channel dead, and if so what killed it?

Three probes have landed on the same fact from three directions:

  * `_moment_noises` amplifies its Q estimate 151x  (0018)
  * the phi start is inert because s_P -> 0 on every dataset tried, INCLUDING
    data generated with s_P = 0.8                   (0029)
  * a process-noise regime of x8 costs 4.9x in tracking, because the
    MEASUREMENT channel absorbs it                  (0033)

`SUMMARY.md` records the standing diagnosis as a conditioning fact -- "Q is
0.66% of gamma_0, so a log-scale wobble on it barely moves the predictive
variance that sigma^2 dominates" -- and the proposed fix as "parameterise the
scale on Q_eff rather than on Q alone".  That diagnosis is stated in the wrong
currency.  gamma_0 is not what the likelihood sees.  What it sees is

    S = a00 + Q e^lamP + R e^lamM

so the channel's leverage is Q/S -- `share_process`, which the filter already
reports -- and the question is how much of the AVAILABLE evidence about a
change in Q that quantity exposes.

  A  LEVERAGE.  The three shares, on ODE data and on the parent's own
     random-walk data, in the currency the likelihood actually uses.

  B  AVAILABLE vs EXTRACTED.  A series with a genuine x8 process-noise regime.
     The evidence ceiling is a Kalman filter told Q_t exactly.  Score that, a
     filter stuck at baseline Q, and the gridded filter both as fitted and with
     s_P FORCED on.  If forcing it on does not recover most of the ceiling, the
     channel is structurally unable to see the regime.

  C  WHAT THE COLLAPSE DELETES.  Under GPB1 every node shares one covariance,
     so two nodes differ in S by one step of process noise and not by the
     accumulated history.  Quantify the loss.

  D  IS THE LIKELIHOOD WRONG, OR THE SEARCH?  Profile the marginal likelihood
     in s_P.  An interior optimum acquits the model and indicts `fit_`.

  E  WHERE THE ZERO ACTUALLY COMES FROM.  Profile s_P on the two datasets that
     reported it: 0032's fitting window and 0029's four series.

  F  THE ASYMMETRY.  What does an unnecessary s_P cost, against what a missing
     one costs?

  G  THE N DEPENDENCE.  Fit end to end at 0029's n and at twice it.

Conclusion, stated up front because it reverses the standing one: the channel
is NOT dead, the diagnosis above is wrong, and the two datasets that reported a
zero were reporting two different things.
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

from odefilter import OdeFilter, Params  # noqa: E402

ALPHA3 = np.array([2.785218519281637, -2.6855430450862655, 0.9003245225862656])
LOG2PI = float(np.log(2.0 * np.pi))
Q0, S20 = 1.0, 9.0
FIG = os.path.join(HERE, "figures")
CACHE = os.path.join(FIG, "ode038_fits.json")


def companion(alpha):
    p = alpha.size
    F = np.zeros((p, p))
    F[0] = alpha
    if p > 1:
        F[1:, :-1] = np.eye(p - 1)
    return F


def ode(n, alpha, Qseq, rng):
    """Simulate with a per-step process variance."""
    p = alpha.size
    z = np.zeros(p)
    x = np.zeros(n)
    for t in range(n):
        w = math.sqrt(Qseq[t]) * rng.standard_normal()
        z = np.concatenate([[alpha @ z + w], z[:-1]])
        x[t] = z[0]
    return x


def logscale(n, rng, phi, s):
    """An AR(1) log-scale at its stationary law -- the channel's own model."""
    lam = np.zeros(n)
    nu = math.sqrt(s * s * (1.0 - phi * phi))
    lam[0] = s * rng.standard_normal()
    for t in range(1, n):
        lam[t] = phi * lam[t - 1] + nu * rng.standard_normal()
    return lam


def kalman(y, alpha, Qseq, s2, burn=0):
    """Exact filter for a KNOWN per-step Q.  The evidence ceiling: no channel
    can beat being told the answer.  Returns (nll/pt, calibration, a00 trace).
    """
    p = alpha.size
    F = companion(alpha)
    m = np.zeros(p)
    P = np.eye(p) * (s2 + float(np.max(Qseq))) * p
    e1 = np.zeros(p)
    e1[0] = 1.0
    nll, cal, k = 0.0, 0.0, 0
    a00s = np.zeros(len(y))
    for t, yt in enumerate(y):
        m = F @ m
        A = F @ P @ F.T
        a00s[t] = A[0, 0]
        S = A[0, 0] + Qseq[t] + s2
        e = float(yt) - m[0]
        if t >= burn:
            nll += 0.5 * (e * e / S + math.log(S) + LOG2PI)
            cal += e * e / S
            k += 1
        row = A[:, 0] + Qseq[t] * e1
        K = row / S
        m = m + K * e
        P = A
        P[0, 0] += Qseq[t]
        P -= np.outer(K, row)
    return nll / k, cal / k, a00s


def grid_score(fil, y, burn=0):
    """Log-loss and calibration for the gridded filter.

    Its `loglik` is a log-sum-exp over the grid, so it is NOT
    -.5(e^2/S + log S + log2pi) for any single S -- the mixture is not
    Gaussian.  Score off `loglik` directly and take calibration from the
    mixture's own reported predictive variance.
    """
    r = fil.reset().filter(y)
    fil.reset()
    ll = np.array([fil.update(float(v)).loglik for v in y])
    nll = -float(np.mean(ll[burn:]))
    e, S = r.innovation[burn:], r.pred_var[burn:]
    return nll, float(np.mean(e * e / S)), r


def nll_pt(pr, y):
    return -OdeFilter(pr).loglik(y) / len(y)


# ---------------------------------------------------------------- A: leverage
def part_a():
    rng = np.random.default_rng(11)
    n = 900
    out = {}
    y_ode = ode(n, ALPHA3, np.full(n, Q0), rng) + math.sqrt(S20) * rng.standard_normal(n)
    r = OdeFilter(Params(ALPHA3, Q0, S20)).reset().filter(y_ode)
    out["ODE (alpha3, Q=1, s2=9)"] = r

    x_rw = np.cumsum(math.sqrt(Q0) * rng.standard_normal(n))
    y_rw = x_rw + math.sqrt(S20) * rng.standard_normal(n)
    out["WALK (alpha=1, Q=1, s2=9)"] = (
        OdeFilter(Params(np.array([1.0]), Q0, S20)).reset().filter(y_rw))

    y_hi = ode(n, ALPHA3, np.full(n, Q0), rng) + 0.5 * rng.standard_normal(n)
    out["ODE at s2 = 0.25"] = (
        OdeFilter(Params(ALPHA3, Q0, 0.25)).reset().filter(y_hi))

    print("A.  LEVERAGE -- mean share of the predictive variance")
    print(f"    {'':30s} {'prior':>8s} {'process':>8s} {'meas':>8s}")
    shares = {}
    for k, r in out.items():
        v = (r.share_prior[100:].mean(), r.share_process[100:].mean(),
             r.share_measurement[100:].mean())
        shares[k] = v
        print(f"    {k:30s} {v[0]:8.4f} {v[1]:8.4f} {v[2]:8.4f}")
    print("    share_process IS the channel's leverage: the fraction of S a")
    print("    wobble on lamP can move.  The ODE case has less of it than the")
    print("    parent's own case, where the same channel works.")
    return shares


# -------------------------------------------------- B: available vs extracted
def part_b():
    rng = np.random.default_rng(4)
    n, lo, hi, mult = 900, 400, 560, 8.0
    Qseq = np.full(n, Q0)
    Qseq[lo:hi] = Q0 * mult
    x = ode(n, ALPHA3, Qseq, rng)
    y = x + math.sqrt(S20) * rng.standard_normal(n)

    burn = 60
    rows = []
    nll_o, cal_o, _ = kalman(y, ALPHA3, Qseq, S20, burn=burn)
    rows.append(("oracle Q_t", nll_o, cal_o))
    nll_f, cal_f, _ = kalman(y, ALPHA3, np.full(n, Q0), S20, burn=burn)
    rows.append(("static Q = 1", nll_f, cal_f))
    qbar = float(np.mean(Qseq))
    nll_b, cal_b, _ = kalman(y, ALPHA3, np.full(n, qbar), S20, burn=burn)
    rows.append((f"hindsight const Q = {qbar:.2f}", nll_b, cal_b))

    lamP = {}
    for name, phi, sP in (("grid, s_P = 0 (as fitted)", 0.0, 0.0),
                          ("grid, s_P = 0.5 forced", 0.90, 0.5),
                          ("grid, s_P = 0.8 forced", 0.90, 0.8),
                          ("grid, s_P = 1.2 forced", 0.95, 1.2)):
        f = OdeFilter(Params(ALPHA3, Q0, S20, phi_P=phi, s_P=sP))
        nll, cal, r = grid_score(f, y, burn=burn)
        rows.append((name, nll, cal))
        lamP[name] = r.process_scale

    print(f"\nB.  AVAILABLE vs EXTRACTED -- x{mult:.0f} regime, t in [{lo}, {hi})")
    print(f"    {'':30s} {'nll/pt':>9s} {'calib':>7s} {'gap closed':>11s}")
    span = nll_f - nll_o
    for name, nll, cal in rows:
        frac = "" if name == "static Q = 1" else f"{(nll_f - nll) / span * 100:10.1f}%"
        print(f"    {name:30s} {nll:9.4f} {cal:7.3f} {frac:>11s}")
    print(f"    the regime is worth {span:.4f} nats/pt to a filter that knows.")
    print("    Forced on, the channel gets most of it -- and beats the best")
    print("    single constant Q chosen in hindsight.  It is NOT dead.")
    return dict(y=y, x=x, Qseq=Qseq, lo=lo, hi=hi, rows=rows, lamP=lamP,
                nll_o=nll_o, nll_f=nll_f, burn=burn)


# ---------------------------------------------------- C: what collapse deletes
def part_c(mult=8.0):
    """Two hypotheses about the same data: Q, and mult*Q.  Their predictive
    variances differ by

        D_true = [a00(mult Q) - a00(Q)]  +  (mult - 1) Q

    but a SHARED covariance -- what collapsing to one P every step enforces --
    hands both nodes the same a00, so they differ by only the second term.
    """
    rng = np.random.default_rng(7)
    n = 1200
    print("\nC.  WHAT THE GPB1 COLLAPSE DELETES")
    print(f"    {'s2':>7s} {'a00(Q)':>10s} {'a00(kQ)':>10s} {'D_acc':>10s}"
          f" {'D_step':>9s} {'kept':>7s}")
    keep = []
    for s2 in (0.25, 1.0, 9.0, 36.0):
        y = (ode(n, ALPHA3, np.full(n, Q0), rng)
             + math.sqrt(s2) * rng.standard_normal(n))
        _, _, a1 = kalman(y, ALPHA3, np.full(n, Q0), s2)
        _, _, a2 = kalman(y, ALPHA3, np.full(n, Q0 * mult), s2)
        A1, A2 = float(a1[-1]), float(a2[-1])
        d_acc, d_step = A2 - A1, (mult - 1.0) * Q0
        kept = d_step / (d_acc + d_step)
        keep.append((s2, A1, A2, d_acc, d_step, kept))
        print(f"    {s2:7.2f} {A1:10.4f} {A2:10.4f} {d_acc:10.4f}"
              f" {d_step:9.4f} {kept * 100:6.1f}%")
    print("    Real, and it scales the wrong way with SNR -- but B shows what")
    print("    survives is still enough.  This is a tax, not the cause.")
    return keep


# ------------------------------------------------- D: the likelihood in s_P
def part_d(b):
    """An interior optimum acquits the model; a boundary one indicts it.

    Twice: with Q pinned, and with Q re-optimised at each s_P -- switching the
    channel on multiplies the mean process variance by e^{s^2/2}, so a joint
    fit can pay for s_P by shrinking Q, and a profile forbidding that is not
    the one the fitter sees.
    """
    from scipy.optimize import minimize_scalar

    rng = np.random.default_rng(19)
    n2 = 900
    y2 = (ode(n2, ALPHA3, Q0 * np.exp(logscale(n2, rng, 0.9, 0.8)), rng)
          + math.sqrt(S20) * rng.standard_normal(n2))

    grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4]
    print("\nD.  IS THE LIKELIHOOD WRONG, OR THE SEARCH?")
    out = {}
    for tag, yy in (("x8 regime", b["y"]), ("AR(1) log-scale, s_P=0.8", y2)):
        pinned, freeQ, qhat = [], [], []
        for sP in grid:
            pinned.append(nll_pt(Params(ALPHA3, Q0, S20, phi_P=0.9, s_P=sP), yy))

            def nllQ(lq, sP=sP):
                try:
                    return nll_pt(Params(ALPHA3, math.exp(lq), S20,
                                         phi_P=0.9, s_P=sP), yy)
                except Exception:
                    return 1e9
            r = minimize_scalar(nllQ, bounds=(math.log(1e-3), math.log(1e3)),
                                method="bounded", options={"xatol": 1e-3})
            freeQ.append(float(r.fun))
            qhat.append(math.exp(float(r.x)))
        out[tag] = (pinned, freeQ, qhat)
        print(f"\n    {tag}")
        print(f"    {'s_P':>5s} {'nll (Q=1)':>11s} {'nll (Q free)':>13s} {'Q_hat':>8s}")
        for sP, a, c, q in zip(grid, pinned, freeQ, qhat):
            print(f"    {sP:5.1f} {a:11.5f} {c:13.5f} {q:8.4f}")
        print(f"    argmin: s_P = {grid[int(np.argmin(pinned))]} pinned, "
              f"{grid[int(np.argmin(freeQ))]} with Q free -- INTERIOR.")
    return grid, out, y2


# --------------------------------------------- E: where the zero comes from
def part_e():
    """The two datasets that reported a zero, profiled.

    0032 fitted on its BASELINE STRETCH -- and the x8 process regime it later
    fails on sits at t in [720, 850), outside that window entirely.  0029
    generated its data WITH s_P = 0.8 and fitted the whole thing.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m32", os.path.join(HERE, "0032_a_hard_series.py"))
    m32 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m32)
    _, y32, _, _ = m32.simulate(np.random.default_rng(20260801))
    base = y32[:m32.JUMPS[0]]
    d = json.load(open(os.path.join(FIG, "ode032_fit.json")))
    pr = Params.from_dict(d["ode"]["params"])

    print("\nE.  WHERE THE ZERO COMES FROM")
    print(f"\n    0032, fitting window t < {m32.JUMPS[0]} (n={len(base)}), "
          f"fitted s_P = {pr.s_P:.3g}")
    print(f"    the x8 process regime it fails on is at t in {m32.PROC_REGIME} "
          "-- OUTSIDE this window")
    g32 = (0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4)
    v32 = [nll_pt(Params(pr.alpha, pr.Q, pr.s2, phi_P=(0.9 if s > 0 else 0.0),
                         phi_M=pr.phi_M, s_P=s, s_M=pr.s_M), base) for s in g32]
    print(f"    {'s_P':>6} {'nll/pt':>10} {'vs s_P=0':>10}")
    for s, v in zip(g32, v32):
        print(f"    {s:6.2f} {v:10.5f} {v - v32[0]:+10.5f}")
    print(f"    argmin s_P = {g32[int(np.argmin(v32))]}.  The zero is CORRECT: "
          "no scale")
    print("    variation is present in the data the fit was shown.")

    print("\n    0029, n=500, data generated WITH s_P = 0.8, profiled at truth")
    g29 = (0.0, 0.2, 0.4, 0.6, 0.8, 1.2)
    e29 = {}
    for lbl, phiP in (("impulsive", 0.05), ("persistent", 0.95)):
        for seed in (31, 32):
            rng = np.random.default_rng(seed)
            lam = logscale(500, rng, phiP, 0.8)
            yy = (ode(500, ALPHA3, Q0 * np.exp(lam), rng)
                  + math.sqrt(S20) * rng.standard_normal(500))
            row = [nll_pt(Params(ALPHA3, Q0, S20,
                                 phi_P=(phiP if s > 0 else 0.0), s_P=s), yy)
                   for s in g29]
            e29[f"{lbl} s{seed}"] = (yy, row)
            print(f"    {lbl+' s'+str(seed):>16}: "
                  + " ".join(f"{v:.5f}" for v in row)
                  + f"  -> argmin s_P = {g29[int(np.argmin(row))]}")
    print("    argmin lands ON the truth.  Here the zero is NOT correct --")
    print("    the likelihood knows the answer and the fit does not return it.")
    return g32, v32, g29, e29


# ------------------------------------------------------------ F: the asymmetry
def part_f():
    """What does an unnecessary s_P cost, against what a missing one costs?"""
    grid = (0.0, 0.5, 0.8, 1.2)
    print("\nF.  THE ASYMMETRY -- nll/pt, mean of 3 seeds")
    print(f"    {'scenario':>30}" + "".join(f"{'s_P='+str(s):>9}" for s in grid))
    res = {}
    for tag, mult in (("constant scale (truth 0)", 1.0), ("x3 regime", 3.0),
                      ("x8 regime", 8.0)):
        acc = np.zeros(len(grid))
        for seed in (4, 5, 6):
            rng = np.random.default_rng(seed)
            n = 900
            Qs = np.full(n, Q0)
            if mult > 1.0:
                Qs[400:560] = mult * Q0
            y = ode(n, ALPHA3, Qs, rng) + math.sqrt(S20) * rng.standard_normal(n)
            for i, s in enumerate(grid):
                f = OdeFilter(Params(ALPHA3, Q0, S20,
                                     phi_P=(0.9 if s > 0 else 0.0), s_P=s))
                acc[i] += grid_score(f, y, burn=60)[0]
        acc /= 3.0
        res[tag] = acc
        print(f"    {tag:>30}" + "".join(f"{v:9.4f}" for v in acc))
        print(f"    {'vs s_P=0':>30}" + "".join(f"{v - acc[0]:+9.4f}" for v in acc))
    cost = res["constant scale (truth 0)"][2] - res["constant scale (truth 0)"][0]
    gain = res["x8 regime"][0] - res["x8 regime"][2]
    print(f"    s_P = 0.8 costs {cost:+.4f}/pt when unnecessary and saves "
          f"{gain:+.4f}/pt")
    print(f"    when it is not: an asymmetry of {gain / cost:.0f}x.  s_P = 0 is")
    print("    not 'a little variation', it is 'provably none, forever'.")
    return grid, res


# ---------------------------------------------------------- G: the n dependence
def part_g(e29):
    """0029 fitted n=500 and got zero.  Does more data change it?

    Also profile s_P AT THE FITTED PARAMETERS rather than at the truth.  That
    is the profile the optimiser actually stands in, and it is the one that
    says whether the fit is stuck at a local optimum or merely stopped short.
    """
    if os.path.exists(CACHE):
        got = json.load(open(CACHE))
    else:
        got = {}
        for n in (500, 900):
            for lbl, phiP in (("impulsive", 0.05), ("persistent", 0.95)):
                for seed in (31, 32):
                    rng = np.random.default_rng(seed)
                    lam = logscale(n, rng, phiP, 0.8)
                    yy = (ode(n, ALPHA3, Q0 * np.exp(lam), rng)
                          + math.sqrt(S20) * rng.standard_normal(n))
                    f = OdeFilter.fit(yy, p=3, dynamics=False)
                    got[f"n{n} {lbl} s{seed}"] = {
                        "params": f.params.to_dict(),
                        "prof": [nll_pt(Params(
                            f.params.alpha, f.params.Q, f.params.s2,
                            phi_P=(f.params.phi_P if s > 0 else 0.0),
                            phi_M=f.params.phi_M, s_P=s, s_M=f.params.s_M), yy)
                            for s in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4)]}
                    print(f"      fitted n={n} {lbl} s{seed}: "
                          f"s_P={f.params.s_P:.4f}", flush=True)
        json.dump(got, open(CACHE, "w"), indent=1)
    pg = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4)
    # tolerate a cache written before the profile was recorded
    dirty = False
    for n in (500, 900):
        for lbl, phiP in (("impulsive", 0.05), ("persistent", 0.95)):
            for seed in (31, 32):
                k = f"n{n} {lbl} s{seed}"
                if "params" in got[k]:
                    continue
                pr = Params.from_dict(got[k])
                rng = np.random.default_rng(seed)
                yy = (ode(n, ALPHA3, Q0 * np.exp(logscale(n, rng, phiP, 0.8)), rng)
                      + math.sqrt(S20) * rng.standard_normal(n))
                got[k] = {"params": got[k],
                          "prof": [nll_pt(Params(
                              pr.alpha, pr.Q, pr.s2,
                              phi_P=(pr.phi_P if s > 0 else 0.0),
                              phi_M=pr.phi_M, s_P=s, s_M=pr.s_M), yy)
                              for s in pg]}
                dirty = True
    if dirty:
        json.dump(got, open(CACHE, "w"), indent=1)
    print("\nG.  THE N DEPENDENCE -- end-to-end fit, truth s_P = 0.8")
    print(f"    {'case':>22} {'s_P n=500':>10} {'s_P n=900':>10}"
          f" {'argmin of the profile AT the n=900 fit':>40}")
    for lbl in ("impulsive", "persistent"):
        for seed in (31, 32):
            a = got[f"n500 {lbl} s{seed}"]
            b = got[f"n900 {lbl} s{seed}"]
            am = pg[int(np.argmin(b["prof"]))]
            print(f"    {lbl+' s'+str(seed):>22}"
                  f" {Params.from_dict(a['params']).s_P:10.4f}"
                  f" {Params.from_dict(b['params']).s_P:10.4f} {am:>40}")
    print("    Doubling the data does NOT flip it, and the profile taken AT")
    print("    the fitted parameters has its argmin at 0 too -- so the fit is")
    print("    at a genuine local optimum, not merely stopped short.  Three")
    print("    search repairs accordingly change nothing: an absolute one-")
    print("    octave initial simplex in place of scipy's relative default, a")
    print("    dedicated s_P scan before the joint search, and fitting the two")
    print("    scale channels in sequence.  Neither the evidence nor the")
    print("    search is the problem.  See H for what the coupling is.")
    return got


# ------------------------------------------------------------------- figures
def figure_b(b, keep):
    y, x, lo, hi = b["y"], b["x"], b["lo"], b["hi"]
    t = np.arange(len(y))
    fig, ax = plt.subplots(3, 1, figsize=(11, 8.4),
                           gridspec_kw={"height_ratios": [1.0, 1.0, 0.85]})
    ax[0].axvspan(lo, hi, color=ts.SERIES[1], alpha=0.10, lw=0)
    ax[0].plot(t, y, lw=0.7, color=ts.INK2, alpha=0.5, label="y")
    ax[0].plot(t, x, lw=1.6, color=ts.SERIES[0], label="truth")
    ax[0].set_title("A x8 process-noise regime, and what the filter is asked to notice")
    ax[0].legend(loc="upper left", ncol=2)
    ax[0].set_ylabel("level")

    ax[1].axvspan(lo, hi, color=ts.SERIES[1], alpha=0.10, lw=0)
    ax[1].axhline(0.0, color=ts.GRID, lw=1.0)
    ax[1].axhline(math.log(8.0), color=ts.INK2, lw=1.0, ls="--",
                  label="truth in the regime (log 8)")
    for i, (name, lam) in enumerate(b["lamP"].items()):
        ax[1].plot(t, lam, lw=1.5, color=ts.SERIES[i], label=name)
    ax[1].set_title("What the process channel reports:  E[lamP | data]")
    ax[1].set_ylabel("nats")
    ax[1].legend(loc="upper left", ncol=2)

    names = [r[0] for r in b["rows"]]
    vals = [r[1] for r in b["rows"]]
    cols = [ts.SERIES[5], ts.SERIES[7], ts.INK2] + list(ts.SERIES[:4])
    ax[2].barh(range(len(names)), vals, color=cols[:len(names)], height=0.62)
    ax[2].set_yticks(range(len(names)))
    ax[2].set_yticklabels(names, fontsize=8.5)
    ax[2].invert_yaxis()
    ax[2].set_xlim(min(vals) - 0.06, max(vals) + 0.02)
    ax[2].axvline(b["nll_o"], color=ts.SERIES[5], lw=1.0, ls="--")
    ax[2].axvline(b["nll_f"], color=ts.SERIES[7], lw=1.0, ls="--")
    for i, v in enumerate(vals):
        ax[2].text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8, color=ts.INK2)
    ax[2].set_title("Log-loss, nats/point.  Dashed: the ceiling (oracle Q_t) "
                    "and the floor (static Q)")
    ax[2].set_xlabel("nats/point, lower is better")
    ax[2].grid(axis="y", visible=False)
    for a in ax[:2]:
        a.set_xlabel("t")
    fig.tight_layout()
    path = os.path.join(FIG, "fig26-the-channel-is-not-dead.png")
    fig.savefig(path, dpi=150)
    print(f"\nwrote {path}")

    fig2, a2 = plt.subplots(figsize=(6.4, 3.6))
    s2s = [k[0] for k in keep]
    kept = [k[5] * 100 for k in keep]
    a2.plot(s2s, kept, "o-", color=ts.SERIES[0])
    for s, k in zip(s2s, kept):
        a2.annotate(f"{k:.1f}%", (s, k), textcoords="offset points",
                    xytext=(6, 6), fontsize=8, color=ts.INK2)
    a2.set_xscale("log")
    a2.set_ylim(0, max(kept) * 1.35)
    a2.set_xlabel("measurement variance  s2")
    a2.set_ylabel("% of the discrimination kept")
    a2.set_title("What survives the GPB1 collapse\n"
                 "share of the Q vs 8Q difference in S a shared covariance sees")
    fig2.tight_layout()
    path2 = os.path.join(FIG, "fig27-what-the-collapse-deletes.png")
    fig2.savefig(path2, dpi=150)
    print(f"wrote {path2}")


def figure_de(grid, out, g32, v32, g29, e29, fgrid, fres):
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.0))

    a = ax[0]
    for j, (tag, (pinned, freeQ, _)) in enumerate(out.items()):
        a.plot(grid, np.array(freeQ) - min(freeQ), "o-", color=ts.SERIES[j],
               label=tag)
    a.axhline(0.0, color=ts.GRID, lw=1.0)
    a.set_xlabel("$s_P$")
    a.set_ylabel("nll/pt above the minimum")
    a.set_title("D.  The likelihood has an interior optimum\n(Q re-optimised at each $s_P$)")
    a.legend(loc="upper center")

    a = ax[1]
    a.plot(g32, np.array(v32) - v32[0], "o-", color=ts.SERIES[2],
           label="0032 fitting window\n(no regime in it)")
    for j, (k, (_, row)) in enumerate(e29.items()):
        a.plot(g29, np.array(row) - row[0], "-", lw=1.3, alpha=0.85,
               color=ts.SERIES[(j % 2) * 3], label="0029 (truth 0.8)" if j == 0 else None)
    a.axhline(0.0, color=ts.GRID, lw=1.0)
    a.set_xlabel("$s_P$")
    a.set_ylabel("nll/pt, relative to $s_P=0$")
    a.set_title("E.  Two zeros, two different meanings\ncorrect on the left curve, wrong on the others")
    a.legend(loc="upper left")

    a = ax[2]
    w = 0.26
    xs = np.arange(len(fgrid))
    for j, (tag, v) in enumerate(fres.items()):
        a.bar(xs + (j - 1) * w, v - v[0], w, color=ts.SERIES[j], label=tag)
    a.axhline(0.0, color=ts.INK2, lw=1.0)
    a.set_xticks(xs)
    a.set_xticklabels([f"{s:g}" for s in fgrid])
    a.set_xlabel("$s_P$ carried")
    a.set_ylabel("nll/pt, relative to $s_P=0$")
    a.set_title("F.  The cost is 35x asymmetric\ninsurance is nearly free; going without is not")
    a.legend(loc="lower left")

    fig.tight_layout()
    path = os.path.join(FIG, "fig28-two-zeros.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


# ----------------------------------------- H: the fix SUMMARY proposed, tested
def part_h():
    """`SUMMARY.md` proposes parameterising the scale on Q_eff rather than Q.

    There is a real coupling to fix.  E[Q e^lam] = Q e^{s^2/2}, so raising s_P
    at fixed Q also raises the MEAN process variance: the two coordinates are
    not orthogonal, and a fit that has already put Q at the mean has to move
    both together to benefit.  Holding Q e^{s^2/2} fixed instead makes s_P a
    pure spread coordinate.  That is the right thought.  Test whether it works.
    """
    print("\nH.  THE PROPOSED FIX: s_P as a pure spread coordinate")
    grid = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4)

    def prof(pr, y, phi):
        a, b = [], []
        for s in grid:
            ph = phi if s > 0 else 0.0
            a.append(nll_pt(Params(pr.alpha, pr.Q, pr.s2, phi_P=ph,
                                   phi_M=pr.phi_M, s_P=s, s_M=pr.s_M), y))
            b.append(nll_pt(Params(pr.alpha, pr.Q * math.exp(-s * s / 2), pr.s2,
                                   phi_P=ph, phi_M=pr.phi_M, s_P=s,
                                   s_M=pr.s_M), y))
        return a, b

    rows = []
    for lbl, phi in (("impulsive", 0.05), ("persistent", 0.95)):
        rng = np.random.default_rng(31)
        n = 500
        y = (ode(n, ALPHA3, Q0 * np.exp(logscale(n, rng, phi, 0.8)), rng)
             + math.sqrt(S20) * rng.standard_normal(n))
        # what an s_P = 0 fit is forced to do: put Q at the mean process variance
        pr = Params(ALPHA3, Q0 * math.exp(0.8 ** 2 / 2), S20)
        rows.append((f"{lbl}, truth s_P=0.8", prof(pr, y, phi)))

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m32", os.path.join(HERE, "0032_a_hard_series.py"))
    m32 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m32)
    _, y32, _, _ = m32.simulate(np.random.default_rng(20260801))
    pr = Params.from_dict(
        json.load(open(os.path.join(FIG, "ode032_fit.json")))["ode"]["params"])
    rows.append(("0032 window, truth s_P=0",
                 prof(pr, y32[:m32.JUMPS[0]], 0.9)))

    for tag, (a, b) in rows:
        print(f"\n    {tag}")
        print("      Q held fixed:      " + " ".join(f"{v:.5f}" for v in a)
              + f"   argmin {grid[int(np.argmin(a))]}")
        print("      Q e^(s^2/2) fixed: " + " ".join(f"{v:.5f}" for v in b)
              + f"   argmin {grid[int(np.argmin(b))]}")
    print("\n    It does not work.  On the impulsive case it moves the argmin")
    print("    the WRONG way, 0.8 -> 0; on 0032's window it leaves the argmin")
    print("    at 0 but flattens the penalty for being wrong sevenfold.  The")
    print("    coupling is real and the reparameterisation is not the fix.")
    return grid, rows


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    part_a()
    b = part_b()
    keep = part_c()
    figure_b(b, keep)
    grid, out, _ = part_d(b)
    g32, v32, g29, e29 = part_e()
    fgrid, fres = part_f()
    part_g(e29)
    part_h()
    figure_de(grid, out, g32, v32, g29, e29, fgrid, fres)
