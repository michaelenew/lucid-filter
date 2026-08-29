"""0001 -- the event model, the elapsed-time maps, and the no-regression audit.

Three things measured here, all of them preconditions for the rest of the ladder:

1. The **propagator** is exact.  A supplied one-step ``F`` is read as the ``a = 1``
   sampling of a fixed generator ``A = log F``; ``F(a) = exp(aA)`` must reproduce the
   known answer on the transitions that actually occur -- including the DEFECTIVE
   constant-velocity block, which has no eigenbasis and which an eigendecomposition
   route silently gets wrong.
2. The **class timescales** carry over: phi^a, forget^a, 1-(1-rho)^a, q_mu * a, Q * a.
   Checked here as a consistency identity: k events of gap a/k must land where one
   event of gap a lands, for the pieces where that is exactly true.
3. **No regression.**  A full row at the nominal step must be BIT-FOR-BIT what it was
   before the change.  Not "close" -- identical, on every rig in the test suite plus
   the dynamics channel.  Anything else means the general path is not a generalisation.

    python research/pointwise-streaming/exploration/0001_the_event_and_the_clock.py
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
from lucid import LucidFilter                                             # noqa: E402
from lucid.statfilter.lucid import _Propagator, _expm, _logm              # noqa: E402

OUT = os.path.join(HERE, "figures", "pw0001.json")
# The filter as it stood before this workstream: the commit this branch was measured
# against.  Pinned rather than "HEAD~1" or "main" so the audit keeps meaning the same
# thing after the merge -- once this work IS main, comparing against main is vacuous.
BASELINE = os.environ.get("PW_BASELINE", "2ecc3d9")


# ------------------------------------------------------------ 1. the propagator
def propagator_exactness():
    rows = []

    def check(name, F, a, exact):
        Fa, _ = _Propagator(np.asarray(F, float)).at(a)
        err = float(np.abs(Fa - np.asarray(exact, float)).max())
        rows.append(dict(case=name, a=a, err=err))
        return err

    # the defective one: a double integrator has a single eigenvector, so F^a is NOT
    # W diag(mu^a) W^-1 -- this is the case that decides the algorithm
    for a in (0.0, 0.25, 1.0, 2.5):
        check("constant-velocity dt=0.1", [[1, 0.1], [0, 1]], a, [[1, 0.1 * a], [0, 1]])
    # a 5-DOF arm's kinematics: the same block, five of them
    Fk = np.kron(np.eye(5), np.array([[1.0, 0.1], [0.0, 1.0]]))
    check("5-DOF arm kinematics", Fk, 0.37,
          np.kron(np.eye(5), np.array([[1.0, 0.037], [0.0, 1.0]])))
    # a rotation: complex eigenvalues, real answer
    th = 0.3
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    check("rotation 0.3 rad", R, 0.5,
          [[math.cos(th / 2), -math.sin(th / 2)], [math.sin(th / 2), math.cos(th / 2)]])
    # integer powers must agree with repeated multiplication
    Fs = np.array([[0.6, 0.2], [0.0, 0.9]])
    check("stable 2x2, a=3", Fs, 3.0, np.linalg.matrix_power(Fs, 3))
    # the semigroup identity F(a/2)^2 = F(a) -- the property that MAKES it a propagator
    p = _Propagator(Fs)
    half, _ = p.at(0.5)
    rows.append(dict(case="semigroup F(1/2)^2 = F(1)", a=0.5,
                     err=float(np.abs(half @ half - Fs).max())))
    # what an eigendecomposition would have done to the defective case, for the record
    Fd = np.array([[1.0, 0.1], [0.0, 1.0]])
    w, W = np.linalg.eig(Fd)
    try:
        naive = (W @ np.diag(w ** 0.5) @ np.linalg.inv(W)).real
        naive_err = float(np.abs(naive - np.array([[1.0, 0.05], [0.0, 1.0]])).max())
    except np.linalg.LinAlgError:
        naive_err = float("inf")
    # dynamics with no real generator must be REFUSED, not silently approximated
    refused = False
    try:
        _Propagator(np.array([[-1.0, 0.0], [0.0, -2.0]])).at(0.5)
    except ValueError:
        refused = True
    return rows, naive_err, refused


# --------------------------------------------------- 2. the class timescale identity
def timescale_identity():
    """A gap of ``a`` split into ``k`` equal sub-gaps must land in the same place, for
    the pieces where that is an exact identity: the transition (semigroup), the process
    accumulation (linear in a), the bank memory (forget^a) and the hazard."""
    F = np.array([[1.0, 0.1], [0.0, 1.0]])
    p = _Propagator(F)
    a, k = 1.7, 5
    Fa, _ = p.at(a)
    Fs, _ = p.at(a / k)
    trans = float(np.abs(np.linalg.matrix_power(Fs, k) - Fa).max())
    forget = abs(0.999 ** a - (0.999 ** (a / k)) ** k)
    rho = 1e-4
    haz = abs(-math.expm1(a * math.log1p(-rho))
              - (1.0 - (1.0 - (-math.expm1((a / k) * math.log1p(-rho)))) ** k))
    proc = abs(a - k * (a / k))
    return dict(transition=trans, forget=forget, hazard=haz, process=proc)


# ------------------------------------------------------------ 3. no regression
def no_regression():
    """Run every rig through the shipped filter and through the filter as it stood
    before this workstream, and require BIT-identity."""
    pw_baseline_cls()                                    # noqa: F841 -- imports it

    r = np.random.default_rng(3)
    T, dt = 200, 0.1
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    B = np.array([[0.0], [dt]])
    rigs = []

    th = np.cumsum(r.standard_normal(T))
    rigs.append(("scalar local level", dict(), (th + r.standard_normal(T))[:, None], None))

    def kin(control=None):
        x = np.zeros(2); Y = np.empty((T, 1))
        U = r.standard_normal((T, 1)) if control is not None else None
        for t in range(T):
            x = F @ x + (B @ U[t] if U is not None else 0.0) + r.standard_normal(2) * 0.05
            Y[t] = H @ x + r.standard_normal(1) * 0.3
        return Y, U

    Y, _ = kin(); rigs.append(("kinematic (H, F)", dict(dynamics=F, H=H), Y, None))
    Y, U = kin(B); rigs.append(("with a control map", dict(dynamics=F, H=H, control=B), Y, U))

    x = 0.0; Ya = np.empty((300, 1))
    for t in range(300):
        x = 0.6 * x + r.standard_normal() * 0.3
        Ya[t] = x + r.standard_normal() * 0.5
    rigs.append(("dynamics=None (learned)", dict(dynamics=None), Ya, None))
    rigs.append(("faults= + a named anchor",
                 dict(dynamics=np.array([[0.9]]), faults=1e-3,
                      anchors=[np.array([[0.6]])]), Ya, None))

    nd = 3
    Fm = np.kron(np.eye(nd), F); Hm = np.kron(np.eye(nd), H)
    x = np.zeros(2 * nd); Y6 = np.empty((T, nd))
    for t in range(T):
        x = Fm @ x + r.standard_normal(2 * nd) * 0.05
        Y6[t] = Hm @ x + r.standard_normal(nd) * 0.3
    rigs.append(("3-DOF, all sensors", dict(dynamics=Fm, H=Hm), Y6, None))
    # H = I pairs every process mode with a sensor, so this exercises the SPLIT LADDER
    # (research/sequence-demix) -- the confounded-pair rungs and their per-axis classes.
    x = np.zeros(2); Y8 = np.empty((T, 2))
    for t in range(T):
        x = F @ x + r.standard_normal(2) * np.array([0.01, 0.05])
        Y8[t] = x + r.standard_normal(2) * np.array([0.30, 0.02])
    rigs.append(("2 sensors, H = I (split ladder)", dict(dynamics=F, H=np.eye(2)), Y8, None))
    Y7 = Y6.copy(); Y7[40:45] = np.nan
    rigs.append(("3-DOF with whole-row gaps", dict(dynamics=Fm, H=Hm), Y7, None))

    out = []
    for name, kw, Y, U in rigs:
        a = pw_baseline_cls()(**kw).filter(Y, U)
        b = LucidFilter(**kw).filter(Y, U)
        worst, where = 0.0, ""
        for fld in ("mean", "var", "innovation", "process_scale", "measurement_scale"):
            u, v = getattr(a, fld), getattr(b, fld)
            if not np.array_equal(np.isnan(u), np.isnan(v)):
                worst, where = float("inf"), fld
                continue
            d_ = np.nanmax(np.abs(u - v)) if np.any(~np.isnan(u)) else 0.0
            if d_ > worst:
                worst, where = float(d_), fld
        out.append(dict(rig=name, worst=worst, field=where,
                        dll=abs(a.loglik - b.loglik)))
    return out


# --------------------------------------------- 4. the one deliberate break
def the_no_information_limit():
    """The all-missing row was the ONE place the change is not bit-identical, and the
    reason is a discontinuity the old code could not see.

    The walk is a scalar Kalman filter on each axis's window centre:

        K_mu = P_mu / (P_mu + 1/info);   P_mu <- (1 - K_mu) P_mu + q_mu

    Take an axis this event carries NO information about -- a sensor that did not read.
    ``info -> 0``, so ``K_mu -> 0`` and the recursion becomes ``P_mu <- P_mu + q_mu``:
    the drift, on its own.  That is not an addition to the walk, it is the walk's own
    limit.  The old code never had to evaluate it, because with a full row every active
    axis always carried information; and on an ALL-missing row it skipped the loop
    entirely and so applied nothing.

    Once partial rows exist that becomes a discontinuity in the sensor count: a sensor
    absent while others report drifts (it takes the ``info -> 0`` branch), the same
    sensor absent while NONE report would not.  The all-missing row has to be the limit
    of the partial row, not a separate case, so the drift is applied on both.

    Measured here: (i) the limit identity, numerically -- the old recursion at zero
    information IS ``P_mu + q_mu``; (ii) what applying it on the all-missing path costs,
    which is the honest price of removing the discontinuity.
    """
    f = LucidFilter()
    e = f._members[0]
    e.update(np.array([0.4]))                       # give it a state to walk from
    Pmu0 = e._Pmu.copy()
    k = 0
    info = 1e-4                                     # the engine's Fisher stabiliser: info -> 0
    K = Pmu0[k] / (Pmu0[k] + 1.0 / info)
    old_recursion = (1.0 - K) * Pmu0[k] + e._qmu[k]
    pure_drift = Pmu0[k] + e._qmu[k]
    return dict(K_mu_at_zero_info=float(K),
                old_recursion=float(old_recursion), pure_drift=float(pure_drift),
                rel_gap=float(abs(old_recursion - pure_drift) / pure_drift))


def the_price_of_the_limit(seeds=12):
    """What applying the drift on the all-missing path costs, over a blackout across
    which the sensor scale MOVES -- the case where the two differ at all.

    PREDICTION (recorded before the run): free when nothing changed across the gap;
    some cost when it did, because coming out of a blackout at the walk's cap spends the
    first reading on one large clipped step and collapses ``P_mu`` behind it.
    """
    T, gap0, win = 320, 120, 60
    out = {}
    for gap in (5, 40):
        for jump in (1.0, 10.0):
            acc = {"new": [], "old": []}
            for sd in range(seeds):
                r = np.random.default_rng(4000 + sd)
                th = np.cumsum(r.standard_normal(T) * 0.3)
                sig = np.where(np.arange(T) < gap0, 0.5, 0.5 * jump)
                Y = (th + r.standard_normal(T) * sig)[:, None]
                Y[gap0:gap0 + gap] = np.nan
                lo, hi = gap0 + gap, gap0 + gap + win
                for tag, cls in (("new", LucidFilter), ("old", pw_baseline_cls())):
                    o = cls(measurement=np.array([0.25])).filter(Y)
                    acc[tag].append(float(np.sqrt(np.mean(
                        (o.mean[lo:hi, 0] - th[lo:hi]) ** 2))))
            out[f"gap{gap}_x{jump:g}"] = dict(
                new=float(np.mean(acc["new"])), old=float(np.mean(acc["old"])),
                se=float(np.std(np.array(acc["new"]) - np.array(acc["old"]))
                         / math.sqrt(seeds)))
    return out


_BASE = {}


def pw_baseline_cls():
    if "cls" not in _BASE:
        old_src = subprocess.check_output(
            ["git", "-C", ROOT, "show", f"{BASELINE}:lucid/statfilter/lucid.py"]).decode()
        d = tempfile.mkdtemp()
        open(os.path.join(d, "pw_baseline.py"), "w").write(old_src)
        sys.path.insert(0, d)
        import pw_baseline
        _BASE["cls"] = pw_baseline.LucidFilter
    return _BASE["cls"]


if __name__ == "__main__":
    print("=" * 78)
    print("1. THE PROPAGATOR -- F(a) = exp(a log F) against the known answer")
    print("=" * 78)
    rows, naive_err, refused = propagator_exactness()
    for row in rows:
        print(f"   {row['case']:<32s} a={row['a']:<5.2f} |err| = {row['err']:.2e}")
    print(f"\n   an eigendecomposition on the defective block: |err| = {naive_err:.2e}"
          "   <- why not that")
    print(f"   dynamics with no real generator refused: {refused}")

    print()
    print("=" * 78)
    print("2. THE CLASS TIMESCALES -- one gap of a == k gaps of a/k")
    print("=" * 78)
    ident = timescale_identity()
    for k, v in ident.items():
        print(f"   {k:<12s} |err| = {v:.2e}")

    print()
    print("=" * 78)
    print("3. NO REGRESSION -- a full row at the nominal step, against the filter")
    print(f"   as it stood before this workstream ({BASELINE}).  Required: bit-for-bit.")
    print("=" * 78)
    reg = no_regression()
    for row in reg:
        tag = "IDENTICAL" if row["worst"] == 0.0 and row["dll"] == 0.0 else "DIFFERS"
        print(f"   {row['rig']:<28s} {tag:>10s}   worst |d| = {row['worst']:.3e}"
              f"   d(loglik) = {row['dll']:.3e}")

    print()
    print("=" * 78)
    print("4. THE ONE DELIBERATE BREAK -- the all-missing row, and why it is not special")
    print("=" * 78)
    lim = the_no_information_limit()
    print(f"   the walk's gain at zero information:  K_mu = {lim['K_mu_at_zero_info']:.2e}")
    print(f"   so its recursion there reads          P_mu -> {lim['old_recursion']:.8f}")
    print(f"   and the pure drift reads              P_mu -> {lim['pure_drift']:.8f}"
          f"   ({lim['rel_gap']:.1e} apart)")
    print("   -- the drift IS the walk's own no-information limit, so a sensor absent")
    print("      while others report already takes it.  Applying it when NONE report")
    print("      removes the discontinuity in the sensor count.  What that costs:")
    pr = the_price_of_the_limit()
    print()
    print(f"   {'blackout':>22}   {'with the limit':>15} {'without (old)':>15}   {'d (se)':>12}")
    for key, v in pr.items():
        gap, jump = key[3:].split("_")
        lab = f"{gap} steps, sensor x{jump[1:]}"
        print(f"   {lab:>22}   {v['new']:>15.4f} {v['old']:>15.4f}"
              f"   {v['new'] - v['old']:+8.4f} ({v['se']:.4f})")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(propagator=rows, naive_eig_err=naive_err, refused=refused,
                   timescales=ident, regression=reg,
                   no_info_limit=lim, price=pr),
              open(OUT, "w"), indent=1)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")
