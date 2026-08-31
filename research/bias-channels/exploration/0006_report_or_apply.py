"""0006 -- the sensor entry: estimable, and not usable.  Both ways, measured.

`0004` established that a sensor bias is identifiable only up to the gauge -- the common mode
of the biases is not in the data at any `m`, so one biased sensor out of `m` leaves an
irreducible `b/m` in the state.  That left a narrower question, and it is the one that decides
what ships: the channel can ESTIMATE the sensor entry accurately (0004: 1.93-1.96 against a
truth of 2.00 at every m).  Can it USE it?

There are exactly two ways, and this probe measures both against the do-nothing baseline.

  APPLY IT (full feedback).  Correct `y` by the estimate before the members see it.  The
  estimate is only defined up to the gauge, so this silently adopts the quotient's own
  convention -- "the offsets average to zero" -- and puts the state at `theta + b/m`.  Measured:
  WORSE than doing nothing, because doing nothing is not nothing.  The scale walk inflates the
  biased sensor's `eta` and down-weights it, which is a partial repair, and the convention
  throws it away.

  ESTIMATE BUT DO NOT APPLY IT (partial feedback).  Leave `y` alone, report the read-out, and
  feed back only the process entry.  Measured: the persistent innovation the bias leaves behind
  loads onto the process entry, WHICH IS applied -- a spurious drift that grows without bound
  and doubles the state error.

So the sensor entry is dropped from the shipped channel, and `_mean_basis(process_only=True)`
is that decision in one line of algebra.  The read-out itself is real and is preserved here;
what it costs to act on it is the finding.

Run: python3 0006_report_or_apply.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from lucid import LucidFilter                                    # noqa: E402
from lucid.statfilter.lucid import (_MeanChannel, _logsumexp,     # noqa: E402
                                    _mean_basis)

Q_TRUE, R_TRUE, N, T0, BIAS, SEEDS = 0.02, 1.0, 700, 300, 2.0, (7, 8, 9, 10)


def rig(m, seed):
    rng = np.random.default_rng(seed)
    theta = np.cumsum(rng.normal(0, np.sqrt(Q_TRUE), N))
    Y = np.stack([theta + rng.normal(0, np.sqrt(R_TRUE), N) for _ in range(m)], axis=1)
    Y[T0:, m - 1] += BIAS
    return theta, Y


class _Variant(_MeanChannel):
    """The shipped channel with the sensor entry restored, in one of the two usable ways.

    Subclassed rather than flagged, so the shipped filter carries no knob that exists only for
    a probe.  ``apply`` chooses between the two: correct ``y`` by the estimate (and score each
    rung on what it alone leaves behind), or leave ``y`` alone and carry the sensor entry in the
    residual instead.  Everything else is the shipped recursion verbatim.
    """

    def __init__(self, *a, apply_sensor=True, **kw):
        super().__init__(*a, **kw)
        self.apply_sensor = apply_sensor

    @property
    def measurement_offset(self):
        return self.C @ self.bbar if self.apply_sensor else np.zeros(self.C.shape[0])

    def step(self, e, S, K, ok=True):
        Vp = self.F @ self.V + self.D
        U = self.H @ Vp + self.C
        self.Pb = self.Pb + self.q[:, :, None] * np.eye(self.k)
        if ok:
            Sb = np.einsum("ia,jab,kb->jik", U, self.Pb, U) + S
            Sbi = np.linalg.inv(Sb)
            r = e - np.einsum("ia,ja->ji", U, self.b - self.bbar)
            if not self.apply_sensor:
                r = r - (self.C @ self.bbar)
            _, ld = np.linalg.slogdet(Sb)
            ll = -0.5 * (ld + np.einsum("ji,jik,jk->j", r, Sbi, r))
            Kb = np.einsum("jab,ib,jik->jak", self.Pb, U, Sbi)
            self.b = self.b + np.einsum("jak,jk->ja", Kb, r)
            self.Pb = self.Pb - np.einsum("jam,mb,jbc->jac", Kb, U, self.Pb)
            self.Pb = 0.5 * (self.Pb + np.swapaxes(self.Pb, 1, 2))
            self.logw = self.forget * (self.logw - _logsumexp(self.logw)) + ll
            self.V = Vp - K @ U
        else:
            self.V = Vp
        w = np.exp(self.logw - _logsumexp(self.logw))
        self.bbar = w @ self.b
        db = self.b - self.bbar
        Pmix = np.einsum("j,jab->ab", w, self.Pb) + np.einsum("j,ja,jb->ab", w, db, db)
        return self.V @ Pmix @ self.V.T


def build(m, mode):
    H, R0 = np.ones((m, 1)), np.ones(m)
    f = LucidFilter(H=H, measurement=R0, offsets=(mode != "off"))
    if mode in ("apply", "estimate"):
        B = _mean_basis(np.eye(1), H, process_only=False)
        f._mean = _Variant(B, 1, np.eye(1), H, np.eye(1), R0, 1e-4, 1000.0,
                           apply_sensor=(mode == "apply"))
        M = len(f._members)
        f._Kb = np.zeros((M, 1, m))
        f._Sb = np.zeros((M, m, m))
    return f


def score(m, mode):
    """`mode` in {'off', 'shipped', 'apply', 'estimate'} -- the four filters compared."""
    acc = np.zeros(4)
    for seed in SEEDS:
        theta, Y = rig(m, seed)
        f = build(m, mode)
        r = f.filter(Y)
        e = r.mean[400:, 0] - theta[400:]
        acc[0] += np.sqrt(np.mean(e ** 2)) / len(SEEDS)
        acc[1] += np.mean(e ** 2 / r.var[400:, 0, 0]) / len(SEEDS)
        acc[2] += (r.offset[-1, 0] if r.offset is not None else 0.0) / len(SEEDS)
        if f._mean is not None and f._mean.C.any():
            c = f._mean.C @ f._mean.bbar
            acc[3] += (c - c.mean())[-1] / len(SEEDS)
    return acc


def main():
    print("=" * 88)
    print(f"One level, m sensors, the last one {BIAS} sigma off from t = {T0}")
    print("=" * 88)
    print(f"{'m':>2} | {'off':>14} | {'shipped':>14} | {'APPLY it':>14} | {'ESTIMATE only':>14}")
    print(f"{'':>2} | {'rmse  calib':>14} | {'rmse  calib':>14} | {'rmse  calib':>14} | "
          f"{'rmse  calib':>14}")
    rows = {}
    for m in (2, 3, 5):
        rows[m] = {mode: score(m, mode) for mode in ("off", "shipped", "apply", "estimate")}
        r = rows[m]
        print(f"{m:2d} | {r['off'][0]:7.3f} {r['off'][1]:6.2f} | "
              f"{r['shipped'][0]:7.3f} {r['shipped'][1]:6.2f} | "
              f"{r['apply'][0]:7.3f} {r['apply'][1]:6.2f} | "
              f"{r['estimate'][0]:7.3f} {r['estimate'][1]:6.2f}")

    print()
    print("Ratios against doing nothing, and the drift each variant invents:")
    print(f"{'m':>2} | {'shipped':>9} | {'APPLY':>9} | {'ESTIMATE':>9} | "
          f"{'drift: shipped / apply / estimate':>34}")
    for m in (2, 3, 5):
        r = rows[m]
        print(f"{m:2d} | {r['shipped'][0]/r['off'][0]:8.3f}x | "
              f"{r['apply'][0]/r['off'][0]:8.3f}x | {r['estimate'][0]/r['off'][0]:8.3f}x | "
              f"{r['shipped'][2]:+10.4f} {r['apply'][2]:+10.4f} {r['estimate'][2]:+10.4f}")

    print()
    print("The read-out itself is accurate in both variants that carry it (0004 again):")
    for m in (2, 3, 5):
        want = BIAS * (m - 1) / m
        print(f"  m = {m}: last sensor vs the rest -> apply {rows[m]['apply'][3]:+.2f}, "
              f"estimate {rows[m]['estimate'][3]:+.2f}   (truth {want:+.2f})")

    print()
    print("So: estimable, and not usable.  APPLY adopts the gauge convention and loses the")
    print("scale walk's partial repair; ESTIMATE leaves a persistent innovation that loads")
    print("onto the process entry, which IS applied.  The shipped channel carries neither.")


if __name__ == "__main__":
    main()
