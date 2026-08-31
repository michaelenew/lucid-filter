"""0013 -- how fast is a bias detected, against the rate an oracle could manage?

Everything in 0001-0012 measures the ESTIMATE -- its accuracy, and what it does to state error.
Nothing measures the RATE, and that is the instrument the rest of this repository reaches for:
`dynamics-learning` prices a fault detection at `log(1/rho) / KL`, `ode-filter` 0041 puts the
temporal confirmation delay on the Lorden frontier, and the offset channel there reports trust
as a directed-information slope.  This probe puts the bias channels on the same footing.

THE MEASUREMENT.  Prequential log-odds against a matched null, exactly as `ode-filter` 0046
defines trust -- never a moment taken from the future:

    Lambda_t = sum_{s<=t} [ log p(y_s | past, OFFSET model) - log p(y_s | past, NO offset) ]

and the ORACLE is the same quantity with the offset known rather than estimated.  Its slope is
by construction the per-step KL between the two models, so it IS the frontier: the fastest any
detector could accrue evidence on this data.  Two numbers come out of the comparison --

    rate ratio  = achieved slope / oracle slope,  what estimation costs against knowing
    latency     = steps to 99:1 (log-odds 4.6), achieved against oracle

-- and the second is the one a caller feels.

Both cells are measured, because "bias" names two different things in this workstream: a
per-sensor bias (the READ-OUT, whose evidence is the observer's own, since it never feeds back
and so contributes nothing to the filter's likelihood) and a process offset (the DRIFT, whose
evidence is the filter's own likelihood, since it does).

The oracle likelihood is obtained by the shift, not by a second model: a constant sensor bias
`c` has `log p(y | c) = log p(y - c | 0)`, and a drift `d` has `log p(y | d) = log p(y - t d | 0)`,
so the oracle is the shipped filter run on data the offset has been taken out of.

Run: python3 0013_the_detection_rate.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from lucid import LucidFilter                                    # noqa: E402
from lucid.statfilter.lucid import _MeanChannel, _logsumexp      # noqa: E402

Q_TRUE, R_TRUE, N, T0, SEEDS = 0.02, 1.0, 1500, 300, (11, 12, 13, 14)
CONFIRM = np.log(99.0)                                           # the repository's 99:1


class _Traced(_MeanChannel):
    """The shipped observer, plus the prequential log-odds it implies against the null.

    Subclassed rather than instrumented in place: the accumulator is a measurement, not part of
    the mechanism.  The two densities differ only in whether the offset is modelled, so the
    `m log 2pi` they share is dropped from both.
    """

    def reset(self):
        super().reset()
        self.lam = 0.0
        self.kl = 0.0
        self.truth = None

    def step(self, e, S, K, ok=True):
        if ok and self.truth is not None:
            # The LOCAL frontier: the KL this step would carry for an observer told the truth,
            # evaluated at THIS filter's own `S` rather than at the oracle's.  The two differ by
            # exactly what the noise walk has done in response to the bias, so the pair
            # separates "the estimator is slow" from "the evidence has been taken away".
            g = np.linalg.lstsq(self.C, self.truth, rcond=None)[0]
            u = self._U @ g
            self.kl += 0.5 * float(u @ np.linalg.solve(S, u))
        Vp, U = self._Vp, self._U
        if ok:
            Sb = np.einsum("ia,jab,kb->jik", U, self.Pb, U) + S
            Sbi = np.linalg.inv(Sb)
            base = self.bbar if self.feedback else np.zeros(self.k)
            r = e - np.einsum("ia,ja->ji", U, self.b - base) - (self.C @ base)
            _, ld = np.linalg.slogdet(Sb)
            ll = -0.5 * (ld + np.einsum("ji,jik,jk->j", r, Sbi, r))
            w = self.logw - _logsumexp(self.logw)
            mix = _logsumexp(w + ll)                             # the offset model's density
            null = -0.5 * (np.linalg.slogdet(S)[1] + float(e @ np.linalg.solve(S, e)))
            self.lam += float(mix - null)
        return super().step(e, S, K, ok)


def latency(lam, t0=T0):
    """Steps after onset until the log-odds first and durably clear 99:1."""
    post = lam[t0:] - lam[t0]
    hit = np.flatnonzero(post >= CONFIRM)
    if not hit.size:
        return float("nan")
    for h in hit:                                                # first crossing that holds
        if np.all(post[h:] >= 0.5 * CONFIRM):
            return float(h)
    return float(hit[-1])


def slope(lam, lo, hi):
    return (lam[hi] - lam[lo]) / (hi - lo)


# ------------------------------------------------------------------ the sensor-bias cell
def sensor_cell(m, bias):
    H, R0 = np.ones((m, 1)), np.ones(m)
    step = np.zeros((N, m))
    step[T0:, m - 1] = bias
    ach_lam = np.zeros(N)
    ora_lam = np.zeros(N)
    loc_lam = np.zeros(N)
    etas = np.zeros(m)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        theta = np.cumsum(rng.normal(0, np.sqrt(Q_TRUE), N))
        Y = np.stack([theta + rng.normal(0, np.sqrt(R_TRUE), N) for _ in range(m)], axis=1)
        Y = Y + step

        f = LucidFilter(H=H, measurement=R0, offsets=True)
        f.reset()
        so = f._sensor
        tr = _Traced(np.vstack([so.D, so.C]), so.n, so.F, so.H, np.eye(so.n), R0,
                     1e-4, 1000.0, feedback=False)
        f._sensor = tr
        lam, kl = np.empty(N), np.empty(N)
        eta = np.zeros(m)
        for t, y in enumerate(Y):
            tr.truth = step[t]
            st = f.update(y)
            lam[t], kl[t] = tr.lam, tr.kl
            if t >= N - 400:
                eta += st.measurement_scale / 400
        ach_lam += lam / len(SEEDS)
        loc_lam += kl / len(SEEDS)
        etas += eta / len(SEEDS)

        # the oracle: the same filter on data the bias has been taken out of
        null = per_step(Y, H=H, measurement=R0)
        told = per_step(Y - step, H=H, measurement=R0)
        ora_lam += np.cumsum(told - null) / len(SEEDS)
    return ach_lam, ora_lam, loc_lam, etas


def per_step(Y, **kw):
    """The streaming per-step predictive log-density -- `filter` keeps only the total."""
    f = LucidFilter(**kw)
    f.reset()
    return np.array([f.update(y).loglik for y in Y])


# ------------------------------------------------------------------ the drift cell
def drift_cell(rate):
    ramp = rate * np.maximum(np.arange(N) - T0 + 1, 0)
    ach_lam = np.zeros(N)
    ora_lam = np.zeros(N)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        theta = np.cumsum(rng.normal(0, np.sqrt(Q_TRUE), N) + rate * (np.arange(N) >= T0))
        Y = (theta + rng.normal(0, np.sqrt(R_TRUE), N))[:, None]
        on = per_step(Y, offsets=True)
        null = per_step(Y)
        told = per_step(Y - ramp[:, None])
        ach_lam += np.cumsum(on - null) / len(SEEDS)
        ora_lam += np.cumsum(told - null) / len(SEEDS)
    return ach_lam, ora_lam


def report(rows):
    """Two windows, because they answer different questions.

    The WHOLE window mixes the rate with the convergence to it; the TAIL is the rate once the
    estimate has arrived.  If the tail ratio goes to 1 and the whole-window one does not, the
    channel is at the frontier and merely gets there late -- which is a different complaint,
    and a different fix, from being slow forever.
    """
    print(f"{'':<24} | {'evidence, nats/step':>34} | {'steps to 99:1':>17}")
    print(f"{'':<24} | {'oracle':>8} {'achieved':>9} {'ratio':>6} {'tail':>8} | "
          f"{'oracle':>7} {'achieved':>8}")
    for label, ach, ora in rows:
        sa, so = slope(ach, T0 + 20, N - 1), slope(ora, T0 + 20, N - 1)
        ta, to = slope(ach, N - 400, N - 1), slope(ora, N - 400, N - 1)
        la, lo = latency(ach), latency(ora)
        print(f"{label:<24} | {so:8.4f} {sa:9.4f} {sa / so:6.2f} {ta / to:8.2f} | "
              f"{lo:7.0f} {la:8.0f}")


def main():
    print("=" * 88)
    print("THE SENSOR-BIAS CELL -- the read-out's own evidence (it never feeds back)")
    print("=" * 88)
    sens = []
    for m in (3, 5):
        for bias in (1.0, 2.0):
            sens.append((f"m = {m}, bias {bias:.0f} sigma", *sensor_cell(m, bias)))
    report([(lab, a, o) for lab, a, o, _l, _e in sens])

    print()
    print("=" * 88)
    print("...and where the rest of it goes: the filter's own defence eats the evidence")
    print("=" * 88)
    print(f"{'case':<22} | {'oracle':>7} {'local':>7} {'achieved':>9} | "
          f"{'ach/oracle':>10} {'ach/local':>10} | eta on the biased sensor")
    for label, ach, ora, loc, eta in sens:
        sa = slope(ach, N - 400, N - 1)
        so = slope(ora, N - 400, N - 1)
        sl = slope(loc, N - 400, N - 1)
        print(f"{label:<22} | {so:7.4f} {sl:7.4f} {sa:9.4f} | {sa / so:10.2f} "
              f"{sa / sl:10.2f} | {eta[-1]:+.2f}  (the others {float(np.mean(eta[:-1])):+.2f})")
    print()
    print("  `local` is the same frontier evaluated at the FILTER'S OWN S rather than the")
    print("  oracle's.  The observer sits at 0.90-0.99 of it, so it is not slow -- the missing")
    print("  evidence has been taken away, by the scale walk inflating the biased sensor's eta")
    print("  and down-weighting it.  Defending the state and naming the culprit are in tension.")

    print()
    print("=" * 88)
    print("THE DRIFT CELL -- the filter's own evidence (this one does feed back)")
    print("=" * 88)
    report([(f"drift {r:.2f}", *drift_cell(r)) for r in (0.05, 0.14, 0.42)])
    print()
    print("  the oracle's slope IS the per-step KL between the two models, so it is the fastest")
    print("  any detector could accrue evidence here.  `ratio` is what estimation costs over the")
    print("  whole run; `tail` is what it costs once the estimate has arrived.")


if __name__ == "__main__":
    main()
