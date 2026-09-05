# 0001 — the event and the clock: the maps, the identities, and the no-regression audit

`0001_the_event_and_the_clock.py`. Before anything is claimed about partial rows or
streams, three preconditions have to hold: the elapsed-time transition is *exact*, the
class timescales compose, and a synchronous full row at the nominal step is **bit-for-bit**
the filter that existed before this workstream. All three were recorded as requirements
before the run.

## 1. The propagator

> **⚖️ ATTRIBUTION —** _Exact elapsed-time transition and forcing via the augmented matrix exponential, computed by scaling-and-squaring / matrix logarithm rather than eigendecomposition (which fails on the defective constant-velocity block). All textbook continuous-discrete / numerical-linear-algebra results, verified to machine precision here._ Prior art: Van Loan 1978 (integrals of the matrix exponential); Moler & Van Loan (matrix-exponential computation); continuous-discrete Kalman — Jazwinski 1970. Status: REPRODUCTION.

A supplied `F` is the `a = 1` sampling of a fixed generator `A = log F`; over an elapsed
`a` nominal steps the transition is `exp(aA)`, read off one exponential of
`a·[[A, I], [0, 0]]` together with the forcing integral `Phi(a) = ∫₀^a exp(Aτ)dτ`.
Numpy-only: scaling-and-squaring for `exp`, Denman–Beavers for the square root, inverse
scaling-and-squaring for `log`.

| case | `a` | \|err\| |
|---|---|---|
| constant-velocity `dt = 0.1` | 0, 0.25, 1, 2.5 | **0** (exact) |
| 5-DOF arm kinematics | 0.37 | **0** (exact) |
| rotation 0.3 rad | 0.5 | 1.1e-16 |
| stable 2×2 vs `matrix_power` | 3 | 8.9e-16 |
| semigroup `F(½)² = F(1)` | 0.5 | 3.3e-16 |

**This is why it is not an eigendecomposition.** The constant-velocity block
`[[1, dt], [0, 1]]` — the single most common non-trivial transition there is, and the one
the repo's own 5-DOF arm is built from — is **defective**: two states, one eigenvector.
`W diag(µ^a) W⁻¹` on it is off by **5.0e-2** at `a = 0.5`, silently, with no warning from
the linear algebra. The inverse-scaling-and-squaring route is exact on it to the last bit.

Dynamics with no real generator (a negative real eigenvalue — no continuous-time system
samples to that) are **refused with an error naming the fix**, not approximated. The
factorisation is lazy, so a filter that is never handed a non-nominal gap never computes a
logarithm and never sees the error.

## 2. The class timescales compose

One gap of `a` must equal `k` gaps of `a/k`. Measured: transition 2.8e-17, `forget**a`
3.3e-16, hazard `1-(1-ρ)^a` 5.1e-17, process accumulation 2.2e-16 — all at machine
precision, which is the statement that these are *rates* and not per-arrival constants.

## 3. No regression

Every rig, the shipped filter against the filter as it stood at the parent commit:

| rig | result |
|---|---|
| scalar local level | **identical**, worst \|d\| = 0, Δloglik = 0 |
| kinematic (`H`, `F`) | **identical** |
| with a control map | **identical** |
| `dynamics=None` (learned) | **identical** |
| `faults=` + a named anchor | **identical** |
| 3-DOF, all sensors | **identical** |
| 3-DOF with whole-row gaps | **differs** — deliberately, §4 |

Bit-for-bit, not "close": the general path is a generalisation, and every reduction to the
old case — no clock, full rows, `a = 1` — returns the same floats. (`a` arrives as exactly
`1.0` when it should: `forget**1.0`, `phi**1.0` and `q_mu*1.0` are exact in IEEE, and the
hazard kernel and the AR(1) kernel short-circuit at `a = 1` rather than recomputing `ρ`
through `expm1(log1p(·))`.)

## 4. The one deliberate break: the all-missing row is not a special case

> **⚖️ ATTRIBUTION —** _The "no-information" limit — an unobserved axis gets predict-only ($P\leftarrow P+q$, the $K\to0$ limit of the update) — is just the Kalman predict step with no measurement, applied uniformly so a partly-observed instant is not a discontinuity in the sensor count. Standard; the measured cost/benefit numbers are the original content._ Prior art: Kalman prediction under missing measurements — standard (Jazwinski 1970). Status: REPRODUCTION.

The walk is a scalar Kalman filter on each axis's window centre:

    K_mu = P_mu / (P_mu + 1/info);    P_mu <- (1 - K_mu) P_mu + q_mu

Take an axis this event carries no information about — a sensor that did not read.
`info → 0`, so `K_mu → 0` and the recursion **is** `P_mu ← P_mu + q_mu`: the drift, on its
own. Measured at the engine's Fisher stabiliser: `K_mu = 1.1e-05`, the recursion gives
`0.12990585`, the pure drift `0.12990706` — **9.3e-06 apart**. The drift is not an addition
to the walk; it is the walk's own no-information limit.

The old code never had to evaluate that limit. With a full row every active axis always
carried information, and on an **all**-missing row it skipped the loop entirely and so
applied nothing. Once partial rows exist that becomes a discontinuity **in the sensor
count**: a sensor absent while others report takes the `info → 0` branch and drifts; the
same sensor absent while *none* report would not. The all-missing row has to be the limit
of the partial row, so the drift is applied on both.

What that costs, over a blackout across which the sensor scale moves (12 seeds):

| blackout | with the limit | without (old) | Δ (se) |
|---|---|---|---|
| 5 steps, scale unchanged | 0.3203 | 0.3203 | −0.0000 (0.0000) |
| 5 steps, sensor ×10 | 2.8104 | 2.8531 | **−0.0427** (0.0135) |
| 40 steps, scale unchanged | 0.3502 | 0.3504 | −0.0002 (0.0001) |
| 40 steps, sensor ×10 | 2.8997 | 3.0537 | **−0.1541** (0.0402) |

**Free when the scale did not move across the gap, and 1.5% / 5.0% BETTER when it moved
during the blackout — 3.2σ and 3.8σ.** So on the current engine the honest limit is also
the cheaper one, and the discontinuity argument does not have to carry the decision alone.

> ~~**Superseded.** On the engine as it stood before the sequence-demix split ladder
> (`05efef1`) this table read the other way: +0.0607 (0.0077) and +0.1973 (0.0228) — the
> drift measured 1.9% / 5.9% **worse** at 8σ, and was kept anyway, as a recorded cost paid
> to remove the discontinuity. The mechanism named then was that coming out of a blackout
> at the walk's cap spends the first reading on one large clipped step and collapses `P_mu`
> behind it. The split ladder retired that: with the process/sensor attribution carried in
> the bank rather than settled by a per-axis Newton step, one noisy step out of a blackout
> costs much less, and the wider window makes the honest prior the better one. The claim
> that changed is the sign of the cost; the reason for the choice did not.~~

The alternative — freezing the walk's confidence for exactly as long as nothing is heard —
is the same latched-freeze error
[`dynamics-learning/0003`](../../dynamics-learning/exploration/0003_excitation_honesty.md)
measured at 20× one level up. The cap that the drift saturates against over a long gap is
the window-localisation bound `(3s)²`, not the scale's stationary variance `s²`; whether
the *no-information* drift should saturate at the latter is still an open, and this table
is what it would have to beat.
