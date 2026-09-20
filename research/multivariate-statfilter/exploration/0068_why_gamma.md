# 0068 — Why a gamma: the kernel is the closed loop's own impulse response, and a gamma is what a chain of lags does to an impulse

> **AI-generated, not peer-reviewed.** Script: [`0068_why_gamma.py`](0068_why_gamma.py) (the discriminating profile
> from the nominal closed loop, scalar level analytically and the arm's pairs numerically, with gamma fits).
> Theory only; no filter change. Written while the stage-A runs of 0067 were in flight, so nothing here is
> informed by their result.

## 1. The question

0067 weights a member's trailing scale readings by a kernel over the lag with its mode away from 0, and used a
chi-squared in the lag (scale 2, mode set by the degrees of freedom) because it is a one-parameter family in
the mode. *Natural* is not *derived*. What is the kernel, from the filter's own model?

## 2. The derivation: the two attributions are two impulse responses of one loop

Let `A = (I − K H) F` be the closed-loop error dynamics of the nominal filter (steady gain `K`, innovation
covariance `S`). At lag 0 an innovation `e` arrives on a sensor. The two hypotheses:

- **process**: the state really moved by `d` along the disturbed mode `B`. The innovations that follow are the
  loop's response to a state impulse: `ν_l = H A^l B d + noise`.
- **sensor**: the reading lied by `s`; the state did not move; the filter pulled its estimate by `K s`, and
  that pull decays out through the same loop: `ν_l = H A^l K s + noise` (with the sign of an estimate error).

Match them at lag 0 (`d = s = e` — the case that is indistinguishable there, which is 0059's theorem). The two
predicted innovation sequences differ by

    D(l) = H A^l (B + K) e   for l ≥ 1,      D(0) = 0.

The optimal test between two Gaussian hypotheses that differ in their mean sequence is the matched filter:
weight the lag-`l` innovation by `D(l) / S`. The **information** the lag-`l` innovation carries about the
attribution is `D(l)² / S`. So the kernel over lags is not a choice of family: it is the discriminating
impulse response of the closed loop, `|D(l)|` for a linear statistic or `D(l)²` for weighting independent
readings (which is what 0067's reading average is). It is **zero at lag 0 by construction** — the spike step
carries no information about which it was — which is the mechanism 0067 stated as a shape.

**Where the gamma comes from.** For a scalar level `A = 1 − K` and `D(l) = e (1 − K)^{l−1}`: an exponential
that *starts at lag 1*, mode 1, tail `τ = 1/K`. For a mode the sensor reads through `n` stages of the state —
an integrator chain, which is a Jordan block `λ I + N` — `H A^l B ∝ C(l, n−1) λ^{l−n+1} ≈ l^{n−1} λ^l / (n−1)!`,
and that is the **Erlang, the gamma in the lag with integer shape `n` and scale `τ = 1/(1 − λ)`**, mode
`(n − 1) τ`; the information profile `D²` is the gamma with shape `2n − 1` and scale `τ/2`. A gamma is the
impulse response of a chain of first-order lags; the closed loop of a filter on an integrator chain *is* a
chain of first-order lags. That is the why. The chi-squared is the scale-2 member of the family, and scale 2
is not the loop's time constant: 0067's kernel had roughly the right *mode* for a directly-read pair and a
tail chosen by hand.

## 3. Checked against the loops

| loop / pair | `D(l)` | mode | tail |
|---|---|---|---|
| scalar level, `q = 0.02, r = 1` (`K = 0.132`) | exactly `(1 − K)^{l−1}`, `l ≥ 1` | 1 | `τ = 7.6` |
| arm, joint 2: jerk vs its accelerometer | peaks at lag 1, half-height 3 lags after; `D²` fits gamma shape 0.84 | 1 | short: the accelerometer reads `α` directly (`n = 1`) |
| arm, joint 0 (yaw): jerk vs its accelerometer | peaks at lag 1, half-height 7 lags | 1 | longer: the yaw loop is slow |
| arm, joint 2: jerk vs its pot | **peaks at lag 21**, half-height 33 lags after; gamma fit shape 2.1–3.3, mode 20 | ~20 | the pot reads `θ`, integrations away from the jerk (`τ_joint ≈ 33`) |
| arm, joint 0: jerk vs its pot | peaks at lag 1 (the sensor-pull term) then decays over ~180 lags | 1, tail `τ = 141` | the yaw's own slow mode |

The chain prediction holds where the pair is a chain (joint 2's pot: a rising kernel with its mode near the
loop's time constant) and the shifted exponential holds where the sensor reads the mode directly. The arm's
discrete-time input matrix moves `θ`, `ω` and `α` together (`dt³/6, dt²/2, dt`), so its profiles are mixtures
rather than pure Erlangs; the exact `D(l)` is what the filter should use, and the gamma is its closed form for
the pure chain.

## 4. What this means for the construction

- The kernel is **per attribution pair** (disturbed mode, resolving sensor) and comes from `(F, H, K)` — the
  same nominal model every structural construction reads. No shape family is needed in the filter: compute
  `D(l)` from the nominal closed loop and cut where its tail mass is below `ε₀`.
- The mode of the kernel is not a ladder coordinate: it is fixed by the loop (`(n − 1) τ` for a chain). What
  a ladder would run over, if anything, is *which pair* a member is resolving — and that is the confounded-pair
  structure the split ladder already finds.
- 0067's mode-1 chi-squared is the right kernel, up to its tail, for the pairs that matter in the arm's SENSOR
  regime (accelerometer vs jerk, `n = 1`), and the wrong kernel for a pot pair. Whatever stage A shows, its
  next form uses `D(l)`.
- The same `D(l)` is the lag profile the **weights** should use for the *attribution* evidence (stage B): the
  memory ladder's exponential is the profile for a stationary "which cell is right" question, and this is the
  profile for "which attribution" — two different questions with two different derived kernels.
