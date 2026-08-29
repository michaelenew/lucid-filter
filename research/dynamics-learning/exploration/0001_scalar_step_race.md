# 0001 — scalar dynamics step: the hazard bank sits on the derived frontier; the random-walk surrogate's whole trade curve is dominated

First rung of the ladder (`0001_scalar_step_race.py`): `x_t = a x_{t-1} + b u_t + w_t`,
`y_t = x_t + v_t`, `a: 0.9 → 0.6` at t*=1500 of T=3000, `q=0.09, r=0.25, b=1`,
`u ~ N(0, su²)`, all dynamics passed to every contender as callables `(m, u) → (F, B)`.
200 seeds for the race, 100 per side panel.  No tuned numbers anywhere: the one labeled
prior is the fault hazard `rho = 1/T` ("about one fault per mission"), and the
augEKF/regwalk drift is `q_a = δ_class² · rho` (the class's matched random-walk rate).

## 1. The frontier, derived and verified

The Bayes detector for the jump class is the hazard-mixed bank (Shiryaev).  Its delay
budget is the launch-to-crossing distance in log-odds, paid at the per-step KL rate
between the two hypothesis filters' predictive densities.  For this rig that rate is
exact: the joint (state, filter-error) pair is linear, so a 2×2 covariance recursion
gives `E[e²]` for each filter and

`KL = ½ [ log(S0/S1) + E[e0²]/S0 − E[e1²]/S1 ]`.

Three checks, all passing:

- **closed form = Monte Carlo**: KL_post 0.4886 derived vs 0.4895 ± 0.0230 measured.
- **transient matters**: the pre-change regime (a=0.9) is more excited than post
  (a=0.6) — `Σxx = (b²su²+q)/(1−a²)` is 5.74 vs 1.70 — so early post-change steps carry
  extra information.  The frontier must iterate the covariance recursion from the
  pre-change stationary point: D* = 15 transient-aware vs 16.4 stationary.
- **optional-stopping audit**: along the measured detection paths, the accumulated llr
  equals the theoretical cumulative KL at the (random) stopping time — su=0.25: 4.12 vs
  4.00; su=0.5: 5.12 vs 5.15; su=1.0: 6.95 vs 7.26 nats.  The bank detects with exactly
  the information the KL accounting says exists, no more.  (At su=2.0 the audit reads
  10.25 vs 13.34 — τ is only ~4 steps there, and the small-τ statistic is skewed by
  discreteness and stopping-time selection; flagged, not resolved.)

Where the measured delay sits *below* the naive `log(1/rho)/KL`, every nat is accounted
for: the pre-change log-odds equilibrium sits **above** the mixing floor `log rho` when
KL_pre is small (launch distance |L_pre| < log(1/rho), paid for in false-alarm rate),
and the soft mixing floor contributes an upward drift near the floor (the "mixing
bonus", 1.1–2.6 nats).  Both are derived objects of the rho commitment, not knobs.

| su   | rho-frontier | measured launch L_pre | Wald-from-launch | measured delay | FA pre-t* |
|------|------|--------|------|--------------|-------|
| 0.25 | 118  | −5.09  | 74.5 | 56.7 ± 2.6   | 37%   |
| 0.50 | 51   | −6.12  | 38.8 | 30.9 ± 2.0   | 29%   |
| 1.00 | 15   | −8.20  | 15.0 | 11.7 ± 0.8   | 16%   |
| 2.00 | 3    | −13.32 | 5.9  | 4.3 ± 0.3    | 7%    |

The frontier is excitation-dependent exactly as the SUMMARY requires (delay ∝ 1/KL,
KL ∝ excitation at leading order), and the bank rides it at every level.

## 2. The race (su = 1.0, 200 seeds)

| mechanism | delay | pre-t* false | calm | RMSE/oracle [0,25) | [25,100) | [100,400) | [400,1500) |
|---|---|---|---|---|---|---|---|
| bank2   | **13.5 ± 0.6** | 21.5% | 1.000 | 1.309 | 1.008 | 1.001 | 1.001 |
| augEKF  | 49.1 ± 1.0 | 0% | 1.013 | 1.449 | 1.164 | 1.013 | 1.008 |
| regwalk | 49.2 ± 1.1 | 0% | 1.015 | 1.497 | 1.184 | 1.014 | 1.009 |
| frozen  | —          | —  | 1.000 | 1.635 | 1.688 | 1.683 | 1.683 |

- **The bank's false alarms are the design point and they cost nothing.**  The
  pre-change crossing rate ≈ 0.5·rho per step — the FA side of the rho operating point,
  as Shiryaev says it must be.  On no-change runs the bank's full-run RMSE ratio is
  **1.0004** with 33% of seeds crossing w>0.5 at least once in 3000 steps: because the
  nominal member stays live (the hedge), a false detection mixes in a wrong-but-stable
  member briefly and costs ~nothing.  **The hedge is what makes the aggressive end of
  the frontier affordable** — this is the never-worse-than-F0 guarantee showing up as a
  detection-speed subsidy, and it is the single most portable finding of this probe.
- **Detection is a reporting convention here** (w > 0.5, or â crossing the midpoint);
  the filters themselves never threshold — they mix.
- **All learning contenders recover to oracle grade**; only frozen pays forever.  The
  bank is at 1.008 within 100 steps of the change.

## 3. The random-walk parameter class is off the frontier at every q_a — the class shape is what's wrong

At the derived `q_a = δ²rho` the augEKF crosses at 49 steps, 3.3× the frontier.  The
sweep shows this is not a constant to fix (`q_a = c·δ²rho`):

| c | delay | calm | pre-t* false |
|---|---|---|---|
| 1    | 48.4 ± 1.6 | 1.014 | 0% |
| 10   | 15.1 ± 0.8 | 1.040 | 24% |
| 100  | 4.7 ± 0.4  | 1.098 | 100% |
| 1000 | 1.8 ± 0.2  | 1.194 | 100% |

To match the bank's 13.5-step delay the random walk must pay ~4% calm RMSE and a 24%
false-crossing rate; the bank gets the same delay at calm 1.0004.  **Every point of the
random-walk trade curve is dominated.**  This is SUMMARY commitment (d) answered for
the scalar rung: a dynamics fault is a jump process, and an AR(1)/random-walk parameter
channel — the shape the noise scales use — is structurally the wrong class for
*detection*.  (It is fine for *refinement*: see §4.)

**The (x,a) cross-covariance is worth nothing here**: regwalk (decoupled scalar walk on
the departure, fed back; exactly "accumulate Σe·m / Σm²" in Kalman form) matches augEKF
to within noise on delay (49.2 vs 49.1), calm, and recovery.  Good news for the
embedded budget — the departure channel can run beside the state filter, not inside it.
Caveat: scalar, well-excited, b known; re-test multivariate before leaning on it (0004).

## 4. The half-way hypothesis: detection is robust, recovery pays

Bank {0.9, 0.75} with truth jumping to 0.6 (the enumerated set misses the truth):
the general KL formula still prices it — KL(0.75-member beats a0 | truth 0.6) = 0.375,
frontier 21.3 — and the bank detects in 14.6 ± 1.0 steps.  But recovery floors at
**1.204× oracle** forever.  Detection degrades gracefully under bank mis-specification;
recovery does not.  That splits the ladder's work exactly as the odefilter precedent
suggests: **detect by bank (jump class, frontier-fast), refine by walk (the regwalk is
3× too slow to detect but converges to 1.009 once the jump is known)** — the hybrid is
the natural mechanism for 0004/0005.

## Opens carried forward

- 0002: the Q↔F confound — run this bank WITH the noise machinery live; does a process
  burst read as a fault?  The whiteness split (wrong-F innovations correlate with the
  regressor) must be measured, not assumed.
- 0003: excitation honesty — at su→0 the frontier delay diverges; the unexcited
  channel's posterior must widen (bounded, never frozen), and the su=2.0 audit anomaly
  should resolve under a cleaner small-τ statistic.
- Hybrid detect+refine (bank + regwalk) is unbuilt; §3/§4 make it the favorite.
- The su=2.0 optional-stopping audit gap (small-τ) — flagged above.
