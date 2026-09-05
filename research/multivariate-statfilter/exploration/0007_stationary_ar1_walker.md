# 0007 — stationary-AR(1) walker, a bug fix, and the real tension

> **⚖️ ATTRIBUTION —** _A bug fix plus the measured "bracket": a stationary-AR(1) scale is stable but under-reaches, an unbounded walk reaches but drifts; both trace to running the state KF at a scale point estimate rather than mixing it (GPB1)._ Prior art: stochastic-volatility AR(1) scale — Taylor 1986; GPB1 collapse — Ackerson & Fu 1970. Status: NEGATIVE-RESULT.

## A bug found (corrects 0006)

`steady_fisher` unpacked the wrong return of `score_fisher` — it took the **score**
(first return) instead of **info** (second), so `I_char` came out **negative**
(`[-0.07,-0.31,-0.25,-0.17]`). That fed `q_mu = K*²/(I_char(1-K*))` a negative drift
variance in 0006's unbounded walker, which is what produced the **catastrophic**
divergence (μ→−50). Fixed (`_, info, … = score_fisher(...)`); `I_char` is now
positive (`[0.01, 0.19, 0.13, 0.06]`). **0006's "diverges to −50" is a bug artifact.**
Re-run below with the fix.

## The real tension (clean, post-fix)

n=2, m=2, correlated PD Q, mixing H. Two walk recursions, both with expected Fisher,
spectral-truncation mask, and the state KF at the point estimate:

| | STATIC data ψ=0 (want ~0) | strong `xi2` hot (grid 0.95) | sensor `eta2` hot (grid 0.94) |
|---|---|---|---|
| **stationary AR(1)** | `[0.00,0.02,0.00,0.03]` ✓ stable | 0.58 (corr 0.87), no leak | 0.33 (corr 0.86), no leak |
| **unbounded walk** | `[-1.58,-0.15,0.44,0.58]` ✗ drifts | 1.30 (corr 0.97), leaks | 1.88 (corr 0.92), leaks |

So the two **bracket** the exact grid, and neither alone matches it:

- **Stationary AR(1) is stable but under-reaches.** No drift (μ→0 on static data), no
  leak, right direction — but a sustained regime is over-shrunk (0.33 vs the grid's
  0.94). The AR(1) prior caps the per-step gain `K = Pμ/(Pμ+1/info)`: with `Pμ ≤ s²`
  and small per-step `info`, K is tiny, so it can't accumulate to a sustained truth
  within the reversion.
- **Unbounded reaches but drifts.** It gets near the truth on a sustained regime
  (even past the grid, which shrinks), but on static data it drifts and it leaks
  across the coupled axes. Note the *scalar* `WalkingFilter` (unbounded) does **not**
  drift on static data — the drift here is from **channel coupling + the
  point-estimate state KF**, which a single direct-observation axis doesn't have.

## What this says about the construction

The grid does both (reaches a sustained regime *and* doesn't drift) because it is a
joint posterior with a **GPB1 collapse that mixes the state KF over the scale
window**. Both walker failures trace to running the state KF at a single scale
point estimate:
- the unbounded drift is a point-estimate coupling bias the mixing would cancel;
- the stationary under-reach is the prior cap the joint evidence would overcome.

So the next probe is the **windowed / GPB1 walker**: carry a small per-axis window
(not a point) and collapse the state KF over it, as the scalar filter does. The open
question is whether that closes the bracket — matching the grid's reach without the
drift — at linear-in-D cost (D windows), or whether the process↔measurement block
coupling forces a joint window there.

Spectral truncation itself is confirmed working (the weak eigenmode carries `I_char`
0.01 vs the strong 0.19; identifiability and the truncation both track λ).

Code: `0007_stationary_ar1_walker.py` (+ the unbounded/stationary comparison inline).
