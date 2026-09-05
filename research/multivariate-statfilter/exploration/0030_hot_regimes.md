# 0030 — solving the hot regimes to the irreducible floor

> **⚖️ ATTRIBUTION —** _Sheds a failing absolute sensor fast with two additions gated on a white channel: an instantaneous ~4σ robust innovation gate (per-update R inflation) and an outlier-boosted raise-rate — standard robust-KF outlier rejection plus an engineering shed; brings the failing-sensor regimes to the oracle-lagged floor._ Prior art: robust Kalman filtering / innovation outlier rejection — Masreliez & Martin 1977, Huber 1964. Status: RECOMBINATION.

The reprofile (0029) found the expensive extreme is **adaptation lag when the absolute position
sensor degrades** (pot-hot 1.86×, process+pot the old 3.72×). This is where the filter breaks.
Target: bring it to the irreducible floor.

## The floor, and the diagnosis

The floor is `oracle-lagged` (a KF handed the EMA-β-lagged true `Q,R`). On the hot regimes it is
**≈ the full oracle** (`lag/orc = 0.98`) — a clean windowed estimator nearly matches the oracle,
so the lag itself is cheap and almost all of the adaptive's **1.98×** gap is reducible.

The mechanism is a **front-loaded** corruption: the pot scale *does* reach the right level by
mid-burst (+5.3 vs true +5.4), but the state error keeps *growing* even after — because once the
pot is (correctly) discarded there is no absolute reference and the position **drifts**, and the
noisy pot injected during the slow ~1/K* ramp got baked into that drifting estimate.

## The fix (shipped) — shed a failing sensor fast

Two complementary additions, both fired **only on a clearly white channel** (whiteness floor
`_WHITE_MIN`), so a process disturbance (correlated) never triggers them:

1. **Instantaneous robust gate.** In the state correction, inflate a sensor's `R` for *this*
   update when its normalised innovation `e_i²/S_ii` exceeds a 4σ outlier cutoff (`_ROBUST_CHI`).
   Protects the state at the *first* corrupted sample, before the scale ramps.
2. **Outlier-boosted raise-rate** (`_SHED`). Accelerate that sensor's scale walk when its
   innovation is a big white outlier, so it sheds in a few steps instead of ~1/K*.

They are complementary: the gate protects the state instantaneously, the shed adapts the scale so
the gate stops rejecting (gate-only keeps rejecting → `R` never adapts → worse). Both need the
whiteness floor — a partly-correlated process disturbance must not look like a failure.

## Result (4 seeds, adaptive / oracle, on the same seeds)

| regime | before | after |
|---|---|---|
| **pot-hot** (var ×100) | 1.98 | **0.94** |
| **process+pot** (the 3.72×) | 1.98 | **0.94** |
| SENSOR burst | 1.41 | **1.24** |
| PROCESS | 1.34 | 1.32 |
| calm recovery | 1.33 | **1.22** |
| **BOTH** | 1.43 | **1.59** |

The failing-absolute-sensor hot regimes are **at the floor** (0.94×); the sensor-burst and
recovery phases also improve. Confound diagnostic and 13 unit tests unchanged.

## Residual — the BOTH tradeoff

The one regression is **BOTH** (a *dynamic* sensor noisy while a process disturbance is active,
+11%). Its channel is only partly white, so the robust gate occasionally over-rejects a
noisy-but-useful accelerometer. This is the collinear confound again — "sensor failing" vs
"noisy-but-useful under process" is genuinely ambiguous on a process-coupled sensor. The clean
refinement is an **observability-weighted** gate: protect robustly only the sensors whose loss
would collapse observability (the absolute references), not the redundant/process-coupled ones.
Deferred; the absolute-sensor regime — the user's stated target — is at the floor.

Code: `0030_hot_regimes.py`.
