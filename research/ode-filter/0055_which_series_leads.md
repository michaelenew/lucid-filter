# 0055 — Which series leads is decided online, and deferral must be uniform

From [`0054`](0054_which_series_leads.py); raw numbers in
`figures/ode054.json`. Discharges `0047` §4 item 5 — the last constructive
item on the stack — using `0042` §4's design: **a lead is a lag in processing
time.** A node with $\tau_j<0$ says $y^{(2)}$ reads the latent's future; its
bracketing states first exist $\lceil-\tau_j\rceil$ steps later, so the node
consumes the $y^{(2)}$ stream with a processing lag and applies the same
bridge row at the shifted fractional position. Log-likelihoods are
accumulated per *sample*, so nodes are compared on a common observation set.

> **⚖️ ATTRIBUTION —** _Handling a lead ($\tau<0$) as a deferred update (processing at the step where the bracketing states first exist) turns lead/lag estimation into fixed-lag smoothing; the "uniform deferral" repair enforces that prequential comparison across members requires a common information set at scoring time — a standard fairness condition for likelihood/scoring comparison. Prior art: fixed-lag smoothing (Rauch–Tung–Striebel 1965; standard); prequential same-information-set comparison (Dawid). The specific bias and its repair are measured on this rig. Status: RECOMBINATION (with a measured NEGATIVE-RESULT)._

## The bias found on the way, and its repair

The naive ledger — each node deferring by its own
$d_j=\lceil-\tau_j\rceil$ — is **biased toward longer deferrals**: a node
processing $y^{(2)}_k$ at time $k+d$ conditions on $d$ more points of
$y^{(1)}$, and the extra conditioning is a likelihood subsidy unrelated to
$\tau$. Measured signature: a persistent spurious posterior band exactly at
the $d{=}1\to2$ class boundary ($\tau\approx-1.1$), inflating lead-side RMS
to 0.265 against 0.098 on the lag side.

The repair is **uniform deferral**: every node processes $y^{(2)}_k$ at time
$k+d_{\max}$, under identical conditioning; lead nodes simply read less far
back into the stored window. After it:

| | A: $y_2$ leads ($\tau{=}{-}0.8$) | B: lags ($+0.8$) | C: flips at $t{=}450$ |
|---|---|---|---|
| sign at 99:1 | **20 points** | 20 | 20 |
| $P(\text{sign})$ final | 0.9999 | 0.9998 | 0.9998 |
| RMS $\tau$ error | **0.024** (was 0.265) | **0.018** (was 0.098) | 0.019 |
| flip relocation | — | — | **3 points** |

Lead/lag symmetry is restored (0.024 vs 0.018), and the spurious band is
gone.

## Two things worth keeping beyond the fix

1. **Deferral helps everywhere.** The lag side also improved 5× — processing
   $y^{(2)}_k$ two steps late reads it against bracketing states already
   refined by two more $y^{(1)}$ points, i.e. against a *partially smoothed*
   path. Offset estimation is a smoothing problem wearing filter clothes; the
   uniform ledger buys fixed-lag smoothing for free. The cost is that
   $y^{(2)}$'s information reaches the state late — irrelevant for $\tau$,
   relevant only if $y^{(2)}$ is also wanted as a forecasting input, where the
   deferral should be a choice, not a default.
2. **The conditioning-fairness principle generalises.** Any mixture whose
   members consume observations on different schedules (different deferrals,
   different windows, different data subsets) pays this subsidy to whichever
   member conditions on more. Prequential comparison requires not just the
   same scored set but the same information set at scoring time. `0046`'s
   hyper-grid satisfied this trivially (identical schedules); the ledger here
   is the first place it had to be enforced by design.

## The stack, closed out

`0047` §4 after this session: ~~1 tube~~ (`0048`), ~~2 persistence~~
(`0050`), ~~3 joint $(\alpha,\tau)$~~ (retired by `0052`/`0053` — the offset
is a symmetry center, first-order immune to dynamics error), ~~5 negative
$\tau$~~ (here). Remaining: **item 4** — a self-consistency score for the
trusted distribution itself (the ramp taught that prequential $y$-likelihood
under-polices $\tau$-band calibration; truth is never revealed online, so
this needs a rolling-PIT-style formulation and is a design question, not an
implementation) — and the engineering ledger for folding the channel into
`output/odefilter` (`0045` §5: the lag-basis GLS bridge row and the
AR-vs-exact-discretisation class gap).
