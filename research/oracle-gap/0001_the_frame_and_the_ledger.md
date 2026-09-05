# 0001 — The frame: one filter, several reported symptoms, and the suspicion they share a hole

> **⚖️ ATTRIBUTION —** _This is a framing/diagnostic note: it scores an adaptive filter against a clairvoyant oracle told $Q_t$, and organises reported symptoms; the oracle-gap methodology and the tools named (GPB1 collapse, self-confirming boundary) are all standard._ Prior art: clairvoyant/oracle bounds and innovation-based diagnostics — Mehra 1970/72, Bar-Shalom; GPB1 — Ackerson & Fu 1970. Status: RECOMBINATION.

## Where this starts

[`ode-filter/0039`](../ode-filter/0039_two_zeros.md)
left the process-scale channel in a strange place. Against a ×8 process-noise
regime, scored against a Kalman filter told $Q_t$ exactly:

| filter | gap closed |
|---|---|
| oracle $Q_t$ | 100% |
| $s_P = 0.8$ **forced** | **80.0%** |
| $s_P$ **as fitted** (lands on 0) | **0.0%** |

Two gaps, not one. The *model* gap: even forced on, the channel stops at 80%.
The *fit* gap: left to itself, the fit puts $s_P$ on the zero boundary and the
channel contributes nothing, exactly when it was needed (`0033`'s 4.9×
tracking regression). `0039` measured the boundary as **self-confirming** —
wherever the fit lands, the profile taken from that point endorses staying —
and priced the asymmetry: carrying the channel unnecessarily costs
**+0.0025 nats/pt**, missing it when needed costs **+0.0872** — 35×.

## The ledger of reported symptoms

An applied workstream ran this filter hard on real data with a genuinely
live scale channel and reported back (its `FILTER-NOTES`). Read together
with `0038`/`0039`, the entries sort into two piles.

**Pile 1 — the likelihood is less decisive than it should be** (online):

- **§4, the sharpest one.** On real data the surface has two optima — the two *ends*
  of the process-scale channel, persistent-moderate ($\varphi_P\!\approx\!0.8$,
  $s_P\!\approx\!1.2$) against impulsive-large ($\varphi_P\!\approx\!0.25$,
  $s_P\!\approx\!1.9$) — separated by **0.003 nats/pt in-sample** and
  **0.042 nats/bar out-of-sample**. The marginal
  likelihood is nearly indifferent between hypotheses that differ 14× more
  than that out of sample. *Which basin the optimiser lands in decides the
  answer, and the data does not.*
- `0039`'s self-confirming boundary is the same shape: the likelihood in
  $s_P$, evaluated through the filter, is flat to four decimals where the
  out-of-sample stakes are 0.087 nats/pt.
- `0038` §C measured a candidate mechanism and then set it aside: the GPB1
  collapse — every grid node handed the **same covariance** every step —
  deletes the *accumulated-history* component of the discrimination between
  scale hypotheses: only 25% kept at $\sigma^2=9$, 11.7% at $\sigma^2=36$.
  Two nodes differ in $S$ by one step of process noise, never by the history
  of the regime they disagree about. `0039` called it "a tax, not the cause"
  because what survives was enough for the *forced* 80% — but a tax on
  evidence is exactly what a flat profile, an indifferent basin choice, and a
  sticky boundary look like from outside.

**Pile 2 — the search proposes the wrong candidates** (fit):

- **§1.** $Q$ is the *median* process variance, so a start-screen candidate
  with $s_P > 0$ at the homoscedastic fit's $Q$ carries a **mean** variance
  inflated by $e^{s_P^2/2}$ — every live-channel candidate is scored at a $Q$
  it would never choose. Measured cost on real data: the uncorrected ranking landed
  the whole fit **10 nats worse**; the one-line correction
  ($\log Q \mathrel{-}= s_P^2/2$) recovered it. Inert when $\hat s_P=0$ —
  which is why the ODE workstream never felt it.

> **⚖️ ATTRIBUTION —** _The median-vs-mean correction is the elementary log-normal identity: if $\log$-scale has spread $s_P$, the mean variance is $e^{s_P^2/2}$ times the median. Standard._ Prior art: log-normal moment relation $E[e^X]=e^{\mu+\sigma^2/2}$ — textbook; the appearance of $\sigma^2/2$ terms in log-scale (stochastic-volatility) models is standard (Harvey/Ruiz/Shephard 1994). Status: REPRODUCTION.
- **§1 related.** `_S_SPLITS` tops out at 0.6; the applied daily fits
  $s_P = 1.24$–2.04. A screen that cannot propose the answer cannot rank it.
- **§5.** `_iv_alpha` defaults $m$ where `0028` measured it as a precondition
  ($m=p$ diverges, $\hat Q = 409$ against 1). Known, listed as `0b`, still
  open.

**Unassigned, kept in view:** §8 (fitted root moduli have a wide null and
nothing reports it), its §22 (the `dynamics` diagnostic tracks a
microstructure artifact within eras, r = +0.902), its §23 (`dynamics` is not a
fraction — it went to −0.404 on 2013 and that reading was *correct*). These
may be downstream of pile 1 — a filter whose scale posterior moves on
one-step evidence only is easier for microstructure to lead around — but that
is a conjecture until measured.

## The hypothesis

**The two piles reinforce.** The GPB1 collapse compresses the evidence that
separates scale hypotheses (pile 1), so the likelihood surface over
$(\varphi_P, s_P)$ is flatter than the data warrants; the screen then starts
the search at candidates handicapped by the median/mean mismatch (pile 2), and
on a flat surface the start decides the endpoint. The observed endpoints are
the reported symptoms: boundary zeros on synthetic data generated with a live
channel (`0029`), basin roulette on real data, 0% of an oracle the forced channel
gets 80% of, and online diagnostics computed from a posterior that never
sharpens.

If this is right, per-node covariances (IMM in place of GPB1 — `0039`'s item
0a′) should do more than close some of the forced-80% gap: it should put
**curvature back into the likelihood** — an interior optimum in $s_P$ at the
truth where the profile is now flat, and a real in-sample separation between
`§4`'s two basins. That second effect is the test that distinguishes "tax"
from "hole": a tax costs nats at fixed parameters, a hole misleads the fit.

## The scoreboard, inherited

From `0039`, any repair must:

1. beat **0.0025 nats/pt** of premium against **0.0872** of exposure,
2. leave `0032`'s window no worse than **+0.0004** nats/pt,
3. and now: close more of the oracle gap than the forced channel's **80%**,
   and break the self-confirmation — the $s_P$ profile evaluated through the
   repaired filter must have its argmin at the generating value on `0038` §D's
   data, from both starting points.

## Order of work

1. [`0002`](0002_per_node_covariances.py) — a reference per-node-covariance
   (IMM) filter beside the shipped GPB1 one, on `0038`'s exact battery: gap
   closed, discrimination kept, the $s_P$ profile from both filters, the
   premium/exposure ledger, and the compute cost.
2. The search half: the §1 screen correction and a widened `_S_SPLITS`,
   measured on `0029`-style data (generated $s_P=0.8$, currently fitted to 0).
3. Only then patch `odefilter/core.py`, with `0032`'s window as the
   do-no-harm gate.
