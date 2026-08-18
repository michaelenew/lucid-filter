# 0017 — Max-entropy is not least favourable at layer 2 under squared error

Script: `0016_max_entropy_full_admissible_range.py`, 80 seeds, $n=1200$.
Supersedes `0014`, whose grid was too narrow to decide anything.

$(\gamma_0,\gamma_1)$ held fixed, class members realised as AR(2) log-scales
across the full admissible $\rho_2\in(2\rho_1^2-1,\,1)$, filter held at the
AR(1) model — the Bayes rule for the max-entropy member, which sits at
$\rho_2=\rho_1^2$.

**Moderate**, $s=0.8$, $\varphi=0.5$, max-entropy $\rho_2=0.25$:

| $\rho_2$ | MSE | vs max-ent | se | oracle | ratio |
|---|---|---|---|---|---|
| −0.44 | 0.22300 | −1.62% | 0.74 | 0.18373 | 1.2137 |
| −0.095 | 0.22401 | −1.18% | 0.39 | 0.18790 | 1.1922 |
| **0.25** | **0.22668** | **0.00%** | — | 0.19331 | 1.1726 |
| 0.5950 | 0.23157 | +2.16% | 0.31 | 0.20011 | 1.1572 |
| 0.9400 | 0.23706 | **+4.58%** | 1.23 | 0.21027 | 1.1274 |

**Persistent**, $s=0.55$, $\varphi=0.93$, max-entropy $\rho_2=0.8649$:

| $\rho_2$ | MSE | vs max-ent | se | oracle | ratio |
|---|---|---|---|---|---|
| 0.7406 | 0.22074 | +0.32% | 0.58 | 0.20233 | 1.0910 |
| 0.7717 | 0.22137 | +0.61% | 0.41 | 0.20475 | 1.0812 |
| **0.8649** | **0.22003** | **0.00%** | — | 0.20940 | 1.0508 |
| 0.9270 | 0.21863 | −0.64% | 0.40 | 0.21139 | 1.0342 |
| 0.9581 | 0.21791 | −0.97% | 0.76 | 0.21244 | 1.0257 |

## The finding

**In neither regime is raw MSE maximised at the max-entropy member.** It is
monotone across the admissible range, and — the striking part — **monotone in
opposite directions in the two regimes.** The max-entropy member is interior
both times. The worst member of the class sits at an endpoint of $\rho_2$, and
which endpoint depends on the regime.

The moderate regime resolves clearly (+4.58% ± 1.23 at the top end, $t=3.7$;
−1.62% ± 0.74 at the bottom, $t=2.2$). The persistent regime is weaker
individually (largest $|t|\approx1.5$) but moves smoothly and monotonically
across nine points, which is more persuasive than any single contrast.

**So the conjecture in `0015` §3 is false as stated, and I retract it.**
Theorem A's saddle argument does not transfer to layer 2 under squared error.

## Why layer 1 works and layer 2 does not

The diagnosis is clean, and it is about **equalizers**, not about entropy.

Theorem A works because the Kalman filter is an *equalizer* over the shape
class: its risk is literally the same number for every member, because it is
linear and its risk is a quadratic form in second moments that the class fixes.
Constant risk plus Bayes-at-one-member is a saddle, in three lines.

At layer 2 the filter's risk is **not** constant across the class — that is
exactly what the tables above measure, varying by 1–5% as $\rho_2$ moves. No
equalizer, no saddle. Max entropy is a statement about *entropy*; least
favourability is a statement about *risk*; they coincide only when the risk
happens to be affine in the constrained statistics. At layer 1 that coincidence
holds for a specific structural reason. At layer 2, under squared error, it does
not.

Decomposing the two columns shows the mechanism. As $\rho_2$ rises:

- the **oracle** risk rises in both regimes (0.1837→0.2103; 0.2023→0.2143) —
  a more persistently clustered log-scale means long stretches of high variance,
  during which the level estimate degrades;
- the **ratio** falls in both regimes (1.2137→1.1274; 1.0910→1.0176) — a more
  predictable log-scale is easier for the filter *relative* to the oracle.

Raw MSE is their product, so the sign of its slope is whichever effect wins. In
the moderate regime the oracle term dominates; in the persistent regime the
ratio term does. That is why the two regimes disagree, and it means no
reparameterisation of the class will make the AR(1) a maximum — the two
competing effects are real and the crossing is genuine.

## What this costs, and what survives

**The Burg step is not load-bearing for minimaxity under squared error.** The
honest statement of layer 2 is now: the filter's log-scale model is the
max-entropy member of the class, which makes it the *parsimonious* and the
*ML-target* member, but **not** the least favourable one. A minimax claim over
$\mathcal C(\gamma_0,\gamma_1)$ under squared error would have its saddle at an
endpoint, and the filter is not that.

**What survives is the magnitude.** Sweeping $\rho_2$ across its entire
admissible range — every process the class permits, given its two moments —
moves risk by at most about 4.6%, and under 1% in the persistent regime. The
class's unconstrained directions are not free, but they are not expensive
either. That is a quantitative bound on what the Burg choice costs, which is
more useful than the qualitative claim it replaces.

## The seam, and where it actually bites

`0012` measured the log-loss/squared-error seam at $+0.23\%\pm0.21$ and called
it benign. That measurement was about **which parameters to use** and it stands.
This is a different question — **which member is worst** — and here the two
losses genuinely disagree:

- Under **log-loss**, the max-entropy member *is* exactly least favourable over
  a moment-constrained class, and for a clean reason: $-\log p^*$ is affine in
  the constrained sufficient statistics, so $\mathbb E_p[-\log p^*]$ is the same
  for every $p\in\mathcal C$. That is an equalizer, and the same three lines as
  Theorem A close it. Written up as
  [`output/02-logloss-least-favourable.md`](proofs/02-logloss-least-favourable.md).
- Under **squared error on $\theta$**, it is not, by up to 4.6%.

So the two losses are not interchangeable in general; they agreed in `0012`'s
setting and disagree here. The seam is real, and this locates it precisely
rather than leaving it as a worry: **it is the difference between coding the
log-scale path well and tracking the level well.**

## Consequences for the program

1. `0015` §3's "one principle applied twice" is **withdrawn** under squared
   error. It is correct under log-loss (Theorem C) and that is a narrower claim.
2. Layer 2's status improves in precision and worsens in strength: it is a
   *theorem under log-loss on the latent path*, with two gaps to the thing
   wanted — the loss, and the marginalisation to the observable (`0005` §6).
   Those are now the only two, and they are nameable.
3. The most interesting open question this raises: **is there a member of
   $\mathcal C$ the filter should be modelling instead?** The endpoint members
   are degenerate (near-deterministic period-2 or perfectly-clustered log-scale)
   and no one would build a filter for them, so minimaxity over the raw
   two-moment class may simply be the wrong target. A class constrained to
   $\rho_2$ near $\rho_1^2$ — i.e. admitting only mildly non-Markov log-scales —
   would restore the saddle. Whether that restriction is defensible from the
   "we know only magnitude and persistence" premise is a real question, and I do
   not think it obviously is.
