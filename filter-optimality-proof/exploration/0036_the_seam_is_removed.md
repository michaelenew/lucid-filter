# 0036 — The seam is removed, in the only direction it could be, and it was not where the oracle gap lived

Probe: [`0035`](0035_layer1_under_code_length.py) ·
theorem: [`output/03`](../output/03-logloss-shape-minimaxity.md) ·
numbers: [`figures/fop0035.json`](figures/fop0035.json)

## What was tried, and what happened

The standing hypothesis (posed from `filter-oracle-gap`, after the IMM patch
brought the filter within ~10% of a causal ceiling of 96.3%): the log-loss/MSE
seam has been a happenstance stabilizer masking the other holes, and removing
it might close the last theoretical gap — unless removal costs stability or
accuracy, in which case live with it.

Removing a seam between two losses means committing to one. Both directions
were already constrained by measurement, and the attempt resolved them
asymmetrically:

**Toward MSE: refuted, and this is where the "stabilizer" intuition is
half-right — inverted.** Committing the fit to an observable squared-error
criterion (PEM) was measured in `0027`: with all six parameters free it
recovers $\sigma^2$ wrong by up to 9× and inflates $s_M$ to 1.54 on data with
no scale variation, because the squared innovation cannot see any direction
that moves the predictive variance without moving the gain. The hybrid's
log-likelihood side *is* the stabilizer — not happenstance but structural,
and it is the side that stays. Removal toward MSE would be the reversion the
decision rule forbids.

**Toward log-loss: nothing to remove in the code, one thing to remove in the
proof — and it removes.** The filter is already the log-loss object end to
end: `fit()` maximises it, the predictive density is the output, and every
oracle-gap number in `filter-oracle-gap` is nll. The seam existed only in
the optimality argument, where layer 1 (Theorem A) was proved under squared
error while layer 2 (Theorem C) was proved under code length. **Theorem A′**
([`output/03`](../output/03-logloss-shape-minimaxity.md)) closes it: the
Kalman code's cross-entropy under any shape in the class depends on $p$ only
through $\mathbb E_p[e_t^2]$ — the same quadratic form as the MSE equalizer —
so it is constant at $\tfrac12\log(2\pi eS_t)$, the Gaussian is exactly least
favourable, and the same three lines give the same saddle under code length.
Verified in `0035`: five shapes on a heteroscedastic path sit on the closed
form within Monte Carlo error (spread 0.0044 nats/pt ≈ 2.5 se), while the
adaptive filter — outside the theorem, correctly — spreads 17× that, with
the heavy-tailed members scoring *better* (Theorem B's relocation, seen from
the code-length side).

## The decision, against the stated criterion

No reversions: zero behavioural change (no runtime code is touched; the full
batteries were green at the merged head), and the refuted direction was never
taken. The seam is removed at the level where it existed. Both layers now
read under one loss, and what remains of the optimality question is
single-loss and was already known: Theorem C does not survive marginalisation
to $x$ (`0023`), the two-moment class is too big for an equalizer anywhere
else (`0024`), and $\mathbb E[e^\lambda]<\infty$ is now load-bearing rather
than optional.

## The bet, honestly scored

"Basically no gap to oracle less lookahead advantage will get to 0 if we fix
the seam" — **not confirmed, and the decomposition says why.** The residual
gap was measured *in code length already*: of `0038` B's static-to-oracle
span, the causal ceiling leaves 3.7% as detection lag (irreducible for any
causal filter — this is the "lookahead advantage", now with a number), and
the remaining reachable piece is 6.8% of channel-model mismatch (AR(1)
log-scale against the true regime chain) — a modelling commitment, not a
loss seam. The seam could not have been carrying the gap because the gap was
never measured across it. What the removal *does* buy: the optimality
argument and the practice now use the same currency, so future gap numbers
and future theorems are commensurable by construction — and the one
remaining theoretical wound (marginalisation) is now a single, precisely
located target instead of being entangled with a loss mismatch.
