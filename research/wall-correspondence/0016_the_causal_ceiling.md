# 0016 — Causal attainability at the bank level (F3)

> **AI-generated, not peer-reviewed.** The last unrun row of the
> original adoption plan. Code: `0016_the_causal_ceiling.py`.

> **⚖️ ATTRIBUTION —** _A real filter-vs-smoother comparison: a fixed-interval (RTS-type) smoother's state gain over the causal filter is captured at lag 1, while transition-timing is the genuinely two-sided structure ("past pins states, future pins boundaries"). Standard smoothing theory, cleanly measured._ Prior art: fixed-interval smoothing (Rauch–Tung–Striebel 1965); filter vs smoother information gain. Status: RECOMBINATION (measurement); the "Euclidean→Lorentzian / causal ceiling" correspondence is SPECULATIVE.

Minimal bank (exact 2-state HMM, dwell ~25), filter vs smoother vs
fixed-lag: state-occupancy gap +0.058, **fully captured at lag 1**
— the smoother's per-time state gain is a one-step-of-future
effect. Transition **timing** is the genuinely two-sided structure:
F1 (±2) 0.405 (filter, 6129 flips — 3× inflated boundary count) vs
0.584 (smoother, 1835 flips). **The past pins states; the future
pins boundaries.** This is the bank-level face of 0003's "sector
identity is a slow observable" and the sibling's 1 − t/T template:
what the Lorentzian restriction costs their theory is boundary
(transition) sharpness, not state tracking.
