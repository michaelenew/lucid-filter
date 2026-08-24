# 0008 — The node barriers: the sibling's ergodicity finding, priced in nats

> **AI-generated, not peer-reviewed.** Experiment 7 (imported with
> 0004): the sibling's coupling scan found their weight's exact
> zeros fracture local MCMC, cured only by their flow's smoothing.
> The filter-side version, at identical compute, no fitted knobs.
> Code: `0008_the_node_barriers.py`.

**Setup**: frequency estimation, y = sin(ω*t) + noise, T = 240 — the
likelihood over ω is a comb of narrow peaks split by deep valleys
(the nodes). Local Metropolis, 200 random restarts, 6000 steps each;
the *sharp* bank targets the likelihood directly, the *tempered*
bank runs the identical chain under a fixed geometric annealing
schedule β: 1/64 → 1 (the filter-side heat-kernel flow).

```
sharp    : trap rate 0.81   prequential price 3.63 ± 0.11 nats/pt
tempered : trap rate 0.40   prequential price 1.98 ± 0.14 nats/pt
truth-ω reference: 0.23 nats/pt
smoothing buys: −0.41 trap rate, −1.65 nats/pt, at equal compute
```

The sibling's prediction holds, priced: the nodes trap, the trap is
a code-length cost, and smoothing anneals it. One honest subtlety
found on the way: with too-small proposals even the *flattened*
surface can't be traversed in budget — barrier-crossing and
diffusion-reach are separate failure modes, and annealing only fixes
the first (the first run showed 0.94 vs 0.82 until the proposal
scale allowed flat-surface traversal). The physics-side echo:
smoothing restores ergodicity only relative to a proposal kernel
that can reach — their lattice's local moves could; a bank's local
search must be checked, not assumed.

**Port back**: their RG's smoothing is not just coarse-graining — it
is the annealing that makes their own measure *learnable* by local
dynamics. A theory whose bare weight has nodes is, in filter terms,
a hypothesis surface that cannot be searched sharply; the flow is
the search schedule. (Their 0102 metastable branch = our trapped
restarts; their τ* window = our β schedule's knee.)

## Honest limits

- Tempered trap rate 0.40 is far from zero: the fixed schedule is
  deliberately unoptimized (house rule) — the comparison is
  sharp-vs-tempered, not tempered-vs-best-possible.
- One surface family (sinusoid comb); depth/width of nodes not
  scanned.

## Open

1. Trap rate vs node depth (σ scan) — the filter-side analogue of
   locating their τ*.
2. Population/replica search vs annealing on the same surface (their
   parallel-replica bond move suggests replicas beat schedules).
