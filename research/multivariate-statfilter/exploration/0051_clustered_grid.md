# 0051 — the recorded open, built: per-channel scales on VectorFilter's grid, clustered, NO beta

> **⚖️ ATTRIBUTION —** _Extends the grid to per-channel scales, factored into per-cluster (per-joint) sub-grids for a block-diagonal H, with per-channel span set by the derived structural confoundability — near-oracle on the confound regimes with no EMA; the one hard case (a genuine failure on the process-collinear accel) is the recorded irreducible confound._ Prior art: block-diagonal / decoupled Kalman factorization (standard); grid/GPB1 multiple-model (as 0050). Status: RECOMBINATION (with a NEGATIVE-RESULT on the collinear SENSOR case).

Extends VectorFilter's grid (the no-EMA machinery) with the per-component ("which sensor is hot")
scales it flags as open (vector.py:33). The per-channel joint grid is exponential, but the confound is
LOCAL: for a block-diagonal H (robotics) it factorises into independent per-cluster sub-filters -- here
one per joint, a small grid over {process-mode, pot, accel} run with VectorFilter's exact GPB1 update
reused verbatim. No _BETA, no C0/C1, no walk, no reach hook.

## Two ingredients, both derived

1. **Wide-span grid, span decoupled from the class s.** A sensor FAILURE (x225 variance) is far outside
   the class swing (s=0.5), so a grid placed at the class resolution has no node to reach it (that was
   the first cut: grid worse than floor everywhere). Decouple: nodes on [-span, span] wide enough to
   cover a failure, stationary weights = N(0,s^2) (bulk at 0, light tail nodes reachable on a burst),
   transition = the AR(1) kernel. span is a resolution/coverage choice, not tuning.
2. **Per-channel span from structural confoundability (0043, derived).** A wide span on EVERY channel
   reaches failures but lets a jerk be mis-attributed to accel-noise -> shed the good accel -> runaway
   (PROCESS/process+pot 17-22x). Fix: span_i from the direct process footprint g_i =
   diag(H Q0 H^T)_i/(that+rho_i). Decoupled sensors (pot, g~0) get the WIDE span (reach); the
   direct-process-readout sensor (accel, g = cluster max) gets ~one class swing (no reach -> a process
   disturbance cannot push its scale up). Process modes get the wide span (they SHOULD reach a jerk).

## Result (5-DOF, 5 per-joint clusters, order 9, no beta)

| regime | floor (shipped, walk+beta) | grid (no beta) |
|---|---|---|
| pot-hot | 1.088 | **1.000** |
| process+pot | 1.088 | **1.017** |
| PROCESS | 1.110 | **0.999** |
| BOTH | 2.351 | **2.338** |
| SENSOR | 1.446 | 2.490 |

Near-ORACLE on pot-hot / process+pot / PROCESS (the confound resolved AND the failing pot reached),
matches BOTH -- all with no EMA. The shipped floor needs _BETA to get these; the grid does not.

## The one hard case that remains: SENSOR (the accel fails)

The accel is the process-readout channel, held narrow for confound-safety, so it cannot shed a GENUINE
accel failure -- and the kept-failing accel corrupts position through acc->vel->pos. That is the
irreducible confound (0043): a channel collinear with the process cannot be both reachable (for its own
failure) and safe (against process), UNLESS a white-vs-correlated discrimination separates them. The
shipped floor got SENSOR 1.446 using exactly that -- the C1 lag whiteness (the beta-EMA). The grid's
own sequence mechanism (per-node P propagation) is the no-EMA version of that discrimination, but the
onset spatial lag defeats it here: an accel jerk looks like an accel failure until the pot drifts
(~integration delay), and by then a wide accel has already run away. Holding the accel narrow trades
that runaway for the SENSOR under-shed.

## The decision / next

Net vs floor: better on 3 regimes (~-0.09 each), matches BOTH, worse on SENSOR (+1.04). The direction
is validated and beta is gone. To recover SENSOR without beta, the grid needs to distinguish a WHITE
accel failure from a CORRELATED process over the sequence -- the pot's co-drift disambiguates it, so a
grid that carries the cross-channel sequence evidence (per-node P, or the pot-coupling in the accel's
likelihood) should shed a true accel failure while staying safe on process. That is the next build; it
is the core.py mechanism pushed through the onset lag, not a new parameter. Nothing merged; the shipped
filter (with _BETA) is untouched pending the SENSOR recovery and the production port that retires _BETA.
