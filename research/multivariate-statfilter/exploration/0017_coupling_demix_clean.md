# 0017 — the de-mix obstacle is deeper: a mixing H couples the sensors to each other

Built the 2-D coupling-grid de-mix cleanly (`0017_coupling_demix_clean.py`) — a distribution
(hedge) over the global process/measurement split, caltrop within-block — and **hammered it
over 6 seeds**. It **fails on the sensors**: when one sensor is hot, both read the same value
(`eta1 ≈ eta2`); the global `gM` absorbs "a sensor is hot" but the per-sensor deviations
`eps` do not split it to the *right* sensor.

Confirmed against the exact grid, which **does** de-mix the sensors under the mixing H
(hot sensor 0.84, clean sensor −0.09; separation ~0.9). So the failure is the approximation's,
not the problem's.

## Root cause, pinned

The "sensors are decoupled" fact (0003) was measured with **H = I**. With a *mixing* H the
sensor-scale Fisher block is **not** diagonal: its off-diagonal is `~0.5 ρ_i ρ_j (S⁻¹)_ij²`,
and `S = H P Hᵀ + R` is off-diagonal whenever H mixes. So the sensors couple **to each other**
through the shared-state observation, exactly as process couples to measurement. For a general
H the coupling is **pervasive** across all scale axes — not one 2-D block. The Q-eigenbasis
diagonalises the *process* (Q) but not this observation-induced coupling.

## Consequence — the honest state

No **fixed** low-dimensional hedge de-mixes a general mixing H; the exact joint grid
(exponential) is what handles it. **State tracking stays solved at linear cost (caltrop, 0013);
the faithful *diagnostic* (which sensor/mode is hot) under a general mixing H is the genuinely
hard, still-open piece.** Three routes remain:

1. **Full-Fisher rotating eigenbasis.** The `D×D` scale Fisher `F = U Λ Uᵀ` diagonalises *all*
   the coupling; a caltrop in the (time-varying) `U`-basis is faithful and linear, reporting
   physical scales by rotating back. This is the wall-correspondence connection (0026–0027)
   gift made concrete — elegant, but a real rotating-frame construction (`U` drifts each step,
   and the walked coordinates are abstract combinations, not "which sensor").
2. **Dynamic-node backup** (user-flagged): add grid nodes where misattribution is detected —
   adapts to arbitrary coupling. A separate workstream.
3. **Accept the split**: exact grid for the faithful diagnostic at small `r`, caltrop for
   state at any `r`.

Hammer result (6 seeds, mixing H): static drift up to 0.66; process-hot→sensor leak mean
+0.16 / worst +0.47; and the decisive failure — sensors do not separate (`eta1 ≈ eta2` when
one is hot). Not prod-faithful. Code: `0017_coupling_demix_clean.py`.
