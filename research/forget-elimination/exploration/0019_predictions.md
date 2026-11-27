# 0019 — predictions, recorded before the run

> **AI-generated, not peer-reviewed.** Written before
> [`0019_grid_free_ladder.py`](0019_grid_free_ladder.py) was first executed.

`0018` found the split and hazard ceilings depend on how finely the ladder is cut. With
spacing `g` and per-step information `I` in the ladder's coordinate, evidence between
neighbours accrues at `½·I·g²`; a fixed level `c` per hop gives hop rate `I·g²/(2c)`, and the
belief spreads along the ladder with diffusion `rate × g² = I·g⁴/(2c)`. Two powers of `g`
too many — both slow a finer grid.

**The proposal (the user's):** a small move should need only small evidence.

- *Literal form:* evidence per hop `∝ g`. Hop rate `∝ g`, diffusion `∝ g³`.
- *Exact form:* diffusion independent of `g`, i.e. hop rate `D/g²` — the standard
  discretisation of a diffusion; evidence per hop `∝ g⁴`; in the limit of a fine grid the
  belief moves in proportion to the evidence it receives.

Both are anchored to agree with the filter at `_LADDER_MEM = 1000` (so `D = 9e-6` per step
in arclength), and run on the ramp at 250 and 4000.

- **G1.** Exact form: the drift's ramp result is the same at 250, 1000 and 4000 to within
  2% (it was 1.031 / 1.033 / 1.107).
- **G2.** Literal form: a partial repair — 4000 lands between the two, around 1.06–1.08.
- **G3.** Neither changes the result at 1000 (by construction), and the default is
  untouched.

**What this does not settle.** Making the ladder grid-free turns a hidden timescale into an
explicit diffusion constant `D`. It does not derive `D`.
