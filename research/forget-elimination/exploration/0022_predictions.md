# 0022 — predictions, recorded before the run

> **AI-generated, not peer-reviewed.** Written before
> [`0022_the_departure_as_a_class.py`](0022_the_departure_as_a_class.py) was first executed.

**The object.** The dynamics departure `g` gets a class of its own, as a noise level has:
a memory and a spread. A cell `(rho, s_g)` is a walker whose coefficients drift with
variance `s_g²·rho` per step and are capped at `s_g²` (the shipped ladder is the row
`s_g = 1`). "No departure" is the zero corner — the edge `s_g → 0` — and it is the nominal
member. The cells are a class axis like `(phi, s)`: weight moves between neighbours no
faster than they can be resolved, at the star's fraction. The zero corner neighbours every
cell of the smallest spread, so a fast small departure is one move away.

**Removed:** the per-rung switch between "nominal" and "departed" (`_hazard_mix`), the
restart on a confirmed fault (`_reprice`), and with them the hazard as an event rate. The
`fault` read-out is the weight away from the zero corner.

**The box** (a choice, like `_PHIS × _SS`): `rho` in {0.42, 0.10, 0.016, 0.0019, 2e-5},
`s_g` in {1/8, 1/4, 1/2, 1, 2} — 25 walkers per noise cell.

Rig: `0014`'s (AR(1), two sensors, dynamics 0.9 → 0.3 after a calm of 600 or 4,000 steps).

- **K1.** It runs and finds the fault: read-out above ½ and held, on every seed.
- **K2.** Flagging delay within 1.5× of the default's after the short calm.
- **K3.** No dependence of the delay on the length of the calm beyond 25% (the default:
  60 → 82; pure-Bayes rungs: 84 → 127) — there is no event-rate belief left to go stale,
  only the star's fraction.
- **K4.** Without the restart the state error in the 300 steps after the fault is within
  5% of the default's; the dynamics estimate at the end is as close to the truth (0.3).
- **K5.** False alarms in the calm at or below 1%.
- **K6.** The weight after the fault sits on wide, slow cells (`s_g ≥ 1/2`, small `rho`):
  a change that happened once and stayed.

Least sure of K2 and K4: the switch and the restart were both put in on measured grounds.
