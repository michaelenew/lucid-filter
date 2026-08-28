# 0045 — the q-free magnitude: net-positive, but full saturation leaves a small BOTH regression

The q-free reach makes the gain explicit: `rate = K* + elig*discount*(1-K*)` (in [K*, 1]), the
saturated limit of 0042's q-sweep with no tuned q and no BFAST (instantaneous e^2 discount).

## Result (20 seeds, pot+accel rig)

| regime | floor | q=4 (stand-in) | q-free |
|---|---|---|---|
| pot-hot | 1.522 | 1.099 | **1.056** |
| process+pot | 1.675 | 1.331 | **1.298** |
| SENSOR | 1.230 | 1.233 | 1.247 |
| PROCESS | 1.089 | 1.096 | 1.096 |
| BOTH | 2.144 | 2.165 | **2.251** |

Best-yet on the reach regimes (pot-hot 1.056 near-oracle, process+pot 1.298 well below floor), but BOTH
regresses to 2.251 (+0.107 vs floor) where q=4 held it at +0.021. This is the q->infinity limit of the
mild BOTH growth the 0042 sweep already showed (2.157->2.172 over q=0.5->8): full saturation continues
it to +0.107.

## Diagnosis: instantaneous discount noise x aggressive jump

BOTH = process + accel-failure. The pot is fine there; its reach should be shut because the accel
(its witness) is lit. On average the discount ~0, but the INSTANTANEOUS e_acc^2 (bfast=1) fluctuates
chi^2(1), so on an occasional step e_acc^2 is small -> discount ~1 -> the pot reaches that step -- and
q-free jumps hard on it. With the gentler q=4 surcharge (~surprise^2) those single spurious steps cost
less. So the confound CRACK is intact (elig*discount shuts the sustained reach), but the magnitude's
aggressiveness interacts with the instantaneous-discount noise on the one regime where the reached
sensor's state is already fragile.

## The clean resolution (next): walk to the robust-MAP target, not the raw residual

Both stand-ins walk toward `target = log(resid/rho)` (the raw C0 residual). The derived robust-MAP
eta_r (0031) is already a bounded, outlier-aware scale with the s^2 spread and the wg tightening -- a
gentler, still-derived target that will not overshoot on a single fluctuating step. The principled
magnitude is: walk mu toward eta_r at the elig*discount-gated saturated rate. That keeps q-free's
pot-hot gain without the BOTH regression, and stays parameter-free (eta_r is derived). To build/test.

Alternatively a small discount smoothing (bfast ~ 0.3) removes the single-step spikes (0043 showed
bfast is nearly free), at the cost of reintroducing one soft constant. The robust-MAP target is
preferred because it is derived.

Net: the reach is net-positive and parameter-free in structure; the remaining work is the target of the
walk (raw residual -> robust-MAP eta_r) to erase the small BOTH cost. Nothing merged; hook off-by-default.
