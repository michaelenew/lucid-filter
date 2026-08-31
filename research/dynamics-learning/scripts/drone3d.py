"""The 3D drone rig: a quadrotor that picks a heavy crate up OFF CENTRE, and puts it down.

The vehicle (n = 12): ``(p, v, att, om)`` -- world position and velocity, ZYX Euler attitude
and body rates.  Inputs ``u = (uT, ux, uy, uz, ug)``: collective thrust in hover units
(``uT = T / m0 g``, so ``uT ~ 1``), three angular-acceleration commands in units of a
reference ``ALPHA`` (``u = tau / (I0 ALPHA)``), and a constant channel carrying gravity
(``ug = G`` always) -- the channel `research/dynamics-learning/SUMMARY.md` records this rig
as needing.  Those units are chosen so that the ``B`` columns the departure directions live
in are comparable in magnitude, which is what makes ``LucidFilter``'s scale-free class size
("this part of the dynamics changed by about its own magnitude") say the same thing on all
six of them.  That is a property of the RIG, not a fitted constant, and it is measured:
`0008_drone3d_payload.py`'s ``units_control`` re-runs the same data with the angular inputs
in newton-metres and reports what it costs.

The one-step map is affine in the state at a given operating point,

    x_{t+1} = F(att_t, om_t; I) x_t + B(att_t; m, I, c) u_t + w_t,

and BOTH halves move with that operating point.  ``F`` does, for two reasons that are rigid-
body physics rather than rig decoration: a rate gyro reads **body** rates, so the attitude
kinematics are ``d(att)/dt = T(att) om_b`` and not ``d(att)/dt = om`` (treating the two as the
same is worth 9-20x the gyro's own noise here, 114x at the worst step); and the gyroscopic
coupling ``om x I om = [om]x I om`` is linear in ``om`` with state-dependent coefficients.
Every physical parameter lives in ``B``:

* ``m``      -- thrust column, position/velocity rows: ``dt g (m0/m) R(att) e3``
* ``I``      -- torque columns, attitude/rate rows: ``dt ALPHA (I0/I)``
* ``c``      -- the CENTRE OF MASS offset.  Rotor thrust acts at the geometric centre, so a
                displaced centre of mass turns collective thrust into a standing torque,
                ``tau = (-c) x (T e3) = T (-c_y, c_x, 0)``: a thrust -> roll/pitch coupling
                that is exactly ZERO on the nominal vehicle.

Grabbing a 0.42 kg crate that ends up hanging 5.9 cm out on the arms moves all three at
once -- ``m`` x1.38, ``I_xx`` x1.83, ``I_yy`` x1.86, ``I_zz`` x1.04, and the centre of mass
1.63 cm off the thrust axis.  The autopilot is never told: it keeps flying on the nominal
mass and trims the standing torque away with its rate integrator (a steady -0.98 / -1.36 on
the roll and pitch commands while carrying, 0.00 when empty), so the crate is not visible in
the vehicle's *behaviour* -- only in the residual, which is the filter's job.

The truth is the same recursion with the true ``(m, I, c)``: it isolates the estimation
question rather than the discretisation one.  What the filter is NOT given, and has to absorb:
the gyroscopic term is computed from the NOMINAL inertia in the filter's model and from the
true one in the truth, so a real (small) model error survives the payload event -- the
departure directions here move ``B`` only.  Remaining rig simplifications, stated so they can
be argued with: no rotor/motor lag, no aerodynamic drag, no slung-load pendulum (the crate is
rigidly gripped), and the wind is a disturbance ACCELERATION rather than a force, so it does
not weaken by ``m0/m = 0.72`` while the crate is aboard.  The flight envelope is +-29 degrees
of bank.

The autopilot flies on an alpha-beta observer of the MEASUREMENTS, never on the truth: 0004
measured that flying on the true state correlates ``u`` with process noise the filter cannot
see and biases identification by +50%.  ``u`` must be measurable from the filter's own
information set.  ``fly()`` is the same mission with an ESTIMATOR in that seat instead --
the aircraft flown on whatever filter you hand it -- which is legal for the same reason: an
estimate is a function of past measurements and inputs, so it stays inside the information
set that observer was standing in for.

The mission's PACE is a parameter.  ``set_regime(f)`` runs every noise regime, every hover
and the mission itself ``f`` times longer, by flying each cruise tour ``f`` times rather than
flying the same waypoints slower -- leg duration sets the bank angle, and the bank is the
excitation that separates ``1/I`` from the centre-of-mass coupling.  The default, ``1.0``, is
the rig `0008` measures, step boundaries included.

Sensors (m = 12), the arm rig's fusion shape one level up -- a bad absolute sensor and a good
dynamic one per axis:

    GPS position   (x, y, z)          sigma 0.30 m        bad, absolute
    GPS velocity   (vx, vy, vz)       sigma 0.035 m/s     good, dynamic
    AHRS attitude  (roll, pitch, yaw) sigma 0.030 rad     bad, absolute
    rate gyro      (p, q, r)          sigma 0.004 rad/s   good, dynamic

Nothing in here is tuned to the filter: it is a vehicle, a job, and a schedule of things
going wrong.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from lucid import LucidFilter                                          # noqa: E402

# ----------------------------------------------------------------- the vehicle
G = 9.81
DT = 0.01
M0 = 1.10                                     # dry mass (kg)
I0 = np.array([0.0150, 0.0150, 0.0270])       # dry inertia (kg m^2)
ALPHA = 10.0                                  # the torque input's unit (rad/s^2)
ARM = 0.26                                    # rotor arm length -- geometry, for the drawing
N, P, M = 12, 5, 12

M_P = 0.42                                    # the crate
D_P = np.array([0.048, -0.034, -0.200])       # where it ends up hanging, in body coords

PX, VX, AT, OM = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12)


def inertia(carry: bool):
    """(mass, principal inertia, centre-of-mass offset) with and without the crate."""
    if not carry:
        return M0, I0.copy(), np.zeros(3)
    m = M0 + M_P
    c = (M_P / m) * D_P
    d = D_P - c
    sq = lambda v: np.array([v[1] ** 2 + v[2] ** 2, v[0] ** 2 + v[2] ** 2,   # noqa: E731
                             v[0] ** 2 + v[1] ** 2])
    return m, I0 + M_P * sq(d) + M0 * sq(c), c


M_FULL, I_FULL, C_FULL = inertia(True)


def Rmat(att):
    """ZYX (yaw-pitch-roll) rotation, body -> world."""
    r, p, y = float(att[0]), float(att[1]), float(att[2])
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def Tmat(att):
    """Body rates -> Euler-angle rates, for ZYX.

    The state carries what a rate gyro actually reads -- **body** rates -- so the attitude
    kinematics are ``d(att)/dt = T(att) om_b``, not ``d(att)/dt = om``.  Treating the two as
    the same is worth 9-20x the gyro's own noise on this rig's flight envelope (max 114x), so
    it is not a rounding error; it is the difference between a fair rig and a flattering one.
    """
    r, p = float(att[0]), float(att[1])
    cr, sr, tp, cp = math.cos(r), math.sin(r), math.tan(p), math.cos(p)
    return np.array([[1.0, sr * tp, cr * tp],
                     [0.0, cr, -sr],
                     [0.0, sr / cp, cr / cp]])


def Bof(att, m, I, c):
    """The one-step input map at attitude ``att`` for a vehicle of ``(m, I, c)``."""
    B = np.zeros((N, P))
    Tm = Tmat(att)
    th = G * (M0 / m) * Rmat(att)[:, 2]                     # thrust: uT = T / (m0 g)
    B[PX, 0], B[VX, 0] = 0.5 * DT ** 2 * th, DT * th
    ang = np.array([-c[1], c[0], 0.0]) * (M0 * G / I)       # off-centre thrust -> torque
    B[AT, 0], B[OM, 0] = 0.5 * DT ** 2 * (Tm @ ang), DT * ang
    eff = np.diag(ALPHA * I0 / I)                           # torque effectiveness
    B[AT, 1:4], B[OM, 1:4] = 0.5 * DT ** 2 * (Tm @ eff), DT * eff
    B[2, 4], B[5, 4] = -0.5 * DT ** 2, -DT                  # gravity (u = G)
    return B


def Fof(att, om, I):
    """The one-step transition at ``(att, om)`` -- state-dependent, and it has to be.

    Two terms make it so, and both are real rigid-body physics rather than rig decoration:
    the attitude kinematics ``T(att)``, and the gyroscopic coupling ``om x I om``, which is
    linear in ``om`` with state-dependent coefficients (``om x I om = [om]x I om``) and so
    lives in ``F`` exactly.
    """
    F = np.eye(N)
    F[PX, VX] = DT * np.eye(3)
    Tm = Tmat(att)
    F[AT, OM] = DT * Tm
    sk = np.array([[0.0, -om[2], om[1]], [om[2], 0.0, -om[0]], [-om[1], om[0], 0.0]])
    gyro = -(sk * I[None, :]) / I[:, None]                  # -I^-1 [om]x I
    F[AT, OM] += 0.5 * DT ** 2 * (Tm @ gyro)
    F[OM, OM] += DT * gyro
    return F


F0 = Fof(np.zeros(3), np.zeros(3), I0)
B_NOM = Bof(np.zeros(3), M0, I0, np.zeros(3))


def base(x):
    """The supplied dynamics: a callable, because BOTH F and B move with the operating point."""
    return Fof(x[AT], x[OM], I0), Bof(x[AT], M0, I0, np.zeros(3))


# ------------------------------------------------- the six physical departure directions
# Each is a callable of the state -- on a real vehicle the direction a physical parameter
# pushes in ROTATES with the operating point (0007's rule; here the thrust axis tilts).
def _thrust_dir(x):
    B = np.zeros((N, P))
    e3 = G * Rmat(x[AT])[:, 2]
    B[PX, 0], B[VX, 0] = 0.5 * DT ** 2 * e3, DT * e3
    return np.zeros((N, N)), B


def _torque_dir(k):
    def d(x):
        B = np.zeros((N, P))
        B[AT, 1 + k] = 0.5 * DT ** 2 * ALPHA * Tmat(x[AT])[:, k]
        B[9 + k, 1 + k] = DT * ALPHA
        return np.zeros((N, N)), B
    return d


def _com_dir(k):
    """Thrust -> torque coupling: ``k=0`` a centre-of-mass offset along body x (pitches),
    ``k=1`` along body y (rolls).  Zero on the nominal vehicle -- this direction exists only
    to be found."""
    row, sgn = (1, 1.0) if k == 0 else (0, -1.0)
    gain = sgn * M0 * G / I0[row]

    def d(x):
        B = np.zeros((N, P))
        B[AT, 0] = 0.5 * DT ** 2 * gain * Tmat(x[AT])[:, row]
        B[9 + row, 0] = DT * gain
        return np.zeros((N, N)), B
    return d


DEPARTURES = [_thrust_dir, _torque_dir(0), _torque_dir(1), _torque_dir(2),
              _com_dir(0), _com_dir(1)]
DEP_NAMES = ["1/m", "1/Ixx", "1/Iyy", "1/Izz", "com-x", "com-y"]

# ------------------------------------------------------------------ noise, base magnitudes
SIG_F, SIG_T = 0.45, 2.0                      # disturbance accel (m/s^2) and ang. accel
_GW = np.zeros((N, 6))
for _i in range(3):
    _GW[_i, _i], _GW[3 + _i, _i] = 0.5 * DT ** 2, DT
    _GW[6 + _i, 3 + _i], _GW[9 + _i, 3 + _i] = 0.5 * DT ** 2, DT
SIGW = np.array([SIG_F] * 3 + [SIG_T] * 3)
Q0 = (_GW * SIGW ** 2) @ _GW.T + 1e-16 * np.eye(N)
H = np.eye(N)
SV = np.array([0.30] * 3 + [0.035] * 3 + [0.030] * 3 + [0.004] * 3)
R0 = SV ** 2

# process-eigenmode -> physical disturbance axis, for the chip grid (Q0 is rank 6)
_lam, _V = np.linalg.eigh(Q0)
MODE_OF_AXIS = {}
for _k in np.argsort(_lam)[-6:]:
    _j = int(np.argmax([abs(_V[:, _k] @ (_GW[:, a] / np.linalg.norm(_GW[:, a])))
                        for a in range(6)]))
    MODE_OF_AXIS[_j] = int(_k)

# ------------------------------------------------------------------ the mission and the job
# The mission is written once, at REGIME = 1 -- a 32 s delivery run -- and then STRETCHED by
# the demo's pacing factor (`set_regime`).  Everything below is what that stretch is derived
# from, so the schedule and the flying stay in step with each other whatever the factor is.
GPS_MULT, GYRO_MULT, WIND_MULT = 12.0, 12.0, 6.0

# The regimes, as fractions of the mission -- the numbers are 950/3200, 1300/3200, ... so
# `set_regime(1.0)` reproduces the step boundaries 0008 measured exactly.
_PHASE_FRAC = [("calm", 0.0, 0.296875), ("WIND", 0.296875, 0.40625),
               ("calm", 0.40625, 0.484375), ("MULTIPATH", 0.484375, 0.59375),
               ("calm", 0.59375, 0.78125), ("VIBRATION", 0.78125, 0.890625),
               ("calm", 0.890625, 1.0)]
# The crate is grabbed and released partway through the two HOVER legs -- the legs whose
# target repeats the one before, the only two in the mission.  Placing the events that way
# rather than at a step number keeps them inside the hover at any pacing factor.
_PICK_FRAC, _DROP_FRAC = (6.2 - 5.9) / 1.3, (21.5 - 21.0) / 1.6

# The mission in stages.  A "tour" is a loop of cruise legs that ends where it began and can
# therefore be FLOWN AGAIN; a "fixed" stage is a manoeuvre whose legs are stretched instead.
# The split matters: leg duration is what sets how hard the vehicle banks, and the bank is
# what separates the torque effectiveness (1/I) from the thrust->torque coupling (c).  A
# longer mission must therefore be MORE LAPS, not slower ones -- flying the same waypoints at
# half speed would quarter the commanded acceleration and take the excitation with it.
#                (duration, (x, y, z, yaw) at the end of the leg)
_STAGES = [("fixed", [(0.8, (0.00, 0.00, 1.20, 0.00))]),                # climb to hover
           ("tour",  [(1.5, (-0.60, -1.15, 1.35, 0.00)),                # out to the pick-up
                      (2.1, (-1.22, -0.79, 1.10, 0.00))]),
           ("fixed", [(1.5, (-1.22, -0.79, 0.77, 0.00)),                # descend
                      (1.3, (-1.22, -0.79, 0.77, 0.00)),                # the pick-up hover
                      (1.8, (-1.22, -0.79, 1.55, 0.00))]),              # climb out, loaded
           ("tour",  [(1.9, (0.05, -0.25, 1.72, 0.50)),                 # the carry legs
                      (1.9, (0.95, 1.10, 1.45, 0.95)),                  # (and a yaw slew)
                      (1.9, (1.50, 0.25, 1.78, 0.35)),
                      (1.8, (0.30, 0.95, 1.40, -0.25)),
                      (1.8, (1.35, 1.30, 1.70, 0.30)),
                      (1.5, (1.40, 0.94, 1.15, 0.00))]),
           ("fixed", [(1.2, (1.40, 0.94, 0.77, 0.00)),                  # descend
                      (1.6, (1.40, 0.94, 0.77, 0.00)),                  # the set-down hover
                      (1.6, (1.40, 0.94, 1.55, 0.00))]),                # climb out, empty
           ("tour",  [(2.0, (0.20, 0.15, 1.35, -0.45)),                 # the way home
                      (2.0, (-0.95, 0.90, 1.65, -0.15)),
                      (1.8, (0.15, -0.60, 1.25, 0.25)),
                      (2.0, (-0.14, -0.22, 1.30, 0.00))])]
_NOMINAL = 32.0                               # seconds of mission at REGIME = 1


def _build(factor):
    """The waypoint list, T, the payload events and the regimes at a pacing ``factor``.

    Tours are flown ``round(factor)`` times and the fixed manoeuvres are stretched to make up
    the rest, so an INTEGER factor scales the whole timeline by exactly that factor -- every
    regime, every hover and the mission itself -- and the fractions above land where they did.
    """
    tour = sum(d for k, legs in _STAGES if k == "tour" for d, _ in legs)
    held = sum(d for k, legs in _STAGES if k == "fixed" for d, _ in legs)
    rep = max(1, int(round(factor)))
    stretch = min(4.0, max(0.5, (factor * _NOMINAL - rep * tour) / held))
    # Leg boundaries are accumulated in STEPS, not seconds: they then land exactly on the
    # sample grid the rest of the rig indexes by, and `set_regime(1.0)` reproduces the
    # hand-written mission bit for bit rather than to within a rounding of 1e-15.
    way, k = [], 0
    for kind, legs in _STAGES:
        for _ in range(rep if kind == "tour" else 1):
            for d, tgt in legs:
                dk = max(1, int(round(d * (1.0 if kind == "tour" else stretch) / DT)))
                way.append((round(k * DT, 10), round((k + dk) * DT, 10), tgt)); k += dk
    n = k
    phases = [(nm, int(round(a * n)), int(round(b * n))) for nm, a, b in _PHASE_FRAC]
    hover = [(t0, t1) for i, (t0, t1, tgt) in enumerate(way) if i and way[i - 1][2] == tgt]
    ev = [int(round((t0 + f * (t1 - t0)) / DT))
          for (t0, t1), f in zip(hover, (_PICK_FRAC, _DROP_FRAC))]
    return way, n, ev[0], ev[1], phases


def set_regime(factor=1.0):
    """Set the demo's pacing: every regime, hover and mission stage runs ``factor`` times
    longer.  ``1.0`` is the rig `0008` measures and is the default -- the animation asks for
    more so that a viewer has time to read each regime before the next one starts."""
    global REGIME, _WAY, T, T_PICK, T_DROP, PHASES
    REGIME = float(factor)
    _WAY, T, T_PICK, T_DROP, PHASES = _build(REGIME)
    return T


REGIME = 1.0
_WAY, T, T_PICK, T_DROP, PHASES = _build(REGIME)


def reference():
    """Minimum-jerk waypoint reference: (T, 4) position+yaw, and its first two derivatives."""
    r0 = np.array(_WAY[0][2], float)
    pos = np.zeros((T, 4)); vel = np.zeros((T, 4)); acc = np.zeros((T, 4))
    t = np.arange(T) * DT
    a = r0
    for (t0, t1, b) in _WAY:
        b = np.array(b, float)
        sel = (t >= t0) & (t < t1)
        Tm = t1 - t0
        if sel.any():
            tau = (t[sel] - t0) / Tm
            s0 = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
            s1 = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / Tm
            s2 = (60 * tau - 180 * tau ** 2 + 120 * tau ** 3) / Tm ** 2
            d = (b - a)[None, :]
            pos[sel] = a[None, :] + d * s0[:, None]
            vel[sel] = d * s1[:, None]
            acc[sel] = d * s2[:, None]
        a = b
    pos[t >= _WAY[-1][1]] = a
    return pos, vel, acc


def schedule(gps=None, gyro=None, wind=None):
    """Per-step sensor sigmas and disturbance sigmas -- the noise regimes.

    The three multipliers default to the rig's own (``GPS_MULT``, ``GYRO_MULT``,
    ``WIND_MULT``); they are arguments so that a demo can say what it is changing at the
    point it changes it, rather than by editing the rig underneath 0008.
    """
    gps = GPS_MULT if gps is None else gps
    gyro = GYRO_MULT if gyro is None else gyro
    wind = WIND_MULT if wind is None else wind
    sv = np.tile(SV, (T, 1))
    sw = np.tile(SIGW, (T, 1))
    for name, a, b in PHASES:
        if name == "MULTIPATH":
            sv[a:b, 0:3] *= gps
        elif name == "VIBRATION":
            sv[a:b, 9:12] *= gyro
        elif name == "WIND":
            sw[a:b, 0:3] *= wind
    return sv, sw


def dropouts(on=0.5, off=0.5, phase="MULTIPATH", sensors=slice(0, 6)):
    """Which sensors READ, per step: a (T, m) boolean, ``False`` where nothing arrives.

    A degraded fix and no fix at all are different failures and the second one is the one an
    airframe actually meets -- a building goes past, the constellation is gone, and both the
    position and the velocity solution go with it, because they come out of the same receiver.
    There is then nothing to distrust: what carries the estimate across the gap is the model,
    and whether that model is the aircraft you are flying is exactly the question this rig
    asks.  Off by default; the demo turns it on inside the ``MULTIPATH`` regime.
    """
    ok = np.ones((T, M), bool)
    per, gap = int(round((on + off) / DT)), int(round(off / DT))
    for name, a, b in PHASES:
        if name == phase:
            for t0 in range(a, b, max(per, 1)):
                ok[t0:min(t0 + gap, b), sensors] = False
    return ok


def draw_noise(seed, sv, sw):
    """The disturbance and sensor noise for one mission, drawn UP FRONT.

    Two aircraft flown on two different filters take two different paths, so the only way to
    put them in the same air is to fix the noise by step index before either takes off.  A
    per-step draw inside the loop would give each of them its own weather.
    """
    rng = np.random.default_rng(seed)
    return sw * rng.standard_normal((T, 6)), sv * rng.standard_normal((T, M))


class Autopilot:
    """Cascade PID on an alpha-beta observer of the measurements.  It is told the NOMINAL
    mass and inertia and never learns: the crate is a disturbance it trims away, which is
    precisely why the vehicle's behaviour does not give the payload away."""

    KP_P, KV_P, KA_A, KW_A = 0.075, 0.30, 0.075, 0.55         # observer gains
    KP = np.array([3.4, 3.4, 11.0])                           # position loop (z is stiffer)
    KD = np.array([3.2, 3.2, 7.6])
    KI = np.array([1.2, 1.2, 14.0])
    KPA, KDA, KIA = 90.0, 16.0, 26.0                          # attitude loop

    def __init__(self):
        self.p = np.array([0.0, 0.0, 1.2]); self.v = np.zeros(3)
        self.a = np.zeros(3); self.w = np.zeros(3)
        self.ei = np.zeros(3); self.eia = np.zeros(3)

    def observe(self, y):
        self.p = self.p + DT * self.v
        self.a = self.a + DT * (Tmat(self.a) @ self.w)
        self.p += self.KP_P * (y[0:3] - self.p)
        self.v += self.KV_P * (y[3:6] - self.v)
        self.a += self.KA_A * (y[6:9] - self.a)
        self.w += self.KW_A * (y[9:12] - self.w)

    def control(self, ref, dref, ddref):
        ad = (ddref[:3] + self.KP * (ref[:3] - self.p) + self.KD * (dref[:3] - self.v)
              + self.KI * self.ei)
        self.ei = np.clip(self.ei + DT * (ref[:3] - self.p), -1.5, 1.5)
        ad[:2] = np.clip(ad[:2], -4.5, 4.5); ad[2] = np.clip(ad[2], -4.0, 7.0)
        fd = ad + np.array([0.0, 0.0, G])
        nf = float(np.linalg.norm(fd))
        b3 = fd / nf
        # Collective is the desired force PROJECTED on the current body z (the standard
        # geometric form): a tilted vehicle then does not shed lift while it rotates.
        uT = float(fd @ Rmat(self.a)[:, 2]) / G        # ... on the NOMINAL mass
        psi = float(ref[3])
        roll = math.asin(float(np.clip(math.sin(psi) * b3[0] - math.cos(psi) * b3[1],
                                       -0.45, 0.45)))
        pitch = math.atan2(math.cos(psi) * b3[0] + math.sin(psi) * b3[1], b3[2])
        want = np.array([roll, np.clip(pitch, -0.45, 0.45), psi])
        err = want - self.a
        alpha = self.KPA * err - self.KDA * self.w + self.KIA * self.eia
        self.eia = np.clip(self.eia + DT * err, -0.6, 0.6)
        u = np.empty(P)
        u[0] = float(np.clip(uT, 0.2, 2.2))
        u[1:4] = np.clip(alpha / ALPHA, -8.0, 8.0)
        u[4] = G
        return u


def simulate(seed=0, carry=True):
    """Fly the mission.  Returns (U, X, Y, carrying)."""
    rng = np.random.default_rng(seed)
    ref, dref, ddref = reference()
    sv, sw = schedule()
    x = np.zeros(N); x[:3] = ref[0, :3]
    ap = Autopilot()
    X = np.zeros((T, N)); Y = np.zeros((T, M)); U = np.zeros((T, P))
    hold = np.zeros(T, bool)
    y = np.concatenate([x[:3], x[3:6], x[6:9], x[9:12]])
    for k in range(T):
        on = carry and (T_PICK <= k < T_DROP)
        hold[k] = on
        m, I, c = inertia(on)
        ap.observe(y)
        u = ap.control(ref[k], dref[k], ddref[k])
        w = sw[k] * rng.standard_normal(6)
        Gw = _GW.copy()                       # a disturbance TORQUE is a body angular
        Gw[AT, 3:6] = 0.5 * DT ** 2 * Tmat(x[AT])   # acceleration; it reaches the Euler
        x = Fof(x[AT], x[OM], I) @ x + Bof(x[AT], m, I, c) @ u + Gw @ w  # angles through T
        y = x + sv[k] * rng.standard_normal(M)
        X[k], Y[k], U[k] = x, y, u
    return U, X, Y, hold


# ------------------------------------------------- flying ON the filter (the closed loop)
# `simulate` above flies the alpha-beta observer of `Autopilot`, which is what 0008 measures.
# The demo asks a different question -- not "how well can each filter track this flight" but
# "how well does this aircraft FLY when that filter is the one in the loop" -- and for that
# the estimator has to be the autopilot's only source of state.  That is legal for exactly the
# reason 0004's closed-loop-bias probe demands: an estimate is a function of past measurements
# and past inputs, so `u` stays inside the filter's own information set.  Flying on the TRUTH
# would not be, and is what biased identification by +50% there.


def fly(estimate, W, V, carry=True, sense=None):
    """Fly the mission with ``estimate`` in the loop.  Returns ``(U, X, Y, XH, carrying)``.

    ``estimate(k, y, u)`` is handed the new measurement and the input that produced it, and
    returns its estimate of the state ``y`` measures.  The autopilot then commands off that
    estimate alone.  ``W, V`` come from :func:`draw_noise` -- pass the same pair to every
    contender.  ``sense`` is an optional :func:`dropouts` mask; a sensor that is not reading
    arrives as ``nan``, which is how ``LucidFilter`` is told a channel is absent rather than
    zero, and ``Y`` carries those entries out as ``nan`` for the drawing.

    The bookkeeping matches ``filter(Y, U)`` exactly: ``X[k]`` is the state after ``U[k]``
    was applied, ``Y[k]`` measures it, and the command at step ``k`` is computed from the
    posterior for ``X[k-1]`` -- the last thing the estimator could possibly have known.
    """
    ref, dref, ddref = reference()
    x = np.zeros(N); x[:3] = ref[0, :3]
    ap = Autopilot()
    X = np.zeros((T, N)); Y = np.zeros((T, M)); U = np.zeros((T, P)); XH = np.zeros((T, N))
    hold = np.zeros(T, bool)
    xh = np.zeros(N); xh[:3] = ref[0, :3]
    for k in range(T):
        on = carry and (T_PICK <= k < T_DROP)
        hold[k] = on
        m, I, c = inertia(on)
        ap.p, ap.v = xh[PX].copy(), xh[VX].copy()
        ap.a, ap.w = xh[AT].copy(), xh[OM].copy()
        u = ap.control(ref[k], dref[k], ddref[k])
        Gw = _GW.copy()                       # a disturbance TORQUE is a body angular
        Gw[AT, 3:6] = 0.5 * DT ** 2 * Tmat(x[AT])   # acceleration; it reaches the Euler
        x = Fof(x[AT], x[OM], I) @ x + Bof(x[AT], m, I, c) @ u + Gw @ W[k]   # angles through T
        y = x + V[k]
        if sense is not None:
            y = np.where(sense[k], y, np.nan)
        X[k], Y[k], U[k] = x, y, u
        xh = np.asarray(estimate(k, y, u), float)
        XH[k] = xh
    return U, X, Y, XH, hold


def fixed_pilot(scale=None):
    """A FIXED Kalman filter as an in-the-loop estimator: the nominal airframe, one ``(Q, R)``
    for the whole mission.  ``scale`` is ``(q, r_gps_pos, r_gps_vel, r_ahrs, r_gyro)`` --
    variance multipliers on the base magnitudes, which is the only freedom a fixed filter has
    and the freedom :func:`tune_fixed` searches."""
    q, rp, rv, ra, rg = (1.0, 1.0, 1.0, 1.0, 1.0) if scale is None else scale
    Rd = R0 * np.concatenate([np.full(3, rp), np.full(3, rv), np.full(3, ra), np.full(3, rg)])
    Q = q * Q0
    st = {"m": np.zeros(N), "P": np.eye(N)}
    st["m"][:3] = reference()[0][0, :3]

    def estimate(k, y, u):
        m, Pm = st["m"], st["P"]
        Fk = Fof(m[AT], m[OM], I0)
        mp = Fk @ m + Bof(m[AT], M0, I0, np.zeros(3)) @ u
        Pp = Fk @ Pm @ Fk.T + Q
        obs = np.flatnonzero(np.isfinite(y))          # the channels that actually read
        if obs.size == M:                             # H is the identity on this rig
            S = Pp + np.diag(Rd)
            K = np.linalg.solve(S.T, Pp.T).T
            st["m"] = mp + K @ (y - mp)
            st["P"] = Pp - K @ Pp
        elif obs.size:
            Ps = Pp[:, obs]
            S = Pp[np.ix_(obs, obs)] + np.diag(Rd[obs])
            K = np.linalg.solve(S.T, Ps.T).T
            st["m"] = mp + K @ (y[obs] - mp[obs])
            st["P"] = Pp - K @ Ps.T
        else:
            st["m"], st["P"] = mp, Pp
        return st["m"]
    return estimate


def lucid_pilot(record=None, hazard=None):
    """``LucidFilter`` as an in-the-loop estimator, stepped one event at a time.

    ``record`` is an optional dict of pre-allocated ``(T, ...)`` arrays -- any of
    ``measurement_scale``, ``process_scale``, ``control``, ``dynamics``, ``fault`` -- filled
    as the flight goes.  The animation reads its right-hand column straight out of it: what
    the filter reports is the same object the autopilot is flying on, not a second pass.
    """
    f = make_filter(hazard)

    def estimate(k, y, u):
        st = f.update(y, u)
        if record is not None:
            for key, arr in record.items():
                arr[k] = getattr(st, key)
        return st.mean
    estimate.filter = f
    return estimate


_TUNE_GRID = (1.0, 4.0, 16.0, 64.0, 256.0, 1024.0)


def tune_fixed(W, V, sense=None, grid=_TUNE_GRID, rounds=3, floor=0.20,
               verbose=False):
    """HINDSIGHT-tune the fixed filter: the ``(q, r...)`` that minimise its own position RMSE
    **on this very mission**, truth included.  Nobody can do this before the flight -- that is
    the point of quoting it, it is the best a fixed filter could have been set to.

    Coordinate descent over a log grid, from several starts, with one constraint that is not
    about accuracy: a flight that puts the aircraft below ``floor`` metres is rejected however
    well it scored, because the metric cannot see a landing it did not survive.  (Tuning on
    estimator error alone, with no such guard, picks a setting that trusts nothing but the
    GPS velocity and flies the aircraft into the ground -- a good score on a flight nobody
    walks away from.)  The multiple starts are the point of the exercise: this number is
    quoted as what a fixed filter COULD have been set to, so leaving it in a local minimum
    would be beating a baseline of one's own making.
    """
    seen = {}

    def cost(c):
        key = tuple(c)
        if key not in seen:
            U, X, Y, XH, _ = fly(fixed_pilot(c), W, V, sense=sense)
            seen[key] = (math.inf if X[:, 2].min() < floor else
                         float(np.sqrt(np.mean(np.sum((XH[:, PX] - X[:, PX]) ** 2, 1)))))
        return seen[key]

    best, arg = math.inf, None
    for start in (grid[0], grid[len(grid) // 2], grid[-1]):
        cur = [float(start)] * 5
        val = cost(cur)
        for _ in range(rounds):
            for j in range(5):
                for g in grid:
                    c = list(cur); c[j] = g
                    v = cost(c)
                    if v < val - 1e-9:
                        val, cur = v, c
                        if verbose:
                            print(f"    tune {tuple(cur)} -> {val:.5f}")
        if val < best:
            best, arg = val, tuple(cur)
    return arg, best


def make_filter(hazard=None):
    """The public filter on this rig.  The default hazard is one fault per mission, so it
    follows ``T`` -- a bound default would still say 1/3200 after ``set_regime(2.0)``."""
    hazard = 1.0 / T if hazard is None else hazard
    return LucidFilter(dynamics=base, control=B_NOM, H=H, process=Q0, measurement=R0,
                       departures=DEPARTURES, faults=hazard)


def kalman(U, Y, sv=None, sw=None, carry=None):
    """A fixed-model Kalman filter.  With no arguments it is the NOMINAL vehicle at the base
    noise -- the same model the lucid filter is given, frozen.  Given the true schedules and
    the true payload window it is the ORACLE: told the noise and the dynamics."""
    Rd = np.tile(R0, (T, 1)) if sv is None else sv ** 2
    Qs = [Q0] * T if sw is None else [(_GW * s ** 2) @ _GW.T + 1e-16 * np.eye(N) for s in sw]
    m = np.zeros(N); m[:3] = reference()[0][0, :3]
    Pm = np.eye(N)
    out = np.zeros((T, N))
    for k in range(T):
        on = False if carry is None else bool(carry[k])
        mm, I, c = inertia(on)
        Fk = Fof(m[AT], m[OM], I)
        mp = Fk @ m + Bof(m[AT], mm, I, c) @ U[k]
        Pp = Fk @ Pm @ Fk.T + Qs[k]
        S = Pp + np.diag(Rd[k])
        K = np.linalg.solve(S.T, Pp.T).T
        m = mp + K @ (Y[k] - mp)
        Pm = Pp - K @ Pp
        out[k] = m
    return out


# ------------------------------------------------- reading the physics back off r.control
def read_payload(ctl):
    """(m_hat, I_hat (3,), c_hat (2,)) from the reported control map, per step.

    Every readout is attitude-free by construction: ``||B[p/v rows, thrust]|| = dt g m0/m``
    is a norm, and the torque and coupling entries do not involve ``R(att)`` at all.  This is
    the public output ``r.control``, not an internal coefficient.

    The coupling entry is a torque per unit thrust *divided by the inertia*, so the physical
    lever arm is recovered by multiplying it back by the inertia the same reading gives.
    """
    ctl = np.asarray(ctl, float)
    thr = np.linalg.norm(ctl[:, VX, 0], axis=1)
    m = DT * G / np.maximum(thr, 1e-12) * M0
    I = DT * ALPHA * I0[None, :] / np.maximum(np.abs(ctl[:, 9:12, 1:4].diagonal(0, 1, 2)),
                                              1e-12)
    cx = ctl[:, 10, 0] * I[:, 1] / (DT * M0 * G)
    cy = -ctl[:, 9, 0] * I[:, 0] / (DT * M0 * G)
    return m, I, np.stack([cx, cy], 1)
