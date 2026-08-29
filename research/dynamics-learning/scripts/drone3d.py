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

The one-step map is affine in the state at a given attitude,

    x_{t+1} = F0 x_t + B(att_t; m, I, c) u_t + w_t,

with ``F0`` the (exact) kinematic integrator and every physical parameter living in ``B``:

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

The truth is the same recursion with the true ``(m, I, c)``, which is the rig convention of
the 5-DOF arm (`make_arm5dof_lucid_gif.py`): it isolates the estimation question rather than
the discretisation one.  The gyroscopic term ``om x I om`` and the Euler-rate/body-rate
distinction are dropped, which is a modelling simplification of the RIG, not of the filter --
the flight envelope here stays inside +-29 degrees of bank.

The autopilot flies on an alpha-beta observer of the MEASUREMENTS, never on the truth: 0004
measured that flying on the true state correlates ``u`` with process noise the filter cannot
see and biases identification by +50%.  ``u`` must be measurable from the filter's own
information set.

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


def Bof(att, m, I, c):
    """The one-step input map at attitude ``att`` for a vehicle of ``(m, I, c)``."""
    B = np.zeros((N, P))
    th = G * (M0 / m) * Rmat(att)[:, 2]                     # thrust: uT = T / (m0 g)
    B[PX, 0], B[VX, 0] = 0.5 * DT ** 2 * th, DT * th
    ang = np.array([-c[1], c[0], 0.0]) * (M0 * G / I)       # off-centre thrust -> torque
    B[AT, 0], B[OM, 0] = 0.5 * DT ** 2 * ang, DT * ang
    eff = np.diag(ALPHA * I0 / I)                           # torque effectiveness
    B[AT, 1:4], B[OM, 1:4] = 0.5 * DT ** 2 * eff, DT * eff
    B[2, 4], B[5, 4] = -0.5 * DT ** 2, -DT                  # gravity (u = G)
    return B


F0 = np.eye(N)
F0[PX, VX] = DT * np.eye(3)
F0[AT, OM] = DT * np.eye(3)
B_NOM = Bof(np.zeros(3), M0, I0, np.zeros(3))


def base(x):
    """The supplied dynamics: a callable of the state, because ``B`` rotates with attitude."""
    return F0, Bof(x[AT], M0, I0, np.zeros(3))


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
        B[6 + k, 1 + k], B[9 + k, 1 + k] = 0.5 * DT ** 2 * ALPHA, DT * ALPHA
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
        B[6 + row, 0], B[9 + row, 0] = 0.5 * DT ** 2 * gain, DT * gain
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
T = 3200
GPS_MULT, GYRO_MULT, WIND_MULT = 12.0, 12.0, 6.0
T_PICK, T_DROP = 620, 2150
PHASES = [("calm", 0, 950), ("WIND", 950, 1300), ("calm", 1300, 1550),
          ("MULTIPATH", 1550, 1900), ("calm", 1900, 2500),
          ("VIBRATION", 2500, 2850), ("calm", 2850, T)]

#            t0     t1    (x, y, z, yaw) at t1 -- a punchy delivery run: the legs are short
#                          enough that roll/pitch commands SWING, which is what separates the
#                          torque effectiveness (1/I) from the thrust->torque coupling (c).
_WAY = [(0.0, 0.8, (0.00, 0.00, 1.20, 0.00)),
        (0.8, 2.3, (-0.60, -1.15, 1.35, 0.00)),
        (2.3, 4.4, (-1.22, -0.79, 1.10, 0.00)),
        (4.4, 5.9, (-1.22, -0.79, 0.77, 0.00)),
        (5.9, 7.2, (-1.22, -0.79, 0.77, 0.00)),         # the pick-up hover
        (7.2, 9.0, (-1.22, -0.79, 1.55, 0.00)),
        (9.0, 10.9, (0.05, -0.25, 1.72, 0.50)),         # the carry legs (and a yaw slew)
        (10.9, 12.8, (0.95, 1.10, 1.45, 0.95)),
        (12.8, 14.7, (1.50, 0.25, 1.78, 0.35)),
        (14.7, 16.5, (0.30, 0.95, 1.40, -0.25)),
        (16.5, 18.3, (1.35, 1.30, 1.70, 0.30)),
        (18.3, 19.8, (1.40, 0.94, 1.15, 0.00)),
        (19.8, 21.0, (1.40, 0.94, 0.77, 0.00)),
        (21.0, 22.6, (1.40, 0.94, 0.77, 0.00)),         # the set-down hover
        (22.6, 24.2, (1.40, 0.94, 1.55, 0.00)),
        (24.2, 26.2, (0.20, 0.15, 1.35, -0.45)),
        (26.2, 28.2, (-0.95, 0.90, 1.65, -0.15)),
        (28.2, 30.0, (0.15, -0.60, 1.25, 0.25)),
        (30.0, 32.0, (-0.14, -0.22, 1.30, 0.00))]


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


def schedule():
    """Per-step sensor sigmas and disturbance sigmas -- the noise regimes."""
    sv = np.tile(SV, (T, 1))
    sw = np.tile(SIGW, (T, 1))
    for name, a, b in PHASES:
        if name == "MULTIPATH":
            sv[a:b, 0:3] *= GPS_MULT
        elif name == "VIBRATION":
            sv[a:b, 9:12] *= GYRO_MULT
        elif name == "WIND":
            sw[a:b, 0:3] *= WIND_MULT
    return sv, sw


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
        self.a = self.a + DT * self.w
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
        B = Bof(x[AT], m, I, c)
        x = F0 @ x + B @ u + _GW @ (sw[k] * rng.standard_normal(6))
        y = x + sv[k] * rng.standard_normal(M)
        X[k], Y[k], U[k] = x, y, u
    return U, X, Y, hold


def make_filter(hazard=1.0 / T):
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
        B = Bof(m[AT], mm, I, c)
        mp = F0 @ m + B @ U[k]
        Pp = F0 @ Pm @ F0.T + Qs[k]
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
