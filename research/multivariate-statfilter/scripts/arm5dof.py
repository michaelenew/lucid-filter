"""The 5-DOF arm rig, with sensors a real arm could actually carry.

**What changed, and why it had to.**  The earlier rig gave every joint a "good accelerometer"
that read that joint's own angular acceleration, ``alpha_j``, through a constant, diagonal
``H``.  No sensor does that.  An inertial sensor is bolted to a *link*, and what it reads is
the motion of the whole chain beneath it, resolved in axes that rotate with the arm.  On this
arm the error is not subtle: joints 2, 3 and 4 all rotate about their local *y*, so their axes
are exactly parallel and a sensor on link 4 reads ``om_2 + om_3 + om_4``, not ``om_4``.  The
coupling between joints 1 and 5 is not even constant -- it sweeps from -0.15 to -0.53 across
the demo trajectory.  Measured against the old rig's own sensor noise, that model error is
**7-13x sigma** on joints 3, 4 and 5.  A filter handed the diagonal ``H`` is being told
something false and loud.

So the second sensor here is the one that actually exists: a **rate gyro on each link**, with
one axis aligned to that joint's rotation axis.  It reads

    y_j = sum_{i <= j} (a_j . a_i)(theta) * om_i

where ``a_i(theta)`` is joint i's axis in world coordinates.  That is linear in the rates but
with **state-dependent coefficients**, and it is not ``H(x) x`` (the ``theta`` block of the
Jacobian does not act on ``theta``), so the rig supplies both the Jacobian and the predicted
measurement -- ``LucidFilter(H=callable)``.  The Jacobian is analytic:

    dy_j / d om_i    = c_ji                                        (i <= j)
    dy_j / d theta_l = sum_{i<=j} om_i * ( [l<j](a_l x a_j).a_i
                                         + [l<i](a_l x a_i).a_j )

using ``d a_i / d theta_l = a_l x a_i`` for ``l < i`` and zero otherwise -- rotating joint l
carries everything downstream of it.  `test_arm_gyro_jacobian` pins it against central
differences.

The rest of the rig: a potentiometer per joint (angle, sigma 0.06 rad ~ 3.4 deg -- the bad
absolute sensor), a triple-integrator plant per joint driven by commanded jerk, and a servo
that tracks minimum-jerk waypoint moves.  **The servo flies on an alpha-beta-gamma tracker of
the potentiometers**, never on the true state: closing the loop on truth hands every filter a
noiseless linear functional of the state for free, which is the closed-loop identification bias
0004 measured on the drone and the same defect one level down.  A failing potentiometer
therefore also disturbs the arm's motion, which is what would really happen.

**What is still not physical here, stated so it can be argued with.**  The plant is five
independent triple integrators in joint space: there is no mass matrix, so no inertial
coupling, no Coriolis and no gravity torque -- the arm is a kinematic benchmark, and a real
manipulator's joints are dynamically coupled as well as kinematically.  The disturbance is a
white JERK, which is smoother than a real vibration (a disturbance torque is an acceleration,
and would enter one derivative lower); jerk is used because it is the input channel, and it is
what keeps each process eigenmode visible to two sensors rather than one.  And the truth is
the same recursion the filter models, so the rig isolates the estimation question rather than
the discretisation one.  None of those flatter the measurement map, which is the thing this
rig exists to get right.
"""
from __future__ import annotations

import math

import numpy as np

NJ, ORDER, DT = 5, 3, 0.01
POT, ACC, JERK = 0.06, 0.020, 0.6             # rad, rad/s^2, rad/s^3
N, M = ORDER * NJ, 2 * NJ

AXES = ["z", "y", "y", "y", "x"]              # each joint's rotation axis, in its own frame
LINK = [0.30, 0.50, 0.40, 0.25, 0.15]         # link lengths (m) -- geometry, for the drawing
HOME = np.array([0.25, 0.55, -0.95, 0.55, 0.0])   # articulated home pose; theta is the deviation

_E = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}

# ------------------------------------------------------------------ the plant
Fb = np.eye(ORDER)
for _i in range(ORDER):
    for _j in range(_i + 1, ORDER):
        Fb[_i, _j] = DT ** (_j - _i) / math.factorial(_j - _i)
G = np.array([DT ** (ORDER - i) / math.factorial(ORDER - i) for i in range(ORDER)])
F = np.kron(np.eye(NJ), Fb)
B = np.kron(np.eye(NJ), G[:, None])
Q0 = np.kron(np.eye(NJ), JERK ** 2 * np.outer(G, G) + 1e-12 * np.eye(ORDER))
R0 = np.tile([POT ** 2, ACC ** 2], NJ)

# eigenmode -> joint map for the process-scale diagnostics (block-diag Q0 -> localised modes)
_lam, _V = np.linalg.eigh(Q0)
MODE_OF_JOINT = {int(np.argmax([np.abs(_V[j * ORDER:(j + 1) * ORDER, k]).sum()
                                for j in range(NJ)])): k
                 for k in np.argsort(_lam)[-NJ:]}


def rot(ax, a):
    c, s = math.cos(a), math.sin(a)
    if ax == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    if ax == "y":
        return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def frames(theta):
    """Joint axes in WORLD coordinates, and each link's world rotation."""
    R = np.eye(3)
    ax = np.empty((NJ, 3))
    Rl = []
    for j in range(NJ):
        ax[j] = R @ _E[AXES[j]]               # joint j's axis, before its own rotation
        R = R @ rot(AXES[j], HOME[j] + theta[j])
        Rl.append(R.copy())
    return ax, Rl


def joints3d(theta):
    """Cartesian joint positions, for the drawing."""
    _, Rl = frames(theta)
    pos = np.zeros(3)
    pts = [pos.copy()]
    for j in range(NJ):
        pos = pos + Rl[j] @ np.array([0, 0, LINK[j]])
        pts.append(pos.copy())
    return np.array(pts)


# ------------------------------------------------- the measurement map, h(x) and its Jacobian
# A link-mounted angular accelerometer on link j, its sensitive axis aligned with joint j's own
# rotation axis.  What it reads is the LINK's angular acceleration resolved on that axis:
#
#     w^world_j   = sum_{i<=j} om_i a_i(theta)
#     y_j         = a_j . d/dt w^world_j
#                 = sum_{i<=j} C[j,i] al_i  +  sum_{i<=j} sum_{l<i} om_l om_i Tt[j,l,i]
#
# with ``C[j,i] = a_j . a_i`` and ``Tt[j,l,i] = a_j . (a_l x a_i)``, because an axis is carried
# by the links before it: ``d a_i/dt = w^world_{i-1} x a_i`` and ``d a_m/d theta_q = a_q x a_m``
# for ``q < m``.  The first sum is the cross-joint coupling the old diagonal rig denied; the
# second is quadratic in the rates, which is why ``h(x)`` is not ``H(x) x`` and the rig hands
# the filter the predicted measurement as well as the Jacobian.
_LOW = np.tril(np.ones((NJ, NJ)))                  # i <= j
_SW = np.triu(np.ones((NJ, NJ)), 1)                # l < i
_CARRY = np.triu(np.ones((NJ, NJ)), 1)             # q < m: joint q carries axis m


def measure(x):
    """``(H, y_pred)`` at state ``x`` -- the callable `LucidFilter` is handed."""
    v = np.asarray(x, float)
    th, om, al = v[0::ORDER], v[1::ORDER], v[2::ORDER]
    ax, _ = frames(th)
    C = ax @ ax.T
    X = np.cross(ax[:, None, :], ax[None, :, :])           # X[l, i] = a_l x a_i
    Tt = np.einsum("jd,lid->jli", ax, X)                   # a_j . (a_l x a_i)
    dax = _CARRY[:, :, None] * X                           # d a_m / d theta_q, zero for q >= m
    Cl = C * _LOW
    W = _LOW[:, None, :] * _SW[None, :, :] * Tt            # the live quadratic weights
    yp_acc = Cl @ al + np.einsum("jli,l,i->j", W, om, om)

    # d C[j,i] / d theta_q, and d Tt[j,l,i] / d theta_q
    dC = np.einsum("qjd,id->qji", dax, ax) + np.einsum("jd,qid->qji", ax, dax)
    dT = (np.einsum("qjd,lid->qjli", dax, X)
          + np.einsum("jd,qlid->qjli", ax, np.cross(dax[:, :, None, :], ax[None, None, :, :]))
          + np.einsum("jd,qlid->qjli", ax, np.cross(ax[None, :, None, :], dax[:, None, :, :])))
    dy_dth = (np.einsum("qji,ji,i->jq", dC, _LOW, al)
              + np.einsum("qjli,jli,l,i->jq", dT, _LOW[:, None, :] * _SW[None, :, :], om, om))
    dy_dom = np.einsum("jpi,i->jp", W, om) + np.einsum("jlp,l->jp", W, om)

    Hm = np.zeros((M, N))
    yp = np.zeros(M)
    Hm[0::2, 0::ORDER] = np.eye(NJ)                        # the potentiometer
    yp[0::2] = th
    Hm[1::2, 0::ORDER] = dy_dth
    Hm[1::2, 1::ORDER] = dy_dom
    Hm[1::2, 2::ORDER] = Cl
    yp[1::2] = yp_acc
    return Hm, yp


H_CHAR = measure(np.zeros(N))[0]               # the characteristic linearisation, at the origin


def sense(x, sd, rng):
    """One noisy measurement vector from the TRUE state."""
    return measure(x)[1] + sd * rng.standard_normal(M)


# ------------------------------------------------------------------ the servo
class Servo:
    """Alpha-beta-gamma tracker on the potentiometers, then a pole-placed jerk command.

    It never sees the true state, and it never sees the gyros either: a joint servo runs on
    its own encoder.  A potentiometer that fails therefore moves the arm, which is what a real
    failure does -- and it is why the estimator is worth having.
    """

    # One number, not three: a critically-damped alpha-beta-gamma tracker fixes beta and
    # gamma from alpha.  Alpha is small because differentiating a 0.06 rad potentiometer twice
    # at 100 Hz is a noisy thing to do -- which is the honest cost of the bad sensor, and is
    # why the servo lags rather than chatters.
    ALPHA_T = 0.03
    KV = 2.0 * (2.0 - ALPHA_T) - 4.0 * math.sqrt(1.0 - ALPHA_T)
    KA = KV ** 2 / (2.0 * ALPHA_T)
    # The servo pole is set by what the FEEDBACK SENSOR can support, which is the constraint a
    # real design has and the old rig quietly did not: closing a triple pole at s = -8 on a
    # 0.06 rad potentiometer injects 1.57 rad/s^3 of jerk noise into the arm -- 2.6x the
    # process noise the filter is told about, so the "oracle told the true schedule" would not
    # have been told the truth.  At s = -4 the injection is 0.18, comfortably under it, and
    # tau ~ 0.25 s is what you would actually ship on a sensor this noisy.
    LAM = 4.0

    def __init__(self):
        self.th = np.zeros(NJ)
        self.om = np.zeros(NJ)
        self.al = np.zeros(NJ)
        self.u = np.zeros(NJ)

    def observe(self, y):
        self.th = self.th + DT * self.om + 0.5 * DT ** 2 * self.al
        self.om = self.om + DT * self.al
        self.al = self.al + DT * self.u
        r = y[0::2] - self.th                  # the standard alpha-beta-gamma corrections
        self.th += self.ALPHA_T * r
        self.om += (self.KV / DT) * r
        self.al += (2.0 * self.KA / DT ** 2) * r

    def command(self, rth, rom, ral, rjk):
        self.u = np.clip(rjk + self.LAM ** 3 * (rth - self.th)
                         + 3 * self.LAM ** 2 * (rom - self.om)
                         + 3 * self.LAM * (ral - self.al), -400.0, 400.0)
        return self.u


# ------------------------------------------------------------------ the job
POSES = [np.zeros(NJ),
         np.array([0.50, 0.30, -0.30, 0.25, 0.60]),
         np.array([-0.40, 0.45, 0.25, -0.30, -0.50]),
         np.array([0.20, -0.25, 0.40, 0.35, 0.30]),
         np.zeros(NJ)]
SEGS = [(0.0, 1.5, 0, 0), (1.5, 4.0, 0, 1), (4.0, 5.5, 1, 1), (5.5, 8.0, 1, 2),
        (8.0, 9.5, 2, 2), (9.5, 12.0, 2, 3), (12.0, 13.5, 3, 3), (13.5, 16.0, 3, 4),
        (16.0, 19.0, 4, 4)]


def reference(T):
    """Minimum-jerk waypoint reference: theta, omega, alpha, jerk, each (T, NJ)."""
    th = np.zeros((T, NJ)); om = np.zeros((T, NJ))
    al = np.zeros((T, NJ)); jk = np.zeros((T, NJ))
    t = np.arange(T) * DT
    for (t0, t1, ia, ib) in SEGS:
        a, b = POSES[ia], POSES[ib]
        sel = (t >= t0) & (t < t1)
        if not sel.any():
            continue
        Tm = t1 - t0
        tau = (t[sel] - t0) / Tm
        d = (b - a)[None, :]
        th[sel] = a[None, :] + d * (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5)[:, None]
        om[sel] = d * ((30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / Tm)[:, None]
        al[sel] = d * ((60 * tau - 180 * tau ** 2 + 120 * tau ** 3) / Tm ** 2)[:, None]
        jk[sel] = d * ((60 - 360 * tau + 360 * tau ** 2) / Tm ** 3)[:, None]
    return th, om, al, jk


def simulate(seed, jstd, pot_s, gyro_s):
    """Fly the job.  ``jstd`` is (T,), ``pot_s``/``gyro_s`` are (T, NJ).  Returns (U, S, Y)."""
    T = len(jstd)
    rng = np.random.default_rng(seed)
    rth, rom, ral, rjk = reference(T)
    servo = Servo()
    s = np.zeros(N)
    S = np.zeros((T, N)); Y = np.zeros((T, M)); U = np.zeros((T, NJ))
    sd = np.empty(M)
    sd[0::2], sd[1::2] = pot_s[0], gyro_s[0]
    y = sense(s, sd, rng)
    for k in range(T):
        servo.observe(y)                       # on the MEASUREMENTS, never on s
        U[k] = servo.command(rth[k], rom[k], ral[k], rjk[k])
        s = F @ s + B @ U[k] + B @ (jstd[k] * rng.standard_normal(NJ))
        sd[0::2], sd[1::2] = pot_s[k], gyro_s[k]
        y = sense(s, sd, rng)
        S[k], Y[k] = s, y
    return U, S, Y


def make_filter():
    """The public filter on this rig -- note ``H`` is the callable, not a matrix."""
    from lucid import LucidFilter
    return LucidFilter(dynamics=F, control=B, H=measure, process=Q0, measurement=R0)


def kalman(U, Y, Qs=None, Rs=None):
    """A Kalman filter on the same model, linearising H per step exactly as the lucid one does.

    With no schedules it is the FIXED baseline (base noise); given the true per-step (Q, R) it
    is the oracle.  Both get the correct measurement map -- the comparison is about the noise,
    not about handing one contender a better sensor model than the other.
    """
    T = len(Y)
    m0 = np.zeros(N); P = np.eye(N); out = np.zeros((T, N))
    for k in range(T):
        mp = F @ m0 + B @ U[k]
        Pp = F @ P @ F.T + (Q0 if Qs is None else Qs[k])
        Hk, yp = measure(mp)
        Rk = np.diag(R0 if Rs is None else Rs[k])
        K = Pp @ Hk.T @ np.linalg.inv(Hk @ Pp @ Hk.T + Rk)
        m0 = mp + K @ (Y[k] - yp)
        P = Pp - K @ Hk @ Pp
        out[k] = m0
    return out
