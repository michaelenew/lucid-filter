"""The 5-DOF arm rig: a common kinematic chain, and sensors an arm could actually carry.

**The chain.**  Two orthogonal rotational DOFs at the base (yaw about the vertical, then
shoulder pitch), an elbow carrying two more (elbow pitch, then a roll about the forearm), and
a short wrist holding the effector on the one remaining flex axis:

    J1 yaw z  | riser 0.30 m  | J2 pitch y | upper arm 0.45 m | J3 pitch y | forearm 0.35 m |
    J4 roll z | collar 0.10 m | J5 flex y  | wrist + effector 0.14 m

Four coplanar pitch joints -- the chain this rig used to have -- is not an arm anyone builds;
this is the layout most 5-DOF arms share.  The roll joint is what makes the wrist
interesting: J5's flex axis is carried by the forearm roll, so where it points is a function
of the configuration, not of the drawing.

**The sensors.**  A potentiometer per joint (angle, sigma 0.06 rad ~ 3.4 deg -- the bad
absolute sensor), and a MEMS accelerometer bolted near the distal end of each link, 6 cm off
the link axis, one sensitive axis read (sigma 0.03 m/s^2 -- the good dynamic sensor).  What
that axis reads is proper acceleration,

    y_j = s . R_j(theta)^T ( p_j..(theta, om, al) + g e_z ),

and every piece of it is nonlinear in the state: the mount's linear acceleration is a
configuration-dependent lever-arm map on the joint accelerations plus centripetal and
Coriolis terms quadratic in the rates, and the gravity term is 9.81 m/s^2 resolved in a link
frame that moves with every joint below it -- which is what makes an accelerometer an
inclinometer.  **There is no constant ``H`` here, not even approximately**, so the
measurement map reaches `LucidFilter` as a callable and is linearised at every step, exactly
as a moving ``F`` is.  (An ANGULAR-rate/acceleration sensor on a chain like this is the
near-miss that first motivated the feature but does not always force it: axis-dot-axis
couplings are constant wherever axes are parallel or orthogonal --
`../exploration/0054_physical_sensors.md` keeps that record.)

The Jacobian is by COMPLEX-STEP differentiation: ``h`` is written complex-safe, so
``Im h(x + i eps e_k) / eps`` is the derivative to machine precision, with no subtractive
cancellation and no step size to choose.  One evaluation per column, exact; `0054` pins it
against central differences.

Two physically honest consequences, kept rather than papered over: the riser accelerometer
reads ~nothing (a vertical link's lateral axis sees no gravity change and no lever arm under
yaw), so yaw redundancy comes from the accelerometers on links 2-5 through the chain --
exactly as on real hardware, where absolute yaw needs the encoder or a magnetometer.  And the
servo flies on an alpha-beta-gamma tracker of the POTENTIOMETERS, never on the true state
(0004's closed-loop bias, one level down), at a bandwidth the potentiometer can support: a
triple pole at s = -8 on a 0.06 rad sensor injects 1.57 rad/s^3 of jerk noise -- 2.6x the
process noise the filter is told about, so an "oracle told the true schedule" would not have
been told the truth -- while s = -4 injects 0.18.

**What is still not physical here, stated so it can be argued with.**  The plant is five
independent triple integrators in joint space: no mass matrix, so no inertial coupling, no
Coriolis torques and no gravity load -- the arm is a kinematic benchmark, and the truth is
the same recursion the filter models, isolating the estimation question from the
discretisation one.  The disturbance is a white jerk on the command channel.  None of that
flatters the measurement map, which is what this rig exists to get right.
"""
from __future__ import annotations

import math

import numpy as np

NJ, ORDER, DT = 5, 3, 0.01
POT, ACC, JERK = 0.06, 0.030, 0.6             # rad, m/s^2, rad/s^3
N, M = ORDER * NJ, 2 * NJ

# the chain: yaw + shoulder pitch at the base, pitch + roll at the elbow, one wrist flex
AXES = ["z", "y", "y", "z", "y"]
LINK = [0.30, 0.45, 0.35, 0.10, 0.14]         # each link extends along its own local z
HOME = np.array([0.25, 0.95, -1.40, 0.40, 0.75])   # articulated home; theta is the deviation

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
GRAV = 9.81
MOUNT = np.array([0.06, 0.0, 0.0])            # IMU offset from the link axis, in link coords
SAXIS = np.array([1.0, 0.0, 0.0])             # the sensitive axis, in link coords


def _rot_c(ax, a):
    """Rotation about a named axis, complex-safe and batched: ``a`` (...,) -> (..., 3, 3)."""
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(c), np.zeros_like(c)
    if ax == "z":
        rows = ((c, -s, z), (s, c, z), (z, z, o))
    elif ax == "y":
        rows = ((c, z, s), (z, o, z), (-s, z, c))
    else:
        rows = ((o, z, z), (z, c, -s), (z, s, c))
    return np.stack([np.stack(r, -1) for r in rows], -2)


def _h(x):
    """The measurement, complex-safe and batched over states: ``x`` (..., N) -> (..., M).

    Potentiometers and link accelerometers, interleaved.  A rigid-body sweep down the chain:
    each joint adds ``al_j`` about its (world) axis to the link's angular acceleration and
    ``om_j`` to its rate; the mount point's linear acceleration is the joint origin's plus
    ``dw x d + w x (w x d)``; and the accelerometer reads the PROPER acceleration ``a + g``
    on its own axis, in its own frame.  Batching is what makes the complex-step Jacobian one
    call rather than N+1.
    """
    th, om, al = x[..., 0::ORDER], x[..., 1::ORDER], x[..., 2::ORDER]
    K = x.shape[:-1]
    R = np.broadcast_to(np.eye(3, dtype=x.dtype), K + (3, 3)).copy()
    w = np.zeros(K + (3,), dtype=x.dtype)      # link angular velocity, world
    dw = np.zeros(K + (3,), dtype=x.dtype)     # its derivative, world
    p_acc = np.zeros(K + (3,), dtype=x.dtype)  # linear acceleration of the joint, world
    g = np.array([0.0, 0.0, GRAV], dtype=x.dtype)
    out = np.empty(K + (M,), dtype=x.dtype)
    for j in range(NJ):
        axis = R @ _E[AXES[j]].astype(x.dtype)             # joint axis, world (..., 3)
        dw = dw + al[..., j, None] * axis + om[..., j, None] * np.cross(w, axis)
        w = w + om[..., j, None] * axis
        R = R @ _rot_c(AXES[j], HOME[j] + th[..., j])
        d = R @ (np.array([0.0, 0.0, LINK[j]], dtype=x.dtype) + MOUNT.astype(x.dtype))
        a_m = p_acc + np.cross(dw, d) + np.cross(w, np.cross(w, d))
        out[..., 2 * j] = th[..., j]                       # the potentiometer
        out[..., 2 * j + 1] = ((R @ SAXIS.astype(x.dtype)) * (a_m + g)).sum(-1)
        link = R @ np.array([0.0, 0.0, LINK[j]], dtype=x.dtype)
        p_acc = p_acc + np.cross(dw, link) + np.cross(w, np.cross(w, link))
    return out


_CSTEP = 1e-20


def measure(x):
    """``(H, y_pred)`` at state ``x`` -- the callable `LucidFilter` is handed.

    The value and all N Jacobian columns come from ONE batched complex-step evaluation."""
    x = np.asarray(x, float)
    X = np.repeat(x[None].astype(complex), N + 1, 0)
    X[1:] += 1j * _CSTEP * np.eye(N)
    out = _h(X)
    return out[1:].imag.T / _CSTEP, out[0].real


H_CHAR = measure(np.zeros(N))[0]               # the characteristic linearisation, at the origin


def sense(x, sd, rng):
    """One noisy measurement vector from the TRUE state."""
    return measure(x)[1] + sd * rng.standard_normal(M)


# ------------------------------------------------------------------ the servo
class Servo:
    """Alpha-beta-gamma tracker on the potentiometers, then a pole-placed jerk command.

    It never sees the true state, and it never sees the accelerometers either: a joint servo
    runs on its own encoder.  A potentiometer that fails therefore moves the arm, which is
    what a real failure does -- and it is why the estimator is worth having.
    """

    # One number, not three: a critically-damped alpha-beta-gamma tracker fixes beta and
    # gamma from alpha.  Alpha is small because differentiating a 0.06 rad potentiometer
    # twice at 100 Hz is a noisy thing to do -- the honest cost of the bad sensor, and why
    # the servo lags rather than chatters.
    ALPHA_T = 0.03
    KV = 2.0 * (2.0 - ALPHA_T) - 4.0 * math.sqrt(1.0 - ALPHA_T)
    KA = KV ** 2 / (2.0 * ALPHA_T)
    # The servo pole is set by what the FEEDBACK SENSOR can support -- see the module
    # docstring (s = -8 injects 2.6x the modelled process noise; s = -4 injects 0.3x).
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
         np.array([0.60, 0.40, -0.50, 0.70, 0.60]),
         np.array([-0.55, 0.55, 0.35, -0.80, -0.50]),
         np.array([0.30, -0.35, 0.60, 0.90, 0.35]),
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


def simulate(seed, jstd, pot_s, acc_s):
    """Fly the job.  ``jstd`` is (T,), ``pot_s``/``acc_s`` are (T, NJ).  Returns (U, S, Y)."""
    T = len(jstd)
    rng = np.random.default_rng(seed)
    rth, rom, ral, rjk = reference(T)
    servo = Servo()
    s = np.zeros(N)
    S = np.zeros((T, N)); Y = np.zeros((T, M)); U = np.zeros((T, NJ))
    sd = np.empty(M)
    sd[0::2], sd[1::2] = pot_s[0], acc_s[0]
    y = sense(s, sd, rng)
    for k in range(T):
        servo.observe(y)                       # on the MEASUREMENTS, never on s
        U[k] = servo.command(rth[k], rom[k], ral[k], rjk[k])
        s = F @ s + B @ U[k] + B @ (jstd[k] * rng.standard_normal(NJ))
        sd[0::2], sd[1::2] = pot_s[k], acc_s[k]
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
