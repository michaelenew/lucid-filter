"""Probe 0024 -- WHY is state estimation barely better than non-adaptive?  The crusher regime.

User's case: a robotic arm sits near an industrial crusher.  When the crusher is ON it injects
BOTH dynamical noise (the arm shakes) AND sensor noise (vibration corrupts the encoders).  The
user is surprised adaptation buys almost nothing for state estimation.

Two hypotheses, tested here:
  (A) WRONG MODEL -- every probe so far uses a local-level random walk (F = I): no velocity
      state, nothing to COAST on when the sensors spike.  A robotic arm has momentum
      (x' = v); a kinematic model can ride through a sensor-noise burst on velocity.  This is
      the derivative mode the user asked about.
  (B) WRONG REGIME -- a single mild excursion on one axis dilutes into the other axes' noise;
      the crusher is a big SIMULTANEOUS burst on process AND all sensors.

Compare, in the crusher regime, state RMSE of:
  local-level  {non-adaptive, oracle-adaptive}
  kinematic    {non-adaptive, oracle-adaptive}
Oracle-adaptive = the filter is handed the TRUE Q(t), R(t) schedule -- the ceiling any learned
adaptation can reach.  If kinematic+oracle >> kinematic+nonadaptive while local+oracle ~
local+nonadaptive, the derivative mode is the missing piece and adaptation matters once you
have a model to coast on.
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True)


def kf(Y, F, H, Qseq, Rseq, m0, P0):
    """Vanilla time-varying Kalman filter; returns filtered state estimates."""
    n = F.shape[0]; m = m0.copy(); P = P0.copy(); out = np.zeros((len(Y), n))
    for t, y in enumerate(Y):
        m = F @ m; P = F @ P @ F.T + Qseq[t]
        S = H @ P @ H.T + Rseq[t]; K = P @ H.T @ np.linalg.inv(S)
        m = m + K @ (y - H @ m); P = P - K @ H @ P
        out[t] = m
    return out


def run(seed=0):
    T = 600; dt = 1.0; on = slice(T // 3, 2 * T // 3)
    rng = np.random.default_rng(seed)
    # --- true system: a 1-DOF arm with momentum, position measured by an encoder
    F = np.array([[1.0, dt], [0.0, 1.0]])            # x' = x + v, v' = v
    G = np.array([[0.5 * dt * dt], [dt]])            # accel enters position and velocity
    H = np.array([[1.0, 0.0]])                       # encoder sees position
    q_base, r_base = 1e-3, 0.04                      # quiet: tiny accel noise, modest encoder noise
    q_on, r_on = 0.30, 4.0                           # crusher ON: arm shakes (x300) AND encoder swamped (x100)
    qt = np.full(T, q_base); rt = np.full(T, r_base); qt[on] = q_on; rt[on] = r_on
    # simulate
    x = np.zeros(2); X = np.zeros((T, 2)); Y = np.zeros((T, 1))
    for t in range(T):
        a = np.sqrt(qt[t]) * rng.standard_normal()
        x = F @ x + (G * a).ravel(); X[t] = x
        Y[t] = H @ x + np.sqrt(rt[t]) * rng.standard_normal()
    Qk = np.array([qt[t] * (G @ G.T) for t in range(T)])
    Rk = np.array([[[rt[t]]] for t in range(T)])
    Qbase = q_base * (G @ G.T); Rbase = np.array([[r_base]])
    P0 = np.eye(2); m0 = np.array([Y[0, 0], 0.0])

    def rmse(est, truth, sl):
        return np.sqrt(((est[sl, 0] - truth[sl, 0]) ** 2).mean())   # position RMSE

    # --- kinematic model (correct F): non-adaptive vs oracle-adaptive
    kin_na = kf(Y, F, H, np.array([Qbase] * T), np.array([Rbase] * T), m0, P0)
    kin_or = kf(Y, F, H, Qk, Rk, m0, P0)
    # --- local-level model (F = I, single position state): non-adaptive vs oracle-adaptive
    Fll = np.array([[1.0]]); Hll = np.array([[1.0]])
    qll_base = q_base; Qll_na = np.array([[[qll_base]]] * T); Rll = np.array([[[r_base]]] * T)
    Qll_or = np.array([[[qt[t] if t in range(*on.indices(T)) else qll_base]]] for t in range(T)) if False else \
        np.array([[[max(qt[t], qll_base)]]] for t in range(T))
    Qll_or = np.array([[[max(qt[t], qll_base)]]] for t in range(T))
    ll_na = kf(Y, Fll, Hll, np.array([[[qll_base]]] * T), np.array([[[r_base]]] * T), np.array([Y[0, 0]]), np.eye(1))
    ll_or = kf(Y, Fll, Hll, np.array([[[qt[t]]] for t in range(T)]), np.array([[[rt[t]]] for t in range(T)]),
               np.array([Y[0, 0]]), np.eye(1))

    for nm, sl in [("crusher ON", on), ("full run", slice(0, T))]:
        print(f"\n position RMSE, {nm}:")
        print(f"   raw encoder (y vs x)        {np.sqrt(((Y[sl,0]-X[sl,0])**2).mean()):.3f}")
        print(f"   LOCAL-LEVEL non-adaptive    {rmse(np.c_[ll_na,ll_na],X,sl):.3f}")
        print(f"   LOCAL-LEVEL oracle-adaptive {rmse(np.c_[ll_or,ll_or],X,sl):.3f}   "
              f"(adapt gain {rmse(np.c_[ll_na,ll_na],X,sl)/rmse(np.c_[ll_or,ll_or],X,sl):.2f}x)")
        print(f"   KINEMATIC   non-adaptive    {rmse(kin_na,X,sl):.3f}")
        print(f"   KINEMATIC   oracle-adaptive {rmse(kin_or,X,sl):.3f}   "
              f"(adapt gain {rmse(kin_na,X,sl)/rmse(kin_or,X,sl):.2f}x)")
        print(f"   >> kinematic vs local-level (both oracle): "
              f"{rmse(np.c_[ll_or,ll_or],X,sl)/rmse(kin_or,X,sl):.2f}x")


if __name__ == "__main__":
    for s in range(3):
        print(f"===== seed {s} ====="); run(s)
