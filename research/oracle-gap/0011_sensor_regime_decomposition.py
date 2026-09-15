"""0011 -- why the arm's SENSOR regime sits at 2.7x the oracle, and how much of that is learning lag.

The 0054 arm rig, SENSOR phase (steps 250-500): every accelerometer's noise sd x15 (+5.42 nats of log-variance),
scored from step 290.  Decomposition, per seed:
  oracle      Kalman told the true per-step (Q, R)
  fixed       Kalman at the base noise (no learning at all)
  plug-in     Kalman fed the FILTER'S OWN estimated scales each step  -> what the filter's scale estimates are worth
                                                                        on their own, without the bank/mixture structure
  lag-oracle  true schedule, but the accelerometer R switches d steps late (d = the filter's measured 90% rise time)
  bias-oracle true schedule, but the accelerometer R at the filter's SETTLED estimate instead of the truth
  filter      the shipped LucidFilter
and the estimated accelerometer log-scale trajectory: rise times to 50% / 90% of +5.42 nats, settled mean and sd,
what the pots and the process scales do meanwhile, and the exit decay.
    python 0011_sensor_regime_decomposition.py [seed]"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "multivariate-statfilter", "scripts")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import arm5dof as A
import importlib.util
spec = importlib.util.spec_from_file_location("p54", os.path.join(HERE, "..", "multivariate-statfilter", "exploration", "0054_physical_sensors.py"))
m54 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
from lucid import LucidFilter
np.seterr(all="ignore")
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
T = m54.T; A0, B0 = 250, 500; SK = m54.SKIP; STEP = 2 * np.log(m54.ACC_MULT)
jstd, pot, acc = m54.schedule(); U, S_, Y = A.simulate(seed, jstd, pot, acc)
Qs = [j ** 2 * (A.B @ A.B.T) for j in jstd]
Rs = [np.concatenate([[pot[k, j] ** 2, acc[k, j] ** 2, acc[k, j] ** 2] for j in range(A.NJ)]) for k in range(T)]
Pt = m54.tip(S_)
def score(Z, a, b):
    return m54.rms(m54.tip(Z), Pt, slice(a, b))
ACC = np.array([i for i in range(A.M) if i % 3 != 0]); POTS = np.array([i for i in range(A.M) if i % 3 == 0])
t0 = time.perf_counter()
f = LucidFilter(dynamics=A.F, control=A.B, H=A.measure, process=A.Q0, measurement=A.R0)
r = f.filter(Y, U); dt = time.perf_counter() - t0
est = np.asarray(r.mean); eta = np.asarray(r.measurement_scale); xi = np.asarray(r.process_scale)
e0 = f._members[0]; V, lam = e0.V, e0.lam
orc = A.kalman(U, Y, Qs, Rs); fix = A.kalman(U, Y)
Qhat = [V @ np.diag(lam * np.exp(np.clip(xi[k], -60, 60))) @ V.T for k in range(T)]
Rhat = [A.R0 * np.exp(np.clip(eta[k], -60, 60)) for k in range(T)]
plug = A.kalman(U, Y, Qhat, Rhat)
plugR = A.kalman(U, Y, Qs, Rhat)                     # true Q, the filter's R: the sensor-scale estimate alone
# the accelerometer scale trajectory
acc_eta = eta[:, ACC].mean(1)
rise = acc_eta[A0:B0] - acc_eta[A0 - 1]
def first(frac):
    i = np.flatnonzero(rise >= frac * STEP); return int(i[0]) if i.size else None
t50, t90 = first(0.5), first(0.9)
settled = acc_eta[400:B0]; bias = settled.mean() - STEP
d = t90 if t90 is not None else 40
Rlag = [Rs[max(k - d, 0)] if A0 <= k < B0 + d else Rs[k] for k in range(T)]
lagorc = A.kalman(U, Y, Qs, Rlag)
Rbias = [Rs[k].copy() for k in range(T)]
for k in range(A0, B0): Rbias[k][ACC] = A.R0[ACC] * np.exp(settled.mean())
biasorc = A.kalman(U, Y, Qs, Rbias)
print(f"seed {seed}: filter {1e3*dt/T:.0f} ms/step")
print(f"accelerometer log-scale during SENSOR: truth +{STEP:.2f} nats; rise to 50% at {t50} steps, 90% at {t90} steps; "
      f"settled (400-500) {settled.mean():+.2f} +- {settled.std():.2f} (bias {bias:+.2f} nats = R factor {np.exp(bias):.2f})")
print(f"  pots' log-scale during SENSOR (350-500): {eta[350:B0][:, POTS].mean():+.2f}; process log-scales (jerk modes 10-14): "
      f"{xi[350:B0][:, 10:].mean():+.2f}, (modes 0-9): {xi[350:B0][:, :10].mean():+.2f}")
exit_eta = acc_eta[B0:B0 + 100] - acc_eta[B0 - 1]
i = np.flatnonzero(exit_eta <= -0.9 * STEP); print(f"  exit: back within 10% of calm after {int(i[0]) if i.size else '>100'} steps")
wins = [("onset 250-290 (unscored)", A0, A0 + SK), ("early 290-350", A0 + SK, 350), ("late 350-500", 350, B0), ("SENSOR scored 290-500", A0 + SK, B0), ("exit 500-540", B0, B0 + 40)]
print(f"{'window':>26} | {'oracle':>7} {'fixed':>7} {'filter':>7} {'plug-in':>7} {'plug-R':>7} {'lag-orc':>7} {'bias-orc':>8}   (tip RMSE / oracle)")
for nm, a, b in wins:
    o = score(orc, a, b)
    print(f"{nm:>26} | {1.0:7.2f} {score(fix,a,b)/o:7.2f} {score(est,a,b)/o:7.2f} {score(plug,a,b)/o:7.2f} {score(plugR,a,b)/o:7.2f} {score(lagorc,a,b)/o:7.2f} {score(biasorc,a,b)/o:8.2f}   (oracle {o:.4f} m)")
np.savez(os.path.join(HERE, f"0011_seed{seed}.npz"), eta=eta, xi=xi, est=est, orc=orc, plug=plug)
