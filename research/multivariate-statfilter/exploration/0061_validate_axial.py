"""0061 -- does the axial stencil recover the true noise levels as well as the
exact tensor grid?  Ground truth is known, so both are scored against it, not
against each other.  See 0060 for the results and their reading.

The tensor-grid reference is the pre-stencil implementation; recreate it with

    git show <pre-stencil-rev>:lucid/statfilter/lucid.py > /tmp/lucid_tensor.py
    LUCID_TENSOR=/tmp/lucid_tensor.py python 0061_validate_axial.py

Without LUCID_TENSOR set, only the axial numbers are produced.
"""
import importlib.util, math, os, sys, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", ".."))
sys.path.insert(0, ROOT)
from lucid import LucidFilter as Axial

Tensor = None
_ref = os.environ.get("LUCID_TENSOR")
if _ref and os.path.exists(_ref):
    spec = importlib.util.spec_from_file_location("lucid_tensor", _ref)
    _t = importlib.util.module_from_spec(spec)
    sys.modules["lucid_tensor"] = _t          # dataclass needs the module registered
    spec.loader.exec_module(_t)
    Tensor = _t.LucidFilter


def local_level(T=1500, q=0.5, s2=2.0, seed=0):
    r = np.random.default_rng(seed)
    th = np.cumsum(r.standard_normal(T) * math.sqrt(q))
    return (th + r.standard_normal(T) * math.sqrt(s2))[:, None], th, q, s2


def servo_joint(T=1500, seed=0, qs=(0.004, 0.05, 0.5), pot=0.03, tach=0.10):
    """Stable PD-servo joint: [theta, omega, alpha], alpha a mean-reverting
    disturbance.  Bounded, unlike a free triple integrator."""
    dt, kp, kd, a = 0.02, 25.0, 7.0, 0.90
    F = np.array([[1., dt, 0.], [-kp*dt, 1-kd*dt, dt], [0., 0., a]])
    B = np.array([[0.], [kp*dt], [0.]])
    H = np.array([[1., 0., 0.], [0., 1., 0.]])
    Q0 = np.diag(np.array(qs) ** 2)
    R0 = np.array([pot**2, tach**2])
    r = np.random.default_rng(seed)
    t = np.arange(T) * dt
    u = 0.8*np.sin(2*np.pi*0.25*t) + 0.4*np.sin(2*np.pi*0.6*t + 1.0)
    s = np.zeros(3); S = np.zeros((T, 3)); Y = np.zeros((T, 2))
    for k in range(T):
        s = F @ s + B[:, 0]*u[k] + np.array(qs)*r.standard_normal(3)
        S[k] = s
        Y[k] = [s[0] + pot*r.standard_normal(), s[1] + tach*r.standard_normal()]
    return dict(F=F, B=B, H=H, Q0=Q0, R0=R0, Y=Y, U=u[:, None], S=S,
                truth_q=np.array(qs), truth_r=np.array([pot, tach]))


def score(cls, kw, Y, U, truth_state, tail=slice(-600, None)):
    f = cls(**kw)
    t0 = time.time()
    r = f.filter(Y, U=U) if U is not None else f.filter(Y)
    secs = time.time() - t0
    rmse = float(np.sqrt(((r.mean[:, 0] - truth_state) ** 2).mean()))
    G = sum(m._G for m in f._members)
    return dict(rmse=rmse, secs=secs, nodes=G,
                ps=r.process_scale[tail].mean(0), ms=r.measurement_scale[tail].mean(0))


print("=" * 74)
print("1) LOCAL LEVEL   true q=0.5  s2=2.0   (base Q0=R0=1 -> truth in log units)")
Y, th, q, s2 = local_level()
kw = dict()
for name, cls in ([("tensor", Tensor)] if Tensor else []) + [("axial ", Axial)]:
    d = score(cls, kw, Y, None, th)
    print(f"  {name}  nodes={d['nodes']:6d}  {d['secs']:6.2f}s  RMSE={d['rmse']:.4f}"
          f"  learned q={math.exp(d['ps'][0]):.3f} (true {q})"
          f"  s2={math.exp(d['ms'][0]):.3f} (true {s2})")

print()
print("2) SERVO JOINT   3-state, 2 sensors, 4 active channels")
P = servo_joint()
kw = dict(dynamics=P['F'], control=P['B'], H=P['H'], process=P['Q0'], measurement=P['R0'])
for name, cls in ([("tensor", Tensor)] if Tensor else []) + [("axial ", Axial)]:
    d = score(cls, kw, P['Y'], P['U'], P['S'][:, 0])
    print(f"  {name}  nodes={d['nodes']:6d}  {d['secs']:6.2f}s  RMSE={d['rmse']:.4f}"
          f"  proc-scale={np.round(d['ps'],2)}  meas-scale={np.round(d['ms'],2)}")
print("  (base Q0/R0 are the truth here, so scales should sit near 0)")

print()
print("3) SERVO JOINT, base off by 100x in Q and 1/25 in R  -> must walk back")
Pw = dict(P); kwn = dict(kw)
kwn['process'] = P['Q0'] * 100.0
kwn['measurement'] = P['R0'] / 25.0
for name, cls in ([("tensor", Tensor)] if Tensor else []) + [("axial ", Axial)]:
    d = score(cls, kwn, P['Y'], P['U'], P['S'][:, 0])
    print(f"  {name}  nodes={d['nodes']:6d}  {d['secs']:6.2f}s  RMSE={d['rmse']:.4f}"
          f"  proc-scale={np.round(d['ps'],2)} (want ~-4.6)"
          f"  meas-scale={np.round(d['ms'],2)} (want ~+3.2)")
