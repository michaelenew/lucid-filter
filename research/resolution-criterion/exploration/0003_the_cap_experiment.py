"""Variant C at the DERIVED bound: gap=min(1.5*s, 0.89) with K raised to keep reach ~3s.
Paired vs shipped on scalar (12 seeds, diff+-sem, ms/step), async (5 seeds), arm seed 0."""
import math, os, sys, time, types, importlib.util
import numpy as np
sys.path.insert(0,".")
SRC="lucid/filter/lucid.py"; CAP=0.89; KC=int(math.ceil(3.0*3.2/CAP))
def load(variant):
    src=open(SRC).read()
    if variant=="C":
        src=src.replace("        self.gap = _GAP_FACTOR * self.s_ax","        self.gap = np.minimum(_GAP_FACTOR * self.s_ax, %r)"%CAP)
        src=src.replace("        K = int(math.ceil(_SPAN_S / _GAP_FACTOR))","        K = %d"%KC)
    mod=types.ModuleType("lcc_"+variant); mod.__file__=SRC
    sys.modules[mod.__name__]=mod; mod.__dict__["__name__"]=mod.__name__
    exec(compile(src,SRC,"exec"),mod.__dict__); return mod
A=load("A"); C=load("C")
print(f"cap={CAP} nats, K={KC} ({2*KC+1} nodes) vs shipped K=2 (5 nodes)", flush=True)
# --- scalar, paired
N,JA,JU,NA=900,380,9.0,600
rows={k:[] for k in ("jump","steady","C")}; tA=tC=0.0
for seed in range(11,23):
    rng=np.random.default_rng(seed); th=np.cumsum(rng.normal(0,math.sqrt(0.02),N)); th[JA:]+=JU
    sd=np.where(np.arange(N)<NA,1.0,3.0); y=th+rng.normal(0,sd)
    t0=time.perf_counter(); ma=np.asarray(A.LucidFilter().filter(y.reshape(-1,1)).mean).reshape(-1); tA+=time.perf_counter()-t0
    t0=time.perf_counter(); mc=np.asarray(C.LucidFilter().filter(y.reshape(-1,1)).mean).reshape(-1); tC+=time.perf_counter()-t0
    for k,sl in (("jump",slice(JA,JA+40)),("steady",slice(80,JA)),("C",slice(NA+40,N))):
        rows[k].append((np.mean((ma[sl]-th[sl])**2), np.mean((mc[sl]-th[sl])**2)))
print("SCALAR paired (12 seeds), RMSE A -> C, paired diff in MSE +- sem, t:")
for k,v in rows.items():
    a=np.array([x[0] for x in v]); c=np.array([x[1] for x in v]); d=c-a; sem=d.std(ddof=1)/math.sqrt(len(d))
    print(f"  {k:7s} RMSE {math.sqrt(a.mean()):.4f} -> {math.sqrt(c.mean()):.4f}   dMSE {d.mean():+.4f} +- {sem:.4f}  t={d.mean()/sem:+.2f}", flush=True)
print(f"  ms/step: A {1e3*tA/(12*N):.2f}  C {1e3*tC/(12*N):.2f}  (x{tC/tA:.1f})", flush=True)
# --- async
spec=importlib.util.spec_from_file_location("rig","research/pointwise-streaming/exploration/0005_the_asynchronous_rig.py")
rig=importlib.util.module_from_spec(spec); spec.loader.exec_module(rig)
for nm,mod in (("A",A),("C",C)):
    whole=[];hot=[]
    for seed in (0,10,20,30,40):
        stream=rig.simulate(seed);truth=np.array([s[3] for s in stream]);times=np.array([s[1] for s in stream])
        F,Q=rig.nominal_model(); f=mod.LucidFilter(dynamics=F,H=rig.H,process=Q,measurement=rig.SIGMA**2,timestep=rig.NOMINAL)
        est=np.empty((len(stream),2))
        for k,(i,t,val,_tr,_sd) in enumerate(stream): est[k]=f.observe(i,val,t=t).mean
        err=est[:,0]-truth[:,0];j,t0,t1,fac=rig.FAIL
        hw=(times>=t0+1)&(times<t1);ww=(times>=1)&(times<rig.DURATION)
        whole.append(np.sqrt(np.mean(err[ww]**2)));hot.append(np.sqrt(np.mean(err[hw]**2)))
    print(f"ASYNC {nm}: whole {np.mean(whole)/0.0345:.2f}x  hot {np.mean(hot)/0.0241:.2f}x  (worst hot {max(hot)/0.0241:.2f}x)", flush=True)
# --- arm seed 0
sys.path.insert(0,"research/multivariate-statfilter/scripts"); import arm5dof as AR
spec=importlib.util.spec_from_file_location("p54","research/multivariate-statfilter/exploration/0054_physical_sensors.py")
m54=importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
np.seterr(all="ignore")
jstd,pot,acc=m54.schedule(); U,S,Y=AR.simulate(0,jstd,pot,acc)
orc=AR.kalman(U,Y,[j**2*(AR.B@AR.B.T) for j in jstd],[np.concatenate([[pot[k,j]**2,acc[k,j]**2,acc[k,j]**2] for j in range(AR.NJ)]) for k in range(m54.T)])
Pt=m54.tip(S); Po=m54.tip(orc); W=m54.windows()
for nm,mod in (("A",A),("C",C)):
    t0=time.perf_counter()
    est=np.asarray(mod.LucidFilter(dynamics=AR.F,control=AR.B,H=AR.measure,process=AR.Q0,measurement=AR.R0).filter(Y,U).mean)
    dt=time.perf_counter()-t0
    fin=bool(np.all(np.isfinite(est)))
    if fin:
        Pl=m54.tip(est)
        print(f"ARM seed0 {nm} ({1e3*dt/m54.T:.0f} ms/step): "+"  ".join(f"{k} {m54.rms(Pl,Pt,m54.mask(sp))/m54.rms(Po,Pt,m54.mask(sp)):.2f}" for k,sp in W.items()), flush=True)
    else:
        print(f"ARM seed0 {nm}: NaN", flush=True)
