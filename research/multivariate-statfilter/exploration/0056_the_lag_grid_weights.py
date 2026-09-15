import os, sys, time, importlib.util, numpy as np
S=os.path.dirname(os.path.abspath(__file__)) + "/"
src=open(S+"0056_the_lag_grid.py").read(); head=src[:src.index('A = load("A"); C = load("C")')]
sys.argv=["x","0,inf","arm"]; ns={}; exec(compile(head,"lag","exec"),ns); C=ns["load"]("C")
sys.path.insert(0,"research/multivariate-statfilter/scripts"); import arm5dof as AR
spec=importlib.util.spec_from_file_location("p54","research/multivariate-statfilter/exploration/0054_physical_sensors.py"); m54=importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
jstd,pot,acc=m54.schedule(); U,S_,Y=AR.simulate(0,jstd,pot,acc); np.seterr(all="ignore")
f=C.LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0)
lag=np.array([getattr(e,"_lag",0.0) for e in f._members]); inf=~np.isfinite(lag)
print("members", len(f._members), "eager", int((~inf).sum()), "never", int(inf.sum()), "forget", f.forget)
prev=None; winf=[]; dll=[]
for t in range(m54.T):
    f.update(Y[t],U[t]); lw=f._logw.copy()
    w=np.exp(lw-lw.max()); w/=w.sum(); winf.append(w[inf].sum())
    if prev is not None:
        inc=lw-f.forget*(prev-prev.max()-np.log(np.exp(prev-prev.max()).sum()))   # per-step ll increment per member (approx: forget*normalised prior + ll)
        # exact: logw_t = forget*(prev normalised) + ll  -> ll = logw_t - forget*(prev - logsumexp(prev))
        ll=inc
        dll.append(np.log(np.exp(ll[~inf]).mean())-np.log(np.exp(ll[inf]).mean()))   # eager-group vs never-group mean predictive density, nats/step
    else: dll.append(0.0)
    prev=lw
winf=np.array(winf); dll=np.array(dll)
for nm,a,b in m54.PHASES:
    print(f"{nm:>8} {a:4d}-{b:4d}: weight on NEVER {winf[a:b].mean():.3f} (end {winf[b-1]:.3f}) | eager - never predictive ll {dll[a:b].mean():+.4f} nats/step (sum {dll[a:b].sum():+.1f})")
