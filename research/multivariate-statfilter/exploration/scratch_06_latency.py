"""scratch_06_latency.py -- transient q* vs (B, latency tau): is it minimax (grows) or convergent?"""
import math, numpy as np, sys
from scratch_05_qwalker import PosteriorVec, run_q, gen_perm, QS
from scratch_core_vec import OracleVec, rmse_ratio

def transient_qstar(B, tau, S=128, n=2000, t0=400):
    rng=np.random.default_rng(55)
    theta,x,eta=gen_perm(rng,S,n,B,t0)
    m=np.zeros(n,bool); m[t0:t0+tau]=True
    vals=np.zeros(len(QS)); se=np.zeros(len(QS))
    for i,q in enumerate(QS):
        em,eo,mus=run_q(theta,x,eta,q)
        vals[i],se[i]=rmse_ratio(em,eo,m)
    i=int(np.argmin(vals))
    return QS[i], vals[i], vals

print("transient q* as a function of (B, latency tau):  q* = arg min RMSE over [t0, t0+tau]")
print(" tau |  " + "   ".join(f"B={B}" for B in [2.0,3.5,5.0,6.5,8.0]))
for tau in [15, 30, 60, 120, 300]:
    row=[]
    for B in [2.0,3.5,5.0,6.5,8.0]:
        qs,val,_=transient_qstar(B,tau)
        row.append(f"{qs:6.3f}")
    print(f" {tau:4d}|  " + "  ".join(row))
    sys.stdout.flush()

print("\nsame, showing whether small (finding-18) q~0.011 suffices, vs q* (tau=30):")
for B in [2.0,5.0,8.0]:
    qs,val,vals=transient_qstar(B,30)
    # ratio at q closest to 0.01
    j=int(np.argmin(np.abs(QS-0.01)))
    print(f"  B={B}: q*={qs:.3f} ratio={val:.3f} | at q=0.01 ratio={vals[j]:.3f} (penalty {100*(vals[j]/val-1):.1f}%)")
