"""Battery of tests for the self-tuning-window filter."""
import numpy as np
from adaptive_window import fixed_window, adaptive_window

T_DEF = 40000

def kfq(x, s2, Q, m0=0.0, v0=100.0):
    T=len(x); out=np.zeros(T); m,v=m0,v0
    for t in range(T):
        vp=v+Q; K=vp/(vp+s2); m=m+K*(x[t]-m); v=(1-K)*vp; out[t]=m
    return out

def oracle_kalman(x, th, s2):
    return min(np.mean((kfq(x,s2,Q)[len(x)//5:]-th[len(x)//5:])**2)
               for Q in [0.0005,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0])

# ---------------- series generators ----------------
def g_stat(seed, q, T=T_DEF):
    r=np.random.default_rng(seed); th=np.cumsum(r.normal(0,np.sqrt(q),T))
    return th, th+r.normal(0,1,T)

def g_regime(seed, seg, T=T_DEF):
    r=np.random.default_rng(seed); th=[0.]; qs=[0.005,0.2]; i=0
    while len(th)<=T:
        for _ in range(seg):
            th.append(th[-1]+r.normal(0,np.sqrt(qs[i%2])))
        i+=1
    th=np.array(th[1:T+1]); return th, th+r.normal(0,1,T)

def g_ramp(seed, T=T_DEF):
    """drift rate ramps smoothly over 3 decades -- no discrete regimes at all"""
    r=np.random.default_rng(seed)
    qs=10**np.linspace(-3,0,T)
    th=np.cumsum(r.normal(0,1,T)*np.sqrt(qs))
    return th, th+r.normal(0,1,T)

def g_step(seed, T=T_DEF, n=8):
    r=np.random.default_rng(seed); th=np.zeros(T)
    for k in range(1,n):
        th[k*T//n:]+=r.normal(0,6)
    return th, th+r.normal(0,1,T)

def g_mixed(seed, T=T_DEF):
    r=np.random.default_rng(seed); th=np.zeros(T); cur=0.
    for t in range(T):
        u=r.random()
        if u<1/2500: cur+=r.normal(0,7)
        else: cur+=r.normal(0,np.sqrt(0.01))
        th[t]=cur
    return th, th+r.normal(0,1,T)

def g_heavy(seed, T=T_DEF, q=0.02):
    r=np.random.default_rng(seed); th=np.cumsum(r.normal(0,np.sqrt(q),T))
    return th, th+r.standard_t(3,T)/np.sqrt(3.)

def g_hetero(seed, T=T_DEF, q=0.02):
    """observation noise itself changes scale midway -- a stress case"""
    r=np.random.default_rng(seed); th=np.cumsum(r.normal(0,np.sqrt(q),T))
    n=r.normal(0,1,T); n[T//2:]*=3.0
    return th, th+n

SERIES = [
    ("stationary q=.005",  lambda s: g_stat(s,0.005)),
    ("stationary q=.02",   lambda s: g_stat(s,0.02)),
    ("stationary q=.2",    lambda s: g_stat(s,0.2)),
    ("regime slow (10k)",  lambda s: g_regime(s,10000)),
    ("regime fast (2k)",   lambda s: g_regime(s,2000)),
    ("smooth ramp 3dec",   lambda s: g_ramp(s)),
    ("repeated steps",     lambda s: g_step(s)),
    ("mixed jump+drift",   lambda s: g_mixed(s)),
    ("heavy-tail noise",   lambda s: g_heavy(s)),
    ("hetero noise",       lambda s: g_hetero(s)),
]

METHODS = [
    ("fixed c=10",   lambda x: fixed_window(x,1.0,c_win=10.)),
    ("fixed c=100",  lambda x: fixed_window(x,1.0,c_win=100.)),
    ("fixed c=1000", lambda x: fixed_window(x,1.0,c_win=1000.)),
    ("ADAPTIVE",     lambda x: adaptive_window(x,1.0)[0]),
]