"""
SELF-TUNING WINDOW via the bias-variance (bandwidth) rule.

    MSE(n) = 1/n + s_a^2 n^2/3    =>    n* = (3/2)^(1/3) s_a^(-2/3)

s_a (per-step RMS movement of the serial correlation a) is itself estimated
from the data, by comparing a fast and a slow EWMA of the same quantity:

    D = a_fast(n/r) - a_slow(n),   E[D^2] ~ r/n  under no movement
    excess = E[D^2] - r/n  ~  s_a^2 n^2 / 3

The claim to test: because s_a enters n* through a 2/3 power AND the MSE bowl
is quadratic at its base, sensitivity to the REMAINING constants (r, c_meta)
should be materially flatter than sensitivity to a hand-set c_win was.
"""
import numpy as np

def q_of(K, a): return K*(2-K)*(a+1.-K)/(1-K)**2 - 2*K
def K_star(q):  return (-q + np.sqrt(q*q + 4*q))/2.


def fixed_window(x, s2, c_win=100., c_damp=0.3, m0=0., K0=0.2, warmup=200):
    T=len(x); out=np.zeros(T); m=m0; K=K0; a=0.; zp=0.
    for t in range(T):
        ne=(2-K)/K; nw=c_win*ne; w=1./max(nw,2.)
        Sb=s2/(1-K); e=x[t]-m; z=e/np.sqrt(Sb); a+=(z*zp-a)*w
        if t>=warmup:
            qq=q_of(K,a); d=c_damp/max(nw,2.)
            K=float(np.clip((1-d)*K+d*K_star(qq),1e-4,0.95)) if qq>1e-12 else max(K*(1-d),1e-4)
        m=m+K*e; zp=z; out[t]=m
    return out


def adaptive_window(x, s2, r=4., c_meta=20., c_damp=0.3, m0=0., K0=0.2,
                    nw0=1000., nw_min=20., nw_max=200000., warmup=500):
    T=len(x); out=np.zeros(T); NW=np.zeros(T)
    m=m0; K=K0; zp=0.
    a_f=0.; a_s=0.; D2=0.; nw=nw0
    for t in range(T):
        Sb=s2/(1-K); e=x[t]-m; z=e/np.sqrt(Sb); g=z*zp
        wf=1./max(nw/r,2.); ws=1./max(nw,2.); wm=1./max(c_meta*nw,2.)
        a_f+=(g-a_f)*wf
        a_s+=(g-a_s)*ws
        D2+=((a_f-a_s)**2-D2)*wm

        if t>=warmup:
            # correct null variance of D for two correlated EWMAs on one series
            c_null = r/2. + 0.5 - 2.*r/(r+1.)
            excess = max(D2 - c_null/max(nw,2.), 0.0)
            # a moving as a random walk with per-step variance s_a^2 gives
            # tracking bias^2 ~ s_a^2 * n, hence MSE(n)=1/n + s_a^2 n and n*=1/s_a
            s_a2 = excess/max(nw, 1.0)
            if s_a2 > 1e-18:
                nw_star = 1.0/np.sqrt(s_a2)
            else:
                nw_star = nw_max
            nw=float(np.clip(nw+(nw_star-nw)*wm, nw_min, nw_max))

            qq=q_of(K,a_s); d=c_damp/max(nw,2.)
            K=float(np.clip((1-d)*K+d*K_star(qq),1e-4,0.95)) if qq>1e-12 else max(K*(1-d),1e-4)
        m=m+K*e; zp=z; out[t]=m; NW[t]=nw
    return out, NW