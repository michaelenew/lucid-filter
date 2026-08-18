"""
MDL-GATED SELF-CONSISTENT FILTER  --  no prior, no hazard rate.

The jump/no-jump decision is made by comparing DESCRIPTION LENGTHS, not by
positing how often jumps occur:

    cost of describing a jump = 0.5 log n   (amplitude: one new parameter)
                              +     log n   (its position inside the window)
    information gained        = 0.5 p^2     (p = aligned component of the pair)

    log-odds = 0.5 p^2 - 1.5 * lam * log(n_window)

n_window is the filter's OWN self-consistent window c_win*(2-K)/K, so the
threshold is a function of the filter's state, not a choice. It GROWS with n,
which is the multiple-comparisons correction arriving for free -- more data
means more chances at a spurious large p, and MDL charges for that.

This closes the loop back to the start of the whole investigation: the
Rissanen/NML universal-code argument, now doing the job the hazard rate did.
"""
import numpy as np

def q_of(K,a): return K*(2-K)*(a+1.-K)/(1-K)**2-2*K
def K_star(q): return (-q+np.sqrt(q*q+4*q))/2.

def mdl_filter(x, s2, c_win=100., c_damp=0.3, lam=1.0, m0=0., K0=0.2,
               K_min=1e-4, K_max=0.95, warmup=200, use_gate=True):
    T=len(x); out=np.zeros(T); Ks=np.zeros(T); pj_tr=np.zeros(T)
    m=m0; K=K0; a=0.; zp=0.
    for t in range(T):
        n_eff=(2-K)/K; nw=c_win*n_eff; w=1./max(nw,2.)
        Sb=s2/(1-K); e=x[t]-m; z=e/np.sqrt(Sb)
        a+=(z*zp-a)*w
        if t>=warmup:
            qq=q_of(K,a); d=c_damp/max(nw,2.)
            K=float(np.clip((1-d)*K+d*K_star(qq),K_min,K_max)) if qq>1e-12 else max(K*(1-d),K_min)
        Ke=K
        if use_gate and t>warmup:
            c=1.-K
            p=(zp+c*z)/np.sqrt(1.+c*c)              # aligned component of the pair
            llr=0.5*p*p - 1.5*lam*np.log(max(nw,3.0))   # MDL: gain minus code length
            pj=1./(1.+np.exp(-np.clip(llr,-50,50)))
            Ke=min(K+(1.-K)*pj*(1.-K),0.98)
            pj_tr[t]=pj
        m=m+Ke*e; zp=z
        out[t]=m; Ks[t]=Ke
    return dict(means=out,K=Ks,pj=pj_tr)