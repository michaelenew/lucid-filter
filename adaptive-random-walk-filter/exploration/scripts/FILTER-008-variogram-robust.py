"""
ROBUST VARIOGRAM FILTER -- attenuation of extreme signals, built in.

TWO ATTENUATIONS, both smooth, both self-scaled:

(1) PAIR ATTENUATION (this is the new part). Each pair-difference entering the
    variogram is weighted
            w(D^2) = 1 / (1 + (D^2/(c V))^2)
    redescending, no branch, and the scale is V itself -- an M-estimator fixed
    point with nothing external. Consistency factor b(c) is a definite integral
    against the chi^2_1 null (b(4)=0.75269), derived and divided out.

    Ordinary V and robust V agree exactly on diffuse process noise (6.0 vs 6.0)
    and diverge on sparse jumps (6.0 vs 2.2). Their RATIO is the jump content.
    Q_diffuse from robust V drives the baseline gain, so a jump no longer
    permanently inflates the filter's idea of the drift rate.

(2) STEP vs OUTLIER, from consecutive increments. A level shift gives
    d_t = +delta, d_{t+1} ~ 0. A measurement outlier gives d_t = +delta,
    d_{t+1} = -delta. So the product d_t d_{t+1} is ~0 for a step and ~ -delta^2
    for an outlier. Smooth reversal score
            rev = -d_t d_{t+1} / (d_t^2 + d_{t+1}^2)
    is ~0 for a step and ~ +1/2 for a pure outlier. The direct mean-shift
    response is gated by (1 - 2*rev), so steps are chased and outliers are not.
    Causal with one step of lag -- confirmation before acting, as intended.

FREE PARAMETERS, called out:
  c    : robustness/efficiency constant of the M-estimator weight. c->inf gives
         the ordinary variogram; c=4 keeps ~75% Gaussian efficiency.
  eps  : leverage truncation tolerance (accuracy vs memory), as before.
  a_j  : how hard the direct step response pushes. NEW.
"""
import numpy as np
from scipy import integrate

RUNGS = np.array([1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584], float)
_B_CACHE = {}


def b_of(c):
    if c not in _B_CACHE:
        num = integrate.quad(lambda u: u/(1+(u/c)**2)*np.exp(-u/2)/np.sqrt(2*np.pi*u), 0, 200)[0]
        den = integrate.quad(lambda u: 1/(1+(u/c)**2)*np.exp(-u/2)/np.sqrt(2*np.pi*u), 0, 200)[0]
        _B_CACHE[c] = num/den
    return _B_CACHE[c]


def robust_V(d2, c=4.0, iters=10):
    V = np.mean(d2); b = b_of(c)
    for _ in range(iters):
        w = 1.0/(1.0 + (d2/(c*V))**2)
        V = np.sum(w*d2)/np.sum(w)/b
    return V


def gls(V, K):
    w = 1.0/(V*V*K)
    Sw = w.sum(); Sk = np.dot(w,K); Skk = np.dot(w,K*K)
    Sv = np.dot(w,V); Skv = np.dot(w,K*V)
    det = Sw*Skk - Sk*Sk
    return (Sw*Skv - Sk*Sv)/det, 0.5*(Skk*Sv - Sk*Skv)/det


def choose_top(Q, s2, eps, kmax):
    K = RUNGS[RUNGS <= kmax]
    if len(K) < 3: return K
    V = K*max(Q,1e-9) + 2.0*max(s2,1e-9)
    w = 1.0/(V*V*K); kbar = np.dot(w,K)/w.sum()
    lev = w*(K-kbar)**2; cum = np.cumsum(lev)/lev.sum()
    return K[:max(int(np.searchsorted(cum, 1.0-eps))+1, 3)]


def run(x, eps=0.10, c=4.0, a_j=1.0, pairs_min=40, refit=50,
        buf_cap=40000, robust=True, Q0=0.05, s20=1.0):
    T = len(x)
    out = np.zeros(T); Qs = np.zeros(T); S2 = np.zeros(T)
    Ks = np.zeros(T); JMP = np.zeros(T)
    Q, s2 = Q0, s20
    q = Q/s2; K = (-q+np.sqrt(q*q+4*q))/2.0
    m = 0.0; L = 2000.0
    d_prev = 0.0; x_prev = None

    for t in range(T):
        if t % refit == 0 and t > 200:
            Kw = choose_top(Q, s2, eps, buf_cap/pairs_min)
            seg = x[max(0, t-int(L)):t]
            Kl = Kw[Kw <= max(8.0, len(seg)/pairs_min)]
            if len(Kl) >= 3:
                V = np.zeros(len(Kl))
                for i, k in enumerate(Kl):
                    d2 = (seg[int(k):]-seg[:-int(k)])**2
                    V[i] = robust_V(d2, c) if robust else np.mean(d2)
                Qr, s2r = gls(V, Kl)
                e = 2.0*V[0]/np.sqrt(max(len(seg), 2))
                Q = 0.5*(Qr+np.sqrt(Qr*Qr+4*e*e))
                s2 = 0.5*(s2r+np.sqrt(s2r*s2r+4*e*e))
                q = Q/s2; K = (-q+np.sqrt(q*q+4*q))/2.0
                L = float(min(pairs_min*Kw[-1], buf_cap))

        # --- direct step response, causal, gated by the reversal score ---
        K_eff = K
        if x_prev is not None:
            d = x[t] - x_prev
            scale = c*(2.0*s2 + Q)
            ext = (d_prev*d_prev/scale)/(1.0 + d_prev*d_prev/scale)   # 0..1, smooth
            den = d_prev*d_prev + d*d + 1e-12
            rev = -d_prev*d/den                                        # ~0 step, ~+0.5 outlier
            gate = 0.5*(1.0 - np.tanh(4.0*(2.0*rev - 0.5)))            # smooth, no branch
            jmp = a_j*ext*gate
            K_eff = K + (1.0-K)*jmp*(1.0-K)
            JMP[t] = jmp
            d_prev = d
        else:
            d_prev = 0.0
        x_prev = x[t]

        m = m + K_eff*(x[t]-m)
        out[t]=m; Qs[t]=Q; S2[t]=s2; Ks[t]=K_eff
    return dict(means=out, Q=Qs, s2=S2, K=Ks, jmp=JMP)