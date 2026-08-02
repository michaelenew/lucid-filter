"""
ONLINE CLEAN FILTER -- no batch refit, no EWMA window parameter.

The batch version recomputed the variogram over L samples every `refit` steps.
That was inelegant and it hid a second choice: the earlier online version used
an EWMA window W set by hand. Both go away at once, because the leverage rule
already produces a memory length L -- so use 1/L as the EWMA rate.

Per step, per rung k:
    V_k  <-  V_k + ((x_t - x_{t-k})^2 - V_k)/L
then a GLS line fit over ~10 rungs gives Q (slope) and sigma^2 (intercept/2),
then the Riccati gain, then the mean update. Cost is O(n_rungs) per step and
O(top rung) memory -- strictly cheaper than the batch version, which was
O(L * n_rungs) every refit.

L itself comes from the leverage truncation, so the forgetting rate is a
consequence of the accuracy tolerance eps rather than an independent dial.

REMAINING KNOBS, complete:
    eps        accuracy vs memory. Monotone.
    pairs_min  how many independent pairs the top rung should see; sets L from
               the top rung. Compute/memory budget.
    buf_cap    hard memory ceiling.
Nothing else. No refit interval, no EWMA window, no c, no a_j, no rate, no prior.
"""
import numpy as np

RUNGS = np.array([1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584], float)


def pos(v, e): return 0.5*(v + np.sqrt(v*v + 4.0*e*e))


def gls(V, K):
    w = 1.0/(V*V*K)
    Sw = w.sum(); Sk = np.dot(w,K); Skk = np.dot(w,K*K)
    Sv = np.dot(w,V); Skv = np.dot(w,K*V)
    det = Sw*Skk - Sk*Sk
    return (Sw*Skv - Sk*Sv)/det, 0.5*(Skk*Sv - Sk*Skv)/det


def leverage_top(Q, s2, eps, kmax):
    K = RUNGS[RUNGS <= kmax]
    if len(K) < 3: return K
    V = K*max(Q,1e-9) + 2.0*max(s2,1e-9)
    w = 1.0/(V*V*K); kbar = np.dot(w,K)/w.sum()
    lev = w*(K-kbar)**2; cum = np.cumsum(lev)/lev.sum()
    return K[:max(int(np.searchsorted(cum, 1.0-eps))+1, 3)]


def run(x, eps=0.10, pairs_min=40, buf_cap=40000):
    T = len(x)
    ks_all = RUNGS[RUNGS <= buf_cap/pairs_min].astype(int)
    top = int(ks_all[-1]); B = top + 1
    buf = np.zeros(B); p = 0; n = 0

    Q, s2 = 0.05, 1.0
    q = Q/s2; K = (-q+np.sqrt(q*q+4*q))/2.0
    m = 0.0
    V = ks_all*Q + 2.0*s2                    # rung state, one EWMA each
    n_act = len(ks_all)
    L = 2000.0

    out = np.zeros(T); Qs = np.zeros(T); S2 = np.zeros(T); Ks = np.zeros(T); Ls = np.zeros(T)
    for t in range(T):
        rate = 1.0/max(L, 4.0)
        # vectorised: all active rungs in one gather, no Python loop
        ka = ks_all[:n_act]
        idx = (p - ka) % B
        live = (n > ka)
        d = x[t] - buf[idx]
        V[:n_act] += np.where(live, (d*d - V[:n_act])*rate, 0.0)

        if n > 50:
            Kl = ks_all[:n_act].astype(float)
            Vv = np.maximum(V[:n_act], 1e-12)
            Qr, s2r = gls(Vv, Kl)
            se = 2.0*Vv[0]*np.sqrt(rate)
            Q = pos(Qr, se); s2 = pos(s2r, se)
            q = Q/s2; K = (-q+np.sqrt(q*q+4*q))/2.0
            Kw = leverage_top(Q, s2, eps, buf_cap/pairs_min)
            n_act = min(len(Kw), len(ks_all))
            L = float(min(pairs_min*ks_all[n_act-1], buf_cap))

        buf[p] = x[t]; p = (p+1) % B; n += 1
        m = m + K*(x[t]-m)
        out[t]=m; Qs[t]=Q; S2[t]=s2; Ks[t]=K; Ls[t]=L
    return dict(means=out, Q=Qs, s2=S2, K=Ks, L=Ls)