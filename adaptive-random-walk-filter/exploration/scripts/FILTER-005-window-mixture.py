"""
MIXTURE OVER WINDOWS -- parameter-free tracking of the best c_win.

Estimating s_a failed because the closed loop drives a -> 0: the disturbance
lives in the control signal, not the error signal. Rather than chase it into
K's trajectory, apply the same one-part-code move that worked for the jump
scale -- refuse to pick c_win at all.

Run a ladder of filters with different c_win in parallel. Weight them by
prequential predictive performance, with the parameter-free share update

    w_i <- w_i * p_i(x_t)          (Bayes)
    w_i <- (1-alpha) w_i + alpha/M ,  alpha = 1/(t+1)     (universal switching)

alpha = 1/t is the canonical parameter-free choice (Herbster-Warmuth /
Volf-Willems "tracking the best expert"): its regret is bounded relative to
the best SEQUENCE of window choices, not just the best fixed one, so the
mixture can follow a c_win that is genuinely different in different regimes.

The ladder depth M is a compute budget: more rungs, better coverage,
saturating. No statistical constant anywhere.
"""
import numpy as np

def q_of(K, a): return K*(2-K)*(a+1.-K)/(1-K)**2 - 2*K
def K_star(q):  return (-q + np.sqrt(q*q + 4*q))/2.


def window_mixture(x, s2, c_wins=(3.,10.,30.,100.,300.,1000.,3000.),
                   c_damp=0.3, m0=0., K0=0.2, warmup=200):
    M = len(c_wins)
    cw = np.array(c_wins, float)
    T = len(x)
    out = np.zeros(T); Kout = np.zeros(T); Weff = np.zeros(T)

    m = np.full(M, m0)
    K = np.full(M, K0)
    a = np.zeros(M)
    zp = np.zeros(M)
    w = np.full(M, 1.0/M)
    # each expert infers its OWN sigma^2 from the two-channel solve, so that a
    # misspecified noise level cannot be absorbed by picking the jitteriest
    # expert. (sigma^2 inference was shown to be free.)
    s2i = np.full(M, s2); E2 = np.ones(M); C1 = np.zeros(M); ep = np.zeros(M)

    for t in range(T):
        Sb = s2i/(1.0-K)
        e  = x[t] - m
        # predictive density of each expert BEFORE it sees x_t
        logp = -0.5*np.log(2*np.pi*Sb) - 0.5*e*e/Sb
        logp -= logp.max()
        p = np.exp(logp)

        # blended output = weighted mean of the experts' predictions
        out[t] = float(np.dot(w, m))
        Kout[t] = float(np.dot(w, K))
        Weff[t] = 1.0/np.dot(w, w)          # effective number of experts alive

        # Bayes update of the weights, then the parameter-free share step
        w = w * p
        s = w.sum()
        w = w/s if s > 1e-300 else np.full(M, 1.0/M)
        alpha = 1.0/(t+2.0)
        w = (1.0-alpha)*w + alpha/M

        # each expert advances its own closed loop
        ne = (2.0-K)/K
        nw = cw*ne
        wt = 1.0/np.maximum(nw, 2.0)
        E2 += (e*e - E2)*wt
        C1 += (e*ep - C1)*wt
        if t >= warmup:
            s2i = np.maximum((1.0-K)*E2 - C1, 1e-4)
        z  = e/np.sqrt(Sb)
        a += (z*zp - a)*wt
        if t >= warmup:
            qq = q_of(K, a)
            d  = c_damp/np.maximum(nw, 2.0)
            Ks = np.where(qq > 1e-12, K_star(np.maximum(qq, 1e-12)), K*(1.0-d))
            K  = np.clip((1.0-d)*K + d*Ks, 1e-4, 0.95)
        m = m + K*e
        zp = z; ep = e
    return dict(means=out, K=Kout, n_experts=Weff, w=w)