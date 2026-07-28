"""
TWO-FACTOR FILTER.

R_N = ||sum u||^2/(Nd) conflates two independent departures from whiteness.
Factor it:

    G_N = ||sum_N e||^2 / (N <e^2>)   COHERENCE  -- joint/correlation, horizon-dependent,
                                                   and SCALE-FREE (sigma^2 cancels)
    M   = <e^2> (1-K) / sigma^2       MARGINAL   -- per-sample information rate, horizon-flat

Both have expectation 1 under correct specification.

Closed form:  rho_k = rho_1 (1-K)^{k-1}, so

    G_N = 1 + rho_1 * Phi_N(K),   Phi_N(K) = 2 sum_{k<N} (1-k/N)(1-K)^{k-1}

LINEAR in rho_1 => weighted least squares across octaves gives the pooled estimate

    rho_1_hat = sum_N w_N (G_N - 1) Phi_N / sum_N w_N Phi_N^2

Then, exactly as before:
    C_0 = 1/((1-K) - rho_1),   q = C_0 K(2-K) - 2K,   K* = (-q + sqrt(q^2+4q))/2
and sigma^2 falls out of the marginal:  sigma^2 = <e^2>(1-K).

FREE PARAMETERS -- all four called out explicitly, none hidden:
  c_win   : EWMA window for the statistics, as a multiple of n_eff=(2-K)/K.
            NOT eliminated by this reformulation. Same object as before.
  c_damp  : stability margin for the algebraic solve (feedback-with-delay).
            NOT eliminated. Same object as before.
  J       : number of octaves in the ladder (N = 1,2,4,...,2^J). Compute/memory
            budget: the buffer must hold 2^J innovations.
  w_N     : scale weights. Set to 1/N, DERIVED from independent-block counting
            (T/N independent blocks at scale N), not chosen. Swept below.
"""
import numpy as np


def K_star(q): return (-q + np.sqrt(q*q + 4*q))/2.


def Phi(N, K):
    """Closed form of 2*sum_{k<N}(1-k/N)a^{k-1}, a=1-K.  O(1) per rung."""
    if N < 2: return 0.0
    a = 1.0-K
    if abs(1.0-a) < 1e-12: return float(N-1)
    S1 = (1.0-a**(N-1))/(1.0-a)
    S2 = (1.0 - a**N - N*a**(N-1)*(1.0-a))/(1.0-a)**2
    return 2.0*(S1 - S2/N)

def Phi_vec(Ns, K):
    return np.array([Phi(N, K) for N in Ns])


def twofactor_filter(x, sigma2=None, J=7, c_win=100., c_damp=0.3,
                     m0=0., K0=0.2, K_min=1e-4, K_max=0.95, warmup=300,
                     w_pow=1.0, infer_sigma=True):
    """
    sigma2       : if None (default) it is INFERRED from the marginal factor.
    w_pow        : scale weights w_N = N^{-w_pow}. 1.0 is the derived value.
    """
    Ns = [2**j for j in range(J+1)]
    Nmax = Ns[-1]
    T = len(x)
    out = np.zeros(T); Kt = np.zeros(T); S2 = np.zeros(T); R1 = np.zeros(T)

    m = m0; K = K0
    s2 = 1.0 if sigma2 is None else sigma2
    buf = np.zeros(Nmax); bi = 0
    run = np.zeros(len(Ns))          # running sum of last N innovations
    E2 = 1.0                          # EWMA of e^2
    PN = np.ones(len(Ns))             # EWMA of (sum_N e)^2 / N

    for t in range(T):
        Sb = s2/(1.0-K)
        e = x[t] - m

        # maintain running sums of the last N innovations for each octave
        old = buf[bi]
        buf[bi] = e
        for i, N in enumerate(Ns):
            j = (bi - N) % Nmax
            run[i] += e - (buf[j] if t >= N else 0.0)
        bi = (bi+1) % Nmax

        ne = (2.0-K)/K
        nw = max(c_win*ne, 4.0)
        w = 1.0/nw
        E2 += (e*e - E2)*w
        PN += (run*run/np.array(Ns) - PN)*w

        if t >= warmup:
            G = PN/max(E2, 1e-12)
            Ph = Phi_vec(Ns, K)
            wN = np.array([N**(-w_pow) for N in Ns])
            den = np.sum(wN*Ph*Ph)
            rho1 = float(np.sum(wN*(G-1.0)*Ph)/den) if den > 1e-12 else 0.0
            rho1 = float(np.clip(rho1, -0.95*(1-K), 0.95))

            denom = (1.0-K) - rho1
            if denom > 1e-9:
                C0 = 1.0/denom
                q = C0*K*(2.0-K) - 2.0*K
                if q > 1e-12:
                    d = c_damp/nw
                    K = float(np.clip((1-d)*K + d*K_star(q), K_min, K_max))
            if infer_sigma:
                s2 = max(E2*(1.0-K), 1e-6)
            R1[t] = rho1

        m = m + K*e
        out[t] = m; Kt[t] = K; S2[t] = s2
    return dict(means=out, K=Kt, s2=S2, rho1=R1)