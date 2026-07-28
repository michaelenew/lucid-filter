"""
ONE-PART (MIXTURE) CODE FILTER  --  no threshold, no decision, no n.

Two-part MDL charged log n to say WHERE a jump was. Sequential prediction
never names a position, so that term does not exist:

    L(x_1..x_T) = - sum_t log p(x_t | x_<t)      (prequential / one-part code)

There is no model to describe and nothing to compare against. All that is
needed is a predictive density that already contains every drift magnitude.

Take the per-step process noise to live on a dyadic ladder anchored to the
filter's own learned baseline:

    Q_j = Q_base * 2^j ,    j in Z

and weight the rungs with Rissanen's universal prior for integers,

    w_j  ∝  1 / ((|j|+1)(|j|+2))

which is PROPER (sums to 1), needs no range, no truncation, and no scale --
it is the code you use when you refuse to say how big something is. The
predictive is then a scale mixture of normals

    p(e) = sum_j w_j N(e | 0, V + Q_j + sigma^2)

and the update is the posterior-weighted blend of the corresponding gains.
Nothing is thresholded. A tiny innovation puts the posterior on low rungs and
the filter behaves normally; a huge one shifts the posterior up the ladder and
the gain rises smoothly toward 1. Jump and drift are the same mechanism read
at different rungs -- there is no branch because there are no two models.

Q_base comes from the closed loop already derived (gain = believed drift,
serial correlation = error in that belief), so the ladder is anchored to the
filter's own state rather than to anything chosen.
"""
import numpy as np

def q_of(K, a): return K * (2 - K) * (a + 1. - K) / (1 - K) ** 2 - 2 * K
def K_star(q):  return (-q + np.sqrt(q * q + 4 * q)) / 2.


def universal_weights(J):
    """Rissanen-style universal prior over dyadic rungs j = -J..J. Proper."""
    js = np.arange(-J, J + 1)
    w = 1.0 / ((np.abs(js) + 1.0) * (np.abs(js) + 2.0))
    return js, w / w.sum()


def mixture_filter(x, s2, J=12, c_win=100., c_damp=0.3, m0=0., K0=0.2,
                   K_min=1e-4, K_max=0.95, warmup=200, use_mixture=True):
    """
    J only sets how far the ladder is enumerated; the universal weights decay
    as 1/j^2 so far rungs contribute negligibly and results are insensitive to
    it. It is a compute budget, not a statistical choice -- verified by sweep.
    """
    js, w = universal_weights(J)
    two_j = 2.0 ** js

    T = len(x)
    out = np.zeros(T); Keff = np.zeros(T); Kbase = np.zeros(T); rung = np.zeros(T)
    m = m0; K = K0; a = 0.; zp = 0.

    for t in range(T):
        n_eff = (2 - K) / K
        nw = c_win * n_eff
        wgt = 1.0 / max(nw, 2.0)

        # --- baseline: the closed loop (gain = believed drift) ---
        Sb = s2 / (1 - K)
        e = x[t] - m
        z = e / np.sqrt(Sb)
        a += (z * zp - a) * wgt
        if t >= warmup:
            qq = q_of(K, a)
            d = c_damp / max(nw, 2.0)
            K = float(np.clip((1 - d) * K + d * K_star(qq), K_min, K_max)) \
                if qq > 1e-12 else max(K * (1 - d), K_min)

        # --- one-part mixture over the dyadic ladder ---
        if use_mixture and t > warmup:
            V = s2 * K                      # steady-state posterior variance (V = sigma^2 K)
            Q_base = K * K * s2 / (1 - K)   # baseline process noise implied by K
            Qj = Q_base * two_j
            Sj = V + Qj + s2
            logL = -0.5 * np.log(Sj) - 0.5 * e * e / Sj
            logp = np.log(w) + logL
            logp -= logp.max()
            post = np.exp(logp); post /= post.sum()
            Kj = (V + Qj) / Sj
            K_use = float(np.dot(post, Kj))
            rung[t] = float(np.dot(post, js))
        else:
            K_use = K

        m = m + K_use * e
        zp = z
        out[t] = m; Keff[t] = K_use; Kbase[t] = K
    return dict(means=out, K=Keff, K_base=Kbase, rung=rung)


# ======================================================================
# The fixed-weight mixture re-pays the universal code cost EVERY step:
# -log w_0 = 1.0 nat per sample, forever, for refusing to name the scale.
# But the rung SEQUENCE is highly compressible -- almost always j=0 with rare
# excursions -- so coding each step as an independent draw is wasteful.
#
# The prequential fix is the Krichevsky-Trofimov estimator, the canonical
# parameter-free universal code for a discrete alphabet:
#
#       w_j^(t)  =  (c_j + 1/2) / (sum_i c_i + |A|/2)
#
# with c_j the soft counts accumulated so far. KT's regret is (|A|/2) log T
# TOTAL, not per step. Decaying the counts on the filter's own self-consistent
# window lets the rung distribution track regime changes; the window is the
# one already in use, so no new quantity is introduced.
# ======================================================================
def kt_mixture_filter(x, s2, J=12, c_win=100., c_damp=0.3, m0=0., K0=0.2,
                      K_min=1e-4, K_max=0.95, warmup=200):
    js = np.arange(-J, J + 1)
    two_j = 2.0 ** js
    A = len(js)
    counts = np.full(A, 0.5)

    T = len(x)
    out = np.zeros(T); Keff = np.zeros(T); rung = np.zeros(T); w0 = np.zeros(T)
    m = m0; K = K0; a = 0.; zp = 0.

    for t in range(T):
        n_eff = (2 - K) / K
        nw = c_win * n_eff
        wgt = 1.0 / max(nw, 2.0)

        Sb = s2 / (1 - K)
        e = x[t] - m
        z = e / np.sqrt(Sb)
        a += (z * zp - a) * wgt
        if t >= warmup:
            qq = q_of(K, a)
            d = c_damp / max(nw, 2.0)
            K = float(np.clip((1 - d) * K + d * K_star(qq), K_min, K_max)) \
                if qq > 1e-12 else max(K * (1 - d), K_min)

        if t > warmup:
            V = s2 * K
            Q_base = K * K * s2 / (1 - K)
            Qj = Q_base * two_j
            Sj = V + Qj + s2
            w = counts / counts.sum()                  # KT weights
            logp = np.log(w) - 0.5 * np.log(Sj) - 0.5 * e * e / Sj
            logp -= logp.max()
            post = np.exp(logp); post /= post.sum()
            Kj = (V + Qj) / Sj
            K_use = float(np.dot(post, Kj))
            counts = (1.0 - wgt) * counts + wgt * A * post + 1e-12
            counts = np.maximum(counts, 0.5 * wgt)      # KT floor
            rung[t] = float(np.dot(post, js)); w0[t] = w[J]
        else:
            K_use = K

        m = m + K_use * e
        zp = z
        out[t] = m; Keff[t] = K_use
    return dict(means=out, K=Keff, rung=rung, w0=w0)