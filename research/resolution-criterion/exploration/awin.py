# The exact-coordinate window: nodes uniform in each axis's Fisher arclength, transformed back to log-scale.
# Random walk of drift q observed in noise r, differenced spectrum f = q + 2r(1-cos w); Whittle Fisher per step:
#   I_q(x) = x^2 (x+2) / (2 (x(x+4))^(3/2)),   I_r(x) = 1/2 - sqrt(x/(x+4)) + I_q(x),   x = q/r.
import math, numpy as np
_LX = np.linspace(-60.0, 60.0, 24001)
_X = np.exp(_LX)
def _Iq(x): return x * x * (x + 2.0) / (2.0 * (x * (x + 4.0)) ** 1.5)
def _Ir(x): return np.maximum(0.5 - np.sqrt(x / (x + 4.0)) + _Iq(x), 0.0)
_dl = _LX[1] - _LX[0]
_AQ = np.concatenate([[0.0], np.cumsum(0.5 * (np.sqrt(_Iq(_X[1:])) + np.sqrt(_Iq(_X[:-1]))) * _dl)])          # a_q(log x): floor at x->0
_ARr = np.sqrt(_Ir(_X)); _AR = np.concatenate([[0.0], np.cumsum(0.5 * (_ARr[1:] + _ARr[:-1]) * _dl)])
_AR = _AR[-1] - _AR                                                                                            # a_r(log x): floor at x->inf
def _P(x): return (x + np.sqrt(x * x + 4.0 * x)) / 2.0
def _gq(x): return x / (_P(x) + 1.0)          # local share of a process scale, q/S
def _gr(x): return 1.0 / (_P(x) + 1.0)        # local share of a sensor scale, r/S
def _invert(fun, g, increasing):
    lo, hi = -60.0, 60.0
    for _ in range(200):
        mid = 0.5 * (lo + hi); v = fun(math.exp(mid))
        if (v < g) == increasing: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)
def _a_of(table, lx): return float(np.interp(lx, _LX, table))
def _lx_of(table, a, increasing):
    t = table if increasing else table[::-1]; l = _LX if increasing else _LX[::-1]
    return float(np.interp(a, t, l))
def axis_nodes(is_proc, g, s, K):
    """Node offsets (2K+1, in log-scale, sorted) for an axis with base local share g and class sd s."""
    if is_proc: lxc = _invert(_gq, g, True); tab, inc = _AQ, True
    else:       lxc = _invert(_gr, g, False); tab, inc = _AR, False
    ac = _a_of(tab, lxc)
    sgn = 1.0 if is_proc else -1.0                          # the axis climbs with log x on a process axis, against it on a sensor axis
    d_up = 1.5 * (_a_of(tab, lxc + sgn * s) - ac)           # one prior sd up the axis, in arclength, times the spacing factor
    d_dn = 1.5 * (ac - _a_of(tab, lxc - sgn * s))
    out = []
    for j in range(-K, K + 1):
        aj = ac + (j * d_up if j >= 0 else j * d_dn)
        if aj <= 0.0: out.append(-60.0 + (K + j))           # below the floor: 'off' nodes, distinct, harmless
        elif aj >= tab.max(): out.append(60.0 + j)
        else: out.append(sgn * (_lx_of(tab, aj, inc) - lxc))   # back to this axis's own log-scale
    return np.array(out)
def _Phi(z): return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
def cell_masses(nodes, mean, sd):
    """Gaussian(mean, sd) mass of each node's cell (edges at midpoints; outer cells to infinity)."""
    e = np.concatenate([[-np.inf], 0.5 * (nodes[1:] + nodes[:-1]), [np.inf]])
    m = np.array([_Phi((e[j + 1] - mean) / sd) - _Phi((e[j] - mean) / sd) for j in range(len(nodes))])
    return m
