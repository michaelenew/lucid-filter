"""wall-correspondence 0008 -- the node barriers: sharp hypothesis
surfaces trap local search; smoothing anneals it.

The sibling's coupling scan (their 0102) found their bare weight's
exact zeros fracture local MCMC: a hot-started chain sticks in a
metastable branch, and only the flow's smoothing restores
ergodicity. Their prediction for this repository (0004, item 3):
a hypothesis bank whose likelihood surface has near-zero valleys
should trap local model-search, and tempering should anneal it, at
identical compute -- scoreable prequentially.

DESIGN. Data y_t = sin(omega* t) + sigma eps (t = 1..T_fit). The
log-likelihood over omega is a comb: sharp peaks at omega* and
aliases, separated by deep valleys (the nodes). Local search =
Metropolis on omega, small proposals, random init in [0.1, 3.0],
NSTEP steps. Two banks, identical compute and proposals:
  sharp    -- targets exp(L(omega)) directly;
  tempered -- geometric annealing beta: 1/64 -> 1 over the run (the
              filter-side copy of the sibling's heat-kernel flow).
Score: trap rate (final omega off by > 0.05) over restarts, and the
prequential price: nats/pt forecasting T_pred fresh points with the
found omega. No fitted knobs: sigma known, schedule fixed a priori.
"""

import numpy as np

OMSTAR = 0.61
SIGMA = 0.35
T_FIT = 240
T_PRED = 100
NSTEP = 6000
NREST = 200
PROP = 0.04
LO, HI = 0.1, 3.0


def loglik(om, t, y):
    r = y - np.sin(om * t)
    return -0.5 * np.sum(r ** 2) / SIGMA ** 2


def search(y, t, rng, tempered):
    om = rng.uniform(LO, HI)
    ll = loglik(om, t, y)
    for i in range(NSTEP):
        beta = (2.0 ** (-6 * (1 - i / NSTEP))) if tempered else 1.0
        omp = om + PROP * rng.normal()
        if not (LO < omp < HI):
            continue
        llp = loglik(omp, t, y)
        if np.log(rng.random() + 1e-300) < beta * (llp - ll):
            om, ll = omp, llp
    return om


def preq_price(om, rng):
    t = np.arange(T_FIT + 1, T_FIT + T_PRED + 1)
    y = np.sin(OMSTAR * t) + SIGMA * rng.normal(size=T_PRED)
    r = y - np.sin(om * t)
    return float(np.mean(0.5 * np.log(2 * np.pi * SIGMA ** 2)
                         + 0.5 * r ** 2 / SIGMA ** 2))


if __name__ == "__main__":
    print("== node barriers: sharp vs tempered local search, equal "
          "compute ==")
    rng = np.random.default_rng(11)
    t = np.arange(1, T_FIT + 1)
    y = np.sin(OMSTAR * t) + SIGMA * rng.normal(size=T_FIT)
    res = {}
    for tempered in (False, True):
        oms = np.array([search(y, t, np.random.default_rng(500 + i),
                               tempered) for i in range(NREST)])
        trapped = np.abs(oms - OMSTAR) > 0.05
        prices = np.array([preq_price(om, np.random.default_rng(
            9000 + i)) for i, om in enumerate(oms)])
        res[tempered] = (trapped.mean(), prices.mean(),
                         prices.std() / np.sqrt(len(prices)))
        name = "tempered (annealed)" if tempered else "sharp      "
        print(f"  {name}: trap rate {trapped.mean():.2f}, "
              f"prequential price {prices.mean():.3f} +- "
              f"{prices.std() / np.sqrt(len(prices)):.3f} nats/pt")
    base = preq_price(OMSTAR, np.random.default_rng(1))
    print(f"  truth-omega reference: {base:.3f} nats/pt")
    dtrap = res[False][0] - res[True][0]
    dcost = res[False][1] - res[True][1]
    print(f"  smoothing buys: trap rate -{dtrap:.2f}, "
          f"{dcost:+.3f} nats/pt")
    assert res[False][0] > res[True][0] + 0.2
    assert dcost > 0.1
    print("  the sibling's prediction holds: the sharp surface's "
          "valleys (the nodes) trap")
    print("  local search and the trap is priced in code length; "
          "annealing -- the filter-side")
    print("  heat-kernel flow -- restores ergodicity at identical "
          "compute.")
    print("all assertions passed")
