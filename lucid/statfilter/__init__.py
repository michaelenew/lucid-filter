"""statfilter -- an adaptive local-level filter with no tuning parameters.

    from statfilter import AdaptiveFilter

    f = AdaptiveFilter.fit(x)           # learns six parameters offline
    r = f.filter(x)
    r.mean                      # tracked level
    r.measurement_regime        # signed, per step, always defined

Or track the noise scale online, with no fit -- supplying only the process
log-scale AR(1) pair ``(phi, s)``:

    from statfilter import WalkingFilter

    f = WalkingFilter(Q=1.0, s2=1.0, phi=0.9, s=0.30)
    r = f.filter(x)
    r.process_scale             # process log-scale, tracked step by step

For a multivariate state read through a measurement matrix you supply, use
``VectorFilter`` -- the same model with an n-vector state, an m-vector
observation, and a supplied ``H`` (m x n); ``fit()`` infers the full-symmetric
base covariances and the noise scales.  At n = m = 1, H = [[1]] it is exactly
``AdaptiveFilter``:

    from statfilter import VectorFilter

    f = VectorFilter.fit(Y, H)          # Y is (T, m); H (m, n) supplied
    r = f.filter(Y)                     # r.mean (T, n), r.var (T, n, n)

To track the noise scale of **each component** online -- a log-scale per process
eigenmode and per sensor, walked with no fit, so you learn *which* component's
noise is up -- use ``WalkingVectorFilter`` (a theory testbed; see its README
section for the honest coupling residual):

    from statfilter import WalkingVectorFilter

    f = WalkingVectorFilter(Q0, R0, H)      # base process cov, per-sensor R0, supplied H
    r = f.filter(Y)                         # r.process_scale (T, n), r.measurement_scale (T, m)

See ``statfilter.core`` / ``statfilter.walking`` / ``statfilter.vector`` /
``statfilter.walkingvector`` for the models and ``theory/`` for their derivations.
"""
from .core import AdaptiveFilter, FilterResult, Params, Step
from .walking import (WalkingFilter, WalkResult, WalkStep,
                      WalkingBank, BankStep, BankResult)
from .vector import VectorFilter, VecParams, VecStep, VecFilterResult
from .walkingvector import WalkingVectorFilter, WalkVecStep, WalkVecResult

__all__ = ["AdaptiveFilter", "FilterResult", "Params", "Step",
           "WalkingFilter", "WalkResult", "WalkStep",
           "WalkingBank", "BankStep", "BankResult",
           "VectorFilter", "VecParams", "VecStep", "VecFilterResult",
           "WalkingVectorFilter", "WalkVecStep", "WalkVecResult"]
__version__ = "1.5.0"
