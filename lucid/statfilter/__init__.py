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

See ``statfilter.core`` / ``statfilter.walking`` / ``statfilter.vector`` for the
models and ``theory/`` for their derivations.
"""
from .core import AdaptiveFilter, FilterResult, Params, Step
from .walking import (WalkingFilter, WalkResult, WalkStep,
                      WalkingBank, BankStep, BankResult)
from .vector import VectorFilter, VecParams, VecStep, VecFilterResult

__all__ = ["AdaptiveFilter", "FilterResult", "Params", "Step",
           "WalkingFilter", "WalkResult", "WalkStep",
           "WalkingBank", "BankStep", "BankResult",
           "VectorFilter", "VecParams", "VecStep", "VecFilterResult"]
__version__ = "1.4.0"
