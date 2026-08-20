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

See ``statfilter.core`` / ``statfilter.walking`` for the models and ``theory/``
for their derivations.
"""
from .core import AdaptiveFilter, FilterResult, Params, Step
from .walking import (WalkingFilter, WalkResult, WalkStep,
                      WalkingBank, BankStep, BankResult)

__all__ = ["AdaptiveFilter", "FilterResult", "Params", "Step",
           "WalkingFilter", "WalkResult", "WalkStep",
           "WalkingBank", "BankStep", "BankResult"]
__version__ = "1.2.0"
