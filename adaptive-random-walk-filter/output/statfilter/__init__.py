"""statfilter -- an adaptive local-level filter with no tuning parameters.

    from statfilter import AdaptiveFilter

    f = AdaptiveFilter.fit(x)
    r = f.filter(x)
    r.mean                      # tracked level
    r.measurement_regime        # signed, per step, always defined

See ``statfilter.core`` for the model and ``theory/`` for its derivation.
"""
from .core import AdaptiveFilter, FilterResult, Params, Step

__all__ = ["AdaptiveFilter", "FilterResult", "Params", "Step"]
__version__ = "1.0.0"
