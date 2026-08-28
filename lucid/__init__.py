"""lucid -- parameter-free adaptive filter with no tuning constants.

    from lucid import LucidFilter

    f = LucidFilter()                               # scalar random-walk level, direct observation
    f = LucidFilter(dynamics=F, H=H)                # supplied dynamics and sensors
    r = f.filter(Y)                                 # Y: (T, m); r.mean (T, n), r.var (T, n, n)
    r.process_scale                                 # (T, n) per process-eigenmode log-scale, online
    r.measurement_scale                             # (T, m) per-sensor log-scale, online

Everything is vector -- pass a length-1 array for scalar problems.
``dynamics=None`` (learn the dynamics) raises ``NotImplementedError``;
it belongs to the future ODE-learning filter.
"""
from .statfilter import LucidFilter, LucidStep, LucidResult

__all__ = ["LucidFilter", "LucidStep", "LucidResult"]
__version__ = "2.0.0"
