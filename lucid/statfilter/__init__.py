"""statfilter -- parameter-free adaptive filter with no tuning constants.

The single public API:

    from statfilter import LucidFilter

    f = LucidFilter()                               # scalar random-walk level, direct observation
    f = LucidFilter(dynamics=F, H=H)                # supplied dynamics and sensors
    f = LucidFilter(dynamics=F, process=Q0, ...)    # also supply base noise magnitudes

    r = f.filter(Y)          # Y: (T, m); r.mean (T, n), r.var (T, n, n)
    r.process_scale          # (T, n) per process-eigenmode log-scale, online
    r.measurement_scale      # (T, m) per-sensor log-scale, online

Everything is vector -- pass a length-1 array for scalar problems.
``dynamics=None`` (learn the dynamics) is not yet implemented; it raises
``NotImplementedError`` and belongs to the future ODE-learning filter.

Prior specimens (AdaptiveFilter, VectorFilter, WalkingFilter, WalkingVectorFilter,
AdaptiveKalmanFilter) are preserved in ``research/multivariate-statfilter/specimens/``
for reference but are no longer part of the public API.
"""
from .lucid import LucidFilter, LucidStep, LucidResult

__all__ = ["LucidFilter", "LucidStep", "LucidResult"]
__version__ = "2.0.0"
