"""The lucid filter -- parameter-free adaptive state estimation, no tuning constants.

The single public API:

    from lucid import LucidFilter

    f = LucidFilter()                               # scalar random-walk level, direct observation
    f = LucidFilter(dynamics=F, H=H)                # supplied dynamics and sensors
    f = LucidFilter(dynamics=F, process=Q0, ...)    # also supply base noise magnitudes
    f = LucidFilter(dynamics=None)                  # LEARN the dynamics online
    f = LucidFilter(dynamics=F, faults=1e-4)        # supplied dynamics that may CHANGE

    r = f.filter(Y)          # Y: (T, m); r.mean (T, n), r.var (T, n, n)
    r.process_scale          # (T, n) per process-eigenmode log-scale, online
    r.measurement_scale      # (T, m) per-sensor log-scale, online

Everything is vector -- pass a length-1 array for scalar problems.
``dynamics=None`` (or ``faults=rho`` around a supplied ``F``) turns on the
dynamics channel: ``r.dynamics`` is the learned ``F`` per step, ``r.control``
the learned ``B``, and ``r.fault`` the posterior probability that the dynamics
have left the nominal.  See ``research/dynamics-learning/SUMMARY.md``.

The earlier fitted and walking filters this one generalises (AdaptiveFilter,
VectorFilter, WalkingFilter, WalkingVectorFilter, AdaptiveKalmanFilter, and the
fit-based odefilter) are preserved in
``research/multivariate-statfilter/specimens/`` and are no longer shipped.
"""
from .lucid import LucidFilter, LucidStep, LucidResult

__all__ = ["LucidFilter", "LucidStep", "LucidResult"]
__version__ = "2.0.0"
