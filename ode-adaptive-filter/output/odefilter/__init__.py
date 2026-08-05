"""Adaptive filter for a process locally described by a linear ODE."""
from .core import OdeFilter, Params, FilterResult, Step, difference_matrix

__all__ = ["OdeFilter", "Params", "FilterResult", "Step", "difference_matrix"]
__version__ = "0.1.0"
