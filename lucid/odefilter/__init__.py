"""Adaptive filter for a process locally described by a linear ODE."""
from .core import OdeFilter, Params, FilterResult, Step, difference_matrix
from .offset import OffsetFilter, OffsetStep, delay_row, cross_anchor

__all__ = ["OdeFilter", "Params", "FilterResult", "Step", "difference_matrix",
           "OffsetFilter", "OffsetStep", "delay_row", "cross_anchor"]
__version__ = "0.2.0"
