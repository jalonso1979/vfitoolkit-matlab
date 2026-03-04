"""Policy representation helpers.

These helpers translate between index-based policy functions used in
MATLAB (e.g., best action indices) and value-based policies, keeping the
array backend abstraction so CPU/GPU paths stay aligned.
"""
from __future__ import annotations

from typing import Sequence

from ..backend import ArrayBackend


def policy_index_to_value(policy_index: any, grid: Sequence[float], backend: ArrayBackend) -> any:
    """Map discrete policy indices to their grid values."""
    xp = backend.xp
    grid_arr = backend.asarray(grid)
    return grid_arr[policy_index]


def clamp_policy(policy_values: any, lower: float, upper: float, backend: ArrayBackend) -> any:
    """Project policy values back into admissible bounds."""
    xp = backend.xp
    return xp.clip(policy_values, lower, upper)


__all__ = ["policy_index_to_value", "clamp_policy"]
