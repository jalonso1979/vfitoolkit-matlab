"""Grid construction utilities mirroring MATLAB discretization helpers."""
from __future__ import annotations

from itertools import product
from typing import Iterable, Sequence, Tuple

from ..backend import ArrayBackend


def evenly_spaced(lower: float, upper: float, size: int, backend: ArrayBackend) -> any:
    """Return a linear grid with ``size`` nodes between ``lower`` and ``upper``."""
    xp = backend.xp
    return xp.linspace(lower, upper, size)


def cartesian_product(grids: Sequence[Iterable[float]], backend: ArrayBackend) -> any:
    """Return the cartesian product of 1D grids as a 2D array.

    This mirrors MATLAB's ``ndgrid`` usage commonly found in the toolkit.
    """
    xp = backend.xp
    mesh = xp.array(list(product(*grids)))
    return mesh


def cumulative_trapz(values: any, grid: any, backend: ArrayBackend) -> any:
    """Compute cumulative trapezoidal integrals along a grid."""
    xp = backend.xp
    diffs = xp.diff(grid)
    avg_heights = 0.5 * (values[1:] + values[:-1])
    increments = diffs * avg_heights
    return xp.concatenate([xp.array([0.0], dtype=values.dtype), xp.cumsum(increments)])


def normalize_weights(weights: any, backend: ArrayBackend) -> any:
    """Normalize quadrature or probability weights."""
    xp = backend.xp
    total = xp.sum(weights)
    if total == 0:
        raise ValueError("Weights must sum to a positive value.")
    return weights / total


__all__ = [
    "evenly_spaced",
    "cartesian_product",
    "cumulative_trapz",
    "normalize_weights",
]
