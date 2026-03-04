"""Backend abstraction for CPU/GPU parity.

This helper centralizes the selection of NumPy vs. CuPy and exposes
narrow utility functions so downstream modules can stay agnostic to the
execution device. The design follows the plan to mirror MATLAB GPU
variants while keeping CPU fallbacks.
"""
from __future__ import annotations

from typing import Any, Iterable, Tuple

try:  # Optional GPU support
    import cupy as _cupy
except Exception:  # pragma: no cover - absence of GPU is acceptable
    _cupy = None

import numpy as _numpy


class ArrayBackend:
    """Lightweight dispatch layer around NumPy/CuPy."""

    def __init__(self, use_gpu: bool = False):
        self._use_gpu = use_gpu and _cupy is not None
        self.xp = _cupy if self._use_gpu else _numpy

    @property
    def using_gpu(self) -> bool:
        return self._use_gpu

    def asarray(self, arr: Any) -> Any:
        """Convert input to the backend array type."""
        return self.xp.asarray(arr)

    def zeros(self, shape: Tuple[int, ...], dtype=None) -> Any:
        return self.xp.zeros(shape, dtype=dtype)

    def ones(self, shape: Tuple[int, ...], dtype=None) -> Any:
        return self.xp.ones(shape, dtype=dtype)

    def to_host(self, arr: Any) -> Any:
        """Move backend arrays to CPU memory when necessary."""
        if self._use_gpu and hasattr(arr, "get"):
            return arr.get()
        return arr

    def rng(self, seed: int | None = None):
        """Return a random number generator bound to the backend."""
        if self._use_gpu:
            return self.xp.random.default_rng(seed)
        return _numpy.random.default_rng(seed)

    def broadcast_to(self, arr: Any, shape: Tuple[int, ...]) -> Any:
        return self.xp.broadcast_to(arr, shape)

    def stack(self, arrays: Iterable[Any], axis: int = 0) -> Any:
        return self.xp.stack(list(arrays), axis=axis)

    def maximum(self, a: Any, b: Any) -> Any:
        return self.xp.maximum(a, b)

    def abs(self, arr: Any) -> Any:
        return self.xp.abs(arr)


def get_backend(use_gpu: bool = False) -> ArrayBackend:
    """Factory returning a backend using GPU when available/desired."""
    return ArrayBackend(use_gpu=use_gpu)


__all__ = ["ArrayBackend", "get_backend"]
