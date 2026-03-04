"""Runtime configuration for VFI Toolkit Python translation.

This module centralizes numerical tolerances, convergence settings,
GPU toggles, and reproducibility controls. It mirrors the MATLAB pattern
of passing option structures into the iterative routines while exposing
Pythonic defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


def _default_logger(message: str) -> None:
    """Default progress logger used when verbose mode is enabled."""
    print(message)


@dataclass
class VFIRuntimeConfig:
    """Runtime settings shared across iteration, simulation, and estimation.

    Attributes
    ----------
    max_iter: int
        Maximum number of iterations for fixed point or root finding loops.
    tol: float
        Convergence tolerance for sup norms or residual norms.
    discount: float
        Discount factor used in Bellman operators; exposed here so helpers
        can validate the input domain.
    use_gpu: bool
        When ``True`` routines attempt to run with CuPy/Numba CUDA
        acceleration. Fallback to NumPy occurs automatically if GPU
        libraries are unavailable.
    rng_seed: Optional[int]
        Optional global seed propagated into random draws for reproducible
        simulations.
    verbose: bool
        Toggle progress logging. When ``True`` each module may emit human
        readable status lines via ``logger``.
    logger: Callable[[str], None]
        Callback used for logging. Defaults to ``print``; can be swapped for
        structured loggers in downstream applications.
    """

    max_iter: int = 1_000
    tol: float = 1e-6
    discount: float = 0.95
    use_gpu: bool = False
    rng_seed: Optional[int] = None
    verbose: bool = False
    logger: Callable[[str], None] = field(default=_default_logger)

    def log(self, message: str) -> None:
        """Emit ``message`` when ``verbose`` is enabled."""
        if self.verbose:
            self.logger(message)


DEFAULT_CONFIG = VFIRuntimeConfig()

__all__ = ["VFIRuntimeConfig", "DEFAULT_CONFIG"]
