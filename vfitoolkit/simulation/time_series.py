"""Time-series simulation utilities for Markov chains and policies."""
from __future__ import annotations

from typing import Any, Tuple

from ..backend import ArrayBackend
from ..config import VFIRuntimeConfig


def simulate_markov_chain(
    transition: Any,
    initial_state: int,
    draws: Any,
    backend: ArrayBackend,
) -> Any:
    """Simulate a finite-state Markov chain given exogenous draws in [0,1)."""
    xp = backend.xp
    transition = backend.asarray(transition)
    draws = backend.asarray(draws)

    n_states = transition.shape[0]
    states = xp.empty(draws.shape[0] + 1, dtype=int)
    states[0] = initial_state

    cum_probs = xp.cumsum(transition, axis=1)
    for t in range(draws.shape[0]):
        u = draws[t]
        states[t + 1] = xp.searchsorted(cum_probs[states[t]], u)
    return states


def simulate_policy(
    policy: Any,
    transition: Any,
    shocks: Any,
    initial_state: int,
    config: VFIRuntimeConfig,
    backend: ArrayBackend,
) -> Tuple[Any, Any]:
    """Simulate state and action paths under a fixed policy."""
    xp = backend.xp
    rng = backend.rng(config.rng_seed)
    draws = rng.random(shocks)

    states = simulate_markov_chain(transition, initial_state, draws, backend)
    actions = policy[states[:-1]]
    return states, actions


__all__ = ["simulate_markov_chain", "simulate_policy"]
