"""Python translation of the VFI Toolkit with optional GPU support."""
from .config import VFIRuntimeConfig, DEFAULT_CONFIG
from .backend import ArrayBackend, get_backend
from .vfi.value_iteration import ValueFunctionIterator, ValueFunctionResult, bellman_operator
from .policy.policies import policy_index_to_value, clamp_policy
from .discretization.grid import (
    evenly_spaced,
    cartesian_product,
    cumulative_trapz,
    normalize_weights,
)
from .simulation.time_series import simulate_markov_chain, simulate_policy

__all__ = [
    "VFIRuntimeConfig",
    "DEFAULT_CONFIG",
    "ArrayBackend",
    "get_backend",
    "ValueFunctionIterator",
    "ValueFunctionResult",
    "bellman_operator",
    "policy_index_to_value",
    "clamp_policy",
    "evenly_spaced",
    "cartesian_product",
    "cumulative_trapz",
    "normalize_weights",
    "simulate_markov_chain",
    "simulate_policy",
]
