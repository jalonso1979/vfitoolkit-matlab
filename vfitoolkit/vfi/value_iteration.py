"""Bellman iteration and policy evaluation routines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from ..backend import ArrayBackend
from ..config import VFIRuntimeConfig


@dataclass
class ValueFunctionResult:
    value: Any
    policy: Any
    iterations: int
    converged: bool


class ValueFunctionIterator:
    """Value function iteration over discrete states/actions.

    Parameters
    ----------
    rewards : array_like
        Immediate returns with shape ``(n_states, n_actions)``.
    transitions : array_like
        Transition probabilities with shape ``(n_states, n_actions, n_states)``
        such that transitions[s, a, s_next] >= 0 and sums to 1 over s_next.
    config : :class:`~vfitoolkit.config.VFIRuntimeConfig`
        Iteration controls including discount factor and tolerances.
    backend : :class:`~vfitoolkit.backend.ArrayBackend`
        Array dispatch layer for CPU/GPU parity.
    """

    def __init__(
        self,
        rewards: Any,
        transitions: Any,
        config: VFIRuntimeConfig,
        backend: ArrayBackend,
    ) -> None:
        self.backend = backend
        self.config = config
        self.rewards = backend.asarray(rewards)
        self.transitions = backend.asarray(transitions)

        if self.rewards.ndim != 2:
            raise ValueError("Rewards must be 2D: (n_states, n_actions).")
        if self.transitions.ndim != 3:
            raise ValueError(
                "Transitions must be 3D: (n_states, n_actions, n_states)."
            )
        if self.rewards.shape[0] != self.transitions.shape[0]:
            raise ValueError("Reward and transition state dimensions must match.")
        if self.rewards.shape[1] != self.transitions.shape[1]:
            raise ValueError("Reward and transition action dimensions must match.")
        if not (0 < self.config.discount < 1 + 1e-12):
            raise ValueError("Discount factor must be in (0, 1].")

    def iterate(self, initial_value: Any | None = None) -> ValueFunctionResult:
        xp = self.backend.xp
        V = xp.zeros(self.rewards.shape[0]) if initial_value is None else self.backend.asarray(initial_value)
        V = V.astype(self.rewards.dtype, copy=False)

        converged = False
        policy = xp.zeros(self.rewards.shape[0], dtype=int)

        for it in range(1, self.config.max_iter + 1):
            expected = xp.tensordot(self.transitions, V, axes=([2], [0]))
            candidate = self.rewards + self.config.discount * expected
            V_new = xp.max(candidate, axis=1)
            policy = xp.argmax(candidate, axis=1)

            diff = xp.max(self.backend.abs(V_new - V))
            self.config.log(f"Iteration {it}: max diff={self.backend.to_host(diff)}")

            if diff <= self.config.tol:
                converged = True
                V = V_new
                break
            V = V_new

        return ValueFunctionResult(
            value=V,
            policy=policy,
            iterations=it,
            converged=converged,
        )

    def evaluate_policy(self, policy: Any, initial_value: Any | None = None) -> ValueFunctionResult:
        """Policy evaluation holding actions fixed.

        Parameters
        ----------
        policy : array_like
            Integer actions for each state, shape ``(n_states,)``.
        initial_value : array_like, optional
            Starting value function guess.
        """
        xp = self.backend.xp
        policy = policy.astype(int) if not hasattr(policy, "dtype") else policy.astype(int)
        V = xp.zeros(self.rewards.shape[0]) if initial_value is None else self.backend.asarray(initial_value)
        V = V.astype(self.rewards.dtype, copy=False)

        converged = False

        for it in range(1, self.config.max_iter + 1):
            chosen_rewards = self.rewards[xp.arange(self.rewards.shape[0]), policy]
            chosen_transitions = self.transitions[xp.arange(self.transitions.shape[0]), policy]
            continuation = chosen_transitions @ V
            V_new = chosen_rewards + self.config.discount * continuation

            diff = xp.max(self.backend.abs(V_new - V))
            self.config.log(f"Policy eval iteration {it}: max diff={self.backend.to_host(diff)}")

            if diff <= self.config.tol:
                converged = True
                V = V_new
                break
            V = V_new

        return ValueFunctionResult(
            value=V,
            policy=policy,
            iterations=it,
            converged=converged,
        )


def bellman_operator(rewards: Any, transitions: Any, value: Any, discount: float, backend: ArrayBackend) -> Any:
    """One-step Bellman update returning the maximal value across actions."""
    xp = backend.xp
    expected = xp.tensordot(transitions, value, axes=([2], [0]))
    candidate = rewards + discount * expected
    return xp.max(candidate, axis=1)


__all__ = ["ValueFunctionIterator", "ValueFunctionResult", "bellman_operator"]
