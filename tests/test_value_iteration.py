import numpy as np

from vfitoolkit.backend import get_backend
from vfitoolkit.config import VFIRuntimeConfig
from vfitoolkit.vfi.value_iteration import ValueFunctionIterator, bellman_operator


def test_value_iteration_converges_to_known_solution():
    # Simple two-state, two-action example
    rewards = np.array([[1.0, 0.5], [0.0, 1.0]])
    transitions = np.array(
        [
            [[0.9, 0.1], [0.7, 0.3]],
            [[0.5, 0.5], [0.2, 0.8]],
        ]
    )

    config = VFIRuntimeConfig(max_iter=500, tol=1e-8, discount=0.95)
    backend = get_backend(use_gpu=False)
    iterator = ValueFunctionIterator(rewards, transitions, config, backend)
    result = iterator.iterate()

    assert result.converged
    # Reference values computed separately
    expected_value = np.array([8.746, 9.5])
    np.testing.assert_allclose(backend.to_host(result.value), expected_value, rtol=1e-3)
    assert result.policy.shape == (2,)


def test_bellman_operator_matches_iterator_step():
    rewards = np.array([[1.0, 0.5], [0.0, 1.0]])
    transitions = np.array(
        [
            [[0.9, 0.1], [0.7, 0.3]],
            [[0.5, 0.5], [0.2, 0.8]],
        ]
    )
    backend = get_backend(use_gpu=False)
    config = VFIRuntimeConfig(max_iter=1, tol=1e-8, discount=0.95)
    iterator = ValueFunctionIterator(rewards, transitions, config, backend)

    V0 = np.zeros(2)
    step = bellman_operator(rewards, transitions, V0, config.discount, backend)
    result = iterator.iterate(initial_value=V0)
    np.testing.assert_allclose(step, backend.to_host(result.value))
