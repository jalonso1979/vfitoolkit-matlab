# Python Translation and GPU Strategy for VFI Toolkit

This document outlines how to port the MATLAB VFI Toolkit to Python while preserving
CUDA-accelerated execution where applicable. The focus is on mirroring the structure
of the existing toolkit, delivering equivalent numerical outputs, and exposing clear
Python APIs for value-function iterations, policy evaluation, simulation, and
estimation routines.

## 1. Inventory of MATLAB components

The repository organizes functionality by task-specific folders:
- `DiscretizationMethods/`: state-space grids and quadrature utilities.
- `ReturnFnMatrix/`, `ValueFnIter/`, `PolicyInd2Val/`, `aprimeFnMatrix/`,
  `PhiaprimeFnMatrix/`, `ValueFnFromPolicy/`: core VFI primitives for
  return computation, policy extraction, and value-function iteration.
- `StationaryDist/`, `SimulateTimeSeries/`, `TransitionPaths/`: distribution and
  simulation routines for steady state and transition analyses.
- `Optimization/`, `Estimation/`, `EvaluateFnOnAgentDist/`: higher-level tools
  for calibration, likelihood evaluation, and numerical optimization.
- `SubCodes/`, `Other/`, and `DataEtc/`: shared helpers, sample data, and
  additional utilities.

Use this mapping to ensure each MATLAB function has a Python counterpart and that
module namespaces remain discoverable.

## 2. High-level Python package design

- Package layout: create a top-level `vfitoolkit` package with submodules
  reflecting the folders above (e.g., `vfitoolkit.discretization`,
  `vfitoolkit.vfi`, `vfitoolkit.simulation`, `vfitoolkit.estimation`).
- Public API: expose key entry points (value function iteration, policy function
  generation, transition simulation, estimation) in `vfitoolkit.__init__` with
  typed signatures and docstrings mirroring MATLAB usage examples.
- Configuration: centralize numerical tolerances, convergence criteria, and GPU
  toggles via a `config` module to keep behaviors consistent across routines.

## 3. Core translation patterns

- Vectorization first: preserve MATLAB vectorized logic using NumPy for CPU paths
  and CuPy for GPU-backed arrays; avoid Python loops unless algorithmically
  necessary.
- Randomness: map MATLAB RNG calls to NumPy/CuPy random generators with explicit
  seeds passed through the public APIs to keep reproducibility.
- Linear algebra: translate MATLAB matrix operations to NumPy/CuPy equivalents;
  lean on `scipy.sparse` and `cupyx.scipy.sparse` for transition matrices when
  large state spaces appear in `StationaryDist` and `TransitionPaths`.
- Interpolation/approximations: for methods in `DiscretizationMethods`, use
  `numpy.interp`, `scipy.interpolate`, or custom kernels; provide GPU variants
  using `cupy.interp` fallbacks when available.

## 4. CUDA and GPU execution strategy

- Array backend abstraction: implement a lightweight `backend` helper that
  dispatches NumPy vs. CuPy based on a user flag and availability; ensure each
  module depends only on this backend to keep CPU/GPU parity.
- Kernel hotspots: for elementwise returns and Bellman updates in `ReturnFnMatrix`
  and `ValueFnIter`, implement Numba CUDA kernels or CuPy RawKernels to match
  MATLAB+CUDA performance. Keep CPU NumPy implementations for environments
  without GPUs.
- Memory layout: favor contiguous arrays (`np.ascontiguousarray`) and preallocate
  buffers reused across iterations to mirror MATLAB preallocation patterns.
- Convergence checks: keep GPU-resident norms (e.g., `cupy.linalg.norm`) to avoid
  host-device transfers; provide `backend.to_host()` utilities when results must
  return to Python scalars.

## 5. Module-by-module guidance

- Value-function iteration (`ValueFnIter/`, `ReturnFnMatrix/`): port Bellman
  operators as functions accepting grids, shocks, and policy indices; design a
  class-based interface (e.g., `ValueFunctionIterator`) with methods for GPU
  and CPU runs.
- Policy extraction (`PolicyInd2Val/`, `aprimeFnMatrix/`, `PhiaprimeFnMatrix/`):
  encode policy matrices as dense arrays; expose helpers to translate between
  index-based policies and value-based policies, reusing shared backend ops.
- Distribution and simulation (`StationaryDist/`, `SimulateTimeSeries/`,
  `TransitionPaths/`): express transition matrices and simulation loops using
  vectorized state updates; for GPU support, keep shocks and states as CuPy
  arrays and limit host transfers to final summaries.
- Estimation and optimization (`Estimation/`, `Optimization/`,
  `EvaluateFnOnAgentDist/`): wrap objective functions to accept a `use_gpu`
  flag; integrate with `scipy.optimize` for CPU and consider PyTorch/JAX
  autograd paths for GPU-friendly gradient-based routines.

## 6. Testing and validation

- Fixture parity: construct regression tests that compare Python outputs against
  MATLAB baselines for representative models (e.g., deterministic growth,
  stochastic growth with discrete shocks, heterogeneous agents).
- Numerical tolerance: adopt relative/absolute tolerance thresholds matching
  MATLAB defaults; report deviations and ensure GPU/CPU results stay within
  the same bounds.
- Performance checks: benchmark critical kernels on CPU vs. GPU to validate that
  CUDA paths deliver expected speedups; profile data transfers to minimize
  overhead.

## 7. Packaging and documentation

- Packaging: use `pyproject.toml` with modern build backends; declare optional
  dependencies (`cupy`, `numba`, `torch`) under extra requirements for GPU.
- Documentation: mirror MATLAB function docs with Sphinx API references and
  usage guides; include notebooks demonstrating CPU and GPU runs for each major
  feature area.
- Examples: port MATLAB demo scripts into `examples/` with equivalent Python
  workflows, showcasing configuration of grids, running iterations, and
  visualizing policies and distributions.

Following these steps will yield a Python toolkit that tracks the MATLAB
implementation closely while offering portable GPU acceleration.
