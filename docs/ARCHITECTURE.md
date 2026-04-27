# unified_ml Architecture and Public Contracts

This document describes the architectural rules that define the production-facing behavior of `unified_ml`.

## Layer model

The repository is organized into three practical layers.

### Core infrastructure

Foundational components expected to remain the most stable:

- `autograd`
- `core`

### Stable production modules

User-facing modules intended for normal production integration through the stable SDK surface:

- `mlp`
- `deep_onet`
- `pinn`
- `rf`
- `xgboost`
- `dbscan`
- `iforest`
- `pca`
- `kriging`
- `transformer`
- `svm`
- `gp`
- `sindy`
- `pideeponet`

### Internal specialized subsystems

Repository-owned subsystems that may be built and tested without being promoted into the top-level production umbrella. UCAO currently belongs to this category.

## Public entry points

Preferred production include:

```cpp
#include <unified_ml_stable.hpp>
```

Compatibility umbrella:

```cpp
#include <unified_ml>
```

Reserved opt-in extension point:

```cpp
#include <unified_ml_experimental.hpp>
```

## Differentiable contract

A trainable module belongs to the stable differentiable surface only if its supported forward path:

1. is composed from graph-preserving `autograd::Tensor` operations,
2. does not silently unwrap into raw buffers and rewrap disconnected trainable tensors,
3. preserves supported first-order gradient connectivity,
4. fails explicitly when it cannot preserve the graph contract.

## Backend isolation

SIMD, OpenMP, packed-kernel strategies, and CUDA-backed internal paths are implementation details.

Consumers should not depend on:

- a specific SIMD instruction set,
- a particular threading strategy,
- the binary layout of internal buffers,
- the exact dispatch strategy of internal kernels.

The public contract is defined by documented types, behavior, and package exports.

## Specialized internal modules

The repository may contain powerful internal modules that are intentionally not promoted into the top-level production SDK surface. UCAO is one such subsystem in this build.

Its internal engine-family declaration is centralized in `include/ucao/engine_registry.hpp`, and runtime selection policy is centralized in `include/ucao/engine_policy.hpp`. That lets internal code enable, disable, or prefer UCAO-backed execution paths by model family without changing the public API surface.

This means:

- it may compile and pass tests,
- it may support internal demos and internal benchmarks,
- it may be used as a foundation for repository-owned workflows,
- it is not automatically part of the stable top-level public umbrella.

## ABI policy

The project exports conventional shared-library versioning and stable target names, but does not promise a rigid C-style ABI guarantee across every minor release.

Treat ABI compatibility as disciplined packaging behavior, not as a guarantee stronger than the documented source-level contract.

## Thread-safety policy

### Safe

- different model instances on different threads
- read-only access to immutable objects after construction
- separate-object inference with no shared mutable state

### Not guaranteed safe

- concurrent training on the same model instance
- concurrent `backward()` or `grad()` on tensors that share a graph
- mutation of parameters while another thread reads or updates them
- sharing optimizer instances across threads without external synchronization

## Production rule of thumb

If a call can mutate gradients, graph state, optimizer state, or model parameters, assume external synchronization is required.


## SLE status and roadmap

SLE is currently treated as an isolated internal R&D track at the top-level distillation API boundary.

For promotion gates, benchmark policy, and maturity levels, see `docs/SLE_RND_ROADMAP.md`.
