# unified_ml Stability Contracts

This document defines what is considered stable in this build of `unified_ml`.

## Stable production surface

A stable component in this build provides:

- documented inputs and usage expectations,
- explicit failure on invalid inputs where applicable,
- predictable fit and inference behavior within documented scope,
- alignment with repository naming and lifecycle conventions,
- no intentional public API design that hides raw-buffer shortcuts behind user-facing abstractions.

## Stable production modules

The production-facing stable surface includes:

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

## Core infrastructure

The most foundational stability layer includes:

- `autograd`
- `core`

## Internal specialized subsystems

Some repository subsystems are intentionally treated as internal specialized components rather than part of the top-level production umbrella. UCAO is in this category in the current build.

That means:

- it can be compiled and tested,
- it can be used internally by repository-owned demos or workflows,
- it is not automatically covered by the top-level stable SDK umbrella contract.

## Compatibility interpretation

The stable contract is primarily source-level and package-level:

- documented headers,
- exported CMake targets,
- installed example workflows,
- documented model lifecycle behavior.

Internal implementation details may evolve as long as the documented stable surface remains coherent.

## Differentiable behavior

Stable trainable modules are expected to preserve the documented autograd contract for the supported paths they expose. Unsupported graph-preserving behavior should fail explicitly rather than degrade silently.

## Logging and diagnostics

Stable public APIs should avoid unsolicited console output. Structured reports, return values, formatting helpers, and callback-driven logging are preferred.
