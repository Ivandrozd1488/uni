# Contributing to unified_ml

This document defines the contributor workflow for `unified_ml`.

## Goals

Contributions should preserve four project qualities:

- production-oriented public API
- clear build and packaging behavior
- warning-clean native builds
- no accidental changes to validated mathematical cores when performing cleanup work

## Repository structure

```text
unified_ml/
├─ include/
├─ src/
├─ examples/
├─ tests/
├─ docs/
├─ cmake/
└─ CMakeLists.txt
```

### Public headers

Production-facing headers live under `include/`.

Important top-level entry points:

- `include/unified_ml_stable.hpp`
- `include/unified_ml.hpp`
- `include/unified_ml`
- `include/unified_ml_experimental.hpp`

### Internal and specialized subsystems

The repository may contain specialized internal modules, including UCAO. These modules can be fully built and tested without being exported as part of the top-level production SDK surface.

## Coding rules

- Use C++20-compatible code.
- Prefer `#pragma once` in headers.
- Use `snake_case` for functions and variables.
- Use `PascalCase` for classes, structs, and enums where appropriate.
- Mark non-mutating methods `const`.
- Use `noexcept` where behavior is truly non-throwing.
- Prefer application-owned formatting and callback-driven diagnostics over direct library console output.
- Use standard containers, smart pointers, or aligned allocators instead of raw ownership through `new` and `delete`.

## API and compatibility rules

- Do not break existing public APIs without a documented compatibility decision.
- Prefer additive overloads and compatibility shims over disruptive renames.
- Keep internal implementation details out of public umbrellas unless there is an explicit product decision to expose them.
- Do not expose internal acceleration details as public contract.

## Build rules

Before considering a change complete, validate the relevant paths:

```bash
cmake -S . -B build -DUNIFIED_ML_BUILD_PYTHON=OFF -DUNIFIED_ML_ENABLE_CUDA=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

For packaging and integration changes, also verify:

- install to a staging prefix
- `find_package(unified_ml REQUIRED)` consumption
- `FetchContent` consumption

## Documentation rules

When you change user-visible behavior, update the corresponding docs so the repository reads as one consistent system.

Primary documentation files:

- `README.md`
- `API_REFERENCE.md`
- `docs/ARCHITECTURE.md`
- `docs/STABILITY_CONTRACTS.md`
- `docs/BUILD_AND_RELEASE.md`
- `CHANGELOG.md`

## Testing expectations

At minimum, add or update tests when you change:

- public API behavior
- package/export/install behavior
- model training or inference semantics
- internal specialized subsystems with dedicated tests, such as UCAO

## Commit guidance

Use small, scoped commits with messages that describe the externally meaningful change, for example:

- `Polish warning hygiene for production SDK`
- `Polish SDK docs and integration surface`

## Pull request checklist

Before opening a PR, confirm:

- the code builds cleanly
- relevant tests pass
- examples still build
- docs reflect the new reality
- the change does not silently alter validated math behavior during cleanup work
