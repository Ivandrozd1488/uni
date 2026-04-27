# unified_ml

C++20 machine learning SDK for native applications, with a stable public surface, installable packages, and optional internal acceleration.

## Why this project

`unified_ml` is meant to be usable in two honest ways:

- **explicit low-level APIs** when you want direct control over a concrete model
- **unified higher-level facades** when you want consistent dataset, metrics, persistence, inference, and reporting flows

It does **not** try to auto-pick models or hide important differences behind fake uniformity.

## Recommended include

Production entry point:

```cpp
#include <unified_ml_stable.hpp>
```

Compatibility umbrella:

```cpp
#include <unified_ml>
```

## What is stable in this build

### Core stable surface

- core math and utility primitives
- autograd
- low-level model families such as RF, SVM, XGBoost, MLP, GP, PCA, Isolation Forest, DBSCAN, SINDy, PINN, DeepONet, PI-DeepONet, Transformer, and related headers exported by `unified_ml_stable.hpp`
- package exports for CMake and pkg-config

### Unified developer-facing surface

- `unified_ml::DatasetView`
- `unified_ml::EvaluationSummary` and `unified_ml::evaluate(...)`
- `unified_ml::ModelArtifact` and `unified_ml::InferenceOutput`
- `unified_ml::TabularModel`
- `unified_ml::ExplainSummary` and `unified_ml::explain(...)`
- `unified_ml::AdvancedModel`
- `unified_ml::UnifiedMLP`
- `unified_ml::distill_to_sle(...)`
- `unified_ml::ModelCapabilities`

## High-level contracts at a glance

### Tabular unified path

Use when you want one consistent train, predict, evaluate, save, and load flow for tabular supervised models.

Current backend coverage:

- `rf::RandomForest`
- `svm::SVM`
- `xgb::XGBModel`

### MLP unified path

Use when you want a practical dense tabular MLP wrapper.

Current scope:

- dense sequential supervised tabular MLPs
- `fit(...)`
- `predict(...)`
- `save(...)`
- `load(...)`

Not promised in the current contract:

- arbitrary custom graph serialization
- `ModelArtifact` integration

### Advanced model path

Use when model outputs are not honestly the same as tabular supervised prediction.

Current family:

- Gaussian Process
- PCA
- Isolation Forest
- DBSCAN
- SINDy

### SLE distillation path

Current scope:

- dataset-based distillation for supported RF, XGBoost, and Isolation Forest paths
- exact logic paths for `mlp::Model` and `pinn::NeuralNetwork`
- larger distillation datasets are reduced to a bounded representative set before synthesis for backend stability
- conservative runtime profile via `unified_ml::make_sle_safe_mode_profile()` for deterministic operations

## Tiny example

```cpp
#include <unified_ml_stable.hpp>

unified_ml::DatasetView dataset(X_train, y_train, unified_ml::LearningTask::Classification);

unified_ml::TabularModel model(unified_ml::RandomForestSpec{});
model.fit(dataset);

auto prediction = model.predict(dataset, 2);
auto metrics = unified_ml::evaluate(dataset, prediction.output, 2);
model.save("forest.umd");
```

## Installation and integration

### Fast local install

```bash
git clone https://github.com/drozdisme/unified_ml
cd unified_ml
./install.sh
```

Custom prefix:

```bash
./install.sh --prefix ~/.local
```

### CMake package

```cmake
find_package(unified_ml REQUIRED)
target_link_libraries(my_app PRIVATE unified_ml::unified_ml)
```

### FetchContent

```cmake
include(FetchContent)

FetchContent_Declare(
  unified_ml
  GIT_REPOSITORY https://github.com/drozdisme/unified_ml.git
  GIT_TAG v1.0.0
)

set(UNIFIED_ML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(UNIFIED_ML_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(unified_ml)

target_link_libraries(my_app PRIVATE unified_ml::unified_ml)
```

### pkg-config

```bash
g++ main.cpp $(pkg-config --cflags --libs unified_ml) -o my_app
```

## Build requirements

- CMake 3.21+
- C++20 compiler
- GCC 11+, Clang 14+, or modern MSVC

Optional backends are autodetected where available:

- OpenMP
- CUDA Toolkit
- pybind11

## Installed layout

```text
<prefix>/
├─ include/
├─ lib/
│  ├─ cmake/unified_ml/
│  └─ pkgconfig/
└─ share/unified_ml/
   ├─ examples/basic_usage/
   ├─ examples/fetchcontent/
   ├─ examples/install_test/
   └─ examples/ucao_showcase/
```

## Documentation map

- `API_REFERENCE.md`, public API structure and limits
- `docs/BUILD_AND_RELEASE.md`, build, test, install, packaging, and release handbook
- `docs/ARCHITECTURE.md`, subsystem rules and public contracts
- `docs/STABILITY_CONTRACTS.md`, stability policy for this build
- `docs/SLE_L3_OPERATIONAL_PLAYBOOK.md`, fallback, diagnostics, and failure handling for SLE Level 3

## Notes

- Development-only tests and benchmarks are repository assets, not release payload.
- Independent model instances are safe to use on different threads, but concurrent mutation of the same model instance is not guaranteed safe without external synchronization.
