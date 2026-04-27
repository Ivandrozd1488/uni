# unified_ml API Reference

This document is the public navigation map for the current `unified_ml` build. It is organized around what is stable, what is unified, what is advanced, and where the exact limits are.

## 1. Entry points

Recommended production include:

```cpp
#include <unified_ml_stable.hpp>
```

Compatibility umbrella:

```cpp
#include <unified_ml>
```

Optional extension point:

```cpp
#include <unified_ml_experimental.hpp>
```

Version macros come from `include/unified_ml_version.hpp`:

```cpp
UNIFIED_ML_VERSION_MAJOR
UNIFIED_ML_VERSION_MINOR
UNIFIED_ML_VERSION_PATCH
UNIFIED_ML_VERSION
UNIFIED_ML_VERSION_STRING
UNIFIED_ML_MAKE_VERSION(maj, min, patch)
```

## 2. Stable public surface

`unified_ml_stable.hpp` re-exports the production-facing surface for:

- core math and utility APIs
- autograd
- low-level model families
- unified higher-level facades
- capability descriptors
- stable namespace aliases under `core::models`

Representative low-level namespaces available through the stable umbrella:

- `mlp`
- `deep_onet`
- `pinn`
- `rf`
- `xgb`
- `dbscan`
- `iforest`
- `pca`
- `kriging`
- `transformer`
- `svm`
- `gp`
- `sindy`
- `pideeponet`

Stable production aliases also exist under `core::models`, for example:

- `core::models::random_forest`
- `core::models::svm`
- `core::models::xgboost`
- `core::models::pca`
- `core::models::isolation_forest`
- `core::models::sle`

## 3. Unified surface

The unified layer is additive. It exists to make common developer workflows more consistent without hiding model identity or introducing auto-selection.

### 3.1 Dataset facade

Defined in `include/unified_ml_dataset.hpp`.

Primary types:

- `unified_ml::LearningTask`
- `unified_ml::DatasetView`

Current contract:

- wraps tabular features as matrix views or owned matrices
- optionally carries targets and task metadata
- converts to and from `rf::Dataset`
- serves as the shared tabular input representation for higher-level facades

### 3.2 Metrics facade

Defined in `include/unified_ml_metrics.hpp`.

Primary types and helpers:

- `unified_ml::MetricTask`
- `unified_ml::ClassificationMetrics`
- `unified_ml::RegressionMetrics`
- `unified_ml::EvaluationSummary`
- `unified_ml::evaluate_classification(...)`
- `unified_ml::evaluate_regression(...)`
- `unified_ml::evaluate(...)`

Current contract:

- shared classification and regression summaries for tabular-style predictions
- built on top of existing production metrics implementations

### 3.3 Artifact and inference facade

Defined in `include/unified_ml_artifact.hpp`.

Primary types:

- `unified_ml::ArtifactFormat`
- `unified_ml::ModelKind`
- `unified_ml::InferenceOutput`
- `unified_ml::ModelArtifact`

Current contract:

- unified `save(...)`, `load(...)`, `export_artifact(...)`, and `run(...)`
- currently wraps:
  - `rf::RandomForest`
  - `svm::SVM`
  - `xgb::XGBModel`
- capability-aware wrapper without hiding the underlying model family

Exact limit:

- `UnifiedMLP` is not yet folded into `ModelArtifact`; it currently owns its own save/load contract

### 3.4 Unified tabular facade

Defined in `include/unified_ml_tabular.hpp`.

Primary types:

- `unified_ml::TabularModelKind`
- `unified_ml::RandomForestSpec`
- `unified_ml::SVMSpec`
- `unified_ml::XGBoostSpec`
- `unified_ml::FitSummary`
- `unified_ml::PredictionSummary`
- `unified_ml::TabularModel`

Current contract:

- unified high-level train, predict, evaluate, save, and load for tabular supervised tasks
- current backend coverage:
  - `rf::RandomForest`
  - `svm::SVM`
  - `xgb::XGBModel`
- model choice remains explicit

### 3.5 Unified explain/report facade

Defined in `include/unified_ml_reports.hpp`.

Primary types and helpers:

- `unified_ml::FeatureImportanceEntry`
- `unified_ml::ExplainSummary`
- `unified_ml::explain(const ModelArtifact&, ...)`
- `unified_ml::explain(const TabularModel&, ...)`

Current contract:

- unified feature importance for RF and XGBoost
- unified attribution/report surface for supported XGBoost explain paths

### 3.6 Unified MLP facade

Defined in `include/unified_ml_mlp.hpp`.

Primary types:

- `unified_ml::MLPActivationKind`
- `unified_ml::MLPSpec`
- `unified_ml::MLPFitSummary`
- `unified_ml::UnifiedMLP`

Current contract:

- dense supervised tabular MLP path
- explicit hidden layer, activation, learning rate, epoch, and binary classification configuration
- supports `fit(...)`, `predict(...)`, `save(...)`, and `load(...)`

Exact limits:

- intended as a practical wrapper for dense sequential tabular MLP use
- not a serializer for arbitrary custom layer graphs
- not yet integrated into `ModelArtifact`

### 3.7 Capability descriptors

Defined in `include/unified_ml_capabilities.hpp`.

Primary types and helpers:

- `unified_ml::ModelCapabilities`
- `unified_ml::capability_descriptor<Model>`
- `unified_ml::capability_descriptor_v<Model>`
- `unified_ml::capabilities_of<Model>()`
- `unified_ml::capabilities_of(model_instance)`

Capability fields:

- `supports_classification`
- `supports_regression`
- `supports_online_inference`
- `supports_batch_inference`
- `supports_serialization`
- `supports_fast_artifact_export`
- `supports_exact_distillation`
- `supports_constraints`

## 4. Advanced surface

The advanced layer is where the project stays honest about models whose outputs do not fit one uniform predictor contract.

### 4.1 Advanced model facade

Defined in `include/unified_ml_phase2.hpp`.

Primary types:

- `unified_ml::AdvancedModelKind`
- `unified_ml::GPSpec`
- `unified_ml::PCASpec`
- `unified_ml::IsolationForestSpec`
- `unified_ml::DBSCANSpec`
- `unified_ml::SINDySpec`
- `unified_ml::AdvancedFitSummary`
- `unified_ml::AdvancedPredictionSummary`
- `unified_ml::AdvancedModel`

Current contract:

- groups GP, PCA, Isolation Forest, DBSCAN, and SINDy into one higher-level family
- keeps outputs honest when they differ, such as transformed matrices, anomaly scores, cluster labels, or symbolic equations

### 4.2 Advanced artifact facade

Defined in `include/unified_ml_phase2_artifact.hpp`.

Current contract:

- unified persistence/artifact path for advanced models where backend support exists
- works with real underlying serialization support rather than pretending it exists

Exact limits:

- DBSCAN persistence remains unsupported and should be treated as unsupported, not implied

## 5. Exact limits and contract boundaries

### 5.1 SLE distillation

Defined in `include/unified_ml_distillation.hpp`.

Primary types and helpers:

- `unified_ml::DistillationConfig`
- `unified_ml::DistillationDataset`
- `unified_ml::DistillationDatasetBuilder`
- `unified_ml::DistilledCircuit`
- `unified_ml::DistillationSummary`
- `unified_ml::distill_to_sle(...)`

Current contract:

- dataset-based SLE distillation for supported RF, XGBoost, and Isolation Forest paths
- exact logic path for `mlp::Model` and `pinn::NeuralNetwork`
- capability-aware bridge into SLE circuits without pretending every model shares the same semantics

Exact limits:

- larger dataset-based distillation requests are reduced to a bounded representative set before synthesis for backend stability
- this is a real contract choice, not an auto-planner
- PCA is not exposed as a generic dataset-distillation path

### 5.2 Persistence boundaries

Current reality:

- `rf::RandomForest`, `svm::SVM`, and `xgb::XGBModel` participate in `ModelArtifact`
- PCA and Isolation Forest now have real native save/load support used by the advanced layer
- `UnifiedMLP` has its own save/load path
- DBSCAN persistence is unsupported

### 5.3 Uniformity boundaries

The project intentionally does not promise:

- automatic model choice
- one fake predictor contract for fundamentally different model families
- serialization support where the backend does not actually implement it

## 6. Core namespaces and representative low-level APIs

### `core`

Core math, memory, and optimizer primitives.

Representative headers:

- `include/core/linalg.hpp`
- `include/core/sdk_common.hpp`
- `include/core/random.hpp`
- `include/core/activations.hpp`
- `include/core/optimizers.hpp`
- `include/core/quantization.hpp`

Representative types:

- `core::Vector`
- `core::Matrix`
- `core::Error`
- `core::ConstRealSpan`
- `core::RealSpan`
- `core::SGD`
- `core::Adam`
- `core::RNG`

### `autograd`

Header root: `include/autograd/`

Primary user-facing type:

- `autograd::Tensor`

Representative capabilities:

- scalar and tensor arithmetic
- backward propagation
- graph-aware functional helpers
- gradient utilities for differentiable models

### Representative low-level model families

- `mlp::Sequential`, `mlp::Linear`, `mlp::Model`
- `deep_onet::DeepONet`
- `pinn::NeuralNetwork`, `pinn::PINNModel`
- `rf::Dataset`, `rf::RandomForest`
- `xgb::XGBModel`
- `svm::SVM`
- `gp::GaussianProcessRegressor`
- `pca::PCA`
- `iforest::IsolationForest`
- `dbscan::DBSCAN`
- `sindy::SINDy`
- `pideeponet::PIDeepONet`

## 7. Packaging and integration surface

Installed package outputs include:

- public headers under `<prefix>/include/`
- extensionless compatibility header `<prefix>/include/unified_ml`
- stable umbrella `<prefix>/include/unified_ml_stable.hpp`
- CMake package files under `<prefix>/lib/cmake/unified_ml/`
- pkg-config file under `<prefix>/lib/pkgconfig/unified_ml.pc`

## 8. Related documents

- `README.md`, landing page and quick integration overview
- `docs/BUILD_AND_RELEASE.md`, build, test, install, and release handbook
- `docs/ARCHITECTURE.md`, subsystem rules and public contracts
- `docs/STABILITY_CONTRACTS.md`, stability policy for this build
