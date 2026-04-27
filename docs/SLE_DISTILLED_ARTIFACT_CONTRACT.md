# SLE Distilled Artifact Contract (Level 1)

The Level 1 SLE distillation contract is explicit and self-contained.

## Artifact schema

`unified_ml::DistilledArtifact` now carries:

- distilled logic circuit (`BooleanCascade`),
- per-input feature quantization metadata (`DistilledFeatureQuantizer`),
- explicit target policy (`DistilledTargetPolicy`),
- versioned metadata (`DistilledArtifactMetadata`).

## Quantization contract

For each distilled input bit, the artifact stores:

- source feature index (`source_feature_index`),
- threshold value (`threshold`),
- binarization scheme (`ThresholdGreaterEqual`).

There are no implicit feature transforms; raw input values are mapped only through this metadata.

## Inference contract

Use `predict_with_distilled_sle(raw_features, artifact)` for one-call inference over raw dense tabular feature vectors.

Batch inference is also available with:

- `predict_with_distilled_sle(span<const vector<double>>, artifact)`.

## Serialization contract

Use:

- `save_distilled_artifact(artifact, path)`
- `load_distilled_artifact(path)`

The binary payload includes a fixed magic header and schema version check. Unsupported versions fail fast.

## Compatibility and failures

- Version mismatch throws `std::invalid_argument`.
- Quantizer/circuit input mismatch throws `std::invalid_argument`.
- Unknown quantization scheme throws `std::invalid_argument`.
- Truncated/corrupt payload throws `std::runtime_error`.
