# SLE Level 3 Operational Playbook

This playbook defines runtime posture and incident handling for the SLE path when running as a **Level 3 promotion candidate**.

## 1) Safe mode runtime profile

Use `unified_ml::make_sle_safe_mode_profile()` for conservative deterministic operation.

Profile intent:

- deterministic synthesis seed (`0x5AFE0001`),
- bounded search (`gate_budget=16`, `iterations=128`),
- conservative rollout limits (`rollout_budget=32`, `rollout_patience=8`),
- JIT synthesis disabled (`prefer_jit=false`) to simplify operational debugging.

Use this profile for production canaries and all incident reproductions.

## 2) Fallback strategy

When SLE behavior is uncertain, route inference through the teacher/compact baseline.

Recommended staged fallback:

1. **Soft fallback (single request):** if `distill_to_sle(...)` or `predict_with_distilled_sle(...)` throws, retry request with baseline model.
2. **Session fallback:** if repeated failures are detected, pin the session/workload to baseline for the remainder of the run.
3. **Fleet fallback:** if CI or production alerts indicate systemic failure, disable SLE distillation by configuration and redeploy using baseline-only path.

## 3) Diagnostics checklist

On any failure or regression, capture:

- compiler family/version and build flags,
- `DistillationConfig` (or explicit safe-mode marker),
- distilled artifact metadata and gate count,
- fidelity/accuracy deltas versus teacher,
- p95 latency deltas,
- sanitizer trace (if available).

### Minimal repro command set

```bash
cmake -S . -B build -DUNIFIED_ML_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build -R 'test_sle_(distillation_diagnostics|distillation_edge_cases|safe_mode_and_repro|level2_utility)' --output-on-failure
```

For sanitizer repros:

```bash
cmake -S . -B build_san -DUNIFIED_ML_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug \
  -DUNIFIED_ML_ENABLE_ASAN=ON -DUNIFIED_ML_ENABLE_UBSAN=ON
cmake --build build_san -j
ctest --test-dir build_san -R 'test_sle_' --output-on-failure
```

## 4) Failure handling policy

- **Hard fail:** crash, sanitizer violation, artifact version mismatch, or gate-budget violation.
  - Action: immediate fallback + block promotion counter + open incident.
- **Soft fail:** utility drift (fidelity/latency) but no correctness break.
  - Action: keep fallback available, investigate drift, do not advance promotion timer until resolved.
- **Transient fail:** environment-only issue (tooling hiccup, host noise) with clean rerun.
  - Action: rerun once, then treat as soft fail if repeated.

## 5) CI mapping for Level 3 gates

- `linux-sle-reliability` workflow job:
  - runs SLE stress/diagnostic tests on **GCC + Clang**,
  - runs **ASan/UBSan** variants,
  - enforces `-Werror`,
  - includes deterministic reproducibility check (`test_sle_safe_mode_and_repro`).

Together with the existing full-test sanitizer job, this forms the CI gate for Level 3 reliability and reproducibility tracking.
