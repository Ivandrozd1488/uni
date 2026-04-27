# SLE Level 3 Preflight Checklist

This checklist captures non-formal, but critical, readiness checks before starting the **Level 3 stability counter**.

## 1) Decision audit (`test_sle_level2_utility`)

Run the L2 harness and audit the *quality* of the win:

- Is there a clear win axis (size, latency, interpretability), or only a weak compression-only win?
- How much margin remains to hard gates (especially fidelity `>= 0.95`)?
- What is the latency regression window for the distilled artifact?

## 2) HPC deep profile (`predict_with_distilled_sle`)

Even with green utility gates, perform micro-architectural verification:

- Inspect cache behavior and branch quality with `perf stat`.
- Validate that vectorized code generation has not regressed for quantized threshold evaluation.

## 3) Sterile environment setup (pre-sanitizing)

Before enabling a 30-day no-red timer, run sanitizer and warning-blocking builds:

- ASan + UBSan
- MSan (Clang-only)
- `-Wall -Wextra -Wpedantic -Werror`

## 4) Safe-mode reproducibility check

Validate deterministic behavior with the conservative runtime profile:

```bash
ctest --test-dir build -R test_sle_safe_mode_and_repro --output-on-failure
```

This test exercises `unified_ml::make_sle_safe_mode_profile()` and verifies deterministic distillation fingerprints across repeated runs.

## 5) Practical integration scenario

Validate on a realistic downstream pipeline (not only synthetic manifests), for example:

- surrogate models in engineering/HPC workflows,
- infrastructure data feeds with noisy/dirty tabular inputs.

## Automation script

Use:

```bash
python3 tools/sle_l3_preflight.py --run-sanitizers
```

The script:

1. Executes `build/test_sle_level2_utility` on `tests/data/sle_level2_manifest.csv`.
2. Prints decision-audit margins and weak-win hints.
3. Runs `perf stat` around `predict_with_distilled_sle` via the L2 harness executable.
4. Optionally configures sanitizer builds (`build_asan_ubsan`, `build_msan`) with `UNIFIED_ML_ENABLE_WERROR=ON`.
5. Complements CI reliability gates described in `docs/SLE_L3_OPERATIONAL_PLAYBOOK.md`.

> Note: `UNIFIED_ML_ENABLE_MSAN` requires Clang and may fail on toolchains that do not provide MemorySanitizer runtime support.
