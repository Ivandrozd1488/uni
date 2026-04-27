# SLE Level 2 Benchmark Harness

This benchmark suite implements the **Level 2** utility gates from `docs/SLE_RND_ROADMAP.md`.

## What is covered

- Fixed-seed benchmark runs driven by a version-controlled dataset manifest.
- Side-by-side comparison between:
  - teacher model (larger RF),
  - compact baseline (smaller RF),
  - SLE distilled artifact.
- Hard gate checks:
  - fidelity to teacher >= 0.95,
  - accuracy drop vs teacher <= 2 percentage points unless compression profile,
  - p95 latency regression <= 10% for non-latency-optimized profiles,
  - at least one clear win axis (size/gates, latency, or interpretability footprint).

## Files

- Harness executable: `tests/test_sle_level2_utility.cpp`
- Dataset manifest: `tests/data/sle_level2_manifest.csv`

## Run

```bash
cmake -S . -B build -DUNIFIED_ML_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build -R test_sle_level2_utility --output-on-failure
```

The test prints per-seed metrics and fails fast on any violated gate.
