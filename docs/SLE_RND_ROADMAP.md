# SLE R&D Roadmap and Promotion Gates

This document defines how to evolve SLE from an isolated internal subsystem into a production-candidate capability.

## Current status (April 24, 2026)

SLE is currently isolated at the `unified_ml_distillation.hpp` surface (`distill_to_sle(...)` throws), which is the right default for stable consumers while the contract is incomplete.

At the same time, the internal distillation path already demonstrates a valid optimization basis (information-gain split selection + explicit gate budget) and has passing diagnostic/edge tests.

## Target outcome

Promote SLE only when it can satisfy all three pillars simultaneously:

1. **Contract completeness**: distilled artifacts are self-contained and reproducible.
2. **Measured utility**: wins are demonstrated against baselines in benchmark suites.
3. **Reliability**: deterministic and sanitizer-clean behavior under stress.

---

## Maturity levels

### Level 0 — Isolated research prototype (current)

**Definition**
- Internal implementation exists and can be tested.
- Public stable entry points remain guarded/disabled.

**Exit criteria**
- Level 1 criteria all green.

### Level 1 — Reproducible R&D artifact

**Required capabilities**
- Distillation output carries full quantization contract:
  - feature binarization scheme,
  - threshold values and feature mapping,
  - target threshold/polarity policy,
  - versioned artifact metadata.
- One-call inference helper over raw features:
  - `predict_with_distilled_sle(raw_features, artifact)`.
- Round-trip serialization/deserialization of full distilled artifact.

**Hard gates**
- 100% parity after round-trip on deterministic fixture sets.
- 0 budget violations in stress loop (`N >= 10_000`) on small synthetic datasets.
- No hidden preprocessing assumptions in API contract docs.

### Level 2 — Utility-proven candidate

**Required capabilities**
- Benchmark harness with fixed seeds and dataset manifests.
- Side-by-side comparison vs teacher and compact baselines.

**Hard gates (minimum acceptance)**
- On at least one task family, SLE demonstrates a **clear win** on one axis:
  - model size / gate count,
  - latency,
  - interpretability footprint,
  while not causing catastrophic drop on accuracy/fidelity.
- Global guardrails:
  - fidelity to teacher >= 0.95 on selected distillation tracks,
  - accuracy drop vs teacher <= 2 percentage points unless explicitly flagged as a compression profile,
  - p95 latency regression <= 10% on profiles where SLE is not expected to optimize latency.

### Level 3 — Stable-track promotion candidate

**Required capabilities**
- CI-integrated stress + sanitizer matrix for SLE paths.
- Clear “safe mode” runtime profile (deterministic, conservative options).
- Operational playbook: fallback strategy, diagnostics, and failure handling.

**Hard gates**
- 30 consecutive CI days with no SLE-specific red builds.
- ASan/UBSan clean on supported Linux toolchains.
- Reproducibility pass on at least two compiler families.

---

## Implementation plan

## Phase A (1–2 weeks): contract completion

### Work items
- Add `DistilledArtifact` schema (or extend `DistillationSummary`) with quantization metadata.
- Add inference adapter over raw dense tabular input.
- Add artifact versioning + compatibility checks.
- Add round-trip tests and negative tests for schema mismatch.

### Deliverables
- New/updated public headers for distilled artifact I/O.
- Unit tests for serialization and prediction parity.
- Contract documentation with examples.

## Phase B (2–4 weeks): benchmark proof

### Benchmark families
1. **Boolean/rule-heavy synthetic**: AND/OR/XOR composites, noisy CNF/DNF.
2. **Tabular real-world**: medium-scale binary classification datasets with threshold-friendly structure.
3. **Edge profile**: strict budget and latency-sensitive inference profiles.

### Metrics
- Teacher fidelity (agreement rate).
- Holdout accuracy / AUROC.
- Gate count and serialized artifact bytes.
- p50/p95 inference latency.
- Build time for distillation.

### Reporting
- Keep benchmark matrix in version-controlled markdown/CSV.
- Track each run with commit SHA, compiler, CPU, seed set.
- Current implementation: `tests/test_sle_level2_utility.cpp` + `tests/data/sle_level2_manifest.csv` (see `docs/SLE_LEVEL2_BENCHMARKS.md`).

## Phase C (2–3 weeks): reliability hardening

### Work items
- Seed sweep and dataset perturbation stress tests.
- Sanitizer jobs for SLE on CI.
- Deterministic safe-mode profile (`unified_ml::make_sle_safe_mode_profile()`).
- Failure taxonomy + user-facing diagnostics (`docs/SLE_L3_OPERATIONAL_PLAYBOOK.md`).

### Deliverables
- CI jobs and failure budget thresholds.
- Reliability report with flaky-test analysis.

---

## Benchmark matrix template

| Suite | Dataset | Seeds | Budget profile | Baselines | Primary metric | Promotion threshold |
|---|---|---:|---|---|---|---|
| LogicSynth | AND/OR/XOR mix | 100 | strict | teacher, tiny RF | fidelity | >= 0.99 |
| TabularA | threshold-friendly real data | 30 | balanced | teacher, distilled tree | accuracy delta | <= 0.02 |
| EdgeLatency | production-like slice | 30 | strict | quantized baseline | p95 latency | win or <= 1.10x |

---

## Definition of “develop to what level”

The minimum justified target is **Level 2**.

- Level 1 alone proves correctness but not business value.
- Level 3 is costly and should only start after Level 2 demonstrates repeatable utility.

So the practical objective is:

1. Reach **Level 1** quickly to close artifact-contract risk.
2. Push to **Level 2** and decide go/no-go based on benchmark evidence.
3. Attempt **Level 3** only if Level 2 shows strong wins.
4. Run the non-gate preflight checks from `docs/SLE_L3_PREFLIGHT.md` before starting the 30-day stability counter.

---

## Go / no-go policy

- **Go**: Level 2 gates pass and at least one deployment profile shows clear advantage.
- **Hold**: correctness is good but no benchmark advantage; keep as internal optional tool.
- **No-go**: repeated reliability or reproducibility failures after Phase C mitigation.

This policy prevents both premature deletion and premature stabilization.
