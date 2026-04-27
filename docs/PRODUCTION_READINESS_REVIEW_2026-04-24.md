# Production Readiness Review — unified_ml (2026-04-24)

## 1) Executive summary

`unified_ml` already demonstrates **strong engineering baseline** for release:

- clean CMake configuration on fresh build directory,
- successful full build of shared + static targets,
- green CTest run on repository test suite.

However, for strict production-readiness (predictability, observability, failure containment, release governance), there are still several high-impact gaps that should be closed before promoting this as a hardened SDK for external users.

Current practical status:

- **Engineering maturity:** medium-high.
- **Production hardening maturity:** medium.
- **Recommendation:** proceed with staged production rollout only after the P0/P1 actions below are implemented.

---

## 2) Review scope and evidence

This review used:

1. Documentation and build/release contracts:
   - `README.md`
   - `docs/BUILD_AND_RELEASE.md`
   - `docs/STABILITY_CONTRACTS.md`
2. Build system and representative code paths:
   - `CMakeLists.txt`
   - `src/models/xgboost/callback/callback.cpp`
   - `include/ucao/kernel/mv_tensor.hpp`
3. Validation run from clean build directory:
   - `cmake -S . -B build_review -DCMAKE_BUILD_TYPE=Release -DUNIFIED_ML_BUILD_TESTS=ON -DUNIFIED_ML_BUILD_BENCHMARKS=OFF`
   - `cmake --build build_review --parallel 4`
   - `ctest --test-dir build_review --output-on-failure`

All 11 tests passed in the run above.

---

## 3) What is already strong

1. **Clear release handbook and clean-build discipline.**
   The project has explicit release checks and smoke verification guidance. This is a strong prerequisite for stable deliveries.

2. **Mature packaging intent.**
   Shared+static builds, install exports, RPATH handling, and CMake/pkg-config integration are documented and implemented.

3. **Healthy baseline test suite.**
   Repository tests pass cleanly in a fresh `Release` build and cover multiple subsystems (autograd, unified facade, SLE, UCAO demos).

---

## 4) Key risks before production (prioritized)

## P0 — must fix before broad production

### P0.1 Uncontrolled console output in runtime library paths

The stability contract says public APIs should avoid unsolicited console output and prefer structured logging/callback-driven diagnostics. In practice, several runtime callbacks print directly to `std::cout` (e.g., early stopping and model checkpoint events).

**Why this is risky in production:**

- contaminates service logs in high-throughput environments,
- makes observability inconsistent across model families,
- complicates compliance/audit logging where log sinks must be centralized.

**Action:** introduce a unified logger interface (sink + severity + structured fields), defaulting to silent/no-op in library mode, with opt-in human-readable console formatter.

### P0.2 Debug-only bounds/contracts in performance-critical tensor access

`MVTensor::at` / `soa_row` rely on `assert` checks under `#ifndef NDEBUG` only. In release builds, invalid indices/layout mismatches will not be guarded.

**Why this is risky in production:**

- latent memory corruption risk under integration misuse,
- failures become non-local and hard to diagnose,
- can create rare crashes under edge-case data pipelines.

**Action:** keep fast path, but add production-safe guarded API variants (or defensive contract mode) returning errors/exceptions with context; preserve current unchecked mode only for explicitly marked hot paths.

## P1 — should fix in first hardening wave

### P1.1 Build graph reproducibility: `GLOB_RECURSE` for XGBoost sources

XGBoost sources are collected via `file(GLOB_RECURSE ...)`.

**Why this is risky in production CI/CD:**

- file-set changes may not invalidate CMake configuration predictably,
- accidental source inclusion/exclusion can slip through PRs,
- harms deterministic release reproducibility.

**Action:** replace with explicit source listing (or at minimum add `CONFIGURE_DEPENDS`, though explicit lists are better for release governance).

### P1.2 Hardening profile is optional, not enforced by release gate

Sanitizers, `WERROR`, and coverage are available but disabled by default.

**Why this matters:**

- release branches can pass standard build while silently accumulating UB/warnings,
- inconsistent quality bar between local development and release candidates.

**Action:** define a mandatory CI “release-hardening matrix” that blocks merges/tags unless:

- GCC/Clang warning-clean with `UNIFIED_ML_ENABLE_WERROR=ON`,
- ASan+UBSan job green,
- optional MSan job for Linux/Clang subset,
- coverage floor on critical modules.

## P2 — medium-term production maturity

### P2.1 ABI governance is documented as best-effort, but lacks automated ABI diff gate

The CMake file explicitly clarifies ABI compatibility is not a hard guarantee.

**Why this matters:**

- downstream binary consumers need deterministic upgrade rules,
- breakages can appear across patch/minor releases without early signal.

**Action:** add ABI compliance checks (e.g., abi-dumper/abi-compliance-checker or equivalent) for public exported symbols and fail release pipeline on incompatible drift unless version policy allows it.

---

## 5) Stabilization plan (production-focused)

## Phase A (1–2 weeks): “No surprises in prod”

1. Centralized logging abstraction + opt-in console sink.
2. Contract-safe access mode for UCAO tensor primitives.
3. Remove unsolicited stdout from stable model paths.
4. Add regression tests asserting silent mode behavior.

**Exit criteria:** stable APIs produce no stdout by default; invalid tensor access reports explicit contract failures in guarded mode.

## Phase B (2–4 weeks): “Deterministic release pipeline”

1. Replace `GLOB_RECURSE` source collection with explicit lists.
2. Add hardening CI matrix (Werror + sanitizers).
3. Add pre-release artifact smoke tests as mandatory status checks.

**Exit criteria:** release candidate is reproducible from clean checkout with identical build graph and passing hardening matrix.

## Phase C (4–8 weeks): “Long-term compatibility”

1. ABI diff gate in CI.
2. Versioning policy document: what can break in major/minor/patch.
3. Public deprecation policy with timeline guarantees.

**Exit criteria:** consumers can upgrade with documented compatibility expectations and machine-verified ABI guardrails.

---

## 6) Production readiness verdict

`unified_ml` is **close to production-ready for controlled deployments**, but **not yet fully hardened for broad external production usage** until P0/P1 items are addressed.

Recommended rollout mode right now:

- allow internal production pilots,
- freeze API expansion for one hardening cycle,
- promote to “production ready” only after hardening pipeline and logging/contracts work are complete.
