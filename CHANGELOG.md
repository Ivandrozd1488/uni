# Changelog

All notable changes to `unified_ml` are documented in this file.

The format follows Keep a Changelog and the project uses Semantic Versioning.

## [Unreleased]

### Changed

- Refined the production-facing SDK surface around `unified_ml_stable.hpp`, `unified_ml.hpp`, and the extensionless `unified_ml` entry point.
- Expanded header-level documentation for stable public API areas, including metrics, logging, and umbrella headers.
- Tightened warning hygiene for production builds without changing mathematical behavior.
- Standardized example projects so installed-package, `find_package`, and `FetchContent` usage all reflect the same integration model.
- Clarified that internal SIMD, OpenMP, CUDA, and specialized subsystems such as UCAO remain implementation details unless explicitly promoted to the public SDK surface.
- Consolidated repository documentation so README, architecture notes, stability policy, contributor guidance, and release guidance describe one consistent system.

### Added

- Installed example set for `basic_usage`, `fetchcontent`, and `install_test`.
- Stable namespace aliases in the production umbrella for a cleaner integration surface.
- Structured reporting helpers for selected model families instead of library-owned console output.

### Fixed

- Removed remaining warning-producing dead locals and dead helpers discovered during strict production rebuilds.
- Fixed package and example integration drift uncovered during install and `FetchContent` verification.
- Resolved warning noise around internal UCAO translation units in strict builds.

## [1.0.0] - 2026-03-26

Initial production release.

### Added

- Stable CMake package export with `find_package(unified_ml REQUIRED)`.
- Imported target `unified_ml::unified_ml`.
- Convenience entry point `#include <unified_ml>`.
- Core ML model families, autograd infrastructure, benchmarks, tests, install scripts, and packaging support.
- Optional internal acceleration through AVX2, AVX-512, OpenMP, and CUDA-backed XGBoost paths where configured.

### Notes

- Public compatibility is defined by the documented production-facing headers and package exports.
- Internal implementation strategies are intentionally free to evolve within the repository.
