# unified_ml Build and Release Handbook

This document is the release-prep handbook for `unified_ml`: how to build it, test it, verify installs, and avoid shipping mismatched artifacts.

## 1. Release goals

A release candidate is only considered healthy when all of the following are true:

- the project configures cleanly in a fresh build directory
- the build succeeds on supported platforms
- tests pass from a clean build tree
- install exports are correct
- smoke examples work against the installed package
- docs describe the package that is actually produced

## 2. Clean build rules

Always prefer a fresh out-of-tree build directory when validating release readiness.

Why this matters:

- stale `CTest` metadata can point at old executables
- switching generators or build layouts in-place can produce misleading failures
- install/export validation should reflect the current tree, not leftovers

Recommended rule:

- use a fresh `build/`, `build_clean/`, or CI-generated build dir for validation
- if the generator or configuration changes, re-run CMake in a clean directory

## 3. Standard local build flows

### Linux and macOS

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DUNIFIED_ML_BUILD_SHARED=ON \
  -DUNIFIED_ML_BUILD_STATIC=ON \
  -DUNIFIED_ML_BUILD_TESTS=ON \
  -DUNIFIED_ML_BUILD_BENCHMARKS=OFF \
  -DUNIFIED_ML_ENABLE_OPENMP=ON \
  -DUNIFIED_ML_ENABLE_AVX=ON \
  -DUNIFIED_ML_ENABLE_AVX512=OFF \
  -DUNIFIED_ML_NATIVE_MARCH=OFF

cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build
```

### Windows

```powershell
cmake -S . -B build `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_INSTALL_PREFIX="C:\unified_ml" `
  -DUNIFIED_ML_BUILD_SHARED=ON `
  -DUNIFIED_ML_BUILD_STATIC=ON `
  -DUNIFIED_ML_BUILD_TESTS=ON `
  -DUNIFIED_ML_BUILD_BENCHMARKS=OFF `
  -DUNIFIED_ML_ENABLE_OPENMP=ON `
  -DUNIFIED_ML_ENABLE_AVX=ON `
  -DUNIFIED_ML_ENABLE_AVX512=OFF `
  -DUNIFIED_ML_NATIVE_MARCH=OFF

cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
cmake --install build --config Release
```

## 4. Optional features

### CUDA-backed XGBoost components

```bash
cmake -S . -B build -DUNIFIED_ML_ENABLE_CUDA=ON
```

### Python bindings

```bash
cmake -S . -B build -DUNIFIED_ML_BUILD_PYTHON=ON
```

## 5. Test policy

Tests are development-time verification assets. They are not part of the installed release payload.

Default local test command:

```bash
ctest --test-dir build --output-on-failure
```

Representative repository targets include:

- `test_autograd`
- `test_models_smoke`
- `test_unified_facade_smoke`
- `test_sle_distillation_diagnostics`
- `test_ucao_kernel`
- `test_ucao_pinn`
- `test_ucao_combat`
- `ucao_demo`

## 6. Install verification checklist

After `cmake --install`, verify the package surface that consumers actually rely on.

### Required installed headers

- `<prefix>/include/unified_ml`
- `<prefix>/include/unified_ml_stable.hpp`

### Required package metadata

- `<prefix>/lib/cmake/unified_ml/unified_mlConfig.cmake`
- `<prefix>/lib/cmake/unified_ml/unified_mlTargets.cmake`
- `<prefix>/lib/pkgconfig/unified_ml.pc`

### Required libraries

Platform-specific examples:

- Linux: `libunified_ml.so` or `libunified_ml.a`
- macOS: `libunified_ml.dylib` or `libunified_ml.a`
- Windows: `unified_ml.lib` and/or `unified_ml.dll`

## 7. Smoke verification

Release prep should verify the installed package, not just the source tree build.

### find_package smoke test

```bash
cmake -S examples/install_test -B /tmp/unified_ml_install_smoke \
  -DCMAKE_PREFIX_PATH=/tmp/unified_ml_stage
cmake --build /tmp/unified_ml_install_smoke
/tmp/unified_ml_install_smoke/install_test
```

### FetchContent smoke test

```bash
cmake -S examples/fetchcontent -B /tmp/unified_ml_fetch_smoke \
  -DFETCHCONTENT_SOURCE_DIR_UNIFIED_ML=$PWD
cmake --build /tmp/unified_ml_fetch_smoke
```

### pkg-config smoke test

```bash
export PKG_CONFIG_PATH=/tmp/unified_ml_stage/lib/pkgconfig:$PKG_CONFIG_PATH
g++ examples/install_test/main.cpp $(pkg-config --cflags --libs unified_ml) -o pkg_test
```

## 8. Platform notes

### Linux

- CI uses Ninja on Ubuntu 22.04
- smoke tests should run against the installed package, not source-tree includes
- if shared libraries are used, runtime loader paths may need `LD_LIBRARY_PATH` in ad hoc smoke runs

### macOS

- CI uses AppleClang on macOS 14
- shared library smoke runs may require `DYLD_LIBRARY_PATH` when executing directly from a staged prefix
- verify `.dylib`, CMake package files, and pkg-config metadata together

### Windows

- CI uses MSVC on Windows Server 2022
- CTest should be run with `-C <config>` for multi-config builds
- smoke tests should prepend the staged `bin/` directory to `PATH` before launching executables
- verify both import/runtime library presence and CMake package metadata

## 9. Packaging and release workflow

Current GitHub workflows cover:

- CI matrix across Linux, macOS, and Windows
- install verification and find_package smoke paths
- FetchContent integration
- sanitizers
- coverage
- tagged release packaging

Release archives should only be built from a staging install tree that has already passed verification.

## 10. Hygiene checklist before tagging

Before cutting a release:

- remove or ignore transient build directories
- ensure docs match the real install surface
- ensure workflow checks match the real install surface
- verify package exports from a clean build
- confirm no temporary local artifacts are being staged

## 11. Minimal release-prep command sequence

```bash
rm -rf build_release /tmp/unified_ml_stage /tmp/unified_ml_install_smoke

cmake -S . -B build_release \
  -DCMAKE_BUILD_TYPE=Release \
  -DUNIFIED_ML_BUILD_SHARED=ON \
  -DUNIFIED_ML_BUILD_STATIC=ON \
  -DUNIFIED_ML_BUILD_TESTS=ON \
  -DUNIFIED_ML_BUILD_BENCHMARKS=OFF \
  -DUNIFIED_ML_ENABLE_OPENMP=ON \
  -DUNIFIED_ML_ENABLE_AVX=ON \
  -DUNIFIED_ML_ENABLE_AVX512=OFF \
  -DUNIFIED_ML_NATIVE_MARCH=OFF

cmake --build build_release --parallel
ctest --test-dir build_release --output-on-failure
cmake --install build_release --prefix /tmp/unified_ml_stage
cmake -S examples/install_test -B /tmp/unified_ml_install_smoke \
  -DCMAKE_PREFIX_PATH=/tmp/unified_ml_stage
cmake --build /tmp/unified_ml_install_smoke
/tmp/unified_ml_install_smoke/install_test
```
