cmake_minimum_required(VERSION 3.21)

# ══════════════════════════════════════════════════════════════════════════════
#  unified_ml — C++ Machine Learning Library
#  ABI note: exported target names are stable, binary compatibility is
#  best-effort within a major series and must not be treated as a hard guarantee.
# ══════════════════════════════════════════════════════════════════════════════
project(unified_ml
  VERSION   1.0.0
  DESCRIPTION "C++ Machine Learning Library"
  HOMEPAGE_URL  "https://github.com/your-org/unified_ml"
  LANGUAGES   CXX
)

set(CMAKE_CXX_STANDARD    20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS    OFF)

if(MSVC)
  add_definitions(-D_USE_MATH_DEFINES)
endif()

# ABI / SONAME policy
# SOVERSION tracks the major version for conventional dynamic-linker lookup.
# This is a packaging aid, not a blanket promise that every 1.x build is ABI-identical.
set(UNIFIED_ML_SOVERSION ${PROJECT_VERSION_MAJOR})

# Options             
option(UNIFIED_ML_BUILD_SHARED  "Build shared library (.so/.dylib/.dll)" ON)
option(UNIFIED_ML_BUILD_STATIC  "Build static library (.a/.lib)"   ON)
option(UNIFIED_ML_BUILD_TESTS   "Build test executables"       ON)
option(UNIFIED_ML_BUILD_BENCHMARKS  "Build benchmark executables"    ON)
option(UNIFIED_ML_ENABLE_OPENMP   "Enable OpenMP parallelism"      ON)
option(UNIFIED_ML_ENABLE_AVX    "Enable internal AVX2+FMA acceleration for library kernels"  ON)
option(UNIFIED_ML_ENABLE_AVX512   "Enable internal AVX-512 acceleration for selected kernels (explicit opt-in only)"     OFF)
# IMPORTANT: UNIFIED_ML_ENABLE_AVX512 defaults to OFF.
# check_cxx_compiler_flag("-mavx512f") only tests that the *compiler* accepts the flag.
# It does NOT probe runtime CPU capabilities.  Enabling AVX-512 on a non-AVX512 machine
# causes SIGILL at runtime.  Only enable this if you know the target CPU supports it.
option(UNIFIED_ML_NATIVE_MARCH  "Use -march=native for local development builds only"     OFF)
option(UNIFIED_ML_ENABLE_ASAN    "Enable AddressSanitizer instrumentation" OFF)
option(UNIFIED_ML_ENABLE_UBSAN   "Enable UndefinedBehaviorSanitizer instrumentation" OFF)
option(UNIFIED_ML_ENABLE_MSAN    "Enable MemorySanitizer instrumentation (Clang-only, Linux/x86_64)" OFF)
option(UNIFIED_ML_ENABLE_WERROR  "Treat compiler warnings as errors" OFF)
option(UNIFIED_ML_ENABLE_COVERAGE "Enable coverage instrumentation (-fprofile-arcs -ftest-coverage)" OFF)
option(UNIFIED_ML_ENABLE_CUDA    "Enable CUDA backend for XGBoost histogram builder" OFF)
option(UNIFIED_ML_BUILD_PYTHON   "Build Python bindings via pybind11" OFF)
option(UNIFIED_ML_INSTALL     "Generate install targets"     ON)
option(UNIFIED_ML_VERSIONED_INSTALL "Install to prefix/unified_ml/VERSION/"  OFF)

if(NOT UNIFIED_ML_BUILD_SHARED AND NOT UNIFIED_ML_BUILD_STATIC)
  message(FATAL_ERROR "At least one of UNIFIED_ML_BUILD_SHARED or UNIFIED_ML_BUILD_STATIC must be ON")
endif()

# Build type           
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    Debug Release RelWithDebInfo MinSizeRel)
endif()

# Global PIC (shared lib needs it; doesn't hurt static)    
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# RPATH configuration           
# During build: use actual build-dir rpath so tests link correctly
# After install: use $ORIGIN-relative rpath so the package is relocatable
# without needing to set LD_LIBRARY_PATH
if(NOT WIN32)
  # Build-tree RPATH
  set(CMAKE_BUILD_RPATH "${CMAKE_BINARY_DIR}/lib")

  # Install-tree RPATH — $ORIGIN = directory of the consuming executable
  # Covers the common layout: bin/ and lib/ are siblings under the same prefix
  if(APPLE)
    # macOS uses @loader_path
    set(CMAKE_INSTALL_RPATH "@loader_path/../lib")
    set(CMAKE_MACOSX_RPATH ON)
  else()
    # Linux / ELF
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
  endif()

  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF) # don't pull in system paths
  set(CMAKE_BUILD_WITH_INSTALL_RPATH  OFF) # build-tree and install-tree are separate
  set(CMAKE_SKIP_BUILD_RPATH    OFF)
endif()

# MSVC: strip /RTC1 from all build configs at directory scope   
# /RTC1 (runtime buffer/stack checks) is injected by CMake into Debug flags.
# It is mutually exclusive with ANY /Ox optimisation flag (/O1, /O2, /Ob, /Os…).
# CMake error: "D8016: '/O2' and '/RTC1' command-line options are incompatible"
# We strip it here once at directory scope, BEFORE add_library/add_executable.
# The string(REPLACE) + CACHE FORCE pattern is the only reliable cross-scope fix.
if(MSVC)
  foreach(_cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    string(REPLACE "/RTC1" "" _flags "${CMAKE_CXX_FLAGS_${_cfg}}")
    set(CMAKE_CXX_FLAGS_${_cfg} "${_flags}" CACHE STRING "" FORCE)
  endforeach()
  unset(_flags)
  unset(_cfg)
endif()

# Sanitizer/coverage flags must be applied consistently to all final link targets
# (libraries and executables). If only the library is instrumented, test binaries
# can fail to link with missing runtime symbols (e.g. MSan).
if(NOT MSVC)
  if(UNIFIED_ML_ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address -fno-omit-frame-pointer)
  endif()
  if(UNIFIED_ML_ENABLE_UBSAN)
    add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=undefined -fno-omit-frame-pointer)
  endif()
  if(UNIFIED_ML_ENABLE_MSAN)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      add_compile_options(-fsanitize=memory -fsanitize-memory-track-origins=2 -fno-omit-frame-pointer)
      add_link_options(-fsanitize=memory -fsanitize-memory-track-origins=2 -fno-omit-frame-pointer)
    else()
      message(FATAL_ERROR "UNIFIED_ML_ENABLE_MSAN requires Clang")
    endif()
  endif()
  if(UNIFIED_ML_ENABLE_COVERAGE)
    add_compile_options(-O0 -g --coverage)
    add_link_options(--coverage)
  endif()
endif()

# Sources             
set(AUTOGRAD_SRCS
  src/autograd/tensor.cpp src/autograd/functional.cpp src/autograd/ops.cpp)
set(CORE_SRCS
  src/core/linalg.cpp src/core/random.cpp
  src/core/activations.cpp src/core/optimizers.cpp
  src/core/quantization.cpp src/core/symbolic.cpp src/core/jit_cache.cpp)
set(MLP_SRCS
  src/models/mlp/linear.cpp src/models/mlp/activation.cpp
  src/models/mlp/loss.cpp src/models/mlp/sequential.cpp src/models/mlp/model.cpp)
set(DEEP_ONET_SRCS
  src/models/deep_onet/activations.cpp src/models/deep_onet/layers.cpp
  src/models/deep_onet/loss.cpp src/models/deep_onet/branch_net.cpp
  src/models/deep_onet/trunk_net.cpp src/models/deep_onet/model.cpp
  src/models/deep_onet/utils.cpp)
set(PINN_SRCS
  src/models/pinn/activations.cpp src/models/pinn/loss.cpp
  src/models/pinn/neural_network.cpp src/models/pinn/optimizers.cpp
  src/models/pinn/pde.cpp src/models/pinn/pinn_model.cpp
  src/models/pinn/pinn_fast.cpp src/models/pinn/pinn_solver.cpp)
set(TRANSFORMER_SRCS
  src/models/transformer/transformer_block.cpp)
set(SVM_SRCS
  src/models/svm/svm.cpp)
set(GP_SRCS
  src/models/gp/gaussian_process.cpp)
set(SINDY_SRCS
  src/models/sindy/sindy.cpp)
set(PIDEEPONET_SRCS
  src/models/pideeponet/pideeponet.cpp)
set(SLE_MODEL_SRCS
  src/models/sle/distillation.cpp)
set(RF_SRCS
  src/models/rf/dataset.cpp src/models/rf/split.cpp src/models/rf/tree.cpp
  src/models/rf/cart.cpp src/models/rf/metrics.cpp
  src/models/rf/random_forest.cpp src/models/rf/utils.cpp)
set(XGB_SRCS
  src/models/xgboost/booster/feature_importance.cpp
  src/models/xgboost/booster/gradient_booster.cpp
  src/models/xgboost/booster/trainer.cpp
  src/models/xgboost/booster/trainer_v2.cpp
  src/models/xgboost/callback/callback.cpp
  src/models/xgboost/core/config.cpp
  src/models/xgboost/core/types.cpp
  src/models/xgboost/data/column_block.cpp
  src/models/xgboost/data/csc_matrix.cpp
  src/models/xgboost/data/data_loader.cpp
  src/models/xgboost/data/dmatrix.cpp
  src/models/xgboost/data/external_memory_dmatrix.cpp
  src/models/xgboost/explain/tree_shap.cpp
  src/models/xgboost/metric/auc.cpp
  src/models/xgboost/metric/metric.cpp
  src/models/xgboost/objective/lambda_mart.cpp
  src/models/xgboost/objective/logistic_loss.cpp
  src/models/xgboost/objective/loss_function.cpp
  src/models/xgboost/objective/multi_softmax.cpp
  src/models/xgboost/objective/squared_error.cpp
  src/models/xgboost/objective/tweedie_regression.cpp
  src/models/xgboost/predictor/predictor.cpp
  src/models/xgboost/tree/decision_tree.cpp
  src/models/xgboost/tree/histogram_builder.cpp
  src/models/xgboost/tree/leaf_wise_builder.cpp
  src/models/xgboost/tree/split_evaluator.cpp
  src/models/xgboost/tree/tree_node.cpp
  src/models/xgboost/utils/binary_serializer.cpp
  src/models/xgboost/utils/csv_reader.cpp
  src/models/xgboost/utils/logger.cpp
  src/models/xgboost/xgboost_enhanced.cpp)
set(XGB_CUDA_SRCS
  src/models/xgboost/cuda/gpu_hist_builder.cpp
  src/models/xgboost/cuda/gpu_hist_kernel.cu)
set(DBSCAN_SRCS src/models/dbscan/dbscan.cpp src/models/dbscan/utils.cpp)
set(IFOREST_SRCS
  src/models/iforest/isolation_forest.cpp src/models/iforest/itree.cpp
  src/models/iforest/math_utils.cpp src/models/iforest/node.cpp
  src/models/iforest/random_utils.cpp)
set(PCA_SRCS src/models/pca/pca.cpp)
set(ABI_SRCS src/core/abi.cpp)

set(UCAO_SRCS
  src/ucao/ucao.cpp)

set(SLE_ENGINE_SRCS
  sle_engine/src/engine.cpp
  sle_engine/src/mixer.cpp
  sle_engine/src/fast_ops.cpp
  sle_engine/src/synthesis.cpp
  sle_engine/src/encoding.cpp
  sle_engine/src/full_engine.cpp
  sle_engine/src/topology.cpp
  sle_engine/src/framework.cpp
  sle_engine/src/topology_search.cpp
  sle_engine/src/perception.cpp
  sle_engine/src/topology_cache.cpp
  sle_engine/src/jit.cpp
  sle_engine/src/jit_batch.cpp
  sle_engine/src/jit_trainer.cpp
  sle_engine/src/validation.cpp)

set(UCAO_HEADER_ONLY_PEDANTIC_SOURCES
  src/ucao/ucao.cpp)

set(ALL_SRCS
  ${AUTOGRAD_SRCS} ${CORE_SRCS} ${MLP_SRCS} ${DEEP_ONET_SRCS}
  ${PINN_SRCS} ${TRANSFORMER_SRCS} ${SVM_SRCS} ${GP_SRCS} ${SINDY_SRCS} ${PIDEEPONET_SRCS} ${SLE_MODEL_SRCS} ${RF_SRCS} ${XGB_SRCS} ${DBSCAN_SRCS} ${IFOREST_SRCS}
  ${PCA_SRCS} ${ABI_SRCS} ${UCAO_SRCS} ${SLE_ENGINE_SRCS})

if(UNIFIED_ML_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND ALL_SRCS ${XGB_CUDA_SRCS})
endif()

# ══════════════════════════════════════════════════════════════════════════════
#  Helper: apply common compile options and link deps to a unified_ml target
# ══════════════════════════════════════════════════════════════════════════════
function(_uml_configure_target target)
  # Public include: BUILD_INTERFACE vs INSTALL_INTERFACE keeps paths relocatable.
  #
  # При сборке из дерева исходников (BUILD_INTERFACE):
  #   include/                  — основные заголовки (unified_ml_stable.hpp, models/, …)
  #   sle_engine/include/       — движок SLE (sle/sle.hpp, sle/jit.hpp, …)
  #
  # После cmake --install оба дерева сливаются в один <prefix>/include/:
  #   <prefix>/include/                  ← из install(DIRECTORY include/)
  #   <prefix>/include/sle/…             ← из install(DIRECTORY sle_engine/include/)
  #
  # Поэтому один INSTALL_INTERFACE:include покрывает оба дерева —
  # компилятору достаточно одного флага -I<prefix>/include.
  target_include_directories(${target}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/sle_engine/include>
    $<INSTALL_INTERFACE:include>
  )
  target_compile_features(${target} PUBLIC cxx_std_20)
  target_compile_definitions(${target} PUBLIC
    UNIFIED_ML_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    UNIFIED_ML_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    UNIFIED_ML_VERSION_PATCH=${PROJECT_VERSION_PATCH}
  )

  # PRIVATE build flags (never exported to consumers)    
  if(MSVC)
    target_compile_options(${target} PRIVATE
    /W4 /permissive- /fp:fast /wd4068
    # /O2 only in Release configs — in Debug MSVC adds /RTC1 which is
    # mutually exclusive with /O2. Use generator expressions to gate it.
    $<$<CONFIG:Release,RelWithDebInfo,MinSizeRel>:/O2 /DNDEBUG>
    $<$<CONFIG:Debug>:/Od>
    )
    target_compile_definitions(${target} PRIVATE "__restrict__=__restrict")
    # NOTE: /RTC1 stripping is done at directory scope before target creation
    # (see the MSVC_RTC1_STRIPPED block near the top of CMakeLists.txt)
  else()
    set(_uml_warning_flags -Wall -Wextra -Wpedantic -Wno-deprecated-declarations)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      # #pragma omp simd may emit -Wpass-failed=transform-warning when the
      # optimizer cannot vectorize a loop; this is informational, not a real bug.
      # Suppress it so -Werror builds don't fail on clang.
      list(APPEND _uml_warning_flags -Wno-pass-failed)
    endif()
    if(UNIFIED_ML_ENABLE_WERROR)
      list(APPEND _uml_warning_flags -Werror)
    endif()

    target_compile_options(${target} PRIVATE
      -O3 -funroll-loops -DNDEBUG
      ${_uml_warning_flags}
      $<$<BOOL:${UNIFIED_ML_NATIVE_MARCH}>:-march=native>
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      set_source_files_properties(${UCAO_HEADER_ONLY_PEDANTIC_SOURCES}
        PROPERTIES COMPILE_OPTIONS "-Wno-pedantic")
    endif()

    if(UNIFIED_ML_ENABLE_ASAN)
      target_compile_options(${target} PRIVATE -fsanitize=address -fno-omit-frame-pointer)
      target_link_options(${target} PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    endif()
    if(UNIFIED_ML_ENABLE_UBSAN)
      target_compile_options(${target} PRIVATE -fsanitize=undefined -fno-omit-frame-pointer)
      target_link_options(${target} PRIVATE -fsanitize=undefined -fno-omit-frame-pointer)
    endif()
    if(UNIFIED_ML_ENABLE_MSAN)
      if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${target} PRIVATE -fsanitize=memory -fsanitize-memory-track-origins=2 -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=memory -fsanitize-memory-track-origins=2 -fno-omit-frame-pointer)
      else()
        message(FATAL_ERROR "UNIFIED_ML_ENABLE_MSAN requires Clang")
      endif()
    endif()
    if(UNIFIED_ML_ENABLE_COVERAGE)
      target_compile_options(${target} PRIVATE -O0 -g --coverage)
      target_link_options(${target} PRIVATE --coverage)
    endif()
  endif()

  # SIMD flags (PRIVATE, x86/x64 only)      
  # Two-level design:
  # UNIFIED_ML_ENABLE_AVX  → -mavx2 -mfma  (safe, Haswell 2013+, default ON)
  # UNIFIED_ML_ENABLE_AVX512 → -mavx512f etc (opt-in, default OFF)
  #
  # NEVER use check_cxx_compiler_flag to gate AVX-512:
  # check_cxx_compiler_flag("-mavx512f") compiles-but-does-not-run a test.
  # It returns TRUE even on CPUs without AVX-512, leading to SIGILL at runtime.
  # GitHub Actions ubuntu-22.04 runners do NOT have AVX-512.
  #
  # ARM64 guard: -mavx512f/-mavx2 are x86-only; on ARM they produce errors or
  # are silently ignored but then fail when <immintrin.h> is included.
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x86|i[3-6]86")
    include(CheckCXXCompilerFlag)

    if(UNIFIED_ML_ENABLE_AVX512)
    if(MSVC)
      check_cxx_compiler_flag("/arch:AVX512" _has_avx512)
      if(_has_avx512)
        target_compile_options(${target} PRIVATE /arch:AVX512)
        message(VERBOSE "unified_ml: AVX-512 enabled (MSVC)")
      else()
        message(WARNING "UNIFIED_ML_ENABLE_AVX512=ON but compiler lacks /arch:AVX512")
      endif()
    else()
      check_cxx_compiler_flag("-mavx512f" _has_avx512)
      if(_has_avx512)
        target_compile_options(${target} PRIVATE
        -mavx512f -mavx512bw -mavx512dq)
        message(VERBOSE "unified_ml: AVX-512 enabled (GCC/Clang)")
      else()
        message(WARNING "UNIFIED_ML_ENABLE_AVX512=ON but compiler lacks -mavx512f")
      endif()
    endif()
    elseif(UNIFIED_ML_ENABLE_AVX)
    # AVX2 + FMA — universally supported on x86 since Haswell (2013)
    if(MSVC)
      check_cxx_compiler_flag("/arch:AVX2" _has_avx2)
      if(_has_avx2)
        target_compile_options(${target} PRIVATE /arch:AVX2)
      endif()
    else()
      check_cxx_compiler_flag("-mavx2" _has_avx2)
      if(_has_avx2)
        target_compile_options(${target} PRIVATE -mavx2 -mfma)
      endif()
    endif()
    endif()
  else()
    # Non-x86 (ARM64, RISC-V, etc.): auto-vectoriser uses native SIMD (NEON, SVE…)
    message(VERBOSE "unified_ml: x86 SIMD disabled on ${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  # OpenMP (PUBLIC because some installed builds may require the runtime,
  # but parallelism remains an internal implementation detail rather than a
  # source-level API contract for consumers.)   
  # MSVC OpenMP note:
  # /openmp   → OpenMP 2.0 (default).  Does NOT support #pragma omp simd.
  # /openmp:experimental → OpenMP SIMD + parallel (needed for our code).
  # We must add /openmp:experimental explicitly; OpenMP::OpenMP_CXX only adds /openmp.
  if(UNIFIED_ML_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
    if(MSVC)
      # /openmp:experimental is a superset: enables both parallel and simd
      target_compile_options(${target} PRIVATE /openmp:experimental)
      # Still need the OpenMP include dirs (for omp.h)
      target_include_directories(${target}
        PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
    else()
      target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
    endif()
    else()
    if(NOT MSVC)
      # Silence unknown-pragma warnings when OpenMP absent
      target_compile_options(${target} PRIVATE -Wno-unknown-pragmas)
    else()
      # MSVC: #pragma omp simd without /openmp:experimental → C7660 error.
      # Silence it with a warning-disable when OpenMP is unavailable.
      target_compile_options(${target} PRIVATE /wd4068)
    endif()
    endif()
  endif()

  if(UNIFIED_ML_ENABLE_CUDA)
    target_compile_definitions(${target} PUBLIC XGB_USE_CUDA=1)
    target_link_libraries(${target} PUBLIC CUDA::cudart)
    set_target_properties(${target} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)
  endif()
endfunction()

# ══════════════════════════════════════════════════════════════════════════════
#  Shared library — libunified_ml.so / .dylib / .dll
# ══════════════════════════════════════════════════════════════════════════════
if(UNIFIED_ML_BUILD_SHARED)
  add_library(unified_ml SHARED ${ALL_SRCS})
  set_target_properties(unified_ml PROPERTIES
    OUTPUT_NAME   "unified_ml"
    # SO versioning: libunified_ml.so.1 (soname) + libunified_ml.so.1.0.0 (file)
    # On macOS: libunified_ml.1.dylib + libunified_ml.1.0.0.dylib
    SOVERSION   ${UNIFIED_ML_SOVERSION}
    VERSION   ${PROJECT_VERSION}
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    # MSVC: distinct PDB name to avoid LNK1201 race with static target
    COMPILE_PDB_NAME "unified_ml_shared"
    PDB_NAME   "unified_ml_shared"
  )
  if(WIN32)
    target_compile_definitions(unified_ml
    PRIVATE UNIFIED_ML_EXPORTS
    INTERFACE UNIFIED_ML_DLL)
  endif()
  _uml_configure_target(unified_ml)
  # Alias for consumers who want to explicitly request shared linkage
  add_library(unified_ml::shared ALIAS unified_ml)
  # Primary alias — unified_ml::unified_ml always resolves to shared when available.
  # This alias must be created here (shared path) because CMake does not allow
  # two ALIAS targets with the same name; the static-only block below creates it
  # only when shared is OFF.
  add_library(unified_ml::unified_ml ALIAS unified_ml)
  message(VERBOSE "unified_ml: shared library enabled (SOVERSION=${UNIFIED_ML_SOVERSION})")
endif()

if(UNIFIED_ML_BUILD_PYTHON)
  add_subdirectory(python)
endif()

# ══════════════════════════════════════════════════════════════════════════════
#  Static library — libunified_ml.a / unified_ml.lib
# ══════════════════════════════════════════════════════════════════════════════
if(UNIFIED_ML_BUILD_STATIC)
  add_library(unified_ml_static STATIC ${ALL_SRCS})
  set_target_properties(unified_ml_static PROPERTIES
    OUTPUT_NAME "unified_ml"
    # Distinct PDB names: avoids LNK1201 race when shared + static build in parallel
    COMPILE_PDB_NAME "unified_ml_static"
    PDB_NAME   "unified_ml_static"
  )
  # On Windows MSVC: static lib gets suffix to avoid collision with DLL import lib
  # (both would produce unified_ml.lib otherwise).
  # NOTE: generator expressions cannot be used as property *names* in
  # set_target_properties, so a plain if() is the correct approach here.
  if(WIN32)
    set_target_properties(unified_ml_static PROPERTIES OUTPUT_NAME "unified_ml_static")
  endif()
  _uml_configure_target(unified_ml_static)
  add_library(unified_ml::static ALIAS unified_ml_static)
  message(VERBOSE "unified_ml: static library enabled")
endif()

# Default alias in the BUILD tree        
# When shared is built, the actual target IS "unified_ml" → exports as unified_ml::unified_ml
# When static-only, create an alias so build-tree usage is consistent.
# Note: ALIAS targets are NOT exported — the actual target name "unified_ml" is what
# appears in unified_mlTargets.cmake under namespace unified_ml::, giving unified_ml::unified_ml.
if(NOT UNIFIED_ML_BUILD_SHARED AND UNIFIED_ML_BUILD_STATIC)
  # Static-only build: set EXPORT_NAME so unified_ml_static exports as
  # "unified_ml::unified_ml" (same name consumers expect in both modes).
  set_target_properties(unified_ml_static PROPERTIES
    EXPORT_NAME "unified_ml")
  add_library(unified_ml::unified_ml ALIAS unified_ml_static)
endif()

# ══════════════════════════════════════════════════════════════════════════════
#  Tests & Benchmarks
# ══════════════════════════════════════════════════════════════════════════════
if(UNIFIED_ML_BUILD_TESTS)
  enable_testing()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_autograd.cpp")
    add_executable(test_autograd test_autograd.cpp)
    target_link_libraries(test_autograd PRIVATE unified_ml::unified_ml)
    add_test(NAME test_autograd COMMAND test_autograd)
    # Ensure DLL is found on Windows test run
    if(WIN32 AND UNIFIED_ML_BUILD_SHARED)
      set_tests_properties(test_autograd PROPERTIES
        ENVIRONMENT "PATH=$<TARGET_FILE_DIR:unified_ml>;$ENV{PATH}")
    endif()
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_models_smoke.cpp")
    add_executable(test_models_smoke test_models_smoke.cpp)
    target_link_libraries(test_models_smoke PRIVATE unified_ml::unified_ml)
    add_test(NAME test_models_smoke COMMAND test_models_smoke)
    if(WIN32 AND UNIFIED_ML_BUILD_SHARED)
      set_tests_properties(test_models_smoke PROPERTIES
        ENVIRONMENT "PATH=$<TARGET_FILE_DIR:unified_ml>;$ENV{PATH}")
    endif()
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_unified_facade_smoke.cpp")
    add_executable(test_unified_facade_smoke tests/test_unified_facade_smoke.cpp)
    target_link_libraries(test_unified_facade_smoke PRIVATE unified_ml::unified_ml)
    add_test(NAME test_unified_facade_smoke COMMAND test_unified_facade_smoke)
    if(WIN32 AND UNIFIED_ML_BUILD_SHARED)
      set_tests_properties(test_unified_facade_smoke PROPERTIES
        ENVIRONMENT "PATH=$<TARGET_FILE_DIR:unified_ml>;$ENV{PATH}")
    endif()
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_sle_distillation_diagnostics.cpp")
    add_executable(test_sle_distillation_diagnostics tests/test_sle_distillation_diagnostics.cpp)
    target_link_libraries(test_sle_distillation_diagnostics PRIVATE unified_ml::unified_ml)
    add_test(NAME test_sle_distillation_diagnostics COMMAND test_sle_distillation_diagnostics)
    set_tests_properties(test_sle_distillation_diagnostics PROPERTIES TIMEOUT 20)
    if(WIN32 AND UNIFIED_ML_BUILD_SHARED)
      set_tests_properties(test_sle_distillation_diagnostics PROPERTIES
        ENVIRONMENT "PATH=$<TARGET_FILE_DIR:unified_ml>;$ENV{PATH}")
    endif()
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_sle_distillation_edge_cases.cpp")
    add_executable(test_sle_distillation_edge_cases tests/test_sle_distillation_edge_cases.cpp)
    target_link_libraries(test_sle_distillation_edge_cases PRIVATE unified_ml::unified_ml)
    add_test(NAME test_sle_distillation_edge_cases COMMAND test_sle_distillation_edge_cases)
    set_tests_properties(test_sle_distillation_edge_cases PROPERTIES TIMEOUT 20)
    if(WIN32 AND UNIFIED_ML_BUILD_SHARED)
      set_tests_properties(test_sle_distillation_edge_cases PROPERTIES
        ENVIRONMENT "PATH=$<TARGET_FILE_DIR:unified_ml>;$ENV{PATH}")
    endif()
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_sle_level2_utility.cpp")
    add_executable(test_sle_level2_utility tests/test_sle_level2_utility.cpp)
    target_link_libraries(test_sle_level2_utility PRIVATE unified_ml::unified_ml)
    add_test(NAME test_sle_level2_utility
      COMMAND test_sle_level2_utility "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/sle_level2_manifest.csv")
    set_tests_properties(test_sle_level2_utility PROPERTIES TIMEOUT 120)
    if(WIN32 AND UNIFIED_ML_BUILD_SHARED)
      set_tests_properties(test_sle_level2_utility PROPERTIES
        ENVIRONMENT "PATH=$<TARGET_FILE_DIR:unified_ml>;$ENV{PATH}")
    endif()
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_sle_safe_mode_and_repro.cpp")
    add_executable(test_sle_safe_mode_and_repro tests/test_sle_safe_mode_and_repro.cpp)
    target_link_libraries(test_sle_safe_mode_and_repro PRIVATE unified_ml::unified_ml)
    add_test(NAME test_sle_safe_mode_and_repro COMMAND test_sle_safe_mode_and_repro)
    set_tests_properties(test_sle_safe_mode_and_repro PROPERTIES TIMEOUT 60)
    if(WIN32 AND UNIFIED_ML_BUILD_SHARED)
      set_tests_properties(test_sle_safe_mode_and_repro PROPERTIES
        ENVIRONMENT "PATH=$<TARGET_FILE_DIR:unified_ml>;$ENV{PATH}")
    endif()
  endif()


  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ucao_kernel.cpp")
    add_executable(test_ucao_kernel tests/test_ucao_kernel.cpp)
    target_link_libraries(test_ucao_kernel PRIVATE unified_ml::unified_ml)
    add_test(NAME test_ucao_kernel COMMAND test_ucao_kernel)
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ucao_pinn.cpp")
    add_executable(test_ucao_pinn tests/test_ucao_pinn.cpp)
    target_link_libraries(test_ucao_pinn PRIVATE unified_ml::unified_ml)
    add_test(NAME test_ucao_pinn COMMAND test_ucao_pinn)
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ucao_combat.cpp")
    add_executable(test_ucao_combat tests/test_ucao_combat.cpp)
    target_link_libraries(test_ucao_combat PRIVATE unified_ml::unified_ml)
    add_test(NAME test_ucao_combat COMMAND test_ucao_combat)
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/ucao_demo.cpp")
    add_executable(ucao_demo tests/ucao_demo.cpp)
    target_link_libraries(ucao_demo PRIVATE unified_ml::unified_ml)
    add_test(NAME ucao_demo COMMAND ucao_demo)
  endif()
endif()

if(UNIFIED_ML_BUILD_BENCHMARKS)
  foreach(_bench benchmark benchmark_hpc)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_bench}.cpp")
    add_executable(${_bench} ${_bench}.cpp)
    target_link_libraries(${_bench} PRIVATE unified_ml::unified_ml)
    endif()
  endforeach()
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/ucao_showcase/main.cpp")
  add_executable(ucao_showcase examples/ucao_showcase/main.cpp)
  target_link_libraries(ucao_showcase PRIVATE unified_ml::unified_ml)
endif()

# ══════════════════════════════════════════════════════════════════════════════
#  Install — versioned layout + symlinks
# ══════════════════════════════════════════════════════════════════════════════
if(UNIFIED_ML_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  # Versioned install path support       
  # When UNIFIED_ML_VERSIONED_INSTALL=ON the effective tree becomes:
  # <prefix>/unified_ml/<version>/{include,lib,share}
  # with a symlink:
  # <prefix>/unified_ml/current -> <version>
  #
  # This allows parallel versions and clean rollback:
  # /usr/local/unified_ml/1.0.0/
  # /usr/local/unified_ml/2.0.0/
  # /usr/local/unified_ml/current -> 1.0.0
  if(UNIFIED_ML_VERSIONED_INSTALL)
    set(_ver_subdir "unified_ml/${PROJECT_VERSION}")
    set(CMAKE_INSTALL_INCLUDEDIR "${_ver_subdir}/include")
    set(CMAKE_INSTALL_LIBDIR   "${_ver_subdir}/lib")
    set(CMAKE_INSTALL_BINDIR   "${_ver_subdir}/bin")
    set(CMAKE_INSTALL_DATAROOTDIR "${_ver_subdir}/share")
  endif()

  set(_cmake_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/unified_ml")

  # Shared library           
  if(UNIFIED_ML_BUILD_SHARED)
    # Install "unified_ml" target — this exports as "unified_ml::unified_ml"
    # because install(EXPORT ... NAMESPACE unified_ml::) is used below.
    install(TARGETS unified_ml
    EXPORT  unified_mlTargets
    ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR} # import lib (.lib on Windows)
    LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR} # .so / .dylib
    RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR} # .dll on Windows
    INCLUDES  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    # On Linux: install the SONAME symlink too (ldconfig needs it)
    # CMake handles .so.1.0.0 and .so.1 automatically via SOVERSION
  endif()

  # Static library           
  if(UNIFIED_ML_BUILD_STATIC)
    install(TARGETS unified_ml_static
    EXPORT unified_mlTargets
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
  endif()

  # Headers
  # ── Основное дерево include/ ───────────────────────────────────────────────
  # Устанавливает всё из include/ в <prefix>/include/
  # Результат: <prefix>/include/unified_ml_stable.hpp, models/, autograd/, …
  install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.h"
  )

  # Extensionless entry-point "unified_ml" (#include <unified_ml>).
  # FILES_MATCHING пропускает файлы без расширения — устанавливаем явно.
  install(FILES include/unified_ml
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  # ── SLE engine: sle_engine/include/ → <prefix>/include/ ───────────────────
  # Содержимое sle_engine/include/ (единственная подпапка: sle/) сливается
  # прямо в <prefix>/include/, не в подкаталог.
  #
  # Итоговая структура на диске:
  #   <prefix>/include/
  #     ├── unified_ml_stable.hpp
  #     ├── models/sle/full_engine.hpp  distillation.hpp  …
  #     └── sle/
  #           ├── sle.hpp               ← #include <sle/sle.hpp>
  #           ├── jit.hpp               ← #include <sle/jit.hpp>
  #           ├── synthesis.hpp
  #           └── … (все остальные заголовки sle_engine)
  #
  # Это согласовано с INSTALL_INTERFACE:include в _uml_configure_target:
  # один флаг -I<prefix>/include даёт компилятору доступ ко всем модулям.
  install(DIRECTORY sle_engine/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.h"
  )

  # CMake package files         
  install(EXPORT unified_mlTargets
    FILE    unified_mlTargets.cmake
    NAMESPACE unified_ml::
    DESTINATION ${_cmake_install_dir}
  )

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/unified_mlConfigVersion.cmake"
    VERSION   ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )

  # Record build-time flags needed by the config file
  if(UNIFIED_ML_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
  endif()

  # OpenMP pkgconfig link flags
  if(OpenMP_CXX_FOUND)
    set(OPENMP_PKGCONFIG_LIBS "${OpenMP_CXX_FLAGS}")
  else()
    set(OPENMP_PKGCONFIG_LIBS "")
  endif()

  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/unified_mlConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/unified_mlConfig.cmake"
    INSTALL_DESTINATION ${_cmake_install_dir}
  )

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/unified_mlConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/unified_mlConfigVersion.cmake"
    DESTINATION ${_cmake_install_dir}
  )

  # pkg-config           
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/unified_ml.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/unified_ml.pc"
    @ONLY
  )
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/unified_ml.pc"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
  )

  # Examples & docs
  install(DIRECTORY examples/basic_usage
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/unified_ml/examples"
    OPTIONAL
  )
  install(DIRECTORY examples/fetchcontent
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/unified_ml/examples"
    OPTIONAL
  )
  install(DIRECTORY examples/install_test
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/unified_ml/examples"
    OPTIONAL
  )
  install(DIRECTORY examples/ucao_showcase
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/unified_ml/examples"
    OPTIONAL
  )
  install(FILES LICENSE README.md
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/unified_ml"
    OPTIONAL
  )

  # Versioned symlink: current -> VERSION      
  # CMake doesn't have a first-class symlink install, so we use a
  # post-install script executed via cmake -P
  if(UNIFIED_ML_VERSIONED_INSTALL AND NOT WIN32)
    configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/create_symlink.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/create_symlink.cmake"
    @ONLY
    )
    # Hook into install: runs after all files are placed
    install(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/create_symlink.cmake")
  endif()

  # ldconfig hint (Linux only)         
  # Tells ldconfig where to find the new shared library.
  # Only needed when installing to a non-standard prefix.
  if(UNIX AND NOT APPLE AND UNIFIED_ML_BUILD_SHARED)
    install(CODE "
    if(EXISTS \"/sbin/ldconfig\")
      message(STATUS \"Running ldconfig...\")
      execute_process(
        COMMAND ldconfig
        ERROR_QUIET
      )
    endif()
    ")
  endif()

  # Uninstall            
  if(NOT TARGET uninstall)
    configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake"
    @ONLY
    )
    add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake"
    COMMENT "Uninstalling unified_ml..."
    )
  endif()
endif() # UNIFIED_ML_INSTALL

# Summary             
# Helper macro: ON → ✔  OFF → ·        
macro(flag_str var out)
  if(${var})
    set(${out} "ON ")
  else()
    set(${out} "OFF")
  endif()
endmacro()

flag_str(UNIFIED_ML_BUILD_SHARED  _f_shared)
flag_str(UNIFIED_ML_BUILD_STATIC  _f_static)
flag_str(UNIFIED_ML_ENABLE_OPENMP _f_omp)
flag_str(UNIFIED_ML_ENABLE_AVX  _f_avx2)
flag_str(UNIFIED_ML_ENABLE_AVX512 _f_avx512)
flag_str(UNIFIED_ML_NATIVE_MARCH  _f_native)
flag_str(UNIFIED_ML_ENABLE_ASAN   _f_asan)
flag_str(UNIFIED_ML_ENABLE_UBSAN  _f_ubsan)
flag_str(UNIFIED_ML_ENABLE_MSAN   _f_msan)
flag_str(UNIFIED_ML_ENABLE_WERROR _f_werror)
flag_str(UNIFIED_ML_ENABLE_COVERAGE _f_cov)
flag_str(UNIFIED_ML_ENABLE_CUDA   _f_cuda)
flag_str(UNIFIED_ML_BUILD_PYTHON  _f_python)

message(STATUS "")
message(STATUS "  ┌  unified_ml ${PROJECT_VERSION}        ")
message(STATUS "  │")
message(STATUS "  │  compiler   ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  │  build type   ${CMAKE_BUILD_TYPE}")
message(STATUS "  │  C++ standard C++${CMAKE_CXX_STANDARD}")
message(STATUS "  │")
message(STATUS "  │  shared lib   ${_f_shared} (soversion ${UNIFIED_ML_SOVERSION})")
message(STATUS "  │  static lib   ${_f_static}")
message(STATUS "  │")
message(STATUS "  │  OpenMP   ${_f_omp}")
message(STATUS "  │  internal AVX2 / FMA  ${_f_avx2}")
message(STATUS "  │  internal AVX-512   ${_f_avx512}")
message(STATUS "  │  native march ${_f_native}")
message(STATUS "  │  ASan      ${_f_asan}")
message(STATUS "  │  UBSan     ${_f_ubsan}")
message(STATUS "  │  MSan      ${_f_msan}")
message(STATUS "  │  Werror    ${_f_werror}")
message(STATUS "  │  coverage    ${_f_cov}")
message(STATUS "  │  CUDA backend ${_f_cuda}")
message(STATUS "  │  Python bindings ${_f_python}")
message(STATUS "  │")
message(STATUS "  │  install prefix ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  │  SLE headers    ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/sle/")
message(STATUS "  │")
message(STATUS "  └           ")
message(STATUS "")
