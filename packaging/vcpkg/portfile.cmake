# unified_ml vcpkg portfile          
# To use this port:
# 1. Copy packaging/vcpkg/ contents to <vcpkg_root>/ports/unified-ml/
# 2. Run: vcpkg install unified-ml
# 3. In CMake: find_package(unified_ml REQUIRED)
#
# For a GitHub-hosted release:
# vcpkg_from_github(
#   OUT_SOURCE_PATH SOURCE_PATH
#   REPO your-org/unified_ml
#   REF v${VERSION}
#   SHA512 <run `vcpkg hash <archive>` to get this>
#   HEAD_REF main
# )
#
# For local development (overlays):
# vcpkg install unified-ml --overlay-ports=./packaging/vcpkg

vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO    your-org/unified_ml
  REF     "v${VERSION}"
  SHA512    0  # REPLACE with actual SHA512 of the release tarball
        # Run: vcpkg hash unified_ml-VERSION-source.tar.gz
  HEAD_REF    main
)

# Feature: shared library         
if("shared" IN_LIST FEATURES)
  set(_build_shared ON)
  vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)
else()
  set(_build_shared OFF)
  vcpkg_check_linkage(ONLY_STATIC_LIBRARY)
endif()

# Feature: OpenMP           
if("openmp" IN_LIST FEATURES)
  set(_openmp ON)
else()
  set(_openmp OFF)
endif()

# CMake configure           
vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}"
  OPTIONS
    -DUNIFIED_ML_BUILD_SHARED=${_build_shared}
    -DUNIFIED_ML_BUILD_STATIC=ON
    -DUNIFIED_ML_BUILD_TESTS=OFF
    -DUNIFIED_ML_BUILD_BENCHMARKS=OFF
    -DUNIFIED_ML_ENABLE_OPENMP=${_openmp}
    -DUNIFIED_ML_ENABLE_AVX=ON
    -DUNIFIED_ML_NATIVE_MARCH=OFF
    -DUNIFIED_ML_INSTALL=ON
    -DUNIFIED_ML_VERSIONED_INSTALL=OFF
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(
  PACKAGE_NAME  unified_ml
  CONFIG_PATH "lib/cmake/unified_ml"
)
vcpkg_fixup_pkgconfig()

# Remove debug includes (vcpkg convention)      
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

# Usage instructions           
file(INSTALL "${SOURCE_PATH}/LICENSE"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
  RENAME copyright)

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/usage"
  "${CURRENT_PACKAGES_DIR}/share/${PORT}/usage"
  COPYONLY)
