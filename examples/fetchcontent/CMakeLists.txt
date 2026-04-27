# unified_ml via FetchContent
#
# Production usage:
#   cmake -S . -B build
#
# CI / local override:
#   cmake -S . -B build \
#     -DFETCHCONTENT_SOURCE_DIR_UNIFIED_ML=/path/to/unified_ml/source
cmake_minimum_required(VERSION 3.21)
project(my_ml_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(
  unified_ml
  GIT_REPOSITORY https://github.com/drozdisme/unified_ml.git
  GIT_TAG    v1.0.0
  # CI override: set -DFETCHCONTENT_SOURCE_DIR_UNIFIED_ML=<path> to skip download
)

set(UNIFIED_ML_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(UNIFIED_ML_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(unified_ml)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE unified_ml::unified_ml)
