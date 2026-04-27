cmake_minimum_required(VERSION 3.21)
project(unified_ml_install_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(unified_ml REQUIRED)

if(NOT DEFINED unified_ml_VERSION)
  message(FATAL_ERROR "unified_ml_VERSION not set — CMake export broken")
endif()
message(STATUS "[install_test] unified_ml_VERSION = ${unified_ml_VERSION}")

add_executable(install_test main.cpp)
target_link_libraries(install_test PRIVATE unified_ml::unified_ml)

enable_testing()
add_test(NAME unified_ml_smoke COMMAND install_test)
