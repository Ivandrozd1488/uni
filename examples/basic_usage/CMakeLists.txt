# unified_ml basic_usage example         
# Build after installing unified_ml:
# cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/prefix
# cmake --build build
# ./build/example
cmake_minimum_required(VERSION 3.21)
project(unified_ml_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(unified_ml REQUIRED)

message(STATUS "Found unified_ml ${unified_ml_VERSION}")

add_executable(example main.cpp)
target_link_libraries(example PRIVATE unified_ml::unified_ml)

# Print install info at build time
add_custom_command(TARGET example POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E echo
    "Built example with unified_ml ${unified_ml_VERSION}")
