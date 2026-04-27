cmake_minimum_required(VERSION 3.21)
project(unified_ml_ucao_showcase LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(unified_ml REQUIRED)

add_executable(ucao_showcase main.cpp)
target_link_libraries(ucao_showcase PRIVATE unified_ml::unified_ml)
