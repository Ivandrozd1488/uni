find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(unified_ml_py bindings.cpp)
target_link_libraries(unified_ml_py PRIVATE unified_ml::unified_ml)
target_compile_features(unified_ml_py PRIVATE cxx_std_17)
set_target_properties(unified_ml_py PROPERTIES OUTPUT_NAME unified_ml)
