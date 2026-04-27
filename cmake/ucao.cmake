set(UCAO_SOURCES
 ${CMAKE_CURRENT_SOURCE_DIR}/src/ucao/ucao.cpp
)

target_sources(unified_ml PRIVATE ${UCAO_SOURCES})

target_include_directories(unified_ml
 PUBLIC
 $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
 $<INSTALL_INTERFACE:include>
)

if(UNIFIED_ML_ENABLE_AVX512)
 set_source_files_properties(
 ${CMAKE_CURRENT_SOURCE_DIR}/src/ucao/ucao.cpp
 PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512vl -mavx512bw"
 )
endif()

if(UNIFIED_ML_BUILD_TESTS)
 add_executable(test_ucao_kernel tests/test_ucao_kernel.cpp)
 target_link_libraries(test_ucao_kernel PRIVATE unified_ml::unified_ml)

 add_executable(test_ucao_pinn tests/test_ucao_pinn.cpp)
 target_link_libraries(test_ucao_pinn PRIVATE unified_ml::unified_ml)

 add_executable(test_ucao_combat tests/test_ucao_combat.cpp)
 target_link_libraries(test_ucao_combat PRIVATE unified_ml::unified_ml)
endif()
