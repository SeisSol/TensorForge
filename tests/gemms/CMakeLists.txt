cmake_minimum_required(VERSION 3.5)
project(gemm-tests)

include(../../cmake/setupOptions.cmake)

set(TARGET ${CMAKE_PROJECT_NAME})
set(TARGET_SOURCE_FILES main.cpp
        common/aux.cpp
        common/gemm.cpp
        common/test_drivers/simple_comparator.cpp
        gen_code/test.cpp)

set(TARGET_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}
        gen_code
        common
        common/test_drivers
        include
        ../../submodules/Device)

set(REFERENCE_IMPL "GEMMFORGE" CACHE STRING "size of the floating point data type")
set_property(CACHE REFERENCE_IMPL PROPERTY STRINGS "GEMMFORGE" "OPENBLAS")

add_subdirectory(../../submodules/Device build)

add_executable(${TARGET} ${TARGET_SOURCE_FILES})
target_compile_options(${TARGET} PUBLIC "-std=c++11")
target_compile_definitions(${TARGET} PUBLIC "REAL_SIZE=${REAL_SIZE}")
target_include_directories(${TARGET} PUBLIC ${TARGET_INCLUDE_DIRS})

if (${REFERENCE_IMPL} STREQUAL "OPENBLAS")
    find_package(OpenBLAS REQUIRED)
    target_include_directories(${TARGET} PRIVATE ${OpenBLAS_INCLUDE_DIRS})
    target_link_libraries(${TARGET} PRIVATE ${OpenBLAS_LIBRARIES})
endif()
target_compile_definitions(${TARGET} PUBLIC "CONCRETE_CPU_BACKEND=${REFERENCE_IMPL}")

set(TEST_SUITE "" CACHE STRING "relative path to test suite spec file")
if ("${TEST_SUITE}" STREQUAL "")
    message(FATAL_ERROR "NO TEST SUITE PROVIDED")
endif()

#add GTest
find_package(GTest REQUIRED)
target_link_libraries(${TARGET} PRIVATE ${GTEST_BOTH_LIBRARIES})
target_include_directories(${TARGET} PRIVATE ${GTEST_INCLUDE_DIR})

#add Yaml
find_package(yaml-cpp REQUIRED)
target_include_directories(${TARGET} PRIVATE ${YAML_CPP_INCLUDE_DIR})
target_link_libraries(${TARGET} PRIVATE ${YAML_CPP_LIBRARIES})

set(GPU_TARGET gpu_part)
set(GPU_TARGET_INCLUDE_DIRS include
        common
        common/test_drivers
        ../../submodules/Device)

if(${DEVICE_BACKEND} STREQUAL "cuda")
    set(GPU_TARGET_SOURCE_FILES common/test_drivers/simple_driver.cpp include/gemmforge_aux.cu gen_code/kernels.cu)
    set(GEN_COPY_PRODUCTS ${CMAKE_SOURCE_DIR}/gen_code/kernels.cu ${CMAKE_SOURCE_DIR}/gen_code/kernels.h ${CMAKE_SOURCE_DIR}/gen_code/test.cpp)
elseif(${DEVICE_BACKEND} STREQUAL "hip")
    set(GPU_TARGET_SOURCE_FILES common/test_drivers/simple_driver.cpp include/gemmforge_aux.cpp gen_code/kernels.cpp)
    set(GEN_COPY_PRODUCTS ${CMAKE_SOURCE_DIR}/gen_code/kernels.cpp ${CMAKE_SOURCE_DIR}/gen_code/kernels.h ${CMAKE_SOURCE_DIR}/gen_code/test.cpp)
elseif((${DEVICE_BACKEND} STREQUAL "oneapi") OR (${DEVICE_BACKEND} STREQUAL "hipsycl"))
    set(GPU_TARGET_SOURCE_FILES common/test_drivers/simple_driver.cpp include/gemmforge_aux_sycl.cpp gen_code/kernels.cpp )
    set(GEN_COPY_PRODUCTS ${CMAKE_SOURCE_DIR}/gen_code/kernels.cpp ${CMAKE_SOURCE_DIR}/gen_code/kernels.h ${CMAKE_SOURCE_DIR}/gen_code/test.cpp)
endif()

set(GEN_COPY_FILES ${CMAKE_BINARY_DIR}/${TEST_SUITE})

include(../../cmake/setupGpuTarget.cmake)
target_include_directories(${TARGET} PRIVATE ../../submodules/Device)
target_link_libraries(${TARGET} PRIVATE ${GPU_TARGET} device -lpthread)
include(../../cmake/generateCode.cmake)

