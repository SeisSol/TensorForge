cmake_minimum_required(VERSION 3.5)
project(initializer-tests)

set(TARGET ${CMAKE_PROJECT_NAME})
set(TARGET_SOURCE_FILES main.cpp
        common/aux.cpp
        common/initialize.cpp
        common/test_drivers/simple_comparator.cpp
        gen_code/test.cpp)
set(TARGET_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}
        gen_code
        common
        common/test_drivers
        include)

set(REFERENCE_IMPL "GEMMFORGE" CACHE STRING "size of the floating point data type")
set_property(CACHE REFERENCE_IMPL PROPERTY STRINGS "GEMMFORGE" "OPENBLAS")

include(../../cmake/setupTarget.cmake)

set(TEST_SUITE "" CACHE STRING "relative path to test suite spec file")
if ("${TEST_SUITE}" STREQUAL "")
    message(FATAL_ERROR "NO TEST SUITE PROVIDED")
endif()

include(../../cmake/addGTest.cmake)
include(../../cmake/addYaml.cmake)

set(GPU_TARGET gpu_part)
set(GPU_TARGET_INCLUDE_DIRS include
        common
        common/test_drivers
        ../../submodules/Device)

if(${DEVICE_BACKEND} STREQUAL "CUDA")
    set(GPU_TARGET_SOURCE_FILES common/test_drivers/simple_driver.cpp include/gemmforge_aux.cu gen_code/kernels.cu)
    set(GEN_COPY_PRODUCTS ${CMAKE_SOURCE_DIR}/gen_code/kernels.cu ${CMAKE_SOURCE_DIR}/gen_code/kernels.h ${CMAKE_SOURCE_DIR}/gen_code/test.cpp)
elseif(${DEVICE_BACKEND} STREQUAL "HIP")
    set(GPU_TARGET_SOURCE_FILES common/test_drivers/simple_driver.cpp include/gemmforge_aux.cpp gen_code/kernels.cpp)
    set(GEN_COPY_PRODUCTS ${CMAKE_SOURCE_DIR}/gen_code/kernels.cpp ${CMAKE_SOURCE_DIR}/gen_code/kernels.h ${CMAKE_SOURCE_DIR}/gen_code/test.cpp)
elseif((${DEVICE_BACKEND} STREQUAL "ONEAPI") OR (${DEVICE_BACKEND} STREQUAL "HIPSYCL"))
    set(GPU_TARGET_SOURCE_FILES common/test_drivers/simple_driver.cpp include/gemmforge_aux_sycl.cpp gen_code/kernels.cpp )
    set(GEN_COPY_PRODUCTS ${CMAKE_SOURCE_DIR}/gen_code/kernels.cpp ${CMAKE_SOURCE_DIR}/gen_code/kernels.h ${CMAKE_SOURCE_DIR}/gen_code/test.cpp)
endif()

set(GEN_COPY_FILES ${CMAKE_BINARY_DIR}/${TEST_SUITE})

include(../../cmake/setupGpuTarget.cmake)
include(../../cmake/linkTargetGpuAndDevice.cmake)
include(../../cmake/generateCode.cmake)

