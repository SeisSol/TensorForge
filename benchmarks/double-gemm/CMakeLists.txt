cmake_minimum_required(VERSION 3.5)
project(double-gemm)

set(TARGET ${CMAKE_PROJECT_NAME})
set(TARGET_SOURCE_FILES main.cpp
                        common/gemm.cpp
                        main.cpp
                        common/gemm.cpp
                        common/aux.cpp
                        common/test_drivers/simple_comparator.cpp)
set(TARGET_INCLUDE_DIRS gen_code
                        common
                        common/test_drivers
                        include)

set(GPU_TARGET gpu_part)
set(GPU_TARGET_INCLUDE_DIRS ../../submodules/Device
                            include
                            common
                            common/test_drivers
                            gencode)

if(${DEVICE_BACKEND} STREQUAL "CUDA")
    set(GPU_TARGET_SOURCE_FILES common/test_drivers/simple_driver.cpp gen_code/kernels.cu include/gemmgen_aux.cu)
    set(GEN_COPY_PRODUCTS ${CMAKE_SOURCE_DIR}/gen_code/kernels.cu ${CMAKE_SOURCE_DIR}/gen_code/kernels.h)
elseif(${DEVICE_BACKEND} STREQUAL "HIP")
    set(GPU_TARGET_SOURCE_FILES common/test_drivers/simple_driver.cpp gen_code/kernels.cpp include/gemmgen_aux.cpp)
    set(GEN_COPY_PRODUCTS ${CMAKE_SOURCE_DIR}/gen_code/kernels.cpp ${CMAKE_SOURCE_DIR}/gen_code/kernels.h)
elseif((${DEVICE_BACKEND} STREQUAL "ONEAPI") OR (${DEVICE_BACKEND} STREQUAL "HIPSYCL"))
    set(GPU_TARGET_SOURCE_FILES common/test_drivers/simple_driver.cpp gen_code/kernels.cpp include/gemmgen_aux_sycl.cpp)
    set(GEN_COPY_PRODUCTS ${CMAKE_SOURCE_DIR}/gen_code/kernels.cpp ${CMAKE_SOURCE_DIR}/gen_code/kernels.h)
endif()

set(GEN_COPY_FILES ${CMAKE_SOURCE_DIR}/params.yaml ${CMAKE_SOURCE_DIR}/config.yaml)

include(../../cmake/setupTarget.cmake)
include(../../cmake/setupGpuTarget.cmake)
include(../../cmake/addYaml.cmake)
include(../../cmake/linkTargetGpuAndDevice.cmake)
include(../../cmake/generateCode.cmake)
include(../../cmake/enableTesting.cmake)