cmake_minimum_required(VERSION 3.5)
project(simple-bandwidth-test)

set(TARGET ${CMAKE_PROJECT_NAME})
set(TARGET_SOURCE_FILES global.cpp)
set(TARGET_INCLUDE_DIRS ../../submodules/Device
        include
        common
        common/test_drivers)

include(../../cmake/setupTarget.cmake)

set(GPU_TARGET gpu_part)
set(GPU_TARGET_INCLUDE_DIRS ../../submodules/Device)

if(${DEVICE_BACKEND} STREQUAL "CUDA")
    set(GPU_TARGET_SOURCE_FILES cuda_kernel.cu)
elseif(${DEVICE_BACKEND} STREQUAL "HIP")
    set(GPU_TARGET_SOURCE_FILES hip_kernel.cpp)
elseif((${DEVICE_BACKEND} STREQUAL "ONEAPI") OR (${DEVICE_BACKEND} STREQUAL "HIPSYCL"))
    set(GPU_TARGET_SOURCE_FILES sycl_kernel.cpp)
endif()

include(../../cmake/setupGpuTarget.cmake)
include(../../cmake/addYaml.cmake)
include(../../cmake/linkTargetGpuAndDevice.cmake)
include(../../cmake/enableTesting.cmake)