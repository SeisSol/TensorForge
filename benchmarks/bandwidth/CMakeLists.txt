cmake_minimum_required(VERSION 3.5)
project(simple-bandwidth-test)

include(../../cmake/setupOptions.cmake)

set(TARGET ${CMAKE_PROJECT_NAME})
set(TARGET_SOURCE_FILES global.cpp)

set(TARGET_INCLUDE_DIRS include
        common
        common/test_drivers
        ../../submodules/Device)

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

set(GPU_TARGET gpu_part)
set(GPU_TARGET_INCLUDE_DIRS ../../submodules/Device)

if(${DEVICE_BACKEND} STREQUAL "cuda")
    set(GPU_TARGET_SOURCE_FILES cuda_kernel.cu)
elseif(${DEVICE_BACKEND} STREQUAL "hip")
    set(GPU_TARGET_SOURCE_FILES hip_kernel.cpp)
elseif((${DEVICE_BACKEND} STREQUAL "oneapi") OR (${DEVICE_BACKEND} STREQUAL "hipsycl"))
    set(GPU_TARGET_SOURCE_FILES sycl_kernel.cpp)
else()
    message(FATAL_ERROR "unknown backend, given: ${DEVICE_BACKEND}")
endif()

#add Yaml
find_package(yaml-cpp REQUIRED)
target_include_directories(${TARGET} PRIVATE ${YAML_CPP_INCLUDE_DIR})
target_link_libraries(${TARGET} PRIVATE ${YAML_CPP_LIBRARIES})

include(../../cmake/setupGpuTarget.cmake)

target_link_libraries(${TARGET} PRIVATE ${GPU_TARGET} device -lpthread)

enable_testing()
add_test(NAME test COMMAND ${CMAKE_PROJECT_NAME})