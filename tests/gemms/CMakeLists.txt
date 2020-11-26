cmake_minimum_required(VERSION 3.5)
project(gemm-tests)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Og")

set(REAL_SIZE "4" CACHE STRING "size of the floating point data type")
set_property(CACHE REAL_SIZE PROPERTY STRINGS "8" "4")

set(SM_ARCH "sm_61" CACHE STRING "size of the floating point data type")
set_property(CACHE SM_ARCH PROPERTY STRINGS "sm_60" "sm_61" "sm_70" "sm_71")

set(TEST_SUITE "" CACHE STRING "relative path to test suite spec file")
if ("${TEST_SUITE}" STREQUAL "")
    message(FATAL_ERROR "NO TEST SUITE PROVIDED")
endif()

set(REFERENCE_IMPL "GEMMGEN" CACHE STRING "size of the floating point data type")
set_property(CACHE REFERENCE_IMPL PROPERTY STRINGS "GEMMGEN" "OPENBLAS")

find_package(GTest REQUIRED)
add_executable(${CMAKE_PROJECT_NAME} main.cpp
                                     common/aux.cpp
                                     common/gemm.cpp
                                     common/test_drivers/simple_comparator.cpp
                                     gen_code/test.cpp)


target_link_libraries(${CMAKE_PROJECT_NAME} ${GTEST_BOTH_LIBRARIES})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${GTEST_INCLUDE_DIR})

if (${REFERENCE_IMPL} STREQUAL "OPENBLAS")
    find_package(OpenBLAS REQUIRED)
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${OpenBLAS_INCLUDE_DIRS})
    target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenBLAS_LIBRARIES})
endif()
target_compile_definitions(${CMAKE_PROJECT_NAME} PUBLIC "CONCRETE_CPU_BACKEND=${REFERENCE_IMPL}")


find_package(yaml-cpp REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} ${YAML_CPP_LIBRARIES})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${YAML_CPP_INCLUDE_DIR})

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}
                                                         gen_code
                                                         common
                                                         common/test_drivers
                                                         include)

target_compile_options(${CMAKE_PROJECT_NAME} PUBLIC "-std=c++11")
target_compile_definitions(${CMAKE_PROJECT_NAME} PUBLIC "REAL_SIZE=${REAL_SIZE}")


find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        -std=c++11;
        -arch=${SM_ARCH};
        #        -restrict;
        -res-usage;
        -O3;
        #        -g;
        #        -lineinfo;
        -Xptxas -v;
        #-maxrregcount 64;
        -DREAL_SIZE=${REAL_SIZE})

cuda_add_library(gpu_part STATIC common/test_drivers/simple_driver.cu
                                 include/gemmgen_aux.cu
                                 gen_code/kernels.cu)

target_include_directories(gpu_part PRIVATE include
                                            common
                                            common/test_drivers)

target_link_libraries(${CMAKE_PROJECT_NAME} gpu_part)

add_custom_target(gpu_generator ALL
                  python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate.py
                            --specfile=${CMAKE_BINARY_DIR}/${TEST_SUITE}
                            --realsize=${REAL_SIZE}
                            --sub_arch=${SM_ARCH}
                            --arch="nvidia"
                  BYPRODUCTS
                    ${CMAKE_SOURCE_DIR}/gen_code/kernels.cu
                    ${CMAKE_SOURCE_DIR}/gen_code/kernels.h
                    ${CMAKE_SOURCE_DIR}/gen_code/test.cpp
                  COMMENT
                    "generating compute kernels"
                  WORKING_DIRECTORY
                    ${CMAKE_SOURCE_DIR})

add_custom_target(copy_params ALL
                  cmake -E copy ${CMAKE_BINARY_DIR}/${TEST_SUITE} ${CMAKE_BINARY_DIR}
                  COMMENT
                  "copying the parameter file"
                  WORKING_DIRECTORY
                  ${CMAKE_SOURCE_DIR})

add_dependencies(gpu_generator copy_params)
add_dependencies(gpu_part gpu_generator)