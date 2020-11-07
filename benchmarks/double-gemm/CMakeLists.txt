cmake_minimum_required(VERSION 3.5)
project(double-gemm)

set(REAL_SIZE "4" CACHE STRING "size of the floating point data type")
set_property(CACHE REAL_SIZE PROPERTY STRINGS "8" "4")

set(SM_ARCH "sm_61" CACHE STRING "size of the floating point data type")
set_property(CACHE SM_ARCH PROPERTY STRINGS "sm_60" "sm_61" "sm_70" "sm_71")

set(REFERENCE_IMPL "GEMMGEN" CACHE STRING "size of the floating point data type")
set_property(CACHE REFERENCE_IMPL PROPERTY STRINGS "GEMMGEN" "OPENBLAS")

add_executable(${CMAKE_PROJECT_NAME}
               main.cpp
               common/gemm.cpp
               common/aux.cpp
               common/test_drivers/simple_comparator.cpp)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC gen_code
                                                 common
                                                 common/test_drivers
                                                 include)

target_compile_options(${CMAKE_PROJECT_NAME} PUBLIC "-std=c++11")
target_compile_definitions(${CMAKE_PROJECT_NAME} PUBLIC "REAL_SIZE=${REAL_SIZE}")


if (${REFERENCE_IMPL} STREQUAL "OPENBLAS")
    find_package(OpenBLAS REQUIRED)
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${OpenBLAS_INCLUDE_DIRS})
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${OpenBLAS_LIBRARIES})
endif()
target_compile_definitions(${CMAKE_PROJECT_NAME} PUBLIC "CONCRETE_CPU_BACKEND=${REFERENCE_IMPL}")


find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        -std=c++11;
        -arch=${SM_ARCH};
#        -restrict;
        -res-usage;
        -O3;
#        -g;
        -lineinfo;
        -Xptxas -v;
#        -maxrregcount 64;
        -DREAL_SIZE=${REAL_SIZE})

cuda_add_library(gpu_part STATIC common/test_drivers/simple_driver.cu
                                 include/gemmgen_aux.cu
                                 gen_code/kernels.cu)
target_include_directories(gpu_part PRIVATE include
                                            common
                                            common/test_drivers
                                            gen_code)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE gpu_part)

find_package(yaml-cpp REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${YAML_CPP_LIBRARIES})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${YAML_CPP_INCLUDE_DIR})

add_custom_target(copy_params ALL
                  cmake -E copy ${CMAKE_SOURCE_DIR}/params.yaml ${CMAKE_BINARY_DIR}
                  COMMAND
                    cmake -E copy ${CMAKE_SOURCE_DIR}/config.yaml ${CMAKE_BINARY_DIR}
                  COMMENT
                    "copying the parameter file"
                  WORKING_DIRECTORY
                    ${CMAKE_SOURCE_DIR})


add_custom_target(gpu_generator ALL
                  python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate.py --realsize=${REAL_SIZE}
                  BYPRODUCTS
                    ${CMAKE_SOURCE_DIR}/gen_code/kernels.cu
                    ${CMAKE_SOURCE_DIR}/gen_code/kernels.h
                  COMMENT
                    "generating compute kernels"
                  WORKING_DIRECTORY
                    ${CMAKE_SOURCE_DIR})

add_dependencies(gpu_part gpu_generator)
add_dependencies(${CMAKE_PROJECT_NAME} copy_params)